import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, num_epochs,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, 
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)
from torch.utils.data import Dataset, DataLoader
from modelCNN import SmallCNN  # Assuming this is your SimpleTimeCNN implementation
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTimeCNN with customizable loss and target transformation')
    parser.add_argument('--loss_type', type=str, default='L2', choices=['L1', 'L2'],
                        help='Loss function: L1 (MAE) or L2 (MSE)')
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'normalize', 'log'],
                        help='Target transformation: none, normalize (zero-centered, std-scaled), or log')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='Number of epochs to train')
    return parser.parse_args()

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
    """
    Create a balanced dataset by binning OC values and resampling
    """
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    validation_indices = []
    training_dfs = []
    
    for bin_idx in range(len(bin_counts)):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) >= 4:
            val_samples = bin_data.sample(n=min(13, len(bin_data)))
            validation_indices.extend(val_samples.index)
            train_samples = bin_data.drop(val_samples.index)
            if len(train_samples) > 0:
                if len(train_samples) < min_samples:
                    resampled = train_samples.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(train_samples)
    
    if not training_dfs or not validation_indices:
        raise ValueError("No training or validation data available after binning")
    
    training_df = pd.concat(training_dfs).drop('bin', axis=1)
    validation_df = df.loc[validation_indices].drop('bin', axis=1)
    
    return training_df, validation_df

def get_target_transform(target_transform_type, train_targets):
    """Define target transformation based on type and compute stats if needed."""
    if target_transform_type == 'normalize':
        mean = train_targets.mean()
        std = train_targets.std()
        if std == 0:
            std = 1.0  # Avoid division by zero
        def transform(x):
            return (x - mean) / std
        def inverse_transform(x):
            return x * std + mean
        return transform, inverse_transform, {"mean": mean.item(), "std": std.item()}
    elif target_transform_type == 'log':
        def transform(x):
            return torch.log1p(x)  # log(1 + x) to handle zero/negative gracefully
        def inverse_transform(x):
            return torch.expm1(x)  # exp(x) - 1
        return transform, inverse_transform, {}
    else:  # 'none'
        def transform(x):
            return x
        def inverse_transform(x):
            return x
        return transform, inverse_transform, {}

def train_model(model, train_loader, val_loader, num_epochs, accelerator, loss_type='L1', target_transform_type='none'):
    # Select loss function
    criterion = nn.L1Loss() if loss_type == 'L1' else nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare with Accelerate
    train_loader, val_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, model, optimizer
    )

    # Compute transformation stats from training data
    all_train_targets = []
    for _, _, _, targets in train_loader:
        all_train_targets.append(targets)
    all_train_targets = torch.cat(all_train_targets).float()
    transform, inverse_transform, transform_params = get_target_transform(target_transform_type, all_train_targets)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()
            transformed_targets = transform(targets)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, transformed_targets)
            accelerator.backward(loss)
            optimizer.step()
            
            running_loss += loss.item()
            
            if accelerator.is_main_process:
                wandb.log({
                    'train_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })

        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets_list = []
        
        with torch.no_grad():
            for longitudes, latitudes, features, targets in val_loader:
                features = features.to(accelerator.device)
                targets = targets.to(accelerator.device).float()
                transformed_targets = transform(targets)
                outputs = model(features)
                loss = criterion(outputs, transformed_targets)
                val_loss += loss.item()
                # Store inverse-transformed outputs and original targets for metrics
                val_outputs.extend(inverse_transform(outputs).cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        
        # Calculate metrics on original scale
        val_outputs = np.array(val_outputs)
        val_targets_list = np.array(val_targets_list)
        correlation = np.corrcoef(val_outputs, val_targets_list)[0, 1] if len(val_outputs) > 1 else 0.0
        r_squared = correlation ** 2
        mse = np.mean((val_outputs - val_targets_list) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(val_outputs - val_targets_list))

        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_avg': train_loss,
                'val_loss': val_loss,
                'correlation': correlation,
                'r_squared': r_squared,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            })

            if val_loss < best_val_loss and epoch % 5 == 0:
                best_val_loss = val_loss
                wandb.run.summary['best_val_loss'] = best_val_loss
                accelerator.save_state(f'simpletimecnn_checkpoint_epoch_{epoch+1}.pth')
                wandb.save(f'simpletimecnn_checkpoint_epoch_{epoch+1}.pth')
        
        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        accelerator.print(f'Validation Loss: {val_loss:.4f}\n')

    return model, val_outputs, val_targets_list

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize Accelerate
    accelerator = Accelerator()
    
    # Initialize wandb
    wandb.init(
        project="socmapping-SimpleTimeCNN",
        config={
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "epochs": args.num_epochs,
            "batch_size": 256,
            "learning_rate": 0.001,
            "input_channels": 6,
            "loss_type": args.loss_type,
            "target_transform": args.target_transform
        }
    )

    # Data preparation
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    def flatten_paths(path_list):
        flattened = []
        for item in path_list:
            if isinstance(item, list):
                flattened.extend(flatten_paths(item))
            else:
                flattened.append(item)
        return flattened

    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    # Create balanced datasets
    train_df, val_df = create_balanced_dataset(df)
    
    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df)
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")

    # Create datasets and dataloaders
    train_dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize model
    model = SmallCNN(input_channels=6)  # Assuming SmallCNN is your SimpleTimeCNN
    
    if accelerator.is_main_process:
        wandb.run.summary["model_parameters"] = model.count_parameters()
        print(f"Model parameters: {model.count_parameters()}")

    # Train model with selected options
    model, val_outputs, val_targets = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.num_epochs, 
        accelerator=accelerator,
        loss_type=args.loss_type,
        target_transform_type=args.target_transform
    )

    # Save final model
    if accelerator.is_main_process:
        final_model_path = (f'simpletimecnn_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                            f'TIME_END_{TIME_END}_LOSS_{args.loss_type}_TRANSFORM_{args.target_transform}.pth')
        accelerator.save(model.state_dict(), final_model_path)
        wandb.save(final_model_path)
        print("Model trained and saved successfully!")

    wandb.finish()

    # bst moel with none/l2