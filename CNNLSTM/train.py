# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse

# Assuming these are imported from your existing files
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (TIME_BEGINNING, TIME_END, MAX_OC, seasons, years_padded,
                    num_epochs, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                    DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                    MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from models import RefittedCovLSTM

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train RefittedCovLSTM model')
    parser.add_argument('--use_validation', action='store_true', default=True,
                        help='Whether to use a separate validation set')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, 
                        help='Number of epochs to train')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['l1', 'mse'], 
                        help='Loss function: l1 (MAE) or mse (MSE)')
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'log'], 
                        help='Target transformation: none or log')
    return parser.parse_args()

# Function to create balanced dataset (unchanged from original)
def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)

    training_dfs = []
    if use_validation:
        validation_indices = []
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
        print('Size of the training set:   ' ,len(training_df))
        print('Size of the validation set:   ' ,len(validation_df))
        return training_df, validation_df
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        if not training_dfs:
            raise ValueError("No training data available after binning")
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        return training_df

# Modified train_model function
def train_model(model, train_loader, val_loader, num_epochs, accelerator, 
                loss_type='mse', target_transform='none'):
    # Set loss function based on argument
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare with Accelerate
    train_loader, val_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, model, optimizer
    )

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()
            # Apply target transformation if specified
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)  # Small constant to avoid log(0)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
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
        
        # Validation phase (only if val_loader is provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets_list = []
            
            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    features = features.to(accelerator.device)
                    targets = targets.to(accelerator.device).float()
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_outputs.extend(outputs.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
            
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            val_outputs = np.array(val_outputs)
            val_targets_list = np.array(val_targets_list)
            if len(val_outputs) > 0:
                correlation = np.corrcoef(val_outputs, val_targets_list)[0, 1]
                r_squared = correlation ** 2
                mse = np.mean((val_outputs - val_targets_list) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(val_outputs - val_targets_list))
            else:
                correlation = None
                r_squared = None
                mse = None
                rmse = None
                mae = None
        else:
            val_loss = None
            correlation = None
            r_squared = None
            mse = None
            rmse = None
            mae = None

        # Logging metrics
        if accelerator.is_main_process:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss_avg': train_loss,
            }
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            if correlation is not None:
                log_dict['correlation'] = correlation
                log_dict['r_squared'] = r_squared
                log_dict['mse'] = mse
                log_dict['rmse'] = rmse
                log_dict['mae'] = mae
            wandb.log(log_dict)

            # Model saving logic
            if val_loader is not None and val_loss is not None and val_loss < best_val_loss and epoch % 5 == 0:
                best_val_loss = val_loss
                wandb.run.summary['best_val_loss'] = best_val_loss
                accelerator.save_state(f'model_checkpoint_epoch_{epoch+1}.pth')
                wandb.save(f'model_checkpoint_epoch_{epoch+1}.pth')
            elif val_loader is None and epoch % 5 == 0:
                accelerator.save_state(f'model_checkpoint_epoch_{epoch+1}.pth')
                wandb.save(f'model_checkpoint_epoch_{epoch+1}.pth')
        
        # Print progress
        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        if val_loss is not None:
            accelerator.print(f'Validation Loss: {val_loss:.4f}\n')

    return model, val_outputs if val_loader is not None else None, val_targets_list if val_loader is not None else None

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize Accelerate
    accelerator = Accelerator()
    
    # Initialize wandb with configuration
    wandb.init(
        project="socmapping-CNNLSTM",
        config={
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands": len(bands_list_order),
            "epochs": args.num_epochs,
            "batch_size": 256,
            "learning_rate": 0.001,
            "lstm_hidden_size": 128,
            "num_layers": 2,
            "dropout_rate": 0.25,
            "loss_type": args.loss_type,
            "target_transform": args.target_transform,
            "use_validation": args.use_validation
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
    
    # Create balanced datasets using create_balanced_dataset
    if args.use_validation:
        train_df, val_df = create_balanced_dataset(df, use_validation=True)
    else:
        train_df = create_balanced_dataset(df, use_validation=False)
        val_df = None

    # Create datasets
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    if val_df is not None:
        val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
    else:
        val_dataset = None

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False) if val_dataset is not None else None
    
    # Get batch size info
    for batch in train_loader:
        _, _, first_batch, _ = batch
        break
    first_batch_size = first_batch.shape
    if accelerator.is_main_process:
        print("Size of the first batch:", first_batch_size)

    # Initialize model
    model = RefittedCovLSTM(
        num_channels=len(bands_list_order),  # e.g., 6
        lstm_input_size=128,  # Fixed by CNN output
        lstm_hidden_size=128,
        num_layers=2,
        dropout=0.25
    )
    
    # Train model
    model, val_outputs, val_targets = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.num_epochs, 
        accelerator=accelerator,
        loss_type=args.loss_type,
        target_transform=args.target_transform
    )

    # Save final model
    if accelerator.is_main_process:
        final_model_path = f'cnnlstm_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
        accelerator.save(model.state_dict(), final_model_path)
        wandb.save(final_model_path)
        print("Model trained and saved successfully!")

    wandb.finish()