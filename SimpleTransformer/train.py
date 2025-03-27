import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears , NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, num_epochs, NUM_HEADS, NUM_LAYERS, 
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from modelSimpleTransformer import SimpleTransformer
import argparse


def composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L1 and scaled chi-squared loss"""
    errors = targets - outputs
    l1_loss = torch.mean(torch.abs(errors))
    
    squared_errors = errors ** 2
    chi2_unscaled = (1/4) * squared_errors * torch.exp(-squared_errors / (2 * sigma))
    chi2_unscaled_mean = torch.mean(chi2_unscaled)
    
    chi2_unscaled_mean = torch.clamp(chi2_unscaled_mean, min=1e-8)
    scale_factor = l1_loss / chi2_unscaled_mean
    chi2_scaled = scale_factor * chi2_unscaled_mean
    
    return alpha * l1_loss + (1 - alpha) * chi2_scaled

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTransformer model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
    return parser.parse_args()

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
    """
    Create a balanced dataset by binning OC values and resampling.
    If use_validation is True, splits into training and validation sets.
    If use_validation is False, returns only a balanced training set.
    """
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
                val_samples = bin_data.sample(n=min(8, len(bin_data)))
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

def train_model(model, train_loader, val_loader, num_epochs=num_epochs, accelerator=None, lr=0.001, 
                loss_alpha=0.6, min_r2=0.5, use_validation=True):
    criterion = lambda outputs, targets: composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=loss_alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare with None for val_loader if not using validation
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    if use_validation and val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    best_r2 = -float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

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
        
        if use_validation and val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets_list = []
            
            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    features = features.to(accelerator.device)
                    targets = targets.to(accelerator.device).float()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_outputs.extend(outputs.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
            
            val_loss = val_loss / len(val_loader)
            val_outputs = np.array(val_outputs)
            val_targets_list = np.array(val_targets_list)
            correlation = np.corrcoef(val_outputs, val_targets_list)[0, 1]
            r_squared = correlation ** 2
            mse = np.mean((val_outputs - val_targets_list) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(val_outputs - val_targets_list))
        else:
            # Default values when no validation
            val_loss = 1.0
            val_outputs = np.array([])
            val_targets_list = np.array([])
            correlation = 1.0
            r_squared = 1.0
            mse = 1.0
            rmse = 1.0
            mae = 1.0

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

            # Save model if it has the best R² and meets minimum threshold (only with validation)
            if use_validation and r_squared > best_r2 and r_squared >= min_r2:
                best_r2 = r_squared
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2
            elif not use_validation and epoch == num_epochs - 1:
                # Save last model when no validation
                best_r2 = 1.0
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2
        
        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        accelerator.print(f'Validation Loss: {val_loss:.4f}')
        if use_validation:
            accelerator.print(f'R²: {r_squared:.4f}\n')

    return model, val_outputs, val_targets_list, best_model_state, best_r2

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    
    wandb.init(
        project="socmapping-SimpleTransformer",
        config={
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands": len(bands_list_order),
            "epochs": num_epochs,
            "batch_size": 256,
            "learning_rate": args.lr,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout_rate": 0.3,
            "loss_alpha": args.loss_alpha,
            "use_validation": args.use_validation
        }
    )

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

    if args.use_validation:
        train_df, val_df = create_balanced_dataset(df, use_validation=True)
        train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
        # Print lengths
        if accelerator.is_main_process:
            print(f"Length of train_dataset: {len(train_dataset)}")
            print(f"Length of val_dataset: {len(val_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    else:
        train_df = create_balanced_dataset(df, use_validation=False)
        train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        # Print length
        if accelerator.is_main_process:
            print(f"Length of train_dataset: {len(train_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = None

    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df) if args.use_validation else 0

    for batch in train_loader:
        _, _, first_batch, _ = batch
        break
    first_batch_size = first_batch.shape
    if accelerator.is_main_process:
        print("Size of the first batch:", first_batch_size)

    model = SimpleTransformer(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=0.3
    )
    
    if accelerator.is_main_process:
        wandb.run.summary["model_parameters"] = model.count_parameters()

    # Train model and get best state
    model, val_outputs, val_targets, best_model_state, best_r2 = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        accelerator=accelerator,
        lr=args.lr,
        loss_alpha=args.loss_alpha,
        min_r2=0.5,
        use_validation=args.use_validation
    )

    if accelerator.is_main_process and best_model_state is not None:
        final_model_path = f'transformer_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}_R2_{best_r2:.4f}.pth'
        accelerator.save(best_model_state, final_model_path)
        wandb.save(final_model_path)
        print(f"Model with best R² ({best_r2:.4f}) saved successfully at: {final_model_path}")
    elif accelerator.is_main_process:
        print("No model saved - R² threshold not met")

    wandb.finish()