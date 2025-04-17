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


def load_data():
    """
    Load the training and validation DataFrames from Parquet files.
    Returns:
        validation_df: DataFrame containing validation data
        training_df: DataFrame containing training data
    """
    # Define the absolute paths to the Parquet files
    validation_path = '/lustre/home/vfourel/SOCProject/SOCmapping/balancedDataset/output/final_validation_df.parquet'
    training_path = '/lustre/home/vfourel/SOCProject/SOCmapping/balancedDataset/output/final_training_df.parquet'

    # Convert to Path objects for robustness
    validation_file = Path(validation_path)
    training_file = Path(training_path)

    # Check if files exist
    if not validation_file.exists():
        raise FileNotFoundError(f"Validation file not found at {validation_file}")
    if not training_file.exists():
        raise FileNotFoundError(f"Training file not found at {training_file}")

    # Load the DataFrames
    validation_df = pd.read_parquet(validation_file)
    training_df = pd.read_parquet(training_file)
    training_df = resample_training_df(training_df)
    return  training_df,validation_df

def resample_training_df(training_df, num_bins=128, target_fraction=3/4):
    """
    Resample the training DataFrame's 'OC' values into 128 bins, ensuring each bin has
    at least 3/4 of the entries of the bin with the highest count.
    Args:
        training_df: DataFrame containing training data
        num_bins: Number of bins for OC values (default: 128)
        target_fraction: Fraction of max bin count for resampling (default: 3/4)
    Returns:
        resampled_df: Resampled training DataFrame
    """
    # Create bins for OC values
    oc_values = training_df['OC'].dropna()
    bins = pd.qcut(oc_values, q=num_bins, duplicates='drop')
    
    # Count entries per bin
    bin_counts = bins.value_counts().sort_index()
    max_count = bin_counts.max()
    target_count = int(max_count * target_fraction)
    
    print(f"Max bin count: {max_count}")
    print(f"Target count per bin (at least): {target_count}")
    
    # Initialize list to collect resampled data
    resampled_dfs = []
    
    # Process each bin
    for bin_label in bin_counts.index:
        # Get points in this bin
        bin_mask = pd.cut(training_df['OC'], bins=bins.cat.categories) == bin_label
        bin_df = training_df[bin_mask]
        
        # If bin has fewer than target_count, oversample with replacement
        if len(bin_df) < target_count:
            additional_samples = target_count - len(bin_df)
            sampled_df = bin_df.sample(n=additional_samples, replace=True, random_state=42)
            resampled_dfs.append(pd.concat([bin_df, sampled_df]))
        else:
            resampled_dfs.append(bin_df)
    
    # Combine all resampled bins
    resampled_df = pd.concat(resampled_dfs, ignore_index=True)
    
    # Verify new bin counts
    new_bins = pd.qcut(resampled_df['OC'], q=num_bins, duplicates='drop')
    new_bin_counts = new_bins.value_counts().sort_index()
    
    print("\nBin counts before resampling:")
    print(bin_counts)
    print("\nBin counts after resampling:")
    print(new_bin_counts)
    
    return resampled_df



# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train RefittedCovLSTM model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['composite_l1', 'l1', 'mse', 'composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
    parser.add_argument('--load_data',default = True,help='We can use pre-computed validation and training sets.')

    return parser.parse_args()

# Function to create balanced dataset
def create_balanced_dataset(df, use_validation=True, n_bins=128, min_ratio=3/4):
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
        print('Size of the training set:   ', len(training_df))
        print('Size of the validation set:   ', len(validation_df))
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
        return training_df, None  # Return None for validation_df when no validation

# Define the composite loss functions
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

def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L2 and scaled chi-squared loss"""
    errors = targets - outputs
    l2_loss = torch.mean(errors ** 2)
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))
    chi2_loss = torch.clamp(chi2_loss, min=1e-8)
    scale_factor = l2_loss / chi2_loss
    chi2_scaled = scale_factor * chi2_loss
    return alpha * l2_loss + (1 - alpha) * chi2_scaled

# Modified train_model function
def train_model(args, model, train_loader, val_loader, num_epochs, accelerator, 
                loss_type='mse', target_transform='none'):
    if target_transform == 'normalize':
        all_targets = []
        for _, _, _, targets in train_loader:
            all_targets.append(targets)
        all_targets = torch.cat(all_targets)
        target_mean = all_targets.mean().item()
        target_std = all_targets.std().item()
        if accelerator.is_main_process:
            print(f"Target mean: {target_mean}, Target std: {target_std}")
    else:
        target_mean, target_std = 0.0, 1.0  # No normalization applied

    # Define loss function based on loss_type
    if loss_type == 'composite_l1':
        criterion = lambda outputs, targets: composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=args.loss_alpha)
    elif loss_type == 'composite_l2':
        criterion = lambda outputs, targets: composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=args.loss_alpha)
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Prepare with Accelerate
    train_loader, model, optimizer = accelerator.prepare(
        train_loader, model, optimizer
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    best_r_squared = -float('inf') if args.use_validation else 1.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)

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
        if args.use_validation and val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_outputs_list = []
            val_targets_list = []
            
            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    features = features.to(accelerator.device)
                    targets = targets.to(accelerator.device).float()
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_outputs_list.append(outputs.cpu())
                    val_targets_list.append(targets.cpu())
            
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            
            # Concatenate outputs and targets from all batches
            val_outputs = torch.cat(val_outputs_list, dim=0).numpy()
            val_targets = torch.cat(val_targets_list, dim=0).numpy()
            
            # Gather data from all processes
            val_outputs_all = torch.from_numpy(val_outputs).to(accelerator.device)
            val_targets_all = torch.from_numpy(val_targets).to(accelerator.device)
            val_outputs_all = accelerator.gather(val_outputs_all).cpu().numpy()
            val_targets_all = accelerator.gather(val_targets_all).cpu().numpy()
            
            if accelerator.is_main_process:
                # Inverse transform to original scale
                if target_transform == 'log':
                    original_val_outputs = np.exp(val_outputs_all)
                    original_val_targets = np.exp(val_targets_all)
                elif target_transform == 'normalize':
                    original_val_outputs = val_outputs_all * target_std + target_mean
                    original_val_targets = val_targets_all * target_std + target_mean
                else:
                    original_val_outputs = val_outputs_all
                    original_val_targets = val_targets_all
                
                # Compute metrics on original scale
                if len(original_val_outputs) > 1 and np.std(original_val_outputs) > 1e-6 and np.std(original_val_targets) > 1e-6:
                    correlation = np.corrcoef(original_val_outputs, original_val_targets)[0, 1]
                    r_squared = correlation ** 2
                    mse = np.mean((original_val_outputs - original_val_targets) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(original_val_outputs - original_val_targets))
                    iqr = np.percentile(original_val_targets, 75) - np.percentile(original_val_targets, 25)
                    rpiq = iqr / rmse if rmse > 0 else float('inf')
                else:
                    correlation = 0.0
                    r_squared = 0.0
                    mse = float('nan')
                    rmse = float('nan')
                    mae = float('nan')
                    rpiq = float('nan')
            else:
                correlation = float('nan')
                r_squared = float('nan')
                mse = float('nan')
                rmse = float('nan')
                mae = float('nan')
                rpiq = float('nan')
            
            # Update best model based on R²
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if accelerator.is_main_process:
                    wandb.run.summary['best_r_squared'] = best_r_squared
        else:
            # No validation, update model state and keep R² as 1.0
            best_r_squared = 1.0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if accelerator.is_main_process:
                wandb.run.summary['best_r_squared'] = best_r_squared
            val_loss = float('nan')
            correlation = float('nan')
            r_squared = 1.0
            mse = float('nan')
            rmse = float('nan')
            mae = float('nan')
            rpiq = float('nan')

        # Logging metrics
        if accelerator.is_main_process:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss_avg': train_loss,
                'val_loss': val_loss,
                'correlation': correlation,
                'r_squared': r_squared,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'rpiq': rpiq
            }
            wandb.log(log_dict)

            # Model saving logic (every 5 epochs)
            if epoch % 5 == 0:
                checkpoint_path = f'model_checkpoint_epoch_{epoch+1}_r2_{best_r_squared:.4f}.pth'
                accelerator.save_state(checkpoint_path)
                wandb.save(checkpoint_path)
        
        # Print progress
        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        if args.use_validation and val_loader is not None:
            accelerator.print(f'Validation Loss: {val_loss:.4f}')
            accelerator.print(f'R²: {r_squared:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RPIQ: {rpiq:.4f}\n')
        else:
            accelerator.print('No validation performed\n')

    return model, val_outputs if args.use_validation and val_loader is not None else None, val_targets_list if args.use_validation and val_loader is not None else None, best_model_state, best_r_squared

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize Accelerate
    accelerator = Accelerator()
    
    # Initialize wandb with configuration
    if accelerator.is_main_process:
        wandb.init(
            project="socmapping-CNNLSTM",
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
    
    if args.use_validation and not args.load_data:
        train_df, val_df = create_balanced_dataset(df, args.use_validation)
    elif args.load_data:
        train_df, val_df = load_data()
    elif not args.use_validation:
        train_df, _ = create_balanced_dataset(df, args.use_validation)
    else:
        train_df, val_df = create_balanced_dataset(df, args.use_validation)

    # Create datasets
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df) if val_df is not None else None

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False) if val_dataset is not None else None
    
    # Log dataset sizes
    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df) if val_df is not None else 0
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df) if val_df is not None else 0}")

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
    
    # Log model parameters
    if accelerator.is_main_process:
        wandb.run.summary["model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train model
    model, val_outputs, val_targets, best_model_state, best_r_squared = train_model(
        args, model, train_loader, val_loader, 
        num_epochs=num_epochs, 
        accelerator=accelerator,
        loss_type=args.loss_type,
        target_transform=args.target_transform
    )

    # Save final model
    if accelerator.is_main_process and best_model_state is not None:
        final_model_path = (f'cnnlstm_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                           f'TIME_END_{TIME_END}_LOSS_{args.loss_type}_TRANSFORM_{args.target_transform}_'
                           f'BEST_R2_{best_r_squared:.4f}.pth')
        torch.save(best_model_state, final_model_path)
        wandb.save(final_model_path)
        print(f"Best model saved with R²: {best_r_squared:.4f}")

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.print("Model trained and saved successfully!")


#loss_type:"mse"
#lstm_hidden_size:128
#max_oc:150
#num_layers:2
#target_transform:"none"
