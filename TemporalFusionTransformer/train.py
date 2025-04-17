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
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears, NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
    seasons, years_padded, num_epochs, NUM_HEADS, NUM_LAYERS,
    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before
)
import argparse
from SimpleTFT import SimpleTFT
from balancedDataset import create_validation_train_sets

# Define the composite loss function
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


# Function to create balanced dataset (unchanged)
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
        return training_df, None

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import wandb
from torch.utils.data import DataLoader

# [parse_args and composite_l2_chi2_loss functions remain unchanged]
# Argument parser with new options for loss type and log transformation
def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTFT model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_alpha', type=float, default=0.8, help='Weight for L1 loss in composite loss')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for TFT model')
    parser.add_argument('--loss_type', type=str, default='composite_l2', choices=['composite_l1', 'l1', 'mse','composite_l2'], 
                        help='Type of loss function to use: composite, mse, or l1')
    parser.add_argument('--apply_log', action='store_true', help='Apply log transformation to targets')
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of times to run the process')    
    return parser.parse_args()

def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L2 and scaled chi-squared loss"""
    errors = targets - outputs
    
    # L2 loss: mean squared error
    l2_loss = torch.mean(errors ** 2)
    
    # Standard chi-squared loss: errors^2 / sigma^2
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))
    
    # Ensure chi2_loss is not too small to avoid division issues
    chi2_loss = torch.clamp(chi2_loss, min=1e-8)
    
    # Scale chi2_loss to match the magnitude of l2_loss
    scale_factor = l2_loss / chi2_loss
    chi2_scaled = scale_factor * chi2_loss
    
    # Combine the losses with the weighting factor alpha
    return alpha * l2_loss + (1 - alpha) * chi2_scaled

# [train_model function remains unchanged]
def train_model(args, model, train_loader, val_loader, num_epochs=num_epochs, accelerator=None, lr=0.001,
                min_r2=0.5, use_validation=True, loss_type='composite_l2', target_transform='none'):
    # Handle target normalization if selected
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

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare with Accelerator (handles DDP automatically)
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    if use_validation and val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    best_r2 = -float('inf')
    best_model_state = None
    best_metrics = {'r_squared': 0.0, 'rmse': float('inf'), 'mae': float('inf'), 'rpiq': 0.0}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            outputs = model(features)  # Ensure outputs are a tensor
            # Apply transformation to targets
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)

            targets = targets.float()
            outputs = outputs.float()
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

        # Validation phase
        if use_validation and val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_outputs_list = []
            val_targets_list = []

            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    outputs = model(features)
                    # Apply transformation to targets
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # Gather outputs and targets across all processes
                    val_outputs_list.append(accelerator.gather(outputs).cpu().numpy())
                    val_targets_list.append(accelerator.gather(targets).cpu().numpy())

            if len(val_outputs_list) == 0 or len(val_targets_list) == 0:
                accelerator.print(f"Warning: Validation data is empty at epoch {epoch+1}")
                val_loss = float('nan')
                r_squared = 0.0
                rmse = float('nan')
                mae = float('nan')
                rpiq = float('nan')
            else:
                val_loss = val_loss / len(val_loader)
                val_outputs = np.concatenate(val_outputs_list)
                val_targets = np.concatenate(val_targets_list)

                # Inverse transform to original scale
                if target_transform == 'log':
                    original_val_outputs = np.exp(val_outputs)
                    original_val_targets = np.exp(val_targets)
                elif target_transform == 'normalize':
                    original_val_outputs = val_outputs * target_std + target_mean
                    original_val_targets = val_targets * target_std + target_mean
                else:
                    original_val_outputs = val_outputs
                    original_val_targets = val_targets

                # Compute metrics on original scale
                if len(original_val_outputs) > 1 and np.std(original_val_outputs) > 1e-6 and np.std(original_val_targets) > 1e-6:
                    corr_matrix = np.corrcoef(original_val_outputs.flatten(), original_val_targets.flatten())
                    correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
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
            val_loss = 0.0
            val_outputs = np.array([])
            val_targets_list = np.array([])
            r_squared = 0.0
            rmse = 0.0
            mae = 0.0
            rpiq = 0.0

        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_avg': train_loss,
                'val_loss': val_loss,
                'correlation': correlation if use_validation else 0.0,
                'r_squared': r_squared,
                'mse': mse if use_validation else 0.0,
                'rmse': rmse,
                'mae': mae,
                'rpiq': rpiq if use_validation else 0.0
            })

            if use_validation and r_squared > best_r2 and r_squared >= min_r2:
                best_r2 = r_squared
                best_model_state = accelerator.unwrap_model(model).state_dict()
                best_metrics = {'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'rpiq': rpiq}
                wandb.run.summary['best_r2'] = best_r2
            elif not use_validation and epoch == num_epochs - 1:
                best_r2 = 0.0
                best_model_state = accelerator.unwrap_model(model).state_dict()
                best_metrics = {'r_squared': r_squared, 'rmse': rmse, 'mae': mae, 'rpiq': rpiq}
                wandb.run.summary['best_r2'] = best_r2

        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        accelerator.print(f'Validation Loss: {val_loss:.4f}')
        if use_validation:
            accelerator.print(f'R²: {r_squared:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RPIQ: {rpiq:.4f}\n')

    return model, val_outputs, val_targets_list, best_model_state, best_r2, best_metrics

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()  # Automatically handles DDP

    # Lists to store best metrics across runs
    all_best_metrics = []

    for run in range(args.num_runs):
        # Initialize wandb for each run
        wandb.init(
            project="socmapping-SimpleTFT",
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
                "use_validation": args.use_validation,
                "hidden_size": args.hidden_size,
                "model_type": "SimpleTFT",
                "loss_type": args.loss_type,
                "target_transform": args.target_transform,
                "run_number": run + 1
            }
        )

        # Load and prepare data
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
            val_df, train_df = create_validation_train_sets(
                df=None,
                output_dir=args.output_dir,
                target_val_ratio=args.target_val_ratio,
                use_gpu=args.use_gpu,
                distance_threshold=args.distance_threshold
            )
        else:
            train_df, val_df = create_balanced_dataset(df, args.use_validation)

        if args.use_validation:
            if len(val_df) == 0:
                raise ValueError("Validation DataFrame is empty after balancing")
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            if accelerator.is_main_process:
                print(f"Run {run+1}: Length of train_dataset: {len(train_dataset)}")
                print(f"Run {run+1}: Length of val_dataset: {len(val_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        else:
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            if accelerator.is_main_process:
                print(f"Run {run+1}: Length of train_dataset: {len(train_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = None

        if accelerator.is_main_process:
            wandb.run.summary["train_size"] = len(train_df)
            wandb.run.summary["val_size"] = len(val_df) if args.use_validation else 0

        # Check first batch size
        for batch in train_loader:
            _, _, first_batch, _ = batch
            break
        if accelerator.is_main_process:
            print(f"Run {run+1}: Size of the first batch:", first_batch.shape)

        # Initialize model
        model = SimpleTFT(
            input_channels=len(bands_list_order),
            height=window_size,
            width=window_size,
            time_steps=time_before,
            d_model=512
        )

        if accelerator.distributed_type == "MULTI_GPU":
            model = accelerator.prepare(model)

        if accelerator.is_main_process:
            wandb.run.summary["model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Train the model with new arguments
        model, val_outputs, val_targets, best_model_state, best_r2, best_metrics = train_model(
            args,
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            accelerator=accelerator,
            lr=args.lr,
            min_r2=0.5,
            use_validation=args.use_validation,
            loss_type=args.loss_type,
            target_transform=args.target_transform
        )

        # Store best metrics for this run
        all_best_metrics.append(best_metrics)

        # Save the best model
        if accelerator.is_main_process and best_model_state is not None:
            final_model_path = f'tft_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}_R2_{best_r2:.4f}_transform_{args.target_transform}_loss_type_{args.loss_type}_run_{run+1}.pth'
            accelerator.save(best_model_state, final_model_path)
            wandb.save(final_model_path)
            print(f"Run {run+1}: Model with best R² ({best_r2:.4f}) saved successfully at: {final_model_path}")
        elif accelerator.is_main_process:
            print(f"Run {run+1}: No model saved - R² threshold not met or no validation used")

        wandb.finish()

    # Compute and log average of best metrics
    if accelerator.is_main_process:
        avg_metrics = {
            'avg_r_squared': np.mean([m['r_squared'] for m in all_best_metrics]),
            'avg_rmse': np.mean([m['rmse'] for m in all_best_metrics if not np.isnan(m['rmse'])]),
            'avg_mae': np.mean([m['mae'] for m in all_best_metrics if not np.isnan(m['mae'])]),
            'avg_rpiq': np.mean([m['rpiq'] for m in all_best_metrics if not np.isnan(m['rpiq'])])
        }
        print("\nAverage Best Metrics Across Runs:")
        print(f"Average R²: {avg_metrics['avg_r_squared']:.4f}")
        print(f"Average RMSE: {avg_metrics['avg_rmse']:.4f}")
        print(f"Average MAE: {avg_metrics['avg_mae']:.4f}")
        print(f"Average RPIQ: {avg_metrics['avg_rpiq']:.4f}")

        # Log average metrics to a final wandb run
        wandb.init(project="socmapping-SimpleTFT", name="average_metrics")
        wandb.log(avg_metrics)
        wandb.finish()