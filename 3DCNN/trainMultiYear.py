import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, num_epochs,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from torch.utils.data import Dataset, DataLoader
from modelCNNMultiYear import Small3DCNN
from accelerate import Accelerator
import argparse

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """
    Load the training and validation DataFrames from Parquet files.
    Returns:
        validation_df: DataFrame containing validation data
        training_df: DataFrame containing training data
    """
    # Define the absolute paths to the Parquet files
    validation_path = '/lustre/home/vfourel/SOCProject/SOCmapping/balancedDataset/output_0.81km_Lognormal_8.37%/final_validation_df.parquet'
    training_path = '/lustre/home/vfourel/SOCProject/SOCmapping/balancedDataset/output_0.81km_Lognormal_8.37%/final_training_df.parquet'

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

def resample_training_df(training_df, num_bins=128, target_fraction=75/100):
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

def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L2 and scaled chi-squared loss"""
    errors = targets - outputs
    l2_loss = torch.mean(errors ** 2)
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))
    chi2_loss = torch.clamp(chi2_loss, min=1e-8)
    scale_factor = l2_loss / chi2_loss
    chi2_scaled = scale_factor * chi2_loss
    return alpha * l2_loss + (1 - alpha) * chi2_scaled

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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator
import wandb

def train_model(args, model, train_loader, val_loader, num_epochs=100, target_transform="none", loss_type="L2"):
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
    
    accelerator = Accelerator()
    device = accelerator.device

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

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
        target_mean, target_std = 0.0, 1.0

    if accelerator.is_main_process:
        print(f"Target transform: {target_transform}, Params: mean: {target_mean} std: {target_std}")
        if target_transform == 'normalize':
            print(f"Training targets - Min: {all_targets.min().item()}, Max: {all_targets.max().item()}")

    if accelerator.is_main_process:
        wandb.config.update({"target_transform": target_transform, "loss_type": loss_type, "lr": args.lr})

    best_r_squared = -float('inf') if args.use_validation else 1.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_size = 0

        for longitudes, latitudes, features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(features)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                accelerator.print(f"Epoch {epoch+1}: NaN/Inf in outputs")
                continue

            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)
            
            outputs = outputs.float()  # Convert model outputs to float32
            targets = targets.float()  # Convert targets to float32
            loss = criterion(outputs, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"Epoch {epoch+1}: NaN/Inf in loss")
                continue
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            gathered_loss = accelerator.gather_for_metrics(loss.detach())
            running_loss += gathered_loss.mean().item()
            train_loader_size += 1

        train_loss = running_loss / train_loader_size if train_loader_size > 0 else float('nan')

        if args.use_validation and val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_original_targets = []  # New list to store original targets
            val_loader_size = 0

            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    original_targets = targets.clone().float()  # Capture targets before transformation
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)
                    outputs = model(features)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        accelerator.print(f"Epoch {epoch+1}: NaN/Inf in val outputs")
                        continue

                    loss = criterion(outputs, targets)

                    val_loss += accelerator.gather_for_metrics(loss.detach()).mean().item()
                    val_loader_size += 1

                    gathered_outputs = accelerator.gather_for_metrics(outputs)
                    gathered_original_targets = accelerator.gather_for_metrics(original_targets)
                    val_outputs.extend(gathered_outputs.cpu().numpy())
                    val_original_targets.extend(gathered_original_targets.cpu().numpy())

            if len(val_outputs) == 0 or len(val_original_targets) == 0:
                accelerator.print(f"Epoch {epoch+1}: Validation data empty")
                val_loss = float('nan')
                r_squared = 0.0
                rmse = float('nan')
                mae = float('nan')
                rpiq = float('nan')  # Initialize RPIQ for empty case
            else:
                val_loss = val_loss / val_loader_size if val_loader_size > 0 else float('nan')
                val_outputs = np.array(val_outputs)
                val_original_targets = np.array(val_original_targets)

                # Inverse transform model outputs to original scale
                if target_transform == 'log':
                    original_outputs = np.exp(val_outputs)
                elif target_transform == 'normalize':
                    original_outputs = val_outputs * target_std + target_mean
                else:
                    original_outputs = val_outputs

                # Compute metrics on original scale
                output_std = np.std(original_outputs)
                target_std = np.std(val_original_targets)
                accelerator.print(f"Epoch {epoch+1}: Output std: {output_std:.4f}, Target std: {target_std:.4f}")

                if output_std < 1e-6 or target_std < 1e-6:
                    correlation = 0.0
                    r_squared = 0.0
                    mse = float('nan')
                    rmse = float('nan')
                    mae = float('nan')
                    rpiq = float('nan')
                    accelerator.print(f"Epoch {epoch+1}: No variability in outputs or targets")
                else:
                    corr_matrix = np.corrcoef(original_outputs.flatten(), val_original_targets.flatten())
                    correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    r_squared = correlation ** 2
                    mse = np.mean((original_outputs - val_original_targets) ** 2) if not np.any(np.isnan(original_outputs)) else float('nan')
                    rmse = np.sqrt(mse) if not np.isnan(mse) else float('nan')
                    mae = np.mean(np.abs(original_outputs - val_original_targets)) if not np.any(np.isnan(original_outputs)) else float('nan')
                    iqr = np.percentile(val_original_targets, 75) - np.percentile(val_original_targets, 25)
                    rpiq = iqr / rmse if rmse > 0 else float('inf')  # Compute RPIQ

                # Update best model based on R² (computed on original scale)
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    if accelerator.is_main_process:
                        wandb.run.summary["best_r_squared"] = best_r_squared
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        accelerator.print(f"Early stopping triggered after {epoch+1} epochs")
                        break

                if accelerator.is_main_process:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'rpiq': rpiq,  # Log RPIQ
                        'output_std': output_std,
                        'target_std': target_std
                    })

                accelerator.print(f'Epoch {epoch+1}:')
                accelerator.print(f'Training Loss: {train_loss:.4f}')
                accelerator.print(f'Validation Loss: {val_loss:.4f}')
                accelerator.print(f'R²: {r_squared:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RPIQ: {rpiq:.4f}\n')
        else:
            # No validation, update model state and set R² to 1.0
            best_r_squared = 1.0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if accelerator.is_main_process:
                wandb.run.summary["best_r_squared"] = best_r_squared
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                })

            accelerator.print(f'Epoch {epoch+1}:')
            accelerator.print(f'Training Loss: {train_loss:.4f}\n')

        scheduler.step(val_loss if args.use_validation and val_loader is not None else train_loss)

    return model, None if not args.use_validation else val_outputs, None if not args.use_validation else val_original_targets, best_model_state, best_r_squared

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3DCNN model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['composite_l1', 'l1', 'mse','composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='normalize', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
    parser.add_argument('--load_data',default = True,help='We can use pre-computed validation and training sets.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(
            project="socmapping-3dcnn",
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
                "target_transform": args.target_transform,
                "loss_type": args.loss_type,
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

    if args.use_validation and not args.load_data:
        train_df, val_df = create_balanced_dataset(df, args.use_validation)
    elif args.load_data:
        train_df, val_df = load_data()
    elif not args.use_validation:
        train_df, val_df = create_balanced_dataset(df, args.use_validation)
    else:
        train_df, val_df = create_balanced_dataset(df, args.use_validation)
    
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df) if val_df is not None else None

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) if val_dataset is not None else None

    accelerator.print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset) if val_dataset is not None else 0}")

    for batch in train_loader:
        _, _, first_batch, _ = batch 
        break
    accelerator.print("Size of the first batch:", first_batch.shape)

    model = Small3DCNN(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before
    )
    
    if accelerator.is_main_process:
        wandb.run.summary["model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df) if val_df is not None else 0

    model, val_outputs, val_targets, best_model_state, best_r_squared = train_model(
        args,
        model, train_loader, val_loader, num_epochs=num_epochs,
        target_transform=args.target_transform, loss_type=args.loss_type
    )

    if accelerator.is_main_process and best_model_state is not None:
        model_path = (f'cnn_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                      f'TIME_END_{TIME_END}_TRANSFORM_{args.target_transform}_'
                      f'LOSS_{args.loss_type}_BEST_R2_{best_r_squared:.4f}.pth')
        torch.save(best_model_state, model_path)
        wandb.save(model_path)
        accelerator.print(f"Best model saved with R²: {best_r_squared:.4f}")

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.print("Model trained and saved successfully!")
    # Best model with none/l2
