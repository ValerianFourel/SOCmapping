import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears , NormalizedMultiRasterDatasetMultiYears
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
from balancedDataset import create_validation_train_sets
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import os


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
    best_mae = float('inf')
    best_rmse = float('inf')
    best_rpiq = -float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    epoch_metrics = []

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
            
            outputs = outputs.float()
            targets = targets.float()
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
            val_original_targets = []
            val_loader_size = 0

            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    original_targets = targets.clone().float()
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
                rpiq = float('nan')
            else:
                val_loss = val_loss / val_loader_size if val_loader_size > 0 else float('nan')
                val_outputs = np.array(val_outputs)
                val_original_targets = np.array(val_original_targets)

                if target_transform == 'log':
                    val_outputs = np.clip(val_outputs, -50, 50)  # FIX APPLIED HERE
                    original_outputs = np.exp(val_outputs)
                elif target_transform == 'normalize':
                    original_outputs = val_outputs * target_std + target_mean
                else:
                    original_outputs = val_outputs

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
                    rpiq = iqr / rmse if rmse > 0 else float('inf')

                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_mae = mae
                    best_rmse = rmse
                    best_rpiq = rpiq
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    if accelerator.is_main_process:
                        wandb.run.summary["best_r_squared"] = best_r_squared
                        wandb.run.summary["best_mae"] = best_mae
                        wandb.run.summary["best_rmse"] = best_rmse
                        wandb.run.summary["best_rpiq"] = best_rpiq
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        accelerator.print(f"Early stopping triggered after {epoch+1} epochs")
                        break

                if accelerator.is_main_process:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'rpiq': rpiq,
                        'output_std': output_std,
                        'target_std': target_std
                    }
                    wandb.log(log_dict)
                    epoch_metrics.append(log_dict)

                accelerator.print(f'Epoch {epoch+1}:')
                accelerator.print(f'Training Loss: {train_loss:.4f}')
                accelerator.print(f'Validation Loss: {val_loss:.4f}')
                accelerator.print(f'R²: {r_squared:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RPIQ: {rpiq:.4f}\n')
        else:
            best_r_squared = 1.0
            best_mae = 0.0
            best_rmse = 0.0
            best_rpiq = float('inf')
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if accelerator.is_main_process:
                wandb.run.summary["best_r_squared"] = best_r_squared
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                }
                wandb.log(log_dict)
                epoch_metrics.append(log_dict)

            accelerator.print(f'Epoch {epoch+1}:')
            accelerator.print(f'Training Loss: {train_loss:.4f}\n')

        scheduler.step(val_loss if args.use_validation and val_loader is not None else train_loss)

    return model, None if not args.use_validation else val_outputs, None if not args.use_validation else val_original_targets, best_model_state, best_r_squared, best_mae, best_rmse, best_rpiq, epoch_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3DCNN model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='composite_l2', choices=['composite_l1', 'l1', 'mse','composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.2, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of times to run the process')
    return parser.parse_args()

def compute_average_metrics(all_runs_metrics):
    if not all_runs_metrics:
        return {}
    
    metric_sums = {}
    metric_counts = {}
    
    for run_metrics in all_runs_metrics:
        for epoch_metrics in run_metrics:
            for metric, value in epoch_metrics.items():
                if metric == 'epoch':
                    continue
                if not np.isnan(value):
                    if metric not in metric_sums:
                        metric_sums[metric] = 0.0
                        metric_counts[metric] = 0
                    metric_sums[metric] += value
                    metric_counts[metric] += 1
    
    avg_metrics = {}
    for metric in metric_sums:
        avg_metrics[metric] = metric_sums[metric] / metric_counts[metric] if metric_counts[metric] > 0 else float('nan')
    
    return avg_metrics

def compute_min_distance_stats(min_distance_stats_all):
    if not min_distance_stats_all:
        return {}
    
    stats = {'mean': [], 'median': [], 'min': [], 'max': [], 'std': []}
    for stat_dict in min_distance_stats_all:
        for key in stats:
            if not np.isnan(stat_dict[key]):
                stats[key].append(stat_dict[key])
    
    avg_stats = {}
    for key in stats:
        avg_stats[f'avg_{key}'] = np.mean(stats[key]) if stats[key] else float('nan')
        avg_stats[f'std_{key}'] = np.std(stats[key]) if stats[key] else float('nan')
    
    return avg_stats

def save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename='training_metrics.txt'):
    with open(filename, 'w') as f:
        f.write("Training Metrics and Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        # Write args
        f.write("Command Line Arguments:\n")
        f.write("-" * 30 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        
        # Write wandb runs info
        f.write("Wandb Runs Information:\n")
        f.write("-" * 30 + "\n")
        for run_idx, run_info in enumerate(wandb_runs_info, 1):
            f.write(f"Run {run_idx}:\n")
            f.write(f"  Project: {run_info['project']}\n")
            f.write(f"  Run Name: {run_info['name']}\n")
            f.write(f"  Run ID: {run_info['id']}\n")
            f.write("\n")
        
        # Write average metrics
        f.write("Average Metrics Across Runs:\n")
        f.write("-" * 30 + "\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
        
        # Write min distance stats
        f.write("Average Min Distance Statistics:\n")
        f.write("-" * 30 + "\n")
        for stat, value in min_distance_stats.items():
            f.write(f"{stat}: {value:.4f}\n")
        f.write("\n")
        
        # Write best metrics
        f.write("Best Metrics Across Runs:\n")
        f.write("-" * 30 + "\n")
        for metric, values in all_best_metrics.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {np.mean(values):.4f}\n")
            f.write(f"  Std: {np.std(values):.4f}\n")
            f.write(f"  Values: {[f'{v:.4f}' for v in values]}\n")
            f.write("\n")

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    
    # Initialize lists to store metrics and best metrics across runs
    all_runs_metrics = []
    all_best_metrics = {
        'r_squared': [],
        'mae': [],
        'rmse': [],
        'rpiq': []
    }
    min_distance_stats_all = []
    wandb_runs_info = []
    
    # Loop through the specified number of runs
    for run in range(args.num_runs):
        if accelerator.is_main_process:
            print(f"\nStarting Run {run + 1}/{args.num_runs}")
        
        # Initialize wandb for this run
        if accelerator.is_main_process:
            wandb_run = wandb.init(
                project="socmapping-3dcnn",
                name=f"run_{run+1}",
                config={
                    "run_number": run + 1,
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
                    "use_validation": args.use_validation,
                    "num_runs": args.num_runs
                }
            )
            wandb_runs_info.append({
                'project': wandb_run.project,
                'name': wandb_run.name,
                'id': wandb_run.id
            })
        
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

        # Create train/validation split
        if args.use_validation:
            val_df, train_df, min_distance_stats = create_validation_train_sets(
                df=df,
                output_dir=args.output_dir,
                target_val_ratio=args.target_val_ratio,
                use_gpu=args.use_gpu,
                distance_threshold=args.distance_threshold
            )
            min_distance_stats_all.append(min_distance_stats)
        else:
            train_df, val_df = create_balanced_dataset(df, args.use_validation)

        # Create datasets
        train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df) if val_df is not None else None

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) if val_dataset is not None else None

        if accelerator.is_main_process:
            print(f"Run {run + 1} - Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset) if val_dataset is not None else 0}")

        # Get batch size info
        for batch in train_loader:
            _, _, first_batch, _ = batch 
            break
        if accelerator.is_main_process:
            print(f"Run {run + 1} - Size of the first batch: {first_batch.shape}")

        # Initialize model
        model = Small3DCNN(
            input_channels=len(bands_list_order),
            input_height=window_size,
            input_width=window_size,
            input_time=time_before
        )
        
        if accelerator.is_main_process:
            wandb_run.summary[f"run_{run+1}_model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb_run.summary[f"run_{run+1}_train_size"] = len(train_df)
            wandb_run.summary[f"run_{run+1}_val_size"] = len(val_df) if val_df is not None else 0

        # Train model
        model, val_outputs, val_targets, best_model_state, best_r_squared, best_mae, best_rmse, best_rpiq, epoch_metrics = train_model(
            args,
            model, train_loader, val_loader, num_epochs=num_epochs,
            target_transform=args.target_transform, loss_type=args.loss_type
        )

        # Store metrics
        all_runs_metrics.append(epoch_metrics)
        if args.use_validation:
            all_best_metrics['r_squared'].append(best_r_squared)
            all_best_metrics['mae'].append(best_mae)
            all_best_metrics['rmse'].append(best_rmse)
            all_best_metrics['rpiq'].append(best_rpiq)

        # Save model
        if accelerator.is_main_process and best_model_state is not None:
            model_path = (f'cnn_model_run_{run+1}_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                          f'TIME_END_{TIME_END}_TRANSFORM_{args.target_transform}_'
                          f'LOSS_{args.loss_type}_BEST_R2_{best_r_squared:.4f}.pth')
            torch.save(best_model_state, model_path)
            wandb_run.save(model_path)
            print(f"Run {run + 1} - Best model saved with R²: {best_r_squared:.4f}, MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}, RPIQ: {best_rpiq:.4f}")

        if accelerator.is_main_process:
            wandb_run.finish()

    # Compute and log average metrics
    if accelerator.is_main_process:
        avg_metrics = compute_average_metrics(all_runs_metrics)
        min_distance_stats = compute_min_distance_stats(min_distance_stats_all)
        
        print("\nAverage Metrics Across Runs:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nAverage Min Distance Statistics:")
        for stat, value in min_distance_stats.items():
            print(f"{stat}: {value:.4f}")
        
        print("\nBest Metrics Across Runs:")
        for metric, values in all_best_metrics.items():
            print(f"{metric} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
        
        # Save all metrics to file
        output_file = os.path.join(args.output_dir, f'training_metrics_3dcnn_{uuid.uuid4().hex[:8]}.txt')
        save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename=output_file)
        print(f"\nMetrics saved to: {output_file}")
        
        # Log average metrics to a new wandb run
        wandb_run = wandb.init(project="socmapping-3dcnn", name="average_metrics")
        wandb_runs_info.append({
            'project': wandb_run.project,
            'name': wandb_run.name,
            'id': wandb_run.id
        })
        wandb_run.log({"average_metrics": avg_metrics, "min_distance_stats": min_distance_stats})
        for metric, values in all_best_metrics.items():
            wandb_run.summary[f"avg_{metric}"] = np.mean(values)
            wandb_run.summary[f"std_{metric}"] = np.std(values)
        
        # Save the best model state from the run with the highest R²
        if args.use_validation and all_best_metrics['r_squared']:
            best_run_idx = np.argmax(all_best_metrics['r_squared'])
            average_best_r2 = np.mean(all_best_metrics['r_squared'])
            best_model_state = {k: v.cpu() for k, v in train_model(args, Small3DCNN(
                input_channels=len(bands_list_order),
                input_height=window_size,
                input_width=window_size,
                input_time=time_before
            ), train_loader, val_loader, num_epochs=num_epochs, target_transform=args.target_transform, loss_type=args.loss_type)[3].items()}
            model_path = (f'cnn_model_best_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                          f'TIME_END_{TIME_END}_TRANSFORM_{args.target_transform}_'
                          f'LOSS_{args.loss_type}_AVG_R2_{average_best_r2:.4f}.pth')
            torch.save(best_model_state, model_path)
            wandb_run.save(model_path)
            print(f"Best model from run {best_run_idx+1} saved with average R²: {np.mean(all_best_metrics['r_squared']):.4f}")
        
        wandb_run.finish()

    accelerator.print("All runs completed, average metrics and min distance statistics computed and saved!")