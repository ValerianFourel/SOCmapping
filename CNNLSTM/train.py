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
                   seasons, years_padded, num_epochs,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
#from modelSimpleTransformer import SimpleTransformer
#from modelSimpleTransformerNew import SimpleTransformerV2
from models import RefittedCovLSTM
import argparse
from balancedDataset import create_validation_train_sets
import uuid
import os

# Uncomment and use this composite loss function if desired
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



def parse_args():
    parser = argparse.ArgumentParser(description='Train CNNLSTM model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    #parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    #parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_type', type=str, default='composite_l2', choices=['composite_l1', 'l1', 'mse','composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', action='store_true', default=False, help='Whether to use validation set')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.2, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of times to run the process')
    return parser.parse_args()

def train_model(model, train_loader, val_loader, num_epochs=num_epochs, accelerator=None, lr=0.001,
                loss_type='l1', loss_alpha=0.5, target_transform='none', min_r2=0.5, use_validation=True):
    # Define loss function based on loss_type
    if loss_type == 'composite_l1':
        criterion = lambda outputs, targets: composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=loss_alpha)
    elif loss_type == 'composite_l2':
        criterion = lambda outputs, targets: composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=loss_alpha)
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare with Accelerator
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    if use_validation and val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
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
    
    best_r2 = -float('inf')
    best_model_state = None
    epoch_metrics = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()
            
            # Apply target transformation
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)  # Add small constant to avoid log(0)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)
            # 'none' requires no transformation
            
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
            val_outputs = []
            val_targets_list = []
            
            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    features = features.to(accelerator.device)
                    targets = targets.to(accelerator.device).float()
                    
                    # Apply the same transformation to validation targets
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)
                    
                    outputs = model(features)
                    val_outputs.extend(outputs.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
            
            # Gather validation outputs and targets across all processes
            val_outputs_tensor = torch.tensor(val_outputs).to(accelerator.device)
            val_targets_tensor = torch.tensor(val_targets_list).to(accelerator.device)
            val_outputs_all = accelerator.gather(val_outputs_tensor).cpu().numpy()
            val_targets_all = accelerator.gather(val_targets_tensor).cpu().numpy()
            
            if accelerator.is_main_process:
                # Apply inverse transformation to get original scale
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
                else:
                    correlation = 0.0
                    r_squared = 0.0
                mse = np.mean((original_val_outputs - original_val_targets) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(original_val_outputs - original_val_targets))
                
                # Compute IQR and RPIQ
                iqr = np.percentile(original_val_targets, 75) - np.percentile(original_val_targets, 25)
                rpiq = iqr / rmse if rmse > 0 else float('inf')
                
                # Compute validation loss on transformed scale
                val_outputs_tensor_all = torch.from_numpy(val_outputs_all).to(accelerator.device)
                val_targets_tensor_all = torch.from_numpy(val_targets_all).to(accelerator.device)
                val_loss = criterion(val_outputs_tensor_all, val_targets_tensor_all).item()
            else:
                val_loss = float('nan')
                correlation = float('nan')
                r_squared = float('nan')
                mse = float('nan')
                rmse = float('nan')
                mae = float('nan')
                rpiq = float('nan')
        else:
            val_loss = float('nan')
            val_outputs = np.array([])
            val_targets_list = np.array([])
            correlation = float('nan')
            r_squared = 1.0
            mse = float('nan')
            rmse = float('nan')
            mae = float('nan')
            rpiq = float('nan')

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
            epoch_metrics.append(log_dict)

            # Save model if it has the best R² and meets minimum threshold
            if use_validation and r_squared > best_r2 and r_squared >= min_r2:
                best_r2 = r_squared
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2
            elif not use_validation and epoch == num_epochs - 1:
                best_r2 = 1.0
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2
        
        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        if use_validation:
            accelerator.print(f'Validation Loss: {val_loss:.4f}')
            accelerator.print(f'R²: {r_squared:.4f}')
            accelerator.print(f'RMSE: {rmse:.4f}')
            accelerator.print(f'MAE: {mae:.4f}')
            accelerator.print(f'RPIQ: {rpiq:.4f}\n')

    return model, val_outputs, val_targets_list, best_model_state, best_r2, epoch_metrics

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
        # Set num_runs to 1 if use_validation is False
    if not args.use_validation:
        args.num_runs = 1
    accelerator = Accelerator()
    
    # Initialize lists to store metrics and best metrics across runs
    all_runs_metrics = []
    all_best_metrics = {
        'r_squared': [],
        'rmse': [],
        'mae': [],
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
                project="socmapping-CNNLSTM",
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
#                    "num_heads": args.num_heads,
#                    "num_layers": args.num_layers,
                    "dropout_rate": 0.3,
                    "loss_type": args.loss_type,
                    "loss_alpha": args.loss_alpha,
                    "target_transform": args.target_transform,
                    "use_validation": args.use_validation
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


        train_df_std_means, _ = create_balanced_dataset(df, use_validation=False)
        train_dataset_std_means = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df_std_means)
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


        # Create datasets
        if args.use_validation:
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            val_dataset.set_feature_means(train_dataset_std_means.get_feature_means())
            val_dataset.set_feature_stds(train_dataset_std_means.get_feature_stds())
            train_dataset.set_feature_means(train_dataset_std_means.get_feature_means())
            train_dataset.set_feature_stds(train_dataset_std_means.get_feature_stds())
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset)}")
                print(f"Run {run + 1} Length of val_dataset: {len(val_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        else:
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df_std_means)
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = None

        if accelerator.is_main_process:
            wandb_run.summary["train_size"] = len(train_dataset_std_means)
            wandb_run.summary["val_size"] = len(val_df) if args.use_validation else 0

        # Get batch size info
        for batch in train_loader:
            _, _, first_batch, _ = batch
            break
        first_batch_size = first_batch.shape
        if accelerator.is_main_process:
            print(f"Run {run + 1} Size of the first batch: {first_batch_size}")

        # Initialize model
        model = RefittedCovLSTM(
            num_channels=len(bands_list_order),
            lstm_input_size=128,
            lstm_hidden_size=128,
            num_layers=2,
            dropout=0.25
        )
        
        if accelerator.is_main_process:
            wandb_run.summary["model_parameters"] = model.count_parameters()

        # Train model
        model, val_outputs, val_targets, best_model_state, best_r2, epoch_metrics = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            accelerator=accelerator,
            lr=args.lr,
            loss_type=args.loss_type,
            loss_alpha=args.loss_alpha,
            target_transform=args.target_transform,
            min_r2=0.5,
            use_validation=args.use_validation
        )

        # Store metrics
        all_runs_metrics.append(epoch_metrics)
        
        # Find best metrics for this run
        best_metrics = {'r_squared': -float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'rpiq': -float('inf')}
        for epoch_metric in epoch_metrics:
            if not np.isnan(epoch_metric['r_squared']) and epoch_metric['r_squared'] > best_metrics['r_squared']:
                best_metrics['r_squared'] = epoch_metric['r_squared']
                best_metrics['rmse'] = epoch_metric['rmse']
                best_metrics['mae'] = epoch_metric['mae']
                best_metrics['rpiq'] = epoch_metric['rpiq']
        
        # Store best metrics
        for metric in all_best_metrics:
            if not np.isnan(best_metrics[metric]):
                all_best_metrics[metric].append(best_metrics[metric])

        # Save model
        if accelerator.is_main_process and best_model_state is not None:
            final_model_path = (f'3dcnn_model_run_{run+1}_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                               f'TIME_END_{TIME_END}_R2_{best_r2:.4f}_TRANSFORM_{args.target_transform}_'
                               f'LOSS_{args.loss_type}.pth')
            accelerator.save(best_model_state, final_model_path)
            wandb_run.save(final_model_path)
            print(f"Run {run + 1} Model with best R² ({best_r2:.4f}) saved at: {final_model_path}")
        elif accelerator.is_main_process:
            print(f"Run {run + 1} No model saved - R² threshold not met")

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
        output_file = os.path.join(args.output_dir, f'training_metrics_transformer_{uuid.uuid4().hex[:8]}.txt')
        save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename=output_file)
        print(f"\nMetrics saved to: {output_file}")
        
        # Log average metrics to a new wandb run
        wandb_run = wandb.init(project="socmapping-CNNLSTM", name="average_metrics")
        wandb_runs_info.append({
            'project': wandb_run.project,
            'name': wandb_run.name,
            'id': wandb_run.id
        })
        wandb_run.log({"average_metrics": avg_metrics, "min_distance_stats": min_distance_stats})
        for metric, values in all_best_metrics.items():
            wandb_run.summary[f"avg_{metric}"] = np.mean(values)
            wandb_run.summary[f"std_{metric}"] = np.std(values)
        wandb_run.finish()

    accelerator.print("All runs completed, average metrics and min distance statistics computed and saved!")

# BEST "composite_l2" / log
