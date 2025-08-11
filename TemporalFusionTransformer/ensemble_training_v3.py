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
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,NUM_EPOCHS_RUN,
                   seasons, years_padded, num_epochs, NUM_HEADS, NUM_LAYERS,hidden_size,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from SimpleTFT import SimpleTFT
# from EnhancedTFT import EnhancedTFT as SimpleTFT
import argparse
from balancedDataset import create_validation_train_sets,create_balanced_dataset
import uuid
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Increase the timeout value
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 
torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=20))


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


def enable_dropout(model):
    """Enable dropout layers during inference for Monte Carlo dropout"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def predict_with_uncertainty(model, dataloader, accelerator, target_mean=0.0, target_std=1.0, 
                           target_transform='none', n_samples=20, use_mc_dropout=True):
    """
    Predict with uncertainty estimation using Monte Carlo dropout
    """
    model.eval()
    if use_mc_dropout:
        enable_dropout(model)

    all_predictions = []
    all_targets = []
    all_coords = []

    with torch.no_grad():
        for longitudes, latitudes, features, targets in dataloader:
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

            # Apply target transformation for consistency with training
            if target_transform == 'log':
                targets_transformed = torch.log(targets + 1e-10)
            elif target_transform == 'normalize':
                targets_transformed = (targets - target_mean) / (target_std + 1e-10)
            else:
                targets_transformed = targets

            # Collect multiple predictions for uncertainty
            batch_predictions = []
            for _ in range(n_samples):
                outputs = model(features)
                batch_predictions.append(outputs.cpu().numpy())

            # Stack predictions and compute statistics
            batch_predictions = np.stack(batch_predictions)  # Shape: (n_samples, batch_size, 1)

            all_predictions.append(batch_predictions)
            all_targets.extend(targets_transformed.cpu().numpy())
            all_coords.extend(list(zip(longitudes.cpu().numpy(), latitudes.cpu().numpy())))

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=1)  # Shape: (n_samples, total_samples, 1)
    predictions = predictions.squeeze(-1)  # Shape: (n_samples, total_samples)

    # Calculate uncertainty metrics
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    return mean_predictions, std_predictions, np.array(all_targets), all_coords


def plot_uncertainty_analysis(predictions, uncertainties, targets, coords=None, save_dir=None, 
                            target_transform='none', run_number=None):
    """
    Create comprehensive uncertainty plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    title_suffix = f" - Run {run_number}" if run_number is not None else ""
    fig.suptitle(f'Model Uncertainty Analysis{title_suffix}', fontsize=16)

    # 1. Predictions vs Targets with Error Bars
    ax1 = axes[0, 0]
    ax1.errorbar(targets, predictions, yerr=uncertainties, fmt='o', alpha=0.6, markersize=3)
    min_val, max_val = min(min(targets), min(predictions)), max(max(targets), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs Actuals with Uncertainty')
    ax1.legend()

    # Calculate R² and add to plot
    if len(predictions) > 1 and np.std(predictions) > 1e-6 and np.std(targets) > 1e-6:
        r_squared = stats.pearsonr(predictions, targets)[0]**2
    else:
        r_squared = 0.0
    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Uncertainty vs Prediction Magnitude
    ax2 = axes[0, 1]
    scatter = ax2.scatter(predictions, uncertainties, c=np.abs(predictions - targets), 
                         cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Prediction Uncertainty (std)')
    ax2.set_title('Uncertainty vs Prediction Magnitude')
    plt.colorbar(scatter, ax=ax2, label='Absolute Error')

    # 3. Residuals vs Uncertainty
    ax3 = axes[1, 0]
    residuals = np.abs(predictions - targets)
    ax3.scatter(uncertainties, residuals, alpha=0.6)

    # Add trend line if we have enough points
    if len(uncertainties) > 1 and np.std(uncertainties) > 1e-6:
        try:
            z = np.polyfit(uncertainties, residuals, 1)
            p = np.poly1d(z)
            ax3.plot(uncertainties, p(uncertainties), "r--", alpha=0.8)
        except:
            pass

    ax3.set_xlabel('Prediction Uncertainty (std)')
    ax3.set_ylabel('Absolute Residuals')
    ax3.set_title('Calibration: Uncertainty vs Absolute Error')

    # Calculate correlation between uncertainty and error
    if len(uncertainties) > 1 and np.std(uncertainties) > 1e-6 and np.std(residuals) > 1e-6:
        corr_coef = stats.pearsonr(uncertainties, residuals)[0]
    else:
        corr_coef = 0.0
    ax3.text(0.05, 0.95, f'Correlation = {corr_coef:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Uncertainty Distribution
    ax4 = axes[1, 1]
    if len(uncertainties) > 0:
        ax4.hist(uncertainties, bins=min(30, len(uncertainties)//2), alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(uncertainties), color='red', linestyle='--', 
                    label=f'Mean = {np.mean(uncertainties):.3f}')
        ax4.axvline(np.median(uncertainties), color='green', linestyle='--', 
                    label=f'Median = {np.median(uncertainties):.3f}')
    ax4.set_xlabel('Prediction Uncertainty (std)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Prediction Uncertainties')
    ax4.legend()

    plt.tight_layout()

    if save_dir:
        filename = f"uncertainty_analysis_run_{run_number}.png" if run_number else "uncertainty_analysis.png"
        plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')

    plt.show()

    # Print summary statistics
    print("\n" + "="*50)
    print(f"UNCERTAINTY ANALYSIS SUMMARY{title_suffix}")
    print("="*50)
    print(f"Mean Uncertainty: {np.mean(uncertainties):.4f}")
    print(f"Median Uncertainty: {np.median(uncertainties):.4f}")
    print(f"Uncertainty Range: {np.min(uncertainties):.4f} - {np.max(uncertainties):.4f}")
    print(f"Correlation (Uncertainty vs |Error|): {corr_coef:.4f}")
    print(f"Mean Absolute Error: {np.mean(residuals):.4f}")
    print(f"R²: {r_squared:.4f}")

    return {
        'mean_uncertainty': np.mean(uncertainties),
        'median_uncertainty': np.median(uncertainties),
        'uncertainty_error_correlation': corr_coef,
        'r_squared': r_squared,
        'mae': np.mean(residuals)
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTFT model with customizable parameters')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training and validation (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['composite_l1', 'l1', 'mse','composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='normalize', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation',  type=bool, default=True, help='Whether to use validation set')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.2, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=4, help='Number of times to run the process')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size for the model')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model')
    parser.add_argument('--save_train_and_val', type=bool, default=False, help='Save training and validation data')

    # Uncertainty analysis parameters
    parser.add_argument('--perform_uncertainty_analysis', type=bool, default=True, help='Perform uncertainty analysis after training')
    parser.add_argument('--uncertainty_samples', type=int, default=20, help='Number of MC dropout samples for uncertainty estimation')
    parser.add_argument('--use_mc_dropout', type=bool, default=True, help='Use Monte Carlo dropout for uncertainty estimation')

    return parser.parse_args()


def train_model(model, train_loader, val_loader,target_mean,target_std, num_epochs=num_epochs, accelerator=None, lr=0.001,
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

                # Assuming original_val_outputs and original_val_targets are NumPy arrays
                min_outputs = np.min(original_val_outputs)
                max_outputs = np.max(original_val_outputs)
                min_targets = np.min(original_val_targets)
                max_targets = np.max(original_val_targets)
                if use_validation:
                    accelerator.print("Min of original_val_outputs:", min_outputs)
                    accelerator.print("Max of original_val_outputs:", max_outputs)
                    accelerator.print("Min of original_val_targets:", min_targets)
                    accelerator.print("Max of original_val_targets:", max_targets)
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
            # Print the results


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


def compute_training_statistics_oc():
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        # Calculate target statistics from balanced dataset
    target_mean = df_train['OC'].mean()
    target_std = df_train['OC'].std()


    return target_mean, target_std


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
        num_epochs = NUM_EPOCHS_RUN
    accelerator = Accelerator()

    # Create experiment folder with descriptive naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (f"TFT_experiment_{timestamp}_"
                      f"OC{MAX_OC}_"
                      f"{TIME_BEGINNING}to{TIME_END}_"
                      f"transform_{args.target_transform}_"
                      f"loss_{args.loss_type}_"
                      f"runs_{args.num_runs}_"
                      f"lr_{args.lr}_"
                      f"batch_{args.batch_size}_"
                      f"heads_{args.num_heads}_"
                      f"layers_{args.num_layers}")

    experiment_dir = os.path.join(args.output_dir, experiment_name)

    # Create experiment configuration (available to all processes)
    experiment_config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "args": vars(args),
        "config_params": {
            "MAX_OC": MAX_OC,
            "TIME_BEGINNING": TIME_BEGINNING,
            "TIME_END": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands_count": len(bands_list_order),
            "num_epochs": num_epochs
        }
    }

    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Created experiment directory: {experiment_dir}")

        # Save experiment configuration
        config_file = os.path.join(experiment_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        print(f"Experiment configuration saved to: {config_file}")

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
    all_uncertainty_stats = []  # New: store uncertainty analysis results

    # **Add these variables to track the best model across all runs**
    best_overall_model_state = None
    best_overall_r_squared = -float('inf')
    best_overall_run_number = None
    best_overall_metrics = {}
    best_overall_val_loader = None  # Store validation loader for uncertainty analysis

    # Loop through the specified number of runs
    for run in range(args.num_runs):
        if accelerator.is_main_process:
            print(f"\nStarting Run {run + 1}/{args.num_runs}")

        # Initialize wandb for this run
        if accelerator.is_main_process:
            wandb_run = wandb.init(
                project="socmapping-SimpleTFT",
                name=f"{experiment_name}_run_{run+1}",
                config={
                    "experiment_name": experiment_name,
                    "experiment_dir": experiment_dir,
                    "run_number": run + 1,
                    "max_oc": MAX_OC,
                    "time_beginning": TIME_BEGINNING,
                    "time_end": TIME_END,
                    "window_size": window_size,
                    "time_before": time_before,
                    "bands": len(bands_list_order),
                    "epochs": num_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout_rate": args.dropout_rate,
                    "loss_type": args.loss_type,
                    "loss_alpha": args.loss_alpha,
                    "target_transform": args.target_transform,
                    "use_validation": args.use_validation,
                    "perform_uncertainty_analysis": args.perform_uncertainty_analysis,
                    "uncertainty_samples": args.uncertainty_samples,
                    "use_mc_dropout": args.use_mc_dropout
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

        train_dataset_features_norm = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, df)
        train_df, _ = create_balanced_dataset(df, min_ratio=3/4,use_validation=False)
        train_dataset_std_means = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        train_dataset_std_means.set_feature_means(train_dataset_features_norm.get_feature_means())
        train_dataset_std_means.set_feature_stds(train_dataset_features_norm.get_feature_stds())
        feature_means = train_dataset_features_norm.get_feature_means()
        feature_stds = train_dataset_features_norm.get_feature_stds()
        target_mean, target_std = compute_training_statistics_oc()

        # Create train/validation split
        if args.use_validation:
            val_df, train_df, min_distance_stats = create_validation_train_sets(
                df=df,
                output_dir=experiment_dir,  # Use experiment directory
                target_val_ratio=args.target_val_ratio,
                use_gpu=args.use_gpu,
                distance_threshold=args.distance_threshold
            )
            min_distance_stats_all.append(min_distance_stats)
            # train_df, val_df = create_balanced_dataset(df, min_ratio=3/4,use_validation=True)

        if args.save_train_and_val:
            # Create data subfolder within experiment directory
            data_dir = os.path.join(experiment_dir, "data")
            if accelerator.is_main_process:
                os.makedirs(data_dir, exist_ok=True)

            # Create a descriptive filename based on run context
            parquet_filename = os.path.join(data_dir, f'train_val_data_run_{run+1}.parquet')

            # Combine train and val dataframes if validation is used
            if args.use_validation:
                # Add a column to identify train vs validation rows
                train_df['dataset_type'] = 'train'
                val_df['dataset_type'] = 'val'
                combined_df = pd.concat([train_df, val_df], ignore_index=True)
            else:
                # Just use train_df and mark all as train
                train_df['dataset_type'] = 'train'
                combined_df = train_df

            # Save to parquet file
            if accelerator.is_main_process:
                combined_df.to_parquet(parquet_filename)

            # Save normalization statistics to a separate file
            stats_filename = os.path.join(data_dir, f'normalization_stats_run_{run+1}.pkl')

            normalization_stats = {
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'target_mean': target_mean,
                'target_std': target_std,
                'experiment_config': experiment_config
            }

            if accelerator.is_main_process:
                import pickle
                with open(stats_filename, 'wb') as f:
                    pickle.dump(normalization_stats, f)

                print(f"Train and validation data saved to: {parquet_filename}")
                print(f"Normalization statistics saved to: {stats_filename}")

        # Create datasets
        if args.use_validation:
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            val_dataset.set_feature_means(train_dataset_features_norm.get_feature_means())
            val_dataset.set_feature_stds(train_dataset_features_norm.get_feature_stds())
            train_dataset.set_feature_means(train_dataset_features_norm.get_feature_means())
            train_dataset.set_feature_stds(train_dataset_features_norm.get_feature_stds())
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset)}")
                print(f"Run {run + 1} Length of val_dataset: {len(val_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset_std_means)}")
            train_loader = DataLoader(train_dataset_std_means, batch_size=args.batch_size, shuffle=True)
            val_loader = None

        if accelerator.is_main_process:
            wandb_run.summary["train_size"] = len(train_df)
            wandb_run.summary["val_size"] = len(val_df) if args.use_validation else 0

        # Get batch size info
        for batch in train_loader:
            _, _, first_batch, _ = batch
            break
        first_batch_size = first_batch.shape
        if accelerator.is_main_process:
            print(f"Run {run + 1} Size of the first batch: {first_batch_size}")

        # Initialize model
        model = SimpleTFT(
            input_channels=len(bands_list_order),
            height=window_size,
            width=window_size,
            time_steps=time_before,
            d_model=args.hidden_size
        )
        print("The number of parameters:", model.count_parameters())

        if accelerator.is_main_process:
            wandb_run.summary["model_parameters"] = model.count_parameters()

        # Train model
        model, val_outputs, val_targets, best_model_state, best_r2, epoch_metrics = train_model(
            model,
            train_loader,
            val_loader,
            target_mean=target_mean, 
            target_std=target_std,
            num_epochs=num_epochs,
            accelerator=accelerator,
            lr=args.lr,
            loss_type=args.loss_type,
            loss_alpha=args.loss_alpha,
            target_transform=args.target_transform,
            min_r2=0.40,
            use_validation=args.use_validation,
        )

        # **UNCERTAINTY ANALYSIS FOR EACH RUN**
        if accelerator.is_main_process and args.perform_uncertainty_analysis and args.use_validation and val_loader is not None and best_model_state is not None:
            print(f"\nPerforming uncertainty analysis for Run {run + 1}...")

            # Load the best model for uncertainty analysis
            model.load_state_dict(best_model_state)

            # Create validation loader for uncertainty (without accelerator preparation to avoid issues)
            val_dataset_uncertainty = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            val_dataset_uncertainty.set_feature_means(feature_means)
            val_dataset_uncertainty.set_feature_stds(feature_stds)
            val_loader_uncertainty = DataLoader(val_dataset_uncertainty, batch_size=args.batch_size, shuffle=False)

            # Predict with uncertainty using Monte Carlo dropout
            mean_preds, std_preds, targets_array, coords = predict_with_uncertainty(
                model, val_loader_uncertainty, accelerator, 
                target_mean=target_mean, target_std=target_std,
                target_transform=args.target_transform,
                n_samples=args.uncertainty_samples, 
                use_mc_dropout=args.use_mc_dropout
            )

            # Apply inverse transformation to get original scale
            if args.target_transform == 'normalize':
                mean_preds_orig = mean_preds * target_std + target_mean
                targets_orig = targets_array * target_std + target_mean
                std_preds_orig = std_preds * target_std  # Scale uncertainty too
            elif args.target_transform == 'log':
                mean_preds_orig = np.exp(mean_preds)
                targets_orig = np.exp(targets_array)
                # For log transform, uncertainty scaling is more complex
                std_preds_orig = mean_preds_orig * std_preds  # Approximate scaling
            else:
                mean_preds_orig = mean_preds
                targets_orig = targets_array
                std_preds_orig = std_preds

            # Create uncertainty plots
            results_dir = os.path.join(experiment_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            uncertainty_stats = plot_uncertainty_analysis(
                mean_preds_orig, std_preds_orig, targets_orig, coords, 
                results_dir, args.target_transform, run_number=run+1
            )

            # Store uncertainty statistics
            all_uncertainty_stats.append(uncertainty_stats)

            # Save uncertainty data for this run
            uncertainty_data = pd.DataFrame({
                'longitude': [coord[0] for coord in coords],
                'latitude': [coord[1] for coord in coords],
                'actual': targets_orig,
                'predicted': mean_preds_orig,
                'uncertainty': std_preds_orig,
                'absolute_error': np.abs(mean_preds_orig - targets_orig)
            })

            uncertainty_file = os.path.join(results_dir, f'uncertainty_analysis_run_{run+1}.csv')
            uncertainty_data.to_csv(uncertainty_file, index=False)
            print(f"Uncertainty data for Run {run+1} saved to: {uncertainty_file}")

            # Log uncertainty stats to wandb
            wandb_run.log({
                "uncertainty_analysis": uncertainty_stats,
                "mean_uncertainty": uncertainty_stats['mean_uncertainty'],
                "uncertainty_error_correlation": uncertainty_stats['uncertainty_error_correlation']
            })

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

        # **Check if this run has the best overall performance**
        if not np.isnan(best_metrics['r_squared']) and best_metrics['r_squared'] > best_overall_r_squared:
            best_overall_r_squared = best_metrics['r_squared']
            best_overall_model_state = best_model_state
            best_overall_run_number = run + 1
            best_overall_metrics = best_metrics.copy()
            if args.use_validation:
                best_overall_val_loader = val_loader

        # Create models subfolder within experiment directory
        models_dir = os.path.join(experiment_dir, "models")
        if accelerator.is_main_process:
            os.makedirs(models_dir, exist_ok=True)

        # Save model for this run
        if accelerator.is_main_process and best_model_state is not None:
            run_model_path = os.path.join(models_dir, f'TFT_model_run_{run+1}_R2_{best_r2:.4f}.pth')

            # Save model with metadata
            model_with_run_metadata = {
                'model_state_dict': best_model_state,
                'run_number': run + 1,
                'best_r2': best_r2,
                'best_metrics': best_metrics,
                'experiment_name': experiment_name,
                'model_config': {
                    'input_channels': len(bands_list_order),
                    'height': window_size,
                    'width': window_size,
                    'time_steps': time_before,
                    'd_model': args.hidden_size
                },
                'normalization_stats': {
                    'feature_means': feature_means,
                    'feature_stds': feature_stds,
                    'target_mean': target_mean,
                    'target_std': target_std
                },
                'training_args': vars(args)
            }

            accelerator.save(model_with_run_metadata, run_model_path)
            wandb_run.save(run_model_path)
            print(f"Run {run + 1} Model with best R² ({best_r2:.4f}) saved at: {run_model_path}")
            print(f"Run {run + 1} - Best metrics: R²: {best_metrics['r_squared']:.4f}, MAE: {best_metrics['mae']:.4f}, RMSE: {best_metrics['rmse']:.4f}, RPIQ: {best_metrics['rpiq']:.4f}")
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

        # Compute average uncertainty statistics if available
        if all_uncertainty_stats:
            avg_uncertainty_stats = {}
            for key in all_uncertainty_stats[0].keys():
                values = [stats[key] for stats in all_uncertainty_stats if not np.isnan(stats[key])]
                if values:
                    avg_uncertainty_stats[f'avg_{key}'] = np.mean(values)
                    avg_uncertainty_stats[f'std_{key}'] = np.std(values)

            print("\nAverage Uncertainty Statistics Across Runs:")
            for stat, value in avg_uncertainty_stats.items():
                print(f"{stat}: {value:.4f}")

        # Create results subfolder
        results_dir = os.path.join(experiment_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save all metrics to file in results directory
        output_file = os.path.join(results_dir, 'training_metrics_summary.txt')
        save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename=output_file)
        print(f"\nMetrics saved to: {output_file}")

        # Save detailed metrics as JSON
        detailed_metrics_file = os.path.join(results_dir, 'detailed_metrics.json')
        detailed_metrics = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': timestamp,
                'total_runs': args.num_runs
            },
            'all_runs_metrics': all_runs_metrics,
            'all_best_metrics': all_best_metrics,
            'average_metrics': avg_metrics,
            'min_distance_stats': min_distance_stats,
            'wandb_runs_info': wandb_runs_info,
            'uncertainty_stats': all_uncertainty_stats if all_uncertainty_stats else None
        }
        with open(detailed_metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        print(f"Detailed metrics saved to: {detailed_metrics_file}")

        # Log average metrics to a new wandb run
        wandb_run = wandb.init(project="socmapping-SimpleTFT", name=f"{experiment_name}_summary")
        wandb_runs_info.append({
            'project': wandb_run.project,
            'name': wandb_run.name,
            'id': wandb_run.id
        })
        wandb_run.log({"average_metrics": avg_metrics, "min_distance_stats": min_distance_stats})
        for metric, values in all_best_metrics.items():
            wandb_run.summary[f"avg_{metric}"] = np.mean(values)
            wandb_run.summary[f"std_{metric}"] = np.std(values)

        # Log uncertainty statistics if available
        if all_uncertainty_stats:
            wandb_run.log({"average_uncertainty_stats": avg_uncertainty_stats})

        # **Save the best model from all runs with complete metadata**
        if best_overall_model_state is not None:
            models_dir = os.path.join(experiment_dir, "models")
            final_model_path = os.path.join(models_dir, f'TFT_model_BEST_OVERALL_run_{best_overall_run_number}_R2_{best_overall_r_squared:.4f}.pth')

            # Save model state with comprehensive metadata
            model_with_metadata = {
                'model_state_dict': best_overall_model_state,
                'best_run_number': best_overall_run_number,
                'best_metrics': best_overall_metrics,
                'average_metrics': {
                    'avg_r_squared': np.mean(all_best_metrics['r_squared']) if all_best_metrics['r_squared'] else 0,
                    'avg_mae': np.mean(all_best_metrics['mae']) if all_best_metrics['mae'] else 0,
                    'avg_rmse': np.mean(all_best_metrics['rmse']) if all_best_metrics['rmse'] else 0,
                    'avg_rpiq': np.mean(all_best_metrics['rpiq']) if all_best_metrics['rpiq'] else 0
                },
                'total_runs': args.num_runs,
                'experiment_info': {
                    'name': experiment_name,
                    'timestamp': timestamp,
                    'directory': experiment_dir
                },
                'model_config': {
                    'input_channels': len(bands_list_order),
                    'height': window_size,
                    'width': window_size,
                    'time_steps': time_before,
                    'd_model': args.hidden_size
                },
                'normalization_stats': {
                    'feature_means': feature_means,
                    'feature_stds': feature_stds,
                    'target_mean': target_mean,
                    'target_std': target_std
                },
                'training_config': {
                    'MAX_OC': MAX_OC,
                    'TIME_BEGINNING': TIME_BEGINNING,
                    'TIME_END': TIME_END,
                    'target_transform': args.target_transform,
                    'loss_type': args.loss_type,
                    'loss_alpha': args.loss_alpha,
                    'learning_rate': args.lr,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'dropout_rate': args.dropout_rate,
                    'batch_size': args.batch_size
                },
                'uncertainty_config': {
                    'perform_uncertainty_analysis': args.perform_uncertainty_analysis,
                    'uncertainty_samples': args.uncertainty_samples,
                    'use_mc_dropout': args.use_mc_dropout
                } if args.perform_uncertainty_analysis else None
            }

            accelerator.save(model_with_metadata, final_model_path)
            wandb_run.save(final_model_path)

            print(f"\n**Best overall model from run {best_overall_run_number} saved**")
            print(f"Model path: {final_model_path}")
            print(f"Best R²: {best_overall_r_squared:.4f}")
            print(f"Best MAE: {best_overall_metrics['mae']:.4f}")
            print(f"Best RMSE: {best_overall_metrics['rmse']:.4f}")
            print(f"Best RPIQ: {best_overall_metrics['rpiq']:.4f}")
            print(f"Average R² across all runs: {np.mean(all_best_metrics['r_squared']) if all_best_metrics['r_squared'] else 0:.4f}")
        else:
            print("No final model saved - R² threshold not met for any run")

        wandb_run.finish()

        # Create a summary file with experiment information
        summary_file = os.path.join(experiment_dir, "EXPERIMENT_SUMMARY.txt")
        with open(summary_file, 'w') as f:
            f.write(f"EXPERIMENT SUMMARY\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Experiment Name: {experiment_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Directory: {experiment_dir}\n\n")

            f.write(f"CONFIGURATION:\n")
            f.write(f"-" * 20 + "\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nDATA PARAMETERS:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"MAX_OC: {MAX_OC}\n")
            f.write(f"TIME_BEGINNING: {TIME_BEGINNING}\n")
            f.write(f"TIME_END: {TIME_END}\n")
            f.write(f"Window Size: {window_size}\n")
            f.write(f"Time Before: {time_before}\n")
            f.write(f"Number of Bands: {len(bands_list_order)}\n")

            if best_overall_model_state is not None:
                f.write(f"\nBEST MODEL:\n")
                f.write(f"-" * 20 + "\n")
                f.write(f"Run Number: {best_overall_run_number}\n")
                f.write(f"R²: {best_overall_r_squared:.4f}\n")
                f.write(f"MAE: {best_overall_metrics['mae']:.4f}\n")
                f.write(f"RMSE: {best_overall_metrics['rmse']:.4f}\n")
                f.write(f"RPIQ: {best_overall_metrics['rpiq']:.4f}\n")

            if all_uncertainty_stats:
                f.write(f"\nUNCERTAINTY ANALYSIS:\n")
                f.write(f"-" * 20 + "\n")
                avg_mean_uncertainty = np.mean([stats['mean_uncertainty'] for stats in all_uncertainty_stats])
                avg_correlation = np.mean([stats['uncertainty_error_correlation'] for stats in all_uncertainty_stats])
                f.write(f"Average Mean Uncertainty: {avg_mean_uncertainty:.4f}\n")
                f.write(f"Average Uncertainty-Error Correlation: {avg_correlation:.4f}\n")

            f.write(f"\nFILES GENERATED:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Models: {models_dir}/\n")
            f.write(f"Results: {results_dir}/\n")
            if args.save_train_and_val:
                f.write(f"Data: {data_dir}/\n")

        print(f"\nExperiment completed successfully!")
        print(f"All outputs saved in: {experiment_dir}")
        print(f"Summary available at: {summary_file}")

    accelerator.print("All runs completed, average metrics, min distance statistics, and uncertainty analysis computed and saved!")
