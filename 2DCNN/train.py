# -*- coding: utf-8 -*-
"""
Optimized script for training a ResNet2DCNN model for Soil Organic Carbon mapping.
Uses PyTorch, Accelerate for distributed training, and WandB for logging.
"""

import argparse
import os
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb # Keep wandb import

# --- Configuration Imports (Assuming these exist) ---
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
    seasons, years_padded, num_epochs, # num_epochs used as default
    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before, NUM_EPOCHS_RUN
)
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears,NormalizedMultiRasterDatasetMultiYears # Assuming this is the primary one used
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from model2DCNN import ResNet2DCNN
from balancedDataset import create_validation_train_sets, create_balanced_dataset # Import both balancing functions


# --- Constants ---
BATCH_SIZE = 256
NUM_WORKERS = 4
LOSS_SIGMA = 3.0 # Sigma for composite losses
MIN_R2_SAVE_THRESHOLD = 0.21 # Minimum validation R2 to save the model


# --- Helper Functions ---

def flatten_paths(path_list: List[Union[str, Path, List]]) -> List[Union[str, Path]]:
    """Flattens a nested list of paths."""
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        elif item is not None: # Ensure None paths are not added
             flattened.append(item)
    return flattened

# --- Loss Functions ---

def composite_l1_chi2_loss(outputs: torch.Tensor, targets: torch.Tensor, sigma: float = LOSS_SIGMA, alpha: float = 0.5) -> torch.Tensor:
    """Composite loss combining L1 and scaled chi-squared like loss."""
    errors = targets - outputs
    l1_loss = torch.mean(torch.abs(errors))

    squared_errors = errors ** 2
    # Chi-squared like term, scaled to be robust to outliers
    chi2_unscaled = (1/4) * squared_errors * torch.exp(-squared_errors / (2 * sigma**2)) # Corrected sigma usage
    chi2_unscaled_mean = torch.mean(chi2_unscaled)

    # Prevent division by zero or near-zero
    chi2_unscaled_mean_clamped = torch.clamp(chi2_unscaled_mean, min=1e-8)
    l1_loss_clamped = torch.clamp(l1_loss, min=1e-8) # Clamp l1_loss too for stability

    # Scale chi2 term to be comparable to l1_loss
    scale_factor = l1_loss_clamped / chi2_unscaled_mean_clamped
    chi2_scaled = scale_factor * chi2_unscaled_mean # Use original mean for scaling

    return alpha * l1_loss + (1 - alpha) * chi2_scaled


def composite_l2_chi2_loss(outputs: torch.Tensor, targets: torch.Tensor, sigma: float = LOSS_SIGMA, alpha: float = 0.5) -> torch.Tensor:
    """Composite loss combining L2 (MSE) and scaled standard chi-squared loss."""
    errors = targets - outputs

    # L2 loss: mean squared error
    l2_loss = torch.mean(errors ** 2)

    # Standard chi-squared like loss component: mean(errors^2 / sigma^2)
    # Adding small epsilon to sigma^2 for stability if sigma is ever zero
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2 + 1e-8))

    # Ensure chi2_loss is not too small to avoid division issues and clamp l2_loss
    chi2_loss_clamped = torch.clamp(chi2_loss, min=1e-8)
    l2_loss_clamped = torch.clamp(l2_loss, min=1e-8) # Clamp l2_loss for stability

    # Scale chi2_loss to match the magnitude of l2_loss
    scale_factor = l2_loss_clamped / chi2_loss_clamped
    chi2_scaled = scale_factor * chi2_loss # Use original chi2_loss for scaling

    # Combine the losses with the weighting factor alpha
    return alpha * l2_loss + (1 - alpha) * chi2_scaled


# --- Metric Calculation ---

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculates regression metrics (R², RMSE, MAE, RPIQ)."""
    metrics = {'r_squared': np.nan, 'rmse': np.nan, 'mae': np.nan, 'rpiq': np.nan}
    if predictions.size < 2 or targets.size < 2 or predictions.size != targets.size:
        print(f"Warning: Insufficient data for metric calculation (preds: {predictions.size}, targets: {targets.size})")
        return metrics

    # Ensure finite values
    valid_indices = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[valid_indices]
    targets = targets[valid_indices]

    if predictions.size < 2:
        print("Warning: Less than 2 valid data points after filtering NaNs/Infs.")
        return metrics

    try:
        # Mean Squared Error and Root Mean Squared Error
        mse = np.mean((predictions - targets) ** 2)
        metrics['rmse'] = np.sqrt(mse)

        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(predictions - targets))

        # R-squared (Coefficient of Determination)
        target_variance = np.var(targets)
        if target_variance > 1e-8: # Avoid division by zero if targets are constant
             # Calculate correlation coefficient first for stability
             correlation_matrix = np.corrcoef(predictions, targets)
             # Handle case where corrcoef returns scalar or has NaNs
             if correlation_matrix.shape == (2, 2) and not np.isnan(correlation_matrix[0, 1]):
                 correlation = correlation_matrix[0, 1]
                 metrics['r_squared'] = correlation ** 2
             else:
                  metrics['r_squared'] = 0.0 # Or np.nan, depending on desired behavior
        else:
            metrics['r_squared'] = 0.0 # Or np.nan

        # Ratio of Performance to Interquartile Range (RPIQ)
        iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
        if iqr > 1e-8 and np.isfinite(metrics['rmse']) and metrics['rmse'] > 1e-8:
            metrics['rpiq'] = iqr / metrics['rmse']
        else:
            metrics['rpiq'] = np.nan # Or 0.0 or inf depending on preference

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return dictionary with NaNs

    return metrics


# --- Target Transformation ---

def transform_target(targets: torch.Tensor, transform_type: str, mean: Optional[float] = None, std: Optional[float] = None) -> torch.Tensor:
    """Applies specified transformation to target tensor."""
    if transform_type == 'log':
        # Add small epsilon for stability if targets can be zero
        return torch.log(targets + 1e-10)
    elif transform_type == 'normalize':
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for normalization.")
        # Add epsilon to std for stability
        return (targets - mean) / (std + 1e-10)
    elif transform_type == 'none':
        return targets
    else:
        raise ValueError(f"Unknown target transformation: {transform_type}")


def inverse_transform_target(outputs: np.ndarray, transform_type: str, mean: Optional[float] = None, std: Optional[float] = None) -> np.ndarray:
    """Applies inverse transformation to model outputs (numpy array)."""
    if transform_type == 'log':
        # Clip predictions to avoid excessively large values from exp
        outputs_clipped = np.clip(outputs, -50, 50) # Adjust clipping range as needed
        return np.exp(outputs_clipped)
    elif transform_type == 'normalize':
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for inverse normalization.")
        return outputs * std + mean
    elif transform_type == 'none':
        return outputs
    else:
        raise ValueError(f"Unknown target transformation: {transform_type}")


# --- Training Function ---
def train_model(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    accelerator: Accelerator,
    num_epochs: int,
) -> Tuple[nn.Module, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]], float, Dict[str, float], List[Dict[str, Any]]]:
    """Trains the model and returns the trained model, validation results, best state, and metrics."""

    # --- Setup: Target Normalization (if needed) ---
    target_mean: Optional[float] = None
    target_std: Optional[float] = None
    if args.target_transform == 'normalize':
        all_targets = []
        for _, _, _, targets_batch in train_loader:
            all_targets.append(targets_batch.cpu())
        if not all_targets:
            raise ValueError("Training loader is empty, cannot calculate target statistics.")
        all_targets_tensor = torch.cat(all_targets).float()
        target_mean = all_targets_tensor.mean().item()
        target_std = all_targets_tensor.std().item()
        if accelerator.is_main_process:
            accelerator.print(f"Target normalization stats: Mean={target_mean:.4f}, Std={target_std:.4f}")
        if target_std < 1e-8:
            accelerator.print("Warning: Target standard deviation is close to zero. Normalization might be unstable.")
            target_std = 1e-8

    # --- Setup: Loss Function ---
    if args.loss_type == 'composite_l1':
        criterion = lambda outputs, targets: composite_l1_chi2_loss(outputs, targets, sigma=LOSS_SIGMA, alpha=args.loss_alpha)
    elif args.loss_type == 'composite_l2':
        criterion = lambda outputs, targets: composite_l2_chi2_loss(outputs, targets, sigma=LOSS_SIGMA, alpha=args.loss_alpha)
    elif args.loss_type == 'l1':
        criterion = nn.L1Loss()
    elif args.loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # --- Setup: Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Prepare with Accelerate ---
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if args.use_validation and val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # --- Training State Initialization ---
    best_val_r2 = -float('inf')
    best_model_state = None
    best_val_metrics = {'r_squared': 0.0, 'rmse': float('inf'), 'mae': float('inf'), 'rpiq': 0.0}
    all_epoch_metrics: List[Dict[str, Any]] = []

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", disable=not accelerator.is_main_process)

        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(train_loop):
            optimizer.zero_grad()
            targets = targets.float()
            transformed_targets = transform_target(targets, args.target_transform, target_mean, target_std)
            outputs = model(features)
            outputs = outputs.float()
            loss = criterion(outputs, transformed_targets)
            accelerator.backward(loss)
            optimizer.step()
            total_train_loss += accelerator.gather(loss).mean().item()

            if accelerator.is_main_process:
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })
                train_loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        val_metrics = {'r_squared': np.nan, 'rmse': np.nan, 'mae': np.nan, 'rpiq': np.nan}
        val_loss_avg = np.nan
        all_val_outputs_np = None
        all_val_targets_np = None

        if args.use_validation and val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_outputs_list = []
            val_targets_list = []
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", disable=not accelerator.is_main_process)

            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loop:
                    targets = targets.float()
                    transformed_targets = transform_target(targets, args.target_transform, target_mean, target_std)
                    outputs = model(features)
                    outputs = outputs.float()
                    loss = criterion(outputs, transformed_targets)
                    gathered_outputs = accelerator.gather(outputs)
                    gathered_targets = accelerator.gather(targets)
                    gathered_loss = accelerator.gather(loss).mean().item()
                    total_val_loss += gathered_loss
                    val_outputs_list.append(gathered_outputs.cpu().numpy())
                    val_targets_list.append(gathered_targets.cpu().numpy())

            if not val_outputs_list or not val_targets_list:
                if accelerator.is_main_process:
                    accelerator.print(f"Warning: Validation data yielded no results at epoch {epoch+1}")
            else:
                val_loss_avg = total_val_loss / len(val_loader)
                all_val_outputs_np = np.concatenate(val_outputs_list)
                all_val_targets_np = np.concatenate(val_targets_list)
                original_scale_val_outputs = inverse_transform_target(
                    all_val_outputs_np, args.target_transform, target_mean, target_std
                )
                if accelerator.is_main_process:
                    val_metrics = calculate_metrics(original_scale_val_outputs.flatten(), all_val_targets_np.flatten())

        # --- Logging Epoch Results (Main Process Only) ---
        if accelerator.is_main_process:
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss_avg': avg_train_loss,
                'val_loss_avg': val_loss_avg if not np.isnan(val_loss_avg) else 0.0,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            wandb.log(epoch_log)
            all_epoch_metrics.append(epoch_log)

            accelerator.print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss_avg:.4f}")
            if args.use_validation:
                accelerator.print(f"Val Metrics: R²={val_metrics['r_squared']:.4f}, RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, RPIQ={val_metrics['rpiq']:.4f}")

            # --- Check for Best Model ---
            current_val_r2 = val_metrics.get('r_squared', -float('inf'))
            if args.use_validation and np.isfinite(current_val_r2) and current_val_r2 > best_val_r2:
                best_val_r2 = current_val_r2
                best_val_metrics = val_metrics
                accelerator.print(f"*** New best validation R²: {best_val_r2:.4f} at epoch {epoch+1} ***")
                if best_val_r2 >= MIN_R2_SAVE_THRESHOLD:
                    best_model_state = accelerator.unwrap_model(model).state_dict()
                    wandb.run.summary['best_val_r2'] = best_val_r2
                    wandb.run.summary['best_epoch'] = epoch + 1
                else:
                    accelerator.print(f"Validation R² {best_val_r2:.4f} is below save threshold {MIN_R2_SAVE_THRESHOLD}. Model not saved yet.")

            if not args.use_validation and epoch == num_epochs - 1:
                best_model_state = accelerator.unwrap_model(model).state_dict()
                best_val_r2 = np.nan
                accelerator.print("Validation disabled. Saving model from last epoch.")

    return model, all_val_outputs_np, all_val_targets_np, best_model_state, best_val_r2, best_val_metrics, all_epoch_metrics


# --- Metrics Aggregation and Saving ---

def compute_average_metrics(all_runs_epoch_metrics: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Computes average metrics across epochs and runs, focusing on the best validation epoch per run."""
    if not all_runs_epoch_metrics:
        return {}

    best_epoch_metrics_list = []
    for run_epochs in all_runs_epoch_metrics:
        if not run_epochs:
            continue
        # Find the epoch with the best validation R² for this run
        best_r2 = -float('inf')
        best_epoch = run_epochs[-1]  # Default to last epoch if no better R² is found
        for epoch_data in run_epochs:
            val_r2 = epoch_data.get('val_r_squared', -float('inf'))
            if np.isfinite(val_r2) and val_r2 > best_r2:
                best_r2 = val_r2
                best_epoch = epoch_data
        best_epoch_metrics_list.append(best_epoch)

    if not best_epoch_metrics_list:
        return {}

    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    for metrics_dict in best_epoch_metrics_list:
        for metric, value in metrics_dict.items():
            if metric == 'epoch':
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

    avg_metrics = {
        metric: metric_sums[metric] / metric_counts[metric]
        for metric in metric_sums if metric_counts[metric] > 0
    }
    return avg_metrics


def compute_min_distance_stats_agg(min_distance_stats_all: List[Dict[str, float]]) -> Dict[str, float]:
    """Computes average and std dev of min distance statistics across runs."""
    if not min_distance_stats_all:
        return {}

    stats_agg: Dict[str, List[float]] = {'mean': [], 'median': [], 'min': [], 'max': [], 'std': []}
    for stat_dict in min_distance_stats_all:
        for key in stats_agg:
            if key in stat_dict and np.isfinite(stat_dict[key]):
                stats_agg[key].append(stat_dict[key])

    avg_stats: Dict[str, float] = {}
    for key, values in stats_agg.items():
        if values:
            avg_stats[f'avg_{key}'] = np.mean(values)
            avg_stats[f'std_{key}'] = np.std(values)
        else:
            avg_stats[f'avg_{key}'] = np.nan
            avg_stats[f'std_{key}'] = np.nan

    return avg_stats


def save_summary_to_file(
    args: argparse.Namespace,
    wandb_runs_info: List[Dict[str, str]],
    avg_best_epoch_metrics: Dict[str, float],
    avg_min_distance_stats: Dict[str, float],
    all_runs_best_metrics: List[Dict[str, float]],
    output_dir: Path,
    filename_prefix: str = "training_summary"
) -> None:
    """Saves a summary of the training runs to a text file with improved formatting and error handling."""
    output_file = output_dir / f"{filename_prefix}_{uuid.uuid4().hex[:8]}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Training Summary Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("Configuration Arguments:\n")
        f.write("-" * 30 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")

        f.write("Wandb Runs Information:\n")
        f.write("-" * 30 + "\n")
        for run_idx, run_info in enumerate(wandb_runs_info, 1):
            f.write(f"Run {run_idx}:\n")
            f.write(f"  Project: {run_info['project']}\n")
            f.write(f"  Name: {run_info['name']}\n")
            f.write(f"  ID: {run_info['id']}\n")
            best_r2 = next((m.get('val_r_squared', np.nan) for m in all_runs_best_metrics if m.get('run_id') == run_info['id']), np.nan)
            f.write(f"  Best Val R²: {best_r2:.4f}\n\n" if np.isfinite(best_r2) else "  Best Val R²: N/A\n\n")

        f.write("Average Metrics (Across Best Epochs of Runs):\n")
        f.write("-" * 30 + "\n")
        if avg_best_epoch_metrics:
            for metric, value in avg_best_epoch_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        else:
            f.write("No average metrics available.\n")
        f.write("\n")

        if args.use_validation:
            f.write("Average Min Distance Statistics (Validation Sets):\n")
            f.write("-" * 30 + "\n")
            if avg_min_distance_stats:
                for stat, value in avg_min_distance_stats.items():
                    f.write(f"{stat}: {value:.4f}\n")
            else:
                f.write("No distance statistics available.\n")
            f.write("\n")

            f.write("Best Validation Metrics Aggregated Across Runs:\n")
            f.write("-" * 30 + "\n")
            for metric in ['val_r_squared', 'val_rmse', 'val_mae', 'val_rpiq']:
                values = [m[metric] for m in all_runs_best_metrics if metric in m and np.isfinite(m[metric])]
                if values:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {np.mean(values):.4f}\n")
                    f.write(f"  Std Dev: {np.std(values):.4f}\n")
                    f.write(f"  Values: {[f'{v:.4f}' for v in values]}\n\n")
                else:
                    f.write(f"{metric}: No finite values available.\n\n")
        else:
            f.write("Validation was disabled. No validation metrics available.\n\n")

    print(f"Summary report saved to: {output_file}")



# --- Argument Parsing ---

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Train ResNet2DCNN model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='Number of training epochs.')
    parser.add_argument('--loss-alpha', type=float, default=0.8, help='Weight for the primary loss term (L1 or L2) in composite losses.')
    parser.add_argument('--no-validation', action='store_true', help='Disable validation set creation and evaluation.')
    parser.add_argument('--loss-type', type=str, default='composite_l2', choices=['composite_l1', 'l1', 'mse', 'composite_l2'],
                        help='Type of loss function: composite_l1, composite_l2, l1 (MAE), or mse (L2).')
    parser.add_argument('--target-transform', type=str, default='none', choices=['none', 'log', 'normalize'],
                        help='Transformation to apply to target values: none, log (log(1+x)), or normalize (z-score).')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for target value resampling/balancing.')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save models and metrics.')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target fraction of data for the validation set (used by create_validation_train_sets).')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage (uses CPU).')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for validation points (used by create_validation_train_sets).')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling (used by create_balanced_dataset).')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of independent training runs to perform.')
    parser.add_argument('--project-name', type=str, default='socmapping-2DCNN', help='WandB project name.')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model state after training.')

    args = parser.parse_args()
    # Post-processing args
    args.use_validation = not args.no_validation
    args.use_gpu = not args.no_gpu
    if args.target_transform == 'log':
         print("Warning: Applying log transformation to target 'OC' values.")

    return args

# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()
    # Set num_runs to 1 if use_validation is False
    if not args.use_validation:
        args.num_runs = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(cpu=args.no_gpu)
    all_runs_epoch_metrics: List[List[Dict[str, Any]]] = []
    all_runs_best_metrics: List[Dict[str, Any]] = []
    all_min_distance_stats: List[Dict[str, float]] = []
    wandb_runs_info: List[Dict[str, str]] = []

    for run_idx in range(args.num_runs):
        run_name = f"run_{run_idx+1}_{uuid.uuid4().hex[:4]}"
        if accelerator.is_main_process:
            print(f"\n{'='*10} Starting Run {run_idx + 1}/{args.num_runs} ({run_name}) {'='*10}")

        current_wandb_run = None
        if accelerator.is_main_process:
            try:
                current_wandb_run = wandb.init(
                    project=args.project_name,
                    name=run_name,
                    config={
                        **vars(args),
                        "max_oc": MAX_OC,
                        "time_beginning": TIME_BEGINNING,
                        "time_end": TIME_END,
                        "window_size": window_size,
                        "time_before": time_before,
                        "bands": len(bands_list_order),
                        "batch_size": BATCH_SIZE,
                        "num_workers": NUM_WORKERS,
                        "loss_sigma": LOSS_SIGMA,
                        "model_type": "ResNet2DCNN",
                        "run_number": run_idx + 1
                    },
                    reinit=True
                )
                wandb_runs_info.append({
                    'project': current_wandb_run.project,
                    'name': current_wandb_run.name,
                    'id': current_wandb_run.id
                })
            except Exception as e:
                print(f"Error initializing WandB for run {run_idx+1}: {e}. Continuing without WandB logging for this run.")
                current_wandb_run = None

        try:
            if accelerator.is_main_process:
                print("Loading and filtering dataframe...")
            df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
            if df.empty:
                raise ValueError("Filtered dataframe is empty. Check filters and data source.")

            if accelerator.is_main_process:
                print("Separating data paths...")
            samples_coords_paths, data_paths = separate_and_add_data()
            samples_coords_paths_flat = list(dict.fromkeys(flatten_paths(samples_coords_paths)))
            data_paths_flat = list(dict.fromkeys(flatten_paths(data_paths)))

            if not samples_coords_paths_flat or not data_paths_flat:
                raise ValueError("No valid data paths found after flattening.")

            if accelerator.is_main_process:
                print(f"Found {len(samples_coords_paths_flat)} unique coordinate files and {len(data_paths_flat)} unique data files.")

            train_df: pd.DataFrame
            val_df: Optional[pd.DataFrame] = None
            min_distance_stats: Optional[Dict[str, float]] = None

            train_df, _ = create_balanced_dataset(
                    df,
                    n_bins=args.num_bins,
                    min_ratio=args.target_fraction,
                    use_validation=False
                )
            train_dataset_means_std = NormalizedMultiRasterDatasetMultiYears(samples_coords_paths_flat, data_paths_flat, train_df)

            if args.use_validation:
                if accelerator.is_main_process:
                    print("Creating validation/train split using distance-based method...")
                split_output_dir = output_dir / f"run_{run_idx+1}_split"
                split_output_dir.mkdir(parents=True, exist_ok=True)
                val_df, train_df, min_distance_stats = create_validation_train_sets(
                    df=df,
                    output_dir=split_output_dir,
                    target_val_ratio=args.target_val_ratio,
                    use_gpu=False,
                    distance_threshold=args.distance_threshold
                )
                if min_distance_stats:
                    all_min_distance_stats.append(min_distance_stats)
                if val_df.empty or train_df.empty:
                    raise ValueError("Validation or Training DataFrame is empty after distance-based split.")

                if train_df.empty:
                    raise ValueError("Training DataFrame is empty after balancing.")

            if accelerator.is_main_process:
                print(f"Training set size: {len(train_df)}")
                if val_df is not None:
                    print(f"Validation set size: {len(val_df)}")
                if current_wandb_run:
                    current_wandb_run.summary["train_size"] = len(train_df)
                    current_wandb_run.summary["val_size"] = len(val_df) if val_df is not None else 0
                    if min_distance_stats:
                        current_wandb_run.log({"validation_min_dist_stats": min_distance_stats})

            if accelerator.is_main_process:
                print("Creating Datasets and DataLoaders...")
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coords_paths_flat, data_paths_flat, train_df)
            train_dataset.set_feature_means(train_dataset_means_std.get_feature_means())
            train_dataset.set_feature_stds(train_dataset_means_std.get_feature_stds())
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

            val_loader = None
            if args.use_validation and val_df is not None:
                val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coords_paths_flat, data_paths_flat, val_df)
                val_dataset.set_feature_means(train_dataset_means_std.get_feature_means())
                val_dataset.set_feature_stds(train_dataset_means_std.get_feature_stds())
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            if accelerator.is_main_process:
                print("Initializing model...")
            input_channels = (len(bands_list_order) - 1) * time_before + 1
            model = ResNet2DCNN(
                input_channels=input_channels,
                input_height=window_size,
                input_width=window_size,
            )

            if accelerator.is_main_process and current_wandb_run:
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model parameters: {num_params:,}")
                current_wandb_run.summary["model_parameters"] = num_params

            if accelerator.is_main_process:
                print(f"Starting training for {args.epochs} epochs...")

            model, _, _, best_model_state, best_r2, best_metrics_run, epoch_metrics_run = train_model(
                args=args,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                accelerator=accelerator,
                num_epochs=args.epochs,
            )

            all_runs_epoch_metrics.append(epoch_metrics_run)
            best_metrics_run_with_id = best_metrics_run.copy()
            best_metrics_run_with_id['run_id'] = current_wandb_run.id if current_wandb_run else run_name
            all_runs_best_metrics.append(best_metrics_run_with_id)

            if accelerator.is_main_process:
                if best_model_state is not None:
                    if not args.no_save:
                        r2_str = f"{best_r2:.4f}" if np.isfinite(best_r2) else "no_val"
                        transform_str = args.target_transform
                        loss_str = args.loss_type
                        model_filename = f"resnet2dcnn_OC_{MAX_OC}_T_{TIME_BEGINNING}-{TIME_END}_R2_{r2_str}_TRANS_{transform_str}_LOSS_{loss_str}_run_{run_idx+1}.pth"
                        final_model_path = output_dir / model_filename
                        accelerator.save(best_model_state, str(final_model_path))
                        accelerator.print(f"Run {run_idx+1}: Best model state saved to: {final_model_path}")
                        if current_wandb_run:
                            try:
                                wandb.save(str(final_model_path), base_path=str(output_dir))
                                accelerator.print(f"Run {run_idx+1}: Model artifact saved to WandB.")
                            except Exception as e:
                                accelerator.print(f"Run {run_idx+1}: Error saving model artifact to WandB: {e}")
                    else:
                        accelerator.print(f"Run {run_idx+1}: Model saving disabled via --no-save flag.")
                else:
                    accelerator.print(f"Run {run_idx+1}: No best model state found or R² threshold not met. Model not saved.")

        except Exception as e:
            accelerator.print(f"\n!!! ERROR in Run {run_idx + 1} !!!")
            accelerator.print(f"Error type: {type(e).__name__}")
            accelerator.print(f"Error message: {e}")
            import traceback
            accelerator.print("Traceback:")
            accelerator.print(traceback.format_exc())
            all_runs_epoch_metrics.append([])
            all_runs_best_metrics.append({'run_id': run_name, 'error': str(e)})

        finally:
            if accelerator.is_main_process and current_wandb_run:
                current_wandb_run.finish()

    if accelerator.is_main_process:
        print(f"\n{'='*10} All Runs Completed - Aggregating Results {'='*10}")
        avg_best_epoch_metrics = compute_average_metrics(all_runs_epoch_metrics)
        print("\nAverage Metrics (Across Best Epochs):")
        if avg_best_epoch_metrics:
            for metric, value in avg_best_epoch_metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("  No average metrics could be computed.")

        avg_min_distance_stats = {}
        if args.use_validation and all_min_distance_stats:
            avg_min_distance_stats = compute_min_distance_stats_agg(all_min_distance_stats)
            print("\nAverage Minimum Distance Statistics (Validation Sets):")
            for stat, value in avg_min_distance_stats.items():
                print(f"  {stat}: {value:.4f}")
        elif args.use_validation:
            print("\nNo minimum distance statistics were recorded.")

        print("\nBest Validation Metrics Summary Across Runs:")
        if args.use_validation and all_runs_best_metrics:
            for metric in ['val_r_squared', 'val_rmse', 'val_mae', 'val_rpiq']:
                values = [m[metric] for m in all_runs_best_metrics if metric in m and np.isfinite(m[metric])]
                if values:
                    print(f"  {metric} - Mean: {np.mean(values):.4f}, StdDev: {np.std(values):.4f}")
                else:
                    print(f"  {metric}: No finite values available.")
        elif not args.use_validation:
            print("  Validation was disabled.")
        else:
            print("  No best metrics recorded.")

        save_summary_to_file(
            args=args,
            wandb_runs_info=wandb_runs_info,
            avg_best_epoch_metrics=avg_best_epoch_metrics,
            avg_min_distance_stats=avg_min_distance_stats,
            all_runs_best_metrics=all_runs_best_metrics,
            output_dir=output_dir
        )

        try:
            print("\nLogging aggregated results to WandB...")
            final_wandb_run = wandb.init(project=args.project_name, name="summary_agg_results", job_type="summary")
            wandb_runs_info.append({
                'project': final_wandb_run.project,
                'name': final_wandb_run.name,
                'id': final_wandb_run.id
            })

            if avg_best_epoch_metrics:
                final_wandb_run.log({"average_best_epoch_metrics": avg_best_epoch_metrics})

            if avg_min_distance_stats:
                final_wandb_run.log({"average_min_distance_stats": avg_min_distance_stats})

            if args.use_validation and all_runs_best_metrics:
                summary_best = {}
                for metric in ['val_r_squared', 'val_rmse', 'val_mae', 'val_rpiq']:
                    values = [m[metric] for m in all_runs_best_metrics if metric in m and np.isfinite(m[metric])]
                    if values:
                        summary_best[f"avg_best_{metric}"] = np.mean(values)
                        summary_best[f"std_best_{metric}"] = np.std(values)
                        final_wandb_run.summary[f"avg_best_{metric}"] = np.mean(values)
                        final_wandb_run.summary[f"std_best_{metric}"] = np.std(values)
                if summary_best:
                    final_wandb_run.log({"aggregated_best_metrics_stats": summary_best})

            final_wandb_run.finish()
            print("Aggregated results logged to WandB.")

        except Exception as e:
            print(f"Failed to log aggregated results to WandB: {e}")

    accelerator.print("\nScript execution finished.")