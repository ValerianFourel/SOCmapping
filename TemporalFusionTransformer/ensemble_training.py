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
import argparse
from balancedDataset import create_validation_train_sets,create_balanced_dataset
import uuid
import os
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import time
import signal
import threading
from contextlib import contextmanager
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class DistributedSafeOperations:
    """Helper class for safe distributed operations"""

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.max_retries = 3
        self.base_timeout = 30

    def safe_gather_for_metrics(self, tensor, max_retries=3):
        """Safely gather tensors with retry mechanism"""
        if self.accelerator.num_processes <= 1:
            return tensor

        for attempt in range(max_retries):
            try:
                # Ensure tensor is on the right device and contiguous
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                # Use timeout to prevent hanging
                with timeout_context(self.base_timeout * (attempt + 1)):
                    gathered = self.accelerator.gather_for_metrics(tensor)
                    return gathered

            except (RuntimeError, TimeoutError) as e:
                logger.warning(f"Gather attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All gather attempts failed, using local tensor")
                    return tensor

                # Wait before retry with exponential backoff
                time.sleep(2 ** attempt)
                self.accelerator.wait_for_everyone()

        return tensor

    def safe_wait_for_everyone(self, timeout=60):
        """Safely wait for all processes with timeout"""
        try:
            with timeout_context(timeout):
                self.accelerator.wait_for_everyone()
                return True
        except TimeoutError:
            logger.error(f"Wait for everyone timed out after {timeout} seconds")
            return False

    def safe_all_reduce(self, tensor, op='sum', timeout=30):
        """Safely perform all_reduce operation"""
        if self.accelerator.num_processes <= 1:
            return tensor

        try:
            with timeout_context(timeout):
                # Ensure tensor requires grad is False for all_reduce
                if tensor.requires_grad:
                    tensor = tensor.detach()

                if hasattr(self.accelerator.state, 'distributed_type'):
                    if self.accelerator.state.distributed_type != 'NO':
                        torch.distributed.all_reduce(tensor)
                        if op == 'mean':
                            tensor = tensor / self.accelerator.num_processes
                        return tensor
                return tensor
        except (RuntimeError, TimeoutError) as e:
            logger.warning(f"All_reduce failed: {e}, returning local tensor")
            return tensor

class ModelEnsemble:
    """Class to handle ensemble of models for uncertainty quantification"""

    def __init__(self, model_config, device='cpu', accelerator=None):
        self.models = []
        self.model_config = model_config
        self.device = device
        self.normalization_stats = None
        self.accelerator = accelerator
        self.distributed_ops = DistributedSafeOperations(accelerator) if accelerator else None

    def add_model(self, model_state_dict, model_metrics=None):
        """Add a trained model to the ensemble"""
        model = SimpleTFT(**self.model_config)
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        self.models.append({
            'model': model,
            'metrics': model_metrics or {}
        })

    def set_normalization_stats(self, feature_means, feature_stds, target_mean, target_std):
        """Set normalization statistics for the ensemble"""
        self.normalization_stats = {
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'target_mean': target_mean,
            'target_std': target_std
        }

    def predict_ensemble(self, dataloader, target_transform='none'):
        """Generate ensemble predictions with uncertainty estimates"""
        if not self.models:
            raise ValueError("No models in ensemble")

        all_predictions = []
        all_targets = []
        all_coords = []

        # Get predictions from each model
        with torch.no_grad():
            for model_idx, model_info in enumerate(self.models):
                model = model_info['model']
                model_predictions = []
                targets = []
                coords = []

                try:
                    for batch_idx, (longitudes, latitudes, features, batch_targets) in enumerate(dataloader):
                        features = features.to(self.device)
                        batch_targets = batch_targets.to(self.device).float()

                        # Apply target transformation if needed
                        if target_transform == 'log':
                            batch_targets = torch.log(batch_targets + 1e-10)
                        elif target_transform == 'normalize' and self.normalization_stats:
                            batch_targets = (batch_targets - self.normalization_stats['target_mean']) / (self.normalization_stats['target_std'] + 1e-10)

                        outputs = model(features)

                        model_predictions.extend(outputs.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())
                        coords.extend(list(zip(longitudes.numpy(), latitudes.numpy())))

                    all_predictions.append(np.array(model_predictions))
                    if len(all_targets) == 0:  # Only store targets once
                        all_targets = np.array(targets)
                        all_coords = coords

                except Exception as e:
                    logger.error(f"Error in model {model_idx} prediction: {e}")
                    # Continue with other models
                    continue

        if not all_predictions:
            raise RuntimeError("All models failed to generate predictions")

        # Convert to numpy array: (n_models, n_samples)
        ensemble_predictions = np.array(all_predictions)

        # Apply inverse transformation to get original scale
        if target_transform == 'log':
            ensemble_predictions = np.exp(ensemble_predictions)
            all_targets = np.exp(all_targets)
        elif target_transform == 'normalize' and self.normalization_stats:
            ensemble_predictions = ensemble_predictions * self.normalization_stats['target_std'] + self.normalization_stats['target_mean']
            all_targets = all_targets * self.normalization_stats['target_std'] + self.normalization_stats['target_mean']

        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0)
        ensemble_var = np.var(ensemble_predictions, axis=0)

        # Calculate confidence intervals
        confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100

            lower_bounds = np.percentile(ensemble_predictions, lower_percentile, axis=0)
            upper_bounds = np.percentile(ensemble_predictions, upper_percentile, axis=0)

            confidence_intervals[conf_level] = {
                'lower': lower_bounds,
                'upper': upper_bounds,
                'width': upper_bounds - lower_bounds
            }

        return {
            'ensemble_predictions': ensemble_predictions,
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'ensemble_var': ensemble_var,
            'confidence_intervals': confidence_intervals,
            'targets': all_targets,
            'coordinates': all_coords,
            'n_models': len(all_predictions)
        }

    def compute_uncertainty_metrics(self, predictions_dict):
        """Compute various uncertainty metrics"""
        ensemble_mean = predictions_dict['ensemble_mean']
        ensemble_std = predictions_dict['ensemble_std']
        targets = predictions_dict['targets']
        confidence_intervals = predictions_dict['confidence_intervals']

        # Basic regression metrics
        mse = np.mean((ensemble_mean - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_mean - targets))

        # R-squared
        if len(targets) > 1 and np.std(targets) > 1e-6:
            correlation = np.corrcoef(ensemble_mean, targets)[0, 1]
            r_squared = correlation ** 2
        else:
            correlation = 0.0
            r_squared = 0.0

        # RPIQ
        iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
        rpiq = iqr / rmse if rmse > 0 else float('inf')

        # Uncertainty-specific metrics
        mean_prediction_uncertainty = np.mean(ensemble_std)
        uncertainty_spread = np.std(ensemble_std)

        # Coverage probability for confidence intervals
        coverage_probabilities = {}
        for conf_level, intervals in confidence_intervals.items():
            in_interval = (targets >= intervals['lower']) & (targets <= intervals['upper'])
            coverage_prob = np.mean(in_interval)
            coverage_probabilities[conf_level] = coverage_prob

        # Prediction interval score (lower is better)
        def calculate_picp_pinaw(targets, lower, upper, confidence_level):
            """Calculate Prediction Interval Coverage Probability and Normalized Average Width"""
            picp = np.mean((targets >= lower) & (targets <= upper))
            pinaw = np.mean(upper - lower) / (np.max(targets) - np.min(targets))
            return picp, pinaw

        interval_scores = {}
        for conf_level, intervals in confidence_intervals.items():
            picp, pinaw = calculate_picp_pinaw(
                targets, intervals['lower'], intervals['upper'], conf_level
            )
            interval_scores[conf_level] = {'PICP': picp, 'PINAW': pinaw}

        return {
            'regression_metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'correlation': correlation,
                'rpiq': rpiq
            },
            'uncertainty_metrics': {
                'mean_prediction_uncertainty': mean_prediction_uncertainty,
                'uncertainty_spread': uncertainty_spread,
                'coverage_probabilities': coverage_probabilities,
                'interval_scores': interval_scores
            }
        }

    def save_ensemble(self, filepath):
        """Save the entire ensemble"""
        try:
            ensemble_data = {
                'model_config': self.model_config,
                'models': [],
                'normalization_stats': self.normalization_stats,
                'n_models': len(self.models)
            }

            for i, model_info in enumerate(self.models):
                ensemble_data['models'].append({
                    'state_dict': model_info['model'].state_dict(),
                    'metrics': model_info['metrics']
                })

            torch.save(ensemble_data, filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            return False

    @classmethod
    def load_ensemble(cls, filepath, device='cpu', accelerator=None):
        """Load a saved ensemble"""
        try:
            ensemble_data = torch.load(filepath, map_location=device)

            ensemble = cls(ensemble_data['model_config'], device, accelerator)
            ensemble.normalization_stats = ensemble_data['normalization_stats']

            for model_data in ensemble_data['models']:
                ensemble.add_model(model_data['state_dict'], model_data['metrics'])

            return ensemble
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return None

# Composite loss functions (unchanged)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTFT ensemble with uncertainty quantification')
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
    parser.add_argument('--num-ensemble-models', type=int, default=5, help='Number of models in ensemble for uncertainty quantification')
    parser.add_argument('--ensemble-strategy', type=str, default='bootstrap', choices=['bootstrap', 'random_init', 'different_splits'], 
                       help='Strategy for creating ensemble models')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size for the model')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model')
    parser.add_argument('--save_train_and_val', type=bool, default=False, help='Save training and validation data')
    parser.add_argument('--min-r2-threshold', type=float, default=0.40, help='Minimum R² threshold for saving models')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--effective-batch-size', type=int, default=2048, help='Target effective batch size across all GPUs')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None, 
                       help='Number of gradient accumulation steps (auto-calculated if None)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout for distributed operations in seconds')

    return parser.parse_args()

def calculate_gradient_accumulation_steps(effective_batch_size, batch_size, num_processes):
    """Calculate gradient accumulation steps to achieve target effective batch size"""
    accumulation_steps = max(1, effective_batch_size // (batch_size * num_processes))
    actual_effective_batch_size = batch_size * num_processes * accumulation_steps
    return accumulation_steps, actual_effective_batch_size

def train_model(model, train_loader, val_loader, target_mean, target_std, num_epochs=num_epochs, 
                accelerator=None, lr=0.001, loss_type='l1', loss_alpha=0.5, target_transform='none', 
                min_r2=0.5, use_validation=True, gradient_accumulation_steps=1):

    distributed_ops = DistributedSafeOperations(accelerator)

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

    # Prepare with Accelerator - CRITICAL: All processes must call this
    try:
        train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
        if use_validation and val_loader is not None:
            val_loader = accelerator.prepare(val_loader)
    except Exception as e:
        logger.error(f"Error preparing with accelerator: {e}")
        raise

    # Ensure all processes are synchronized after preparation
    if not distributed_ops.safe_wait_for_everyone():
        logger.error("Failed to synchronize processes after accelerator preparation")

    # Handle target normalization if selected
    if target_transform == 'normalize':
        if accelerator.is_main_process:
            logger.info(f"Target mean: {target_mean}, Target std: {target_std}")
            logger.info(f"Using gradient accumulation steps: {gradient_accumulation_steps}")
    else:
        target_mean, target_std = 0.0, 1.0  # No normalization applied

    best_r2 = -float('inf')
    best_model_state = None
    epoch_metrics = []

    for epoch in range(num_epochs):
        # Ensure all processes start the epoch together
        distributed_ops.safe_wait_for_everyone()

        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        # Clear gradients at the start of epoch
        optimizer.zero_grad()

        # Use regular tqdm only on main process
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if accelerator.is_main_process else train_loader

        try:
            for batch_idx, (longitudes, latitudes, features, targets) in enumerate(train_iter):
                try:
                    features = features.to(accelerator.device, non_blocking=True)
                    targets = targets.to(accelerator.device, non_blocking=True).float()

                    # Apply target transformation
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)  # Add small constant to avoid log(0)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)

                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, targets)

                    # Scale loss by accumulation steps for proper averaging
                    scaled_loss = loss / gradient_accumulation_steps

                    # Backward pass (accumulate gradients)
                    accelerator.backward(scaled_loss)

                    # Update weights every gradient_accumulation_steps or at the end of epoch
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        # Clip gradients to prevent explosion
                        if hasattr(accelerator.unwrap_model(model), 'parameters'):
                            torch.nn.utils.clip_grad_norm_(accelerator.unwrap_model(model).parameters(), max_norm=1.0)

                        optimizer.step()
                        optimizer.zero_grad()

                        # Accumulate the original loss value
                        total_loss += loss.item()
                        num_batches += 1

                        if accelerator.is_main_process:
                            try:
                                wandb.log({
                                    'train_batch_loss': loss.item(),
                                    'batch': batch_idx + 1 + epoch * len(train_loader),
                                    'epoch': epoch + 1,
                                })
                            except Exception as e:
                                logger.warning(f"Failed to log to wandb: {e}")

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    # Skip this batch and continue
                    continue

        except Exception as e:
            logger.error(f"Error in training epoch {epoch}: {e}")
            # Continue to next epoch
            continue

        # Synchronize all processes before validation
        distributed_ops.safe_wait_for_everyone()

        # Calculate average training loss across all processes
        if num_batches > 0:
            avg_train_loss = total_loss / num_batches

            # Average across all processes
            loss_tensor = torch.tensor(avg_train_loss, device=accelerator.device)
            avg_loss_tensor = distributed_ops.safe_all_reduce(loss_tensor, op='mean')
            avg_train_loss = avg_loss_tensor.item()
        else:
            avg_train_loss = float('inf')

        # Validation phase
        val_loss = float('nan')
        correlation = float('nan') 
        r_squared = float('nan')
        mse = float('nan')
        rmse = float('nan')
        mae = float('nan')
        rpiq = float('nan')

        if use_validation and val_loader is not None:
            try:
                model.eval()
                val_outputs_list = []
                val_targets_list = []
                val_loss_total = 0.0
                val_batches = 0

                with torch.no_grad():
                    for val_batch_idx, (longitudes, latitudes, features, targets) in enumerate(val_loader):
                        try:
                            features = features.to(accelerator.device, non_blocking=True)
                            targets = targets.to(accelerator.device, non_blocking=True).float()

                            # Apply the same transformation to validation targets
                            original_targets = targets.clone()  # Keep original for metrics
                            if target_transform == 'log':
                                targets = torch.log(targets + 1e-10)
                            elif target_transform == 'normalize':
                                targets = (targets - target_mean) / (target_std + 1e-10)

                            outputs = model(features)
                            batch_val_loss = criterion(outputs, targets)
                            val_loss_total += batch_val_loss.item()
                            val_batches += 1

                            # Convert back to original scale for metrics calculation
                            if target_transform == 'log':
                                outputs_original = torch.exp(outputs)
                                targets_original = original_targets
                            elif target_transform == 'normalize':
                                outputs_original = outputs * target_std + target_mean
                                targets_original = original_targets
                            else:
                                outputs_original = outputs
                                targets_original = targets

                            # Keep tensors on device for gathering
                            val_outputs_list.append(outputs_original.detach())
                            val_targets_list.append(targets_original.detach())

                        except Exception as e:
                            logger.warning(f"Error in validation batch {val_batch_idx}: {e}")
                            continue

                # Average validation loss across processes
                if val_batches > 0:
                    val_loss = val_loss_total / val_batches
                    loss_tensor = torch.tensor(val_loss, device=accelerator.device)
                    val_loss_tensor = distributed_ops.safe_all_reduce(loss_tensor, op='mean')
                    val_loss = val_loss_tensor.item()

                # Process validation results
                if val_outputs_list and val_targets_list:
                    # Concatenate all validation results (still on GPU)
                    val_outputs_all = torch.cat(val_outputs_list, dim=0)
                    val_targets_all = torch.cat(val_targets_list, dim=0)

                    # Safely gather across all processes
                    val_outputs_gathered = distributed_ops.safe_gather_for_metrics(val_outputs_all)
                    val_targets_gathered = distributed_ops.safe_gather_for_metrics(val_targets_all)

                    # Compute metrics only on main process
                    if accelerator.is_main_process:
                        try:
                            val_outputs_np = val_outputs_gathered.cpu().numpy()
                            val_targets_np = val_targets_gathered.cpu().numpy()

                            # Compute metrics on original scale
                            if len(val_outputs_np) > 1 and np.std(val_outputs_np) > 1e-6 and np.std(val_targets_np) > 1e-6:
                                correlation = np.corrcoef(val_outputs_np.flatten(), val_targets_np.flatten())[0, 1]
                                r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
                            else:
                                correlation = 0.0
                                r_squared = 0.0

                            mse = np.mean((val_outputs_np - val_targets_np) ** 2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(val_outputs_np - val_targets_np))

                            # Compute IQR and RPIQ
                            iqr = np.percentile(val_targets_np, 75) - np.percentile(val_targets_np, 25)
                            rpiq = iqr / rmse if rmse > 0 else float('inf')

                        except Exception as e:
                            logger.error(f"Error computing validation metrics: {e}")
                            # Set default values
                            correlation = 0.0
                            r_squared = 0.0
                            mse = float('inf')
                            rmse = float('inf')
                            mae = float('inf')
                            rpiq = 0.0

            except Exception as e:
                logger.error(f"Error in validation phase: {e}")
                # Set default validation values
                val_loss = float('inf')
        else:
            # No validation - assume perfect performance for training-only mode
            r_squared = 1.0

        # Broadcast validation metrics from main process to all processes
        if accelerator.num_processes > 1:
            try:
                metrics_tensor = torch.tensor([r_squared, rmse, mae, rpiq, correlation, val_loss], 
                                            device=accelerator.device)
                if accelerator.is_main_process:
                    # Main process has the computed metrics
                    pass
                else:
                    # Other processes get default values that will be overwritten
                    metrics_tensor.fill_(0.0)

                # Broadcast from main process (rank 0)
                if hasattr(torch.distributed, 'broadcast') and accelerator.num_processes > 1:
                    torch.distributed.broadcast(metrics_tensor, src=0)
                    r_squared, rmse, mae, rpiq, correlation, val_loss = metrics_tensor.cpu().numpy()

            except Exception as e:
                logger.warning(f"Failed to broadcast validation metrics: {e}")

        # Wait for all processes to complete validation before logging
        distributed_ops.safe_wait_for_everyone()

        # Logging and model saving (only main process)
        if accelerator.is_main_process:
            try:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss_avg': avg_train_loss,
                    'val_loss': val_loss,
                    'correlation': correlation,
                    'r_squared': r_squared,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'rpiq': rpiq,
                    'epoch_time': time.time() - epoch_start_time
                }

                try:
                    wandb.log(log_dict)
                except Exception as e:
                    logger.warning(f"Failed to log to wandb: {e}")

                epoch_metrics.append(log_dict)

                # Save model if it has the best R² and meets minimum threshold
                if use_validation and not np.isnan(r_squared) and r_squared > best_r2 and r_squared >= min_r2:
                    try:
                        best_r2 = r_squared
                        best_model_state = accelerator.unwrap_model(model).state_dict().copy()
                        wandb.run.summary['best_r2'] = best_r2
                    except Exception as e:
                        logger.error(f"Failed to save best model state: {e}")

                elif not use_validation and epoch == num_epochs - 1:
                    try:
                        best_r2 = 1.0
                        best_model_state = accelerator.unwrap_model(model).state_dict().copy()
                        wandb.run.summary['best_r2'] = best_r2
                    except Exception as e:
                        logger.error(f"Failed to save final model state: {e}")

                # Print epoch summary
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Training Loss: {avg_train_loss:.4f}')
                if use_validation:
                    print(f'  Validation Loss: {val_loss:.4f}')
                    print(f'  R²: {r_squared:.4f}')
                    print(f'  RMSE: {rmse:.4f}')
                    print(f'  MAE: {mae:.4f}')
                    print(f'  RPIQ: {rpiq:.4f}')

            except Exception as e:
                logger.error(f"Error in logging/saving for epoch {epoch}: {e}")

        # Final synchronization at end of epoch
        distributed_ops.safe_wait_for_everyone()

    # Return final validation outputs for ensemble if available
    try:
        if use_validation and 'val_outputs_gathered' in locals() and accelerator.is_main_process:
            final_val_outputs = val_outputs_gathered.cpu().numpy()
            final_val_targets = val_targets_gathered.cpu().numpy()
        else:
            final_val_outputs = np.array([])
            final_val_targets = np.array([])
    except Exception as e:
        logger.error(f"Error preparing final validation outputs: {e}")
        final_val_outputs = np.array([])
        final_val_targets = np.array([])

    return model, final_val_outputs, final_val_targets, best_model_state, best_r2, epoch_metrics

def create_bootstrap_dataset(df, seed=None):
    """Create a bootstrap sample of the dataset"""
    if seed is not None:
        np.random.seed(seed)
    bootstrap_indices = np.random.choice(len(df), size=len(df), replace=True)
    return df.iloc[bootstrap_indices].reset_index(drop=True)

def compute_training_statistics_oc():
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    target_mean = df_train['OC'].mean()
    target_std = df_train['OC'].std()
    return target_mean, target_std

def plot_uncertainty_analysis(predictions_dict, save_path):
    """Create comprehensive uncertainty analysis plots"""
    try:
        ensemble_mean = predictions_dict['ensemble_mean']
        ensemble_std = predictions_dict['ensemble_std']
        targets = predictions_dict['targets']
        confidence_intervals = predictions_dict['confidence_intervals']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Predictions vs Targets with uncertainty
        ax = axes[0, 0]
        scatter = ax.scatter(targets, ensemble_mean, c=ensemble_std, cmap='viridis', alpha=0.6)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predictions vs True Values\n(Color = Prediction Uncertainty)')
        plt.colorbar(scatter, ax=ax, label='Prediction Std')

        # Add R² to plot
        if len(targets) > 1 and np.std(targets) > 1e-6:
            r_squared = np.corrcoef(ensemble_mean, targets)[0, 1] ** 2
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Residuals vs Predictions
        ax = axes[0, 1]
        residuals = targets - ensemble_mean
        ax.scatter(ensemble_mean, residuals, c=ensemble_std, cmap='viridis', alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predictions\n(Color = Prediction Uncertainty)')

        # 3. Uncertainty distribution
        ax = axes[0, 2]
        ax.hist(ensemble_std, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Uncertainty (Std)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Uncertainty')
        ax.axvline(np.mean(ensemble_std), color='red', linestyle='--', label=f'Mean: {np.mean(ensemble_std):.3f}')
        ax.legend()

        # 4. Confidence intervals coverage
        ax = axes[1, 0]
        conf_levels = list(confidence_intervals.keys())
        coverage_probs = []
        expected_coverage = []

        for conf_level in conf_levels:
            intervals = confidence_intervals[conf_level]
            in_interval = (targets >= intervals['lower']) & (targets <= intervals['upper'])
            coverage_prob = np.mean(in_interval)
            coverage_probs.append(coverage_prob)
            expected_coverage.append(conf_level)

        ax.plot(expected_coverage, coverage_probs, 'bo-', label='Actual Coverage')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Actual Coverage')
        ax.set_title('Confidence Interval Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Prediction uncertainty vs residual magnitude
        ax = axes[1, 1]
        abs_residuals = np.abs(residuals)
        ax.scatter(ensemble_std, abs_residuals, alpha=0.6)
        ax.set_xlabel('Prediction Uncertainty (Std)')
        ax.set_ylabel('Absolute Residuals')
        ax.set_title('Uncertainty vs Absolute Error')

        # Add correlation coefficient
        if len(ensemble_std) > 1:
            corr_coef = np.corrcoef(ensemble_std, abs_residuals)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 6. Model agreement analysis
        ax = axes[1, 2]
        ensemble_predictions = predictions_dict['ensemble_predictions']
        model_agreement = np.std(ensemble_predictions, axis=0) / np.mean(ensemble_predictions, axis=0)
        model_agreement = model_agreement[np.isfinite(model_agreement)]  # Remove inf/nan values

        if len(model_agreement) > 0:
            ax.hist(model_agreement, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Coefficient of Variation')
        ax.set_ylabel('Frequency')
        ax.set_title('Model Agreement\n(Lower = Better Agreement)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True

    except Exception as e:
        logger.error(f"Error creating uncertainty analysis plots: {e}")
        return False

def save_uncertainty_metrics(uncertainty_metrics, save_path):
    """Save uncertainty metrics to file"""
    try:
        with open(save_path, 'w') as f:
            f.write("UNCERTAINTY QUANTIFICATION METRICS\n")
            f.write("=" * 50 + "\n\n")

            # Regression metrics
            f.write("REGRESSION METRICS:\n")
            f.write("-" * 30 + "\n")
            reg_metrics = uncertainty_metrics['regression_metrics']
            for metric, value in reg_metrics.items():
                f.write(f"{metric.upper()}: {value:.6f}\n")

            f.write("\nUNCERTAINTY METRICS:\n")
            f.write("-" * 30 + "\n")
            unc_metrics = uncertainty_metrics['uncertainty_metrics']
            f.write(f"Mean Prediction Uncertainty: {unc_metrics['mean_prediction_uncertainty']:.6f}\n")
            f.write(f"Uncertainty Spread (Std): {unc_metrics['uncertainty_spread']:.6f}\n")

            f.write("\nCOVERAGE PROBABILITIES:\n")
            f.write("-" * 30 + "\n")
            for conf_level, coverage in unc_metrics['coverage_probabilities'].items():
                f.write(f"{conf_level*100:.0f}% CI Coverage: {coverage:.4f} (Expected: {conf_level:.2f})\n")

            f.write("\nPREDICTION INTERVAL SCORES:\n")
            f.write("-" * 30 + "\n")
            for conf_level, scores in unc_metrics['interval_scores'].items():
                f.write(f"{conf_level*100:.0f}% CI:\n")
                f.write(f"  PICP (Coverage): {scores['PICP']:.4f}\n")
                f.write(f"  PINAW (Width): {scores['PINAW']:.4f}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving uncertainty metrics: {e}")
        return False

if __name__ == "__main__":
    args = parse_args()

    # Set num_ensemble_models to 1 if use_validation is False
    if not args.use_validation:
        args.num_ensemble_models = 1
        num_epochs = NUM_EPOCHS_RUN

    # Initialize accelerator FIRST, with better configuration for deadlock prevention
    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=1,  # We handle this manually
            split_batches=False,  # Don't split batches across processes
            mixed_precision='no',  # Disable automatic mixed precision for stability
       #     even_batches=False,  # Ensure even batch distribution
       #     dispatch_batches=False  # Handle batching ourselves for better control
        )
        logger.info(f"Accelerator initialized successfully with {accelerator.num_processes} processes")

        # Wait for all processes to initialize accelerator
        accelerator.wait_for_everyone()

    except Exception as e:
        logger.error(f"Failed to initialize accelerator: {e}")
        raise

    # Initialize distributed operations helper
    distributed_ops = DistributedSafeOperations(accelerator)

    # Calculate gradient accumulation steps if not provided
    if args.gradient_accumulation_steps is None:
        gradient_accumulation_steps, actual_effective_batch_size = calculate_gradient_accumulation_steps(
            args.effective_batch_size, args.batch_size, accelerator.num_processes
        )
        args.gradient_accumulation_steps = gradient_accumulation_steps
    else:
        actual_effective_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        print(f"Batch configuration:")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Number of GPUs: {accelerator.num_processes}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {actual_effective_batch_size}")
        print(f"  Target effective batch size: {args.effective_batch_size}")

    # Create experiment folder with descriptive naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (f"TFT_ensemble_{timestamp}_"
                      f"OC{MAX_OC}_"
                      f"{TIME_BEGINNING}to{TIME_END}_"
                      f"transform_{args.target_transform}_"
                      f"loss_{args.loss_type}_"
                      f"ensemble_{args.num_ensemble_models}_"
                      f"strategy_{args.ensemble_strategy}_"
                      f"lr_{args.lr}_"
                      f"heads_{args.num_heads}_"
                      f"layers_{args.num_layers}_"
                      f"effbs_{actual_effective_batch_size}")

    experiment_dir = os.path.join(args.output_dir, experiment_name)

    # Create experiment configuration
    experiment_config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "args": vars(args),
        "batch_config": {
            "batch_size_per_gpu": args.batch_size,
            "num_gpus": accelerator.num_processes,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": actual_effective_batch_size,
            "target_effective_batch_size": args.effective_batch_size
        },
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

    # Create experiment directory (only main process)
    if accelerator.is_main_process:
        try:
            os.makedirs(experiment_dir, exist_ok=True)
            logger.info(f"Created experiment directory: {experiment_dir}")

            # Save experiment configuration
            config_file = os.path.join(experiment_dir, "experiment_config.json")
            with open(config_file, 'w') as f:
                json.dump(experiment_config, f, indent=2)
            logger.info(f"Experiment configuration saved to: {config_file}")
        except Exception as e:
            logger.error(f"Failed to create experiment directory: {e}")
            raise

    # Synchronize all processes after directory creation
    distributed_ops.safe_wait_for_everyone()

    # Initialize ensemble tracking
    ensemble_models = []
    ensemble_metrics = []
    wandb_runs_info = []
    all_val_predictions = []

    # Data preparation (all processes)
    try:
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

        # Get feature normalization statistics from full dataset
        train_dataset_features_norm = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, df)
        feature_means = train_dataset_features_norm.get_feature_means()
        feature_stds = train_dataset_features_norm.get_feature_stds()
        target_mean, target_std = compute_training_statistics_oc()

        if accelerator.is_main_process:
            logger.info(f"Dataset size: {len(df)}")
            logger.info(f"Feature normalization computed")

    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise

    # Synchronize after data preparation
    distributed_ops.safe_wait_for_everyone()

    # Create validation set (same for all ensemble models if using validation)
    val_loader = None
    base_train_df = df

    if args.use_validation:
        try:
            # Only main process creates the validation set
            if accelerator.is_main_process:
                val_df, base_train_df, min_distance_stats = create_validation_train_sets(
                    df=df,
                    output_dir=experiment_dir,
                    target_val_ratio=args.target_val_ratio,
                    use_gpu=args.use_gpu,
                    distance_threshold=args.distance_threshold
                )
                logger.info(f"Validation set size: {len(val_df)}")
                logger.info(f"Base training set size: {len(base_train_df)}")

                # Save the splits for other processes
                val_df_path = os.path.join(experiment_dir, 'val_df.pkl')
                base_train_df_path = os.path.join(experiment_dir, 'base_train_df.pkl')
                with open(val_df_path, 'wb') as f:
                    pickle.dump(val_df, f)
                with open(base_train_df_path, 'wb') as f:
                    pickle.dump(base_train_df, f)

            # Wait for main process to create the files
            distributed_ops.safe_wait_for_everyone()

            # All processes load the same splits
            val_df_path = os.path.join(experiment_dir, 'val_df.pkl')
            base_train_df_path = os.path.join(experiment_dir, 'base_train_df.pkl')

            with open(val_df_path, 'rb') as f:
                val_df = pickle.load(f)
            with open(base_train_df_path, 'rb') as f:
                base_train_df = pickle.load(f)

            # Create validation dataset and loader (all processes)
            val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            val_dataset.set_feature_means(feature_means)
            val_dataset.set_feature_stds(feature_stds)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                  pin_memory=True, num_workers=2)

        except Exception as e:
            logger.error(f"Error creating validation set: {e}")
            # Fallback to no validation
            args.use_validation = False
            val_loader = None
            base_train_df = df

    # Synchronize before starting ensemble training
    distributed_ops.safe_wait_for_everyone()

    # Train ensemble models
    for model_idx in range(args.num_ensemble_models):
        if accelerator.is_main_process:
            logger.info(f"Training Ensemble Model {model_idx + 1}/{args.num_ensemble_models}")

        # Initialize wandb for this model (only main process)
        wandb_run = None
        if accelerator.is_main_process:
            try:
                wandb_run = wandb.init(
                    project="socmapping-SimpleTFT-Ensemble",
                    name=f"{experiment_name}_model_{model_idx+1}",
                    config={
                        "experiment_name": experiment_name,
                        "ensemble_model_number": model_idx + 1,
                        "total_ensemble_size": args.num_ensemble_models,
                        "ensemble_strategy": args.ensemble_strategy,
                        "max_oc": MAX_OC,
                        "time_beginning": TIME_BEGINNING,
                        "time_end": TIME_END,
                        "window_size": window_size,
                        "time_before": time_before,
                        "bands": len(bands_list_order),
                        "epochs": num_epochs,
                        "batch_size_per_gpu": args.batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "effective_batch_size": actual_effective_batch_size,
                        "learning_rate": args.lr,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "dropout_rate": args.dropout_rate,
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
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

        # Synchronize before data creation
        distributed_ops.safe_wait_for_everyone()

        # Create training dataset based on ensemble strategy (all processes must do this identically)
        try:
            if args.ensemble_strategy == 'bootstrap':
                # Bootstrap sampling
                train_df = create_bootstrap_dataset(base_train_df, seed=model_idx)
            elif args.ensemble_strategy == 'different_splits':
                # Different train/val splits (if using validation)
                if args.use_validation:
                    if accelerator.is_main_process:
                        train_df, _, _ = create_validation_train_sets(
                            df=df,
                            output_dir=experiment_dir,
                            target_val_ratio=args.target_val_ratio,
                            use_gpu=args.use_gpu,
                            distance_threshold=args.distance_threshold,
                            random_seed=model_idx
                        )
                        # Save for other processes
                        train_df_path = os.path.join(experiment_dir, f'train_df_model_{model_idx}.pkl')
                        with open(train_df_path, 'wb') as f:
                            pickle.dump(train_df, f)

                    # Wait and load
                    distributed_ops.safe_wait_for_everyone()
                    train_df_path = os.path.join(experiment_dir, f'train_df_model_{model_idx}.pkl')
                    with open(train_df_path, 'rb') as f:
                        train_df = pickle.load(f)
                else:
                    train_df = base_train_df
            else:  # 'random_init' - same data, different random initialization
                train_df = base_train_df

            # Apply balanced sampling if needed (all processes)
            train_df, _ = create_balanced_dataset(train_df, min_ratio=3/4, use_validation=False)

            # Create training dataset and loader (all processes)
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            train_dataset.set_feature_means(feature_means)
            train_dataset.set_feature_stds(feature_stds)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                    pin_memory=True, num_workers=2)

            if accelerator.is_main_process:
                logger.info(f"Model {model_idx + 1} training set size: {len(train_df)}")

        except Exception as e:
            logger.error(f"Error creating training dataset for model {model_idx}: {e}")
            continue

        # Synchronize before model initialization
        distributed_ops.safe_wait_for_everyone()

        # Initialize model with different random seed for each ensemble member (all processes)
        try:
            torch.manual_seed(model_idx * 42 + 123)  # Different seed pattern
            np.random.seed(model_idx * 42 + 123)

            model = SimpleTFT(
                input_channels=len(bands_list_order),
                height=window_size,
                width=window_size,
                time_steps=time_before,
                d_model=args.hidden_size
            )

            if accelerator.is_main_process:
                logger.info(f"Model {model_idx + 1} parameters: {model.count_parameters()}")
                if wandb_run:
                    wandb_run.summary["model_parameters"] = model.count_parameters()
                    wandb_run.summary["train_size"] = len(train_df)

        except Exception as e:
            logger.error(f"Error initializing model {model_idx}: {e}")
            continue

        # Train model
        try:
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
                min_r2=args.min_r2_threshold,
                use_validation=args.use_validation,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
        except Exception as e:
            logger.error(f"Error training model {model_idx}: {e}")
            # Clean up and continue
            if accelerator.is_main_process and wandb_run:
                wandb_run.finish()
            continue

        # Store model if it meets criteria (only main process)
        if accelerator.is_main_process:
            if best_model_state is not None and best_r2 >= args.min_r2_threshold:
                # Find best metrics for this model
                best_metrics = {'r_squared': -float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'rpiq': -float('inf')}
                for epoch_metric in epoch_metrics:
                    if not np.isnan(epoch_metric['r_squared']) and epoch_metric['r_squared'] > best_metrics['r_squared']:
                        best_metrics = {
                            'r_squared': epoch_metric['r_squared'],
                            'rmse': epoch_metric['rmse'],
                            'mae': epoch_metric['mae'],
                            'rpiq': epoch_metric['rpiq']
                        }

                ensemble_models.append(best_model_state)
                ensemble_metrics.append(best_metrics)
                logger.info(f"Model {model_idx + 1} accepted with R²: {best_r2:.4f}")
            else:
                logger.info(f"Model {model_idx + 1} rejected - R² threshold not met (R²: {best_r2:.4f})")

            # Save individual model
            if best_model_state is not None:
                try:
                    models_dir = os.path.join(experiment_dir, "individual_models")
                    os.makedirs(models_dir, exist_ok=True)

                    model_path = os.path.join(models_dir, f'TFT_ensemble_model_{model_idx+1}_R2_{best_r2:.4f}.pth')
                    model_with_metadata = {
                        'model_state_dict': best_model_state,
                        'model_number': model_idx + 1,
                        'best_r2': best_r2,
                        'ensemble_strategy': args.ensemble_strategy,
                        'model_config': {
                            'input_channels': len(bands_list_order),
                            'height': window_size,
                            'width': window_size,
                            'time_steps': time_before,
                            'd_model': args.hidden_size
                        },
                        'batch_config': {
                            'batch_size_per_gpu': args.batch_size,
                            'gradient_accumulation_steps': args.gradient_accumulation_steps,
                            'effective_batch_size': actual_effective_batch_size
                        }
                    }
                    torch.save(model_with_metadata, model_path)
                    if wandb_run:
                        wandb_run.save(model_path)
                except Exception as e:
                    logger.error(f"Error saving individual model {model_idx}: {e}")

        # Clean up wandb
        if accelerator.is_main_process and wandb_run:
            try:
                wandb_run.finish()
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")

        # Clean up memory and synchronize
        del model, train_loader, train_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for all processes before next iteration
        distributed_ops.safe_wait_for_everyone()

    # Create and evaluate ensemble (only main process)
    if accelerator.is_main_process and len(ensemble_models) > 0:
        try:
            logger.info(f"Creating ensemble with {len(ensemble_models)} models")

            # Initialize ensemble
            model_config = {
                'input_channels': len(bands_list_order),
                'height': window_size,
                'width': window_size,
                'time_steps': time_before,
                'd_model': args.hidden_size
            }

            ensemble = ModelEnsemble(model_config, device=accelerator.device, accelerator=accelerator)
            ensemble.set_normalization_stats(feature_means, feature_stds, target_mean, target_std)

            # Add models to ensemble
            for i, (model_state, metrics) in enumerate(zip(ensemble_models, ensemble_metrics)):
                ensemble.add_model(model_state, metrics)

            # Evaluate ensemble on validation set
            if args.use_validation and val_loader is not None:
                try:
                    logger.info("Evaluating ensemble on validation set...")
                    predictions_dict = ensemble.predict_ensemble(val_loader, target_transform=args.target_transform)
                    uncertainty_metrics = ensemble.compute_uncertainty_metrics(predictions_dict)

                    # Create results directory
                    results_dir = os.path.join(experiment_dir, "ensemble_results")
                    os.makedirs(results_dir, exist_ok=True)

                    # Save uncertainty analysis plots
                    plot_path = os.path.join(results_dir, "uncertainty_analysis.png")
                    if plot_uncertainty_analysis(predictions_dict, plot_path):
                        logger.info(f"Uncertainty analysis plots saved to: {plot_path}")

                    # Save uncertainty metrics
                    metrics_path = os.path.join(results_dir, "uncertainty_metrics.txt")
                    if save_uncertainty_metrics(uncertainty_metrics, metrics_path):
                        logger.info(f"Uncertainty metrics saved to: {metrics_path}")

                    # Save detailed results
                    results_dict = {
                        'predictions_summary': {
                            'ensemble_mean': predictions_dict['ensemble_mean'].tolist(),
                            'ensemble_std': predictions_dict['ensemble_std'].tolist(),
                            'targets': predictions_dict['targets'].tolist(),
                            'n_models': predictions_dict['n_models']
                        },
                        'uncertainty_metrics': uncertainty_metrics,
                        'ensemble_info': {
                            'n_models': len(ensemble_models),
                            'strategy': args.ensemble_strategy,
                            'min_r2_threshold': args.min_r2_threshold
                        },
                        'batch_config': {
                            'batch_size_per_gpu': args.batch_size,
                            'gradient_accumulation_steps': args.gradient_accumulation_steps,
                            'effective_batch_size': actual_effective_batch_size
                        }
                    }

                    results_json_path = os.path.join(results_dir, "ensemble_results.json")
                    try:
                        with open(results_json_path, 'w') as f:
                            json.dump(results_dict, f, indent=2, default=str)
                        logger.info(f"Detailed results saved to: {results_json_path}")
                    except Exception as e:
                        logger.error(f"Error saving results JSON: {e}")

                    # Print summary
                    print("\nENSEMBLE EVALUATION SUMMARY:")
                    print("=" * 50)
                    reg_metrics = uncertainty_metrics['regression_metrics']
                    unc_metrics = uncertainty_metrics['uncertainty_metrics']
                    print(f"Ensemble Size: {len(ensemble_models)} models")
                    print(f"R²: {reg_metrics['r_squared']:.4f}")
                    print(f"RMSE: {reg_metrics['rmse']:.4f}")
                    print(f"MAE: {reg_metrics['mae']:.4f}")
                    print(f"RPIQ: {reg_metrics['rpiq']:.4f}")
                    print(f"Mean Prediction Uncertainty: {unc_metrics['mean_prediction_uncertainty']:.4f}")
                    print(f"Uncertainty Spread: {unc_metrics['uncertainty_spread']:.4f}")

                    # Coverage probabilities
                    print("\nCOVERAGE PROBABILITIES:")
                    for conf_level, coverage in unc_metrics['coverage_probabilities'].items():
                        print(f"{conf_level*100:.0f}% CI: {coverage:.3f} (Expected: {conf_level:.2f})")

                    # Log to wandb
                    try:
                        wandb_run = wandb.init(
                            project="socmapping-SimpleTFT-Ensemble",
                            name=f"{experiment_name}_ensemble_summary"
                        )
                        wandb_run.log({
                            "ensemble_size": len(ensemble_models),
                            "ensemble_r_squared": reg_metrics['r_squared'],
                            "ensemble_rmse": reg_metrics['rmse'],
                            "ensemble_mae": reg_metrics['mae'],
                            "ensemble_rpiq": reg_metrics['rpiq'],
                            "mean_prediction_uncertainty": unc_metrics['mean_prediction_uncertainty'],
                            "uncertainty_spread": unc_metrics['uncertainty_spread'],
                            "effective_batch_size": actual_effective_batch_size,
                            "gradient_accumulation_steps": args.gradient_accumulation_steps
                        })

                        # Log coverage probabilities
                        for conf_level, coverage in unc_metrics['coverage_probabilities'].items():
                            wandb_run.log({f"coverage_{int(conf_level*100)}pct": coverage})

                        if os.path.exists(plot_path):
                            wandb_run.save(plot_path)
                        if os.path.exists(metrics_path):
                            wandb_run.save(metrics_path)
                        if os.path.exists(results_json_path):
                            wandb_run.save(results_json_path)
                        wandb_run.finish()
                    except Exception as e:
                        logger.warning(f"Error with wandb logging: {e}")

                except Exception as e:
                    logger.error(f"Error evaluating ensemble: {e}")

            # Save complete ensemble
            try:
                ensemble_path = os.path.join(experiment_dir, "complete_ensemble.pth")
                if ensemble.save_ensemble(ensemble_path):
                    logger.info(f"Complete ensemble saved to: {ensemble_path}")
                else:
                    logger.error("Failed to save complete ensemble")
            except Exception as e:
                logger.error(f"Error saving ensemble: {e}")

        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")

    elif accelerator.is_main_process:
        logger.info("No models met the R² threshold - no ensemble created")

    # Final synchronization
    distributed_ops.safe_wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\nEnsemble experiment completed!")
        print(f"Results saved in: {experiment_dir}")
        if len(ensemble_models) > 0:
            print(f"Ensemble contains {len(ensemble_models)} models")
        else:
            print("No ensemble created due to insufficient model quality")

    accelerator.print("Ensemble training completed successfully!")
