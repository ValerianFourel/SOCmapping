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
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, NUM_EPOCHS_RUN,
                   seasons, years_padded, num_epochs, NUM_HEADS, NUM_LAYERS, hidden_size,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from SimpleTFT import SimpleTFT
import argparse
from balancedDataset import create_validation_train_sets, create_balanced_dataset
import uuid
import os
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Class to handle ensemble of models for uncertainty quantification"""

    def __init__(self, model_config, device='cpu'):
        self.models = []
        self.model_config = model_config
        self.device = device
        self.normalization_stats = None

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

        # Basic regression metrics (same as train.py)
        mse = np.mean((ensemble_mean - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_mean - targets))

        # R-squared (same as train.py)
        if len(targets) > 1 and np.std(targets) > 1e-6:
            correlation = np.corrcoef(ensemble_mean, targets)[0, 1]
            r_squared = correlation ** 2
        else:
            correlation = 0.0
            r_squared = 0.0

        # RPIQ (same as train.py)
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

        # Prediction interval scores
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
    def load_ensemble(cls, filepath, device='cpu'):
        """Load a saved ensemble"""
        try:
            ensemble_data = torch.load(filepath, map_location=device)
            ensemble = cls(ensemble_data['model_config'], device)
            ensemble.normalization_stats = ensemble_data['normalization_stats']

            for model_data in ensemble_data['models']:
                ensemble.add_model(model_data['state_dict'], model_data['metrics'])

            return ensemble
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return None

# Loss functions (same as train.py)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTFT ensemble with uncertainty quantification')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['composite_l1', 'l1', 'mse', 'composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='normalize', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', type=bool, default=True, help='Whether to use validation set')
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

    return parser.parse_args()

def train_model(model, train_loader, val_loader, target_mean, target_std, num_epochs=num_epochs, 
                device='cpu', lr=0.001, loss_type='l1', loss_alpha=0.5, target_transform='none', 
                min_r2=0.5, use_validation=True):
    """
    Training function that matches train.py logic exactly - simplified for single process
    """
    # Define loss function based on loss_type (same as train.py)
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
    model = model.to(device)

    # Handle target normalization if selected (same as train.py)
    if target_transform == 'normalize':
        print(f"Target mean: {target_mean}, Target std: {target_std}")
    else:
        target_mean, target_std = 0.0, 1.0  # No normalization applied

    best_r2 = -float('inf')
    best_model_state = None
    epoch_metrics = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop exactly like train.py
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(device)
            targets = targets.to(device).float()

            # Apply target transformation (same as train.py)
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)  # Add small constant to avoid log(0)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            try:
                wandb.log({
                    'train_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })
            except:
                pass  # Ignore wandb errors

        train_loss = running_loss / len(train_loader)

        # Validation phase exactly like train.py
        if use_validation and val_loader is not None:
            model.eval()
            val_outputs = []
            val_targets_list = []

            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device).float()

                    # Apply the same transformation to validation targets
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)

                    outputs = model(features)
                    val_outputs.extend(outputs.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())

            # Convert to numpy arrays (same as train.py)
            val_outputs_all = np.array(val_outputs)
            val_targets_all = np.array(val_targets_list)

            # Apply inverse transformation to get original scale (same as train.py)
            if target_transform == 'log':
                original_val_outputs = np.exp(val_outputs_all)
                original_val_targets = np.exp(val_targets_all)
            elif target_transform == 'normalize':
                original_val_outputs = val_outputs_all * target_std + target_mean
                original_val_targets = val_targets_all * target_std + target_mean
            else:
                original_val_outputs = val_outputs_all
                original_val_targets = val_targets_all

            # Compute metrics on original scale (same as train.py)
            if len(original_val_outputs) > 1 and np.std(original_val_outputs) > 1e-6 and np.std(original_val_targets) > 1e-6:
                correlation = np.corrcoef(original_val_outputs, original_val_targets)[0, 1]
                r_squared = correlation ** 2
            else:
                correlation = 0.0
                r_squared = 0.0
            mse = np.mean((original_val_outputs - original_val_targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(original_val_outputs - original_val_targets))

            # Compute IQR and RPIQ (same as train.py)
            iqr = np.percentile(original_val_targets, 75) - np.percentile(original_val_targets, 25)
            rpiq = iqr / rmse if rmse > 0 else float('inf')

            # Compute validation loss on transformed scale (same as train.py)
            val_outputs_tensor = torch.from_numpy(val_outputs_all).to(device)
            val_targets_tensor = torch.from_numpy(val_targets_all).to(device)
            val_loss = criterion(val_outputs_tensor, val_targets_tensor).item()

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

        try:
            wandb.log(log_dict)
        except:
            pass  # Ignore wandb errors

        epoch_metrics.append(log_dict)

        # Save model if it has the best R² and meets minimum threshold (same as train.py)
        if use_validation and r_squared > best_r2 and r_squared >= min_r2:
            best_r2 = r_squared
            best_model_state = model.state_dict()
            try:
                wandb.run.summary['best_r2'] = best_r2
            except:
                pass  # Ignore wandb errors
        elif not use_validation and epoch == num_epochs - 1:
            best_r2 = 1.0
            best_model_state = model.state_dict()
            try:
                wandb.run.summary['best_r2'] = best_r2
            except:
                pass  # Ignore wandb errors

        # Print epoch summary (same as train.py)
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss:.4f}')
        if use_validation:
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'R²: {r_squared:.4f}')
            print(f'RMSE: {rmse:.4f}')
            print(f'MAE: {mae:.4f}')
            print(f'RPIQ: {rpiq:.4f}\n')

    return model, val_outputs, val_targets_list, best_model_state, best_r2, epoch_metrics

def create_bootstrap_dataset(df, seed=None):
    """Create a bootstrap sample of the dataset"""
    if seed is not None:
        np.random.seed(seed)
    bootstrap_indices = np.random.choice(len(df), size=len(df), replace=True)
    return df.iloc[bootstrap_indices].reset_index(drop=True)

def compute_training_statistics_oc():
    """Compute training statistics (same as train.py)"""
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

    # Set device (simple GPU setup like train.py)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

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
                      f"layers_{args.num_layers}")

    experiment_dir = os.path.join(args.output_dir, experiment_name)

    # Create experiment configuration
    experiment_config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "args": vars(args),
        "device": str(device),
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

    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Created experiment directory: {experiment_dir}")

    # Save experiment configuration
    config_file = os.path.join(experiment_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    logger.info(f"Experiment configuration saved to: {config_file}")

    # Initialize ensemble tracking
    ensemble_models = []
    ensemble_metrics = []
    wandb_runs_info = []

    # Data preparation (same as train.py)
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

    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Feature normalization computed")

    # Create validation set (same for all ensemble models if using validation)
    val_loader = None
    base_train_df = df

    if args.use_validation:
        val_df, base_train_df, min_distance_stats = create_validation_train_sets(
            df=df,
            output_dir=experiment_dir,
            target_val_ratio=args.target_val_ratio,
            use_gpu=args.use_gpu,
            distance_threshold=args.distance_threshold
        )
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Base training set size: {len(base_train_df)}")

        # Create validation dataset and loader - use batch_size 256 like train.py
        val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
        val_dataset.set_feature_means(feature_means)
        val_dataset.set_feature_stds(feature_stds)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train ensemble models
    for model_idx in range(args.num_ensemble_models):
        logger.info(f"Training Ensemble Model {model_idx + 1}/{args.num_ensemble_models}")

        # Initialize wandb for this model
        wandb_run = None
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
                    "batch_size": 256,  # Same as train.py
                    "learning_rate": args.lr,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout_rate": args.dropout_rate,
                    "loss_type": args.loss_type,
                    "loss_alpha": args.loss_alpha,
                    "target_transform": args.target_transform,
                    "use_validation": args.use_validation,
                    "device": str(device)
                }
            )
            wandb_runs_info.append({
                'project': wandb_run.project,
                'name': wandb_run.name,
                'id': wandb_run.id
            })
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

        # Create training dataset based on ensemble strategy
        if args.ensemble_strategy == 'bootstrap':
            # Bootstrap sampling
            train_df = create_bootstrap_dataset(base_train_df, seed=model_idx)
        elif args.ensemble_strategy == 'different_splits':
            # Different train/val splits (if using validation)
            if args.use_validation:
                train_df, _, _ = create_validation_train_sets(
                    df=df,
                    output_dir=experiment_dir,
                    target_val_ratio=args.target_val_ratio,
                    use_gpu=args.use_gpu,
                    distance_threshold=args.distance_threshold,
                    random_seed=model_idx
                )
            else:
                train_df = base_train_df
        else:  # 'random_init' - same data, different random initialization
            train_df = base_train_df

        # Apply balanced sampling if needed - same as train.py
        train_df, _ = create_balanced_dataset(train_df, min_ratio=3/4, use_validation=False)

        # Create training dataset and loader - use batch_size 256 like train.py
        train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        train_dataset.set_feature_means(feature_means)
        train_dataset.set_feature_stds(feature_stds)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        logger.info(f"Model {model_idx + 1} training set size: {len(train_df)}")

        # Initialize model with different random seed for each ensemble member
        torch.manual_seed(model_idx * 42 + 123)  # Different seed pattern
        np.random.seed(model_idx * 42 + 123)

        model = SimpleTFT(
            input_channels=len(bands_list_order),
            height=window_size,
            width=window_size,
            time_steps=time_before,
            d_model=args.hidden_size
        )

        logger.info(f"Model {model_idx + 1} parameters: {model.count_parameters()}")
        if wandb_run:
            wandb_run.summary["model_parameters"] = model.count_parameters()
            wandb_run.summary["train_size"] = len(train_df)

        # Train model using the simplified single-process function
        model, val_outputs, val_targets, best_model_state, best_r2, epoch_metrics = train_model(
            model,
            train_loader,
            val_loader,
            target_mean=target_mean,
            target_std=target_std,
            num_epochs=num_epochs,
            device=device,
            lr=args.lr,
            loss_type=args.loss_type,
            loss_alpha=args.loss_alpha,
            target_transform=args.target_transform,
            min_r2=args.min_r2_threshold,
            use_validation=args.use_validation,
        )

        # Store model if it meets criteria
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
                }
            }
            torch.save(model_with_metadata, model_path)
            if wandb_run:
                wandb_run.save(model_path)

        # Clean up wandb
        if wandb_run:
            wandb_run.finish()

        # Clean up memory
        del model, train_loader, train_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create and evaluate ensemble
    if len(ensemble_models) > 0:
        logger.info(f"Creating ensemble with {len(ensemble_models)} models")

        # Initialize ensemble
        model_config = {
            'input_channels': len(bands_list_order),
            'height': window_size,
            'width': window_size,
            'time_steps': time_before,
            'd_model': args.hidden_size
        }

        ensemble = ModelEnsemble(model_config, device=device)
        ensemble.set_normalization_stats(feature_means, feature_stds, target_mean, target_std)

        # Add models to ensemble
        for i, (model_state, metrics) in enumerate(zip(ensemble_models, ensemble_metrics)):
            ensemble.add_model(model_state, metrics)

        # Evaluate ensemble on validation set
        if args.use_validation and val_loader is not None:
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
                }
            }

            results_json_path = os.path.join(results_dir, "ensemble_results.json")
            with open(results_json_path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            logger.info(f"Detailed results saved to: {results_json_path}")

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
                    "uncertainty_spread": unc_metrics['uncertainty_spread']
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

        # Save complete ensemble
        ensemble_path = os.path.join(experiment_dir, "complete_ensemble.pth")
        if ensemble.save_ensemble(ensemble_path):
            logger.info(f"Complete ensemble saved to: {ensemble_path}")

    else:
        logger.info("No models met the R² threshold - no ensemble created")

    print(f"\nEnsemble experiment completed!")
    print(f"Results saved in: {experiment_dir}")
    if len(ensemble_models) > 0:
        print(f"Ensemble contains {len(ensemble_models)} models")
    else:
        print("No ensemble created due to insufficient model quality")

    print("Ensemble training completed successfully!")
