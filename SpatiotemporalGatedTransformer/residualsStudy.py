import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
import argparse
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from scipy import stats

from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import separate_and_add_data
from torch.utils.data import DataLoader
from SimpleSGT import SimpleSGT
# from EnhancedSGT import EnhancedSGT as SimpleSGT
from config import (TIME_BEGINNING, TIME_END, MAX_OC,
                   NUM_HEADS, NUM_LAYERS, hidden_size, bands_list_order, 
                   window_size, time_before)

# Set publication-quality font sizes and styling globally
plt.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 22,      # Axis label font size
    'xtick.labelsize': 18,     # X-axis tick labels
    'ytick.labelsize': 18,     # Y-axis tick labels
    'legend.fontsize': 18,     # Legend font size
    'figure.titlesize': 26,    # Figure title
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.linewidth': 2.0,     # Thicker axis lines
    'grid.linewidth': 1.2,     # Thicker grid lines
    'lines.linewidth': 2.5,    # Thicker plot lines
    'lines.markersize': 8,     # Larger markers
})

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SimpleSGT model residuals and generate visualizations')
    parser.add_argument('--model-path', type=str, default='/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels360k_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_none_LOSS_composite_l2/TFT_model_BEST_OVERALL_from_run_3_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_none_LOSS_composite_l2_R2_0.5173.pth', help='Path to the trained model .pth file')
    parser.add_argument('--data-path', type=str, default='/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels360k_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_none_LOSS_composite_l2/train_val_data_run_3_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_none_LOSS_composite_l2.parquet', help='Path to the parquet file containing train/val data')
    parser.add_argument('--stats-path', type=str, default='/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels360k_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_none_LOSS_composite_l2/normalization_stats_run_3_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_none_LOSS_composite_l2.pkl', help='Path to normalization stats pickle file')
    parser.add_argument('--output-dir', type=str, default='residual_analysis', help='Directory to save output visualizations')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU for inference')
    parser.add_argument('--hidden-size', type=int, default=hidden_size, help='Hidden size for the model')
    return parser.parse_args()

def extract_transform_info(model_path):
    """Extract transformation info from model filename"""
    if 'log' in model_path:
        return 'log'
    elif 'normalize' in model_path:
        return 'normalize'
    else:
        return 'none'


def load_model(model_path, device, hidden_size):
    """Load the trained SGT model"""
    model = SimpleSGT(
        input_channels=len(bands_list_order),
        height=window_size,
        width=window_size,
        time_steps=time_before,
        d_model=hidden_size
    )

    # Load the checkpoint and extract the model state dict
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # If checkpoint is directly the state dict
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded with {model.count_parameters():,} parameters")
    return model


def run_inference(model, dataloader, target_transform, target_mean, target_std, device):
    """Run inference on a dataset and return predictions and targets"""
    all_outputs = []
    all_targets = []
    all_longitudes = []
    all_latitudes = []

    with torch.no_grad():
        for longitudes, latitudes, features, targets in tqdm(dataloader, desc="Running inference"):
            features = features.to(device)
            original_targets = targets.clone().cpu().numpy()

            # Apply target transformation if needed (for comparison with model output)
            if target_transform == 'log':
                targets_transformed = torch.log(targets + 1e-10)
            elif target_transform == 'normalize':
                targets_transformed = (targets - target_mean) / (target_std + 1e-10)
            else:
                targets_transformed = targets

            outputs = model(features)

            # Transform outputs back to original scale
            if target_transform == 'log':
                outputs = torch.exp(outputs)
            elif target_transform == 'normalize':
                outputs = outputs * target_std + target_mean

            # Collect results
            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(original_targets)
            all_longitudes.extend(longitudes.cpu().numpy())
            all_latitudes.extend(latitudes.cpu().numpy())

    return np.array(all_outputs), np.array(all_targets), np.array(all_longitudes), np.array(all_latitudes)

def calculate_metrics(predictions, targets):
    """Calculate various performance metrics"""
    residuals = targets - predictions

    # Calculate R² as squared correlation coefficient
    correlation_coeff = np.corrcoef(targets, predictions)[0, 1]
    r2 = correlation_coeff ** 2

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
    rpiq = iqr / rmse if rmse > 0 else float('inf')

    return {
        'residuals': residuals,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'rpiq': rpiq,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'median_residual': np.median(residuals),
        'residual_iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25)
    }


def create_visualizations(train_results, val_results, output_dir):
    """Create and save visualizations of model performance and residuals"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Scatter plot of predicted vs actual for both train and val
    plt.figure(figsize=(14, 12))
    plt.scatter(train_results['targets'], train_results['predictions'], 
                alpha=0.5, s=50, label=f'Training (R² = {train_results["metrics"]["r2"]:.4f})')
    plt.scatter(val_results['targets'], val_results['predictions'], 
                alpha=0.5, s=50, label=f'Validation (R² = {val_results["metrics"]["r2"]:.4f})')

    # Add perfect prediction line
    min_val = min(np.min(train_results['targets']), np.min(val_results['targets']))
    max_val = max(np.max(train_results['targets']), np.max(val_results['targets']))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=3, label='Perfect prediction')

    plt.xlabel('Actual Organic Carbon', labelpad=12)
    plt.ylabel('Predicted Organic Carbon', labelpad=12)
    plt.title('Predicted vs Actual Organic Carbon', pad=20, weight='bold')
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Residual histogram for train and validation
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.histplot(train_results['metrics']['residuals'], kde=True, ax=axes[0], bins=30)
    axes[0].set_title(f'Training Residuals (Mean: {train_results["metrics"]["mean_residual"]:.4f})', 
                     pad=15, weight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted)', labelpad=12)
    axes[0].set_ylabel('Count', labelpad=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)

    sns.histplot(val_results['metrics']['residuals'], kde=True, ax=axes[1], bins=30)
    axes[1].set_title(f'Validation Residuals (Mean: {val_results["metrics"]["mean_residual"]:.4f})', 
                     pad=15, weight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted)', labelpad=12)
    axes[1].set_ylabel('Count', labelpad=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Residuals vs predicted values
    plt.figure(figsize=(14, 12))
    plt.scatter(train_results['predictions'], train_results['metrics']['residuals'], 
                alpha=0.5, s=50, label='Training')
    plt.scatter(val_results['predictions'], val_results['metrics']['residuals'], 
                alpha=0.5, s=50, label='Validation')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=3)
    plt.xlabel('Predicted Organic Carbon', labelpad=12)
    plt.ylabel('Residual (Actual - Predicted)', labelpad=12)
    plt.title('Residuals vs Predicted Values', pad=20, weight='bold')
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'residuals_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Spatial distribution of residuals
    plt.figure(figsize=(16, 14))
    sc = plt.scatter(val_results['longitudes'], val_results['latitudes'], 
               c=val_results['metrics']['residuals'], cmap='coolwarm', 
               alpha=0.7, s=40, vmin=-2*np.std(val_results['metrics']['residuals']), 
               vmax=2*np.std(val_results['metrics']['residuals']))
    cbar = plt.colorbar(sc, label='Residual')
    cbar.ax.tick_params(labelsize=18, width=2.0, length=6, pad=8)
    cbar.set_label('Residual', fontsize=22, labelpad=15)
    plt.xlabel('Longitude', labelpad=12)
    plt.ylabel('Latitude', labelpad=12)
    plt.title('Spatial Distribution of Validation Residuals', pad=20, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'spatial_residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Create OC bins and analyze residuals by bin
    def create_oc_bins(targets, predictions, residuals, n_bins=10):
        min_target = np.min(targets)
        max_target = np.max(targets)
        bin_edges = np.linspace(min_target, max_target, n_bins + 1)
        bin_indices = np.digitize(targets, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        binned_data = []
        bin_centers = []
        bin_counts = []
        bin_mean_residuals = []
        bin_std_residuals = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                binned_data.append(residuals[mask])
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                bin_counts.append(np.sum(mask))
                bin_mean_residuals.append(np.mean(residuals[mask]))
                bin_std_residuals.append(np.std(residuals[mask]))

        return binned_data, bin_centers, bin_counts, bin_mean_residuals, bin_std_residuals

    # 5a. Training data boxplot
    train_binned_residuals, train_bin_centers, train_bin_counts, train_mean_residuals, train_std_residuals = create_oc_bins(
        train_results['targets'], train_results['predictions'], train_results['metrics']['residuals'])

    plt.figure(figsize=(16, 9))
    bp = plt.boxplot(train_binned_residuals, labels=[f"{c:.2f}\n(n={n})" for c, n in zip(train_bin_centers, train_bin_counts)],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(2)
    for cap in bp['caps']:
        cap.set_linewidth(2)
    for median in bp['medians']:
        median.set_linewidth(3)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=3)
    plt.xlabel('Organic Carbon Bins', labelpad=12)
    plt.ylabel('Residuals', labelpad=12)
    plt.title('Distribution of Training Residuals by Organic Carbon Content Bins', pad=20, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_residuals_by_oc_bins.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5b. Validation data boxplot
    val_binned_residuals, val_bin_centers, val_bin_counts, val_mean_residuals, val_std_residuals = create_oc_bins(
        val_results['targets'], val_results['predictions'], val_results['metrics']['residuals'])

    plt.figure(figsize=(16, 9))
    bp = plt.boxplot(val_binned_residuals, labels=[f"{c:.2f}\n(n={n})" for c, n in zip(val_bin_centers, val_bin_counts)],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(2)
    for cap in bp['caps']:
        cap.set_linewidth(2)
    for median in bp['medians']:
        median.set_linewidth(3)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=3)
    plt.xlabel('Organic Carbon Bins', labelpad=12)
    plt.ylabel('Residuals', labelpad=12)
    plt.title('Distribution of Validation Residuals by Organic Carbon Content Bins', pad=20, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_residuals_by_oc_bins.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5c. Line plot of mean residuals by bin
    plt.figure(figsize=(14, 9))
    plt.errorbar(train_bin_centers, train_mean_residuals, yerr=train_std_residuals, 
                 fmt='o-', capsize=8, capthick=2, linewidth=2.5, markersize=10, label='Training')
    plt.errorbar(val_bin_centers, val_mean_residuals, yerr=val_std_residuals, 
                 fmt='o-', capsize=8, capthick=2, linewidth=2.5, markersize=10, label='Validation')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=3)
    plt.xlabel('Organic Carbon Bin Center', labelpad=12)
    plt.ylabel('Mean Residual ± Std Dev', labelpad=12)
    plt.title('Mean Residuals by Organic Carbon Content', pad=20, weight='bold')
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'mean_residuals_by_oc_bins.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. QQ plot for residuals
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Training QQ plot
    stats.probplot(train_results['metrics']['residuals'], dist="norm", plot=axes[0])
    axes[0].set_title('Training Residuals Q-Q Plot', pad=15, weight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    axes[0].set_xlabel('Theoretical Quantiles', labelpad=12)
    axes[0].set_ylabel('Sample Quantiles', labelpad=12)

    # Validation QQ plot
    stats.probplot(val_results['metrics']['residuals'], dist="norm", plot=axes[1])
    axes[1].set_title('Validation Residuals Q-Q Plot', pad=15, weight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    axes[1].set_xlabel('Theoretical Quantiles', labelpad=12)
    axes[1].set_ylabel('Sample Quantiles', labelpad=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_qq_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Heat map of predictions vs actual values
    def create_heatmap(preds, actuals, title, filename):
        # Create 2D histogram (heatmap)
        plt.figure(figsize=(12, 12))

        # Define bins for both axes
        bins = np.linspace(min(np.min(preds), np.min(actuals)), 
                          max(np.max(preds), np.max(actuals)), 
                          30)

        plt.hist2d(actuals, preds, bins=[bins, bins], cmap='viridis', norm=plt.cm.colors.LogNorm())
        cbar = plt.colorbar(label='Count')
        cbar.ax.tick_params(labelsize=18, width=2.0, length=6, pad=8)
        cbar.set_label('Count', fontsize=22, labelpad=15)
        plt.plot([bins[0], bins[-1]], [bins[0], bins[-1]], 'r--', linewidth=3)
        plt.xlabel('Actual Organic Carbon', labelpad=12)
        plt.ylabel('Predicted Organic Carbon', labelpad=12)
        plt.title(title, pad=20, weight='bold')
        plt.grid(False)
        plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    create_heatmap(train_results['predictions'], train_results['targets'], 
                  'Training Data: Predicted vs Actual (Density)', 'train_pred_actual_heatmap.png')
    create_heatmap(val_results['predictions'], val_results['targets'], 
                  'Validation Data: Predicted vs Actual (Density)', 'val_pred_actual_heatmap.png')

    # 8. Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'RMSE', 'MAE', 'RPIQ', 'Mean Residual', 'Std Residual', 
                  'Min Residual', 'Max Residual', 'Median Residual', 'Residual IQR'],
        'Training': [train_results['metrics']['r2'], train_results['metrics']['rmse'],
                    train_results['metrics']['mae'], train_results['metrics']['rpiq'],
                    train_results['metrics']['mean_residual'], train_results['metrics']['std_residual'],
                    train_results['metrics']['min_residual'], train_results['metrics']['max_residual'],
                    train_results['metrics']['median_residual'], train_results['metrics']['residual_iqr']],
        'Validation': [val_results['metrics']['r2'], val_results['metrics']['rmse'],
                      val_results['metrics']['mae'], val_results['metrics']['rpiq'],
                      val_results['metrics']['mean_residual'], val_results['metrics']['std_residual'],
                      val_results['metrics']['min_residual'], val_results['metrics']['max_residual'],
                      val_results['metrics']['median_residual'], val_results['metrics']['residual_iqr']]
    })

    metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)

    # 9. Calculate and plot relative errors
    train_rel_errors = 100 * np.abs(train_results['metrics']['residuals'] / np.maximum(train_results['targets'], 1e-10))
    val_rel_errors = 100 * np.abs(val_results['metrics']['residuals'] / np.maximum(val_results['targets'], 1e-10))

    # Handle division by zero or near-zero
    train_rel_errors = np.clip(train_rel_errors, 0, 1000)  # Cap at 1000% error
    val_rel_errors = np.clip(val_rel_errors, 0, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.histplot(train_rel_errors, kde=True, ax=axes[0], bins=30)
    axes[0].set_title(f'Training Relative Errors % (Median: {np.nanmedian(train_rel_errors):.2f}%)', 
                     pad=15, weight='bold')
    axes[0].set_xlabel('Relative Error (%)', labelpad=12) 
    axes[0].set_ylabel('Count', labelpad=12)
    axes[0].set_xlim(0, min(300, np.nanpercentile(train_rel_errors, 95)))
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)

    sns.histplot(val_rel_errors, kde=True, ax=axes[1], bins=30)
    axes[1].set_title(f'Validation Relative Errors % (Median: {np.nanmedian(val_rel_errors):.2f}%)', 
                     pad=15, weight='bold')
    axes[1].set_xlabel('Relative Error (%)', labelpad=12)
    axes[1].set_ylabel('Count', labelpad=12)
    axes[1].set_xlim(0, min(300, np.nanpercentile(val_rel_errors, 95)))
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Plot relative errors vs actual OC content
    plt.figure(figsize=(14, 10))
    plt.scatter(train_results['targets'], train_rel_errors, alpha=0.5, s=50, label='Training')
    plt.scatter(val_results['targets'], val_rel_errors, alpha=0.5, s=50, label='Validation')
    plt.xlabel('Actual Organic Carbon', labelpad=12)
    plt.ylabel('Relative Error (%)', labelpad=12)
    plt.title('Relative Errors vs Actual Organic Carbon', pad=20, weight='bold')
    plt.ylim(0, min(300, max(np.nanpercentile(train_rel_errors, 95), np.nanpercentile(val_rel_errors, 95))))
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'relative_errors_vs_oc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 11. Visualize prediction uncertainty
    binned_uncertainty = []
    binned_rel_errors = []
    bin_centers_uncertainty = []

    # Calculate uncertainty based on the standard deviation of residuals in each bin
    for i, (center, residuals_bin) in enumerate(zip(val_bin_centers, val_binned_residuals)):
        if len(residuals_bin) > 5:  # Ensure enough samples for reliable statistics
            bin_centers_uncertainty.append(center)
            binned_uncertainty.append(np.std(residuals_bin))

            # Get the corresponding relative errors for this bin
            bin_mask = np.logical_and(
                val_results['targets'] >= center - (val_bin_centers[1] - val_bin_centers[0])/2,
                val_results['targets'] < center + (val_bin_centers[1] - val_bin_centers[0])/2
            )
            bin_rel_errors = val_rel_errors[bin_mask]
            binned_rel_errors.append(np.median(bin_rel_errors))

    plt.figure(figsize=(14, 9))
    plt.bar(bin_centers_uncertainty, binned_uncertainty, width=(val_bin_centers[1] - val_bin_centers[0])*0.8, 
            alpha=0.6, label='Std Dev of Residuals', edgecolor='black', linewidth=1.5)
    plt.plot(bin_centers_uncertainty, binned_rel_errors, 'ro-', linewidth=2.5, markersize=10, 
             label='Median Relative Error (%)')
    plt.xlabel('Organic Carbon Content', labelpad=12)
    plt.ylabel('Uncertainty / Error', labelpad=12)
    plt.title('Prediction Uncertainty by Organic Carbon Content', pad=20, weight='bold')
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'prediction_uncertainty.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 12. SGT-specific: Time-based error analysis
    plt.figure(figsize=(12, 8))
    plt.hist(val_results['metrics']['residuals'], bins=30, alpha=0.7, color='blue', edgecolor='black', linewidth=1.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=3)
    plt.xlabel('Residual (Actual - Predicted)', labelpad=12)
    plt.ylabel('Frequency', labelpad=12)
    plt.title('Distribution of Temporal Fusion Transformer Residuals', pad=20, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=8, width=2.0, length=6)
    plt.savefig(os.path.join(output_dir, 'SGT_residual_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract transformation type from model path
    target_transform = extract_transform_info(args.model_path)
    print(f"Detected target transform: {target_transform}")

    # Load the model
    model = load_model(args.model_path, device, args.hidden_size)
    print(f"Model loaded from {args.model_path}")

    # Load normalization statistics
    with open(args.stats_path, 'rb') as f:
        stats = pickle.load(f)

    feature_means = stats['feature_means']
    feature_stds = stats['feature_stds']
    target_mean = stats.get('target_mean', 0.0)
    target_std = stats.get('target_std', 1.0)

    print(f"Loaded normalization stats - Target mean: {target_mean}, Target std: {target_std}")

    # Load data
    data_df = pd.read_parquet(args.data_path)

    # Split data into train and validation sets
    train_df = data_df[data_df['dataset_type'] == 'train']
    val_df = data_df[data_df['dataset_type'] == 'val']

    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples")

    # Get samples_coordinates_array_path and data_array_path
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    # Flatten the paths if they are nested lists
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

    # Create datasets
    train_dataset = NormalizedMultiRasterDatasetMultiYears(
        samples_coordinates_array_path, data_array_path, train_df
    )
    train_dataset.set_feature_means(feature_means)
    train_dataset.set_feature_stds(feature_stds)

    val_dataset = NormalizedMultiRasterDatasetMultiYears(
        samples_coordinates_array_path, data_array_path, val_df
    )
    val_dataset.set_feature_means(feature_means)
    val_dataset.set_feature_stds(feature_stds)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    # Run inference on train set
    print("Running inference on training set...")
    train_preds, train_targets, train_lons, train_lats = run_inference(
        model, train_loader, target_transform, target_mean, target_std, device
    )

    # Run inference on validation set
    print("Running inference on validation set...")
    val_preds, val_targets, val_lons, val_lats = run_inference(
        model, val_loader, target_transform, target_mean, target_std, device
    )

    # Calculate metrics
    train_metrics = calculate_metrics(train_preds, train_targets)
    val_metrics = calculate_metrics(val_preds, val_targets)

    # Prepare results dictionaries
    train_results = {
        'predictions': train_preds,
        'targets': train_targets,
        'longitudes': train_lons,
        'latitudes': train_lats,
        'metrics': train_metrics
    }

    val_results = {
        'predictions': val_preds,
        'targets': val_targets,
        'longitudes': val_lons,
        'latitudes': val_lats,
        'metrics': val_metrics
    }

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(train_results, val_results, args.output_dir)

    # Save results to pickle for further analysis
    with open(os.path.join(args.output_dir, 'analysis_results.pkl'), 'wb') as f:
        pickle.dump({
            'train_results': train_results,
            'val_results': val_results,
            'model_path': args.model_path,
            'stats': stats
        }, f)

    print(f"Analysis complete! Results saved to {args.output_dir}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Training set (n={len(train_targets)}):")
    print(f"  R² = {train_metrics['r2']:.4f}")
    print(f"  RMSE = {train_metrics['rmse']:.4f}")
    print(f"  MAE = {train_metrics['mae']:.4f}")
    print(f"  RPIQ = {train_metrics['rpiq']:.4f}")
    print(f"  Mean Residual = {train_metrics['mean_residual']:.4f}")

    print(f"\nValidation set (n={len(val_targets)}):")
    print(f"  R² = {val_metrics['r2']:.4f}")
    print(f"  RMSE = {val_metrics['rmse']:.4f}")
    print(f"  MAE = {val_metrics['mae']:.4f}")
    print(f"  RPIQ = {val_metrics['rpiq']:.4f}")
    print(f"  Mean Residual = {val_metrics['mean_residual']:.4f}")

if __name__ == "__main__":
    main()