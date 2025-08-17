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
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import geopandas as gpd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as patches
from matplotlib import cm

from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import separate_and_add_data
from torch.utils.data import DataLoader
from SimpleTFT import SimpleTFT
# from EnhancedTFT import EnhancedTFT as SimpleTFT
from config import (TIME_BEGINNING, TIME_END, MAX_OC,
                   NUM_HEADS, NUM_LAYERS, hidden_size, bands_list_order, 
                   window_size, time_before)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SimpleTFT model residuals with advanced statistical analysis')
    parser.add_argument('--model-path', type=str, default='/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/TFT360kparams_OC150_2007to2023_transform_normalize_loss_l1/models/TFT_model_BEST_OVERALL_run_4_R2_0.6195.pth', help='Path to the trained model .pth file')
    parser.add_argument('--data-path', type=str, default='/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/TFT360kparams_OC150_2007to2023_transform_normalize_loss_l1/data/train_val_data_run_4.parquet', help='Path to the parquet file containing train/val data')
    parser.add_argument('--stats-path', type=str, default='/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/TFT360kparams_OC150_2007to2023_transform_normalize_loss_l1/data/normalization_stats_run_4.pkl', help='Path to normalization stats pickle file')
    parser.add_argument('--output-dir', type=str, default='residual_Maps_Bavaria', help='Directory to save output visualizations')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU for inference')
    parser.add_argument('--hidden-size', type=int, default=hidden_size, help='Hidden size for the model')
    parser.add_argument('--buffer-distance', type=float, default=0.1, help='Buffer distance around Bavaria boundaries')
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
    """Load the trained TFT model"""
    model = SimpleTFT(
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
    """Calculate various performance metrics including bias-variance decomposition"""
    residuals = targets - predictions

    # Calculate R² as squared correlation coefficient
    correlation_coeff = np.corrcoef(targets, predictions)[0, 1]
    r2 = correlation_coeff ** 2

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
    rpiq = iqr / rmse if rmse > 0 else float('inf')

    # Bias-variance decomposition
    bias = np.mean(residuals)  # Systematic error
    variance = np.var(residuals)  # Random error variance
    total_error = np.var(residuals - bias)  # Total prediction variance

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
        'residual_iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
        'bias': bias,
        'variance': variance,
        'total_error': total_error
    }

def load_landcover_data():
    """Load landcover data and coordinates"""
    landcover_path = "/home/vfourel/SOCProject/SOCmapping/Archive/landcover_bavaria_Visual_results_npy_s/landcover_values_merged.npy"
    landcover_coordinates_path = "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy"

    print("Loading landcover data...")
    landcover_values = np.load(landcover_path, allow_pickle=True)
    landcover_coordinates = np.load(landcover_coordinates_path)

    return landcover_coordinates, landcover_values

def interpolate_landcover(landcover_coords, landcover_values, grid_x, grid_y):
    """Interpolate landcover values to the same grid"""
    landcover_lons = landcover_coords[:, 0]
    landcover_lats = landcover_coords[:, 1]
    points = np.column_stack((landcover_lons, landcover_lats))
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    landcover_interpolated = griddata(points, landcover_values, grid_points, method='nearest', fill_value=0)
    return landcover_interpolated.reshape(grid_x.shape)

def create_spatial_maps(train_results, val_results, output_dir, buffer_distance=0.1):
    """Create spatial distribution maps for both training and validation"""

    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']

    # Load landcover data
    landcover_coordinates, landcover_values = load_landcover_data()

    # Create grid for landcover
    bounds = bavaria.total_bounds
    grid_resolution = 100
    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[0] - buffer_distance, bounds[2] + buffer_distance, grid_resolution),
        np.linspace(bounds[1] - buffer_distance, bounds[3] + buffer_distance, grid_resolution)
    )
    landcover_grid = interpolate_landcover(landcover_coordinates, landcover_values, grid_x, grid_y)

    # Color scheme for residuals
    all_residuals = np.concatenate([train_results['metrics']['residuals'], val_results['metrics']['residuals']])
    vmin, vmax = np.percentile(all_residuals, [2, 98])
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax

    # Custom colormap
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
              '#f7f7f7', '#fdbf6f', '#ff7f00', '#e31a1c', '#b10026']
    residual_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('residuals', colors, N=256)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

    for ax, results, title_suffix in [(ax1, train_results, 'Training'), (ax2, val_results, 'Validation')]:
        lons = results['longitudes']
        lats = results['latitudes']
        residuals = results['metrics']['residuals']

        # Plot landcover overlays first (no colored background)
        # Built-up areas (class 50) in black
        built_up_mask = landcover_grid == 50
        if np.any(built_up_mask):
            ax.contourf(grid_x, grid_y, built_up_mask.astype(float), 
                       levels=[0.5, 1.5], colors=['black'], alpha=0.7)

        # Water bodies (class 80) in navy blue
        water_mask = landcover_grid == 80
        if np.any(water_mask):
            ax.contourf(grid_x, grid_y, water_mask.astype(float), 
                       levels=[0.5, 1.5], colors=['#000080'], alpha=0.7)

        # Plot Bavaria boundaries
        bavaria.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.9)
        bavaria.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.9)

        # Plot residuals as bigger scatter points (size 9)
        scatter = ax.scatter(lons, lats, c=residuals, cmap=residual_cmap, norm=norm,
                            s=9, alpha=0.7, edgecolors='none', zorder=5)  # Bigger dots (s=9)

        # Set map extent
        ax.set_xlim(bounds[0] - buffer_distance, bounds[2] + buffer_distance)
        ax.set_ylim(bounds[1] - buffer_distance, bounds[3] + buffer_distance)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'{title_suffix} Set Residuals\n(n={len(residuals):,})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)

    # Add single colorbar for both subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add main title
    fig.suptitle('TFT Model Residuals: Spatial Distribution Comparison', fontsize=16, fontweight='bold')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spatial_residuals_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/spatial_residuals_comparison.pdf', bbox_inches='tight')
    plt.close()

def create_individual_distribution_plots(train_results, val_results, output_dir):
    """Create separate distribution plots as individual PNG files"""

    train_residuals = train_results['metrics']['residuals']
    val_residuals = val_results['metrics']['residuals']

    n_bins = 200  # Much more bins for better resolution

    # Training residuals distribution
    plt.figure(figsize=(12, 8), dpi=300)

    # Create high-resolution histogram
    counts, bins, _ = plt.hist(train_residuals, bins=n_bins, alpha=0.7, color='skyblue', 
                              edgecolor='darkblue', linewidth=0.3, density=True)

    # Fit Gaussian with better parameters
    mu, sigma = stats.norm.fit(train_residuals)

    # Create a finer x-axis for smoother curve that follows the distribution more closely
    x_fine = np.linspace(train_residuals.min(), train_residuals.max(), 1000)

    # Use the actual bin centers for better alignment with histogram
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit Gaussian to actual histogram shape for better alignment at the beginning
    # Use a more sophisticated approach with kernel density estimation
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(train_residuals)
    kde_curve = kde(x_fine)

    # Also plot the theoretical Gaussian
    gaussian_theoretical = stats.norm.pdf(x_fine, mu, sigma)

    # Plot both curves
    plt.plot(x_fine, gaussian_theoretical, 'red', linewidth=3, linestyle='--', 
             label=f'Theoretical Gaussian\nμ={mu:.4f}, σ={sigma:.4f}', alpha=0.8)
    plt.plot(x_fine, kde_curve, 'darkgreen', linewidth=3, linestyle='-',
             label='KDE Fit (follows data closely)', alpha=0.9)

    # Add reference lines
    plt.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Ideal (μ=0)')
    plt.axvline(mu, color='orange', linestyle='-', linewidth=2, label=f'Actual Mean')

    plt.xlabel('Residual (Actual - Predicted)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title(f'Training Set Residuals Distribution\n(n={len(train_residuals):,} samples)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistical information
    skewness = stats.skew(train_residuals)
    kurtosis = stats.kurtosis(train_residuals)
    shapiro_stat, shapiro_p = stats.shapiro(train_residuals[:5000] if len(train_residuals) > 5000 else train_residuals)

    stats_text = f'Skewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}\nShapiro-Wilk p: {shapiro_p:.2e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Validation residuals distribution
    plt.figure(figsize=(12, 8), dpi=300)

    # Create high-resolution histogram
    counts, bins, _ = plt.hist(val_residuals, bins=n_bins, alpha=0.7, color='lightcoral', 
                              edgecolor='darkred', linewidth=0.3, density=True)

    # Fit Gaussian
    mu, sigma = stats.norm.fit(val_residuals)

    # Create fine x-axis
    x_fine = np.linspace(val_residuals.min(), val_residuals.max(), 1000)

    # KDE for better fitting
    kde = gaussian_kde(val_residuals)
    kde_curve = kde(x_fine)

    # Theoretical Gaussian
    gaussian_theoretical = stats.norm.pdf(x_fine, mu, sigma)

    # Plot both curves
    plt.plot(x_fine, gaussian_theoretical, 'blue', linewidth=3, linestyle='--', 
             label=f'Theoretical Gaussian\nμ={mu:.4f}, σ={sigma:.4f}', alpha=0.8)
    plt.plot(x_fine, kde_curve, 'darkgreen', linewidth=3, linestyle='-',
             label='KDE Fit (follows data closely)', alpha=0.9)

    # Add reference lines
    plt.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Ideal (μ=0)')
    plt.axvline(mu, color='orange', linestyle='-', linewidth=2, label=f'Actual Mean')

    plt.xlabel('Residual (Actual - Predicted)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title(f'Validation Set Residuals Distribution\n(n={len(val_residuals):,} samples)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistical information
    skewness = stats.skew(val_residuals)
    kurtosis = stats.kurtosis(val_residuals)
    shapiro_stat, shapiro_p = stats.shapiro(val_residuals[:5000] if len(val_residuals) > 5000 else val_residuals)

    stats_text = f'Skewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}\nShapiro-Wilk p: {shapiro_p:.2e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined distribution comparison
    plt.figure(figsize=(14, 10), dpi=300)

    # Calculate common range for both distributions
    combined_min = min(train_residuals.min(), val_residuals.min())
    combined_max = max(train_residuals.max(), val_residuals.max())
    bins_combined = np.linspace(combined_min, combined_max, n_bins)

    # Plot both histograms
    plt.hist(train_residuals, bins=bins_combined, alpha=0.6, label='Training', color='skyblue', 
             density=True, edgecolor='darkblue', linewidth=0.3)
    plt.hist(val_residuals, bins=bins_combined, alpha=0.6, label='Validation', color='lightcoral', 
             density=True, edgecolor='darkred', linewidth=0.3)

    # Create fine x-axis for curves
    x_fine = np.linspace(combined_min, combined_max, 1000)

    # Fit and plot curves for both datasets
    for residuals, color, label_suffix, linestyle in [(train_residuals, 'darkblue', 'Train', '-'), 
                                                     (val_residuals, 'darkred', 'Val', '--')]:
        mu, sigma = stats.norm.fit(residuals)

        # KDE for close following
        kde = gaussian_kde(residuals)
        kde_curve = kde(x_fine)

        # Theoretical Gaussian
        gaussian_curve = stats.norm.pdf(x_fine, mu, sigma)

        plt.plot(x_fine, kde_curve, color=color, linewidth=3, linestyle=linestyle,
                label=f'{label_suffix} KDE', alpha=0.9)
        plt.plot(x_fine, gaussian_curve, color=color, linewidth=2, linestyle=':',
                label=f'{label_suffix} Gaussian (μ={mu:.3f})', alpha=0.7)

    # Add reference line
    plt.axvline(0, color='green', linestyle='-', linewidth=3, alpha=0.8, label='Ideal (μ=0)')

    plt.xlabel('Residual (Actual - Predicted)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title('Combined Residual Distributions Comparison\nwith High-Resolution Fitting', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)

    # Add comparison statistics
    train_stats = f'Training: μ={np.mean(train_residuals):.4f}, σ={np.std(train_residuals):.4f}'
    val_stats = f'Validation: μ={np.mean(val_residuals):.4f}, σ={np.std(val_residuals):.4f}'
    comparison_text = f'{train_stats}\n{val_stats}'

    plt.text(0.02, 0.02, comparison_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_advanced_residual_analysis(train_results, val_results, output_dir):
    """Create advanced residual analysis with fitted curves and bias-variance decomposition"""

    train_residuals = train_results['metrics']['residuals']
    val_residuals = val_results['metrics']['residuals']

    # Create comprehensive figure (without distribution plots - those are separate now)
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Q-Q plots for normality assessment
    ax1 = fig.add_subplot(gs[0, 0])
    stats.probplot(train_residuals, dist="norm", plot=ax1)
    ax1.set_title('Training Residuals Q-Q Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(val_residuals, dist="norm", plot=ax2)
    ax2.set_title('Validation Residuals Q-Q Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 2. Bias-Variance Decomposition
    ax3 = fig.add_subplot(gs[0, 2])

    metrics_comparison = pd.DataFrame({
        'Dataset': ['Training', 'Validation'],
        'Bias²': [train_results['metrics']['bias']**2, val_results['metrics']['bias']**2],
        'Variance': [train_results['metrics']['variance'], val_results['metrics']['variance']],
        'Total Error': [train_results['metrics']['total_error'], val_results['metrics']['total_error']]
    })

    x = np.arange(len(metrics_comparison))
    width = 0.25

    ax3.bar(x - width, metrics_comparison['Bias²'], width, label='Bias²', color='red', alpha=0.7)
    ax3.bar(x, metrics_comparison['Variance'], width, label='Variance', color='blue', alpha=0.7)
    ax3.bar(x + width, metrics_comparison['Total Error'], width, label='Total Error', color='green', alpha=0.7)

    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Error Magnitude')
    ax3.set_title('Bias-Variance Decomposition', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_comparison['Dataset'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 3. Fitted curve of residuals vs predicted values
    ax4 = fig.add_subplot(gs[1, 0:2])  # Span two columns

    # Combine data for trend analysis
    all_predictions = np.concatenate([train_results['predictions'], val_results['predictions']])
    all_residuals = np.concatenate([train_residuals, val_residuals])

    # Sort by predictions for smooth curve
    sort_idx = np.argsort(all_predictions)
    sorted_preds = all_predictions[sort_idx]
    sorted_residuals = all_residuals[sort_idx]

    # Calculate moving average for trend line (more points for smoother curve)
    window_size = max(len(sorted_preds) // 100, 50)  # More points for smoother curve
    trend_x = []
    trend_y = []
    trend_std = []

    for i in range(0, len(sorted_preds) - window_size, window_size//2):  # Overlap for smoother curve
        window_end = min(i + window_size, len(sorted_preds))
        trend_x.append(np.mean(sorted_preds[i:window_end]))
        trend_y.append(np.mean(sorted_residuals[i:window_end]))
        trend_std.append(np.std(sorted_residuals[i:window_end]))

    trend_x = np.array(trend_x)
    trend_y = np.array(trend_y)
    trend_std = np.array(trend_std)

    # Scatter plot with smaller points
    ax4.scatter(train_results['predictions'], train_residuals, alpha=0.3, s=1, color='blue', label='Training')
    ax4.scatter(val_results['predictions'], val_residuals, alpha=0.3, s=1, color='red', label='Validation')

    # Smooth trend line with confidence interval
    ax4.plot(trend_x, trend_y, 'black', linewidth=4, label='Moving Average Trend', alpha=0.9)
    ax4.fill_between(trend_x, trend_y - trend_std, trend_y + trend_std, 
                     color='gray', alpha=0.3, label='±1 Std')

    # High-order polynomial fit for overall trend
    if len(all_predictions) > 100:
        coeffs = np.polyfit(all_predictions, all_residuals, 3)  # Cubic fit
        poly_x = np.linspace(all_predictions.min(), all_predictions.max(), 200)
        poly_y = np.polyval(coeffs, poly_x)
        ax4.plot(poly_x, poly_y, 'orange', linewidth=3, label='Cubic Polynomial Fit', alpha=0.8)

    # LOWESS smoothing for even better curve following
    from scipy import interpolate
    # Create a spline that follows the data closely
    if len(trend_x) > 10:
        spline = interpolate.UnivariateSpline(trend_x, trend_y, s=len(trend_x)*0.1)
        spline_x = np.linspace(trend_x.min(), trend_x.max(), 300)
        spline_y = spline(spline_x)
        ax4.plot(spline_x, spline_y, 'purple', linewidth=2, label='Spline Fit', alpha=0.8)

    ax4.axhline(y=0, color='green', linestyle='--', linewidth=3, label='Ideal (y=0)')

    ax4.set_xlabel('Predicted Values', fontsize=12)
    ax4.set_ylabel('Residuals', fontsize=12)
    ax4.set_title('Residuals vs Predicted Values with Multiple Fitted Curves', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 4. Performance metrics comparison
    ax5 = fig.add_subplot(gs[1, 2])

    metrics_names = ['R²', 'RMSE', 'MAE', 'Bias', 'Std']
    train_metrics = [train_results['metrics']['r2'], train_results['metrics']['rmse'],
                    train_results['metrics']['mae'], abs(train_results['metrics']['bias']),
                    train_results['metrics']['std_residual']]
    val_metrics = [val_results['metrics']['r2'], val_results['metrics']['rmse'],
                  val_results['metrics']['mae'], abs(val_results['metrics']['bias']),
                  val_results['metrics']['std_residual']]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax5.bar(x - width/2, train_metrics, width, label='Training', color='blue', alpha=0.7)
    ax5.bar(x + width/2, val_metrics, width, label='Validation', color='red', alpha=0.7)

    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Value')
    ax5.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_names, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 5. Residual patterns by target value ranges
    ax6 = fig.add_subplot(gs[2, 0])

    # Create target bins
    n_target_bins = 25
    val_targets = val_results['targets']
    target_bins = np.linspace(val_targets.min(), val_targets.max(), n_target_bins)
    bin_centers = (target_bins[:-1] + target_bins[1:]) / 2

    binned_residual_means = []
    binned_residual_stds = []

    for i in range(len(target_bins)-1):
        mask = (val_targets >= target_bins[i]) & (val_targets < target_bins[i+1])
        if np.sum(mask) > 0:
            binned_residual_means.append(np.mean(val_residuals[mask]))
            binned_residual_stds.append(np.std(val_residuals[mask]))
        else:
            binned_residual_means.append(0)
            binned_residual_stds.append(0)

    ax6.errorbar(bin_centers, binned_residual_means, yerr=binned_residual_stds, 
                fmt='o-', capsize=5, linewidth=2, markersize=6)
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Target Value Bins')
    ax6.set_ylabel('Mean Residual ± Std')
    ax6.set_title('Residual Bias by Target Range', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 6. Statistical summary table
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')

    summary_stats = [
        ['Metric', 'Training', 'Validation'],
        ['Sample Size', f"{len(train_residuals):,}", f"{len(val_residuals):,}"],
        ['Mean Residual', f"{train_results['metrics']['mean_residual']:.4f}", f"{val_results['metrics']['mean_residual']:.4f}"],
        ['Std Residual', f"{train_results['metrics']['std_residual']:.4f}", f"{val_results['metrics']['std_residual']:.4f}"],
        ['Skewness', f"{stats.skew(train_residuals):.4f}", f"{stats.skew(val_residuals):.4f}"],
        ['Kurtosis', f"{stats.kurtosis(train_residuals):.4f}", f"{stats.kurtosis(val_residuals):.4f}"],
        ['Min Residual', f"{train_results['metrics']['min_residual']:.4f}", f"{val_results['metrics']['min_residual']:.4f}"],
        ['Max Residual', f"{train_results['metrics']['max_residual']:.4f}", f"{val_results['metrics']['max_residual']:.4f}"]
    ]

    table = ax7.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax7.set_title('Statistical Summary', fontsize=12, fontweight='bold')

    # 7. Residual autocorrelation
    ax8 = fig.add_subplot(gs[2, 2])

    # Calculate residual autocorrelation for validation set
    from scipy.stats import pearsonr

    # Sort by longitude to check spatial correlation
    val_lons = val_results['longitudes']
    val_residuals_sorted = val_residuals[np.argsort(val_lons)]

    # Calculate lagged correlations
    lags = range(1, min(200, len(val_residuals_sorted)//10))
    correlations = []

    for lag in lags:
        if lag < len(val_residuals_sorted):
            corr, _ = pearsonr(val_residuals_sorted[:-lag], val_residuals_sorted[lag:])
            correlations.append(corr)

    ax8.plot(lags[:len(correlations)], correlations, 'b-', linewidth=2)
    ax8.axhline(y=0, color='red', linestyle='--')
    ax8.set_xlabel('Spatial Lag')
    ax8.set_ylabel('Autocorrelation')
    ax8.set_title('Spatial Autocorrelation of Residuals', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Add main title
    fig.suptitle('TFT Model: Advanced Residual Analysis & Statistical Decomposition', fontsize=18, fontweight='bold', y=0.98)

    # Save the comprehensive analysis
    plt.savefig(f'{output_dir}/advanced_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/advanced_residual_analysis.pdf', bbox_inches='tight')
    plt.close()

def save_detailed_metrics(train_results, val_results, output_dir):
    """Save detailed metrics and statistical analysis to files"""

    # Create comprehensive metrics DataFrame
    detailed_metrics = pd.DataFrame({
        'Metric': ['R²', 'RMSE', 'MAE', 'RPIQ', 'Mean Residual (Bias)', 'Std Residual', 
                  'Min Residual', 'Max Residual', 'Median Residual', 'IQR Residual',
                  'Variance', 'Skewness', 'Kurtosis'],
        'Training': [
            train_results['metrics']['r2'], train_results['metrics']['rmse'],
            train_results['metrics']['mae'], train_results['metrics']['rpiq'],
            train_results['metrics']['mean_residual'], train_results['metrics']['std_residual'],
            train_results['metrics']['min_residual'], train_results['metrics']['max_residual'],
            train_results['metrics']['median_residual'], train_results['metrics']['residual_iqr'],
            train_results['metrics']['variance'], stats.skew(train_results['metrics']['residuals']),
            stats.kurtosis(train_results['metrics']['residuals'])
        ],
        'Validation': [
            val_results['metrics']['r2'], val_results['metrics']['rmse'],
            val_results['metrics']['mae'], val_results['metrics']['rpiq'],
            val_results['metrics']['mean_residual'], val_results['metrics']['std_residual'],
            val_results['metrics']['min_residual'], val_results['metrics']['max_residual'],
            val_results['metrics']['median_residual'], val_results['metrics']['residual_iqr'],
            val_results['metrics']['variance'], stats.skew(val_results['metrics']['residuals']),
            stats.kurtosis(val_results['metrics']['residuals'])
        ]
    })

    detailed_metrics.to_csv(f'{output_dir}/detailed_performance_metrics.csv', index=False)
    print(f"Detailed metrics saved to {output_dir}")

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
        stats_dict = pickle.load(f)

    feature_means = stats_dict['feature_means']
    feature_stds = stats_dict['feature_stds']
    target_mean = stats_dict.get('target_mean', 0.0)
    target_std = stats_dict.get('target_std', 1.0)

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

    # Run inference on both sets
    print("Running inference on training set...")
    train_preds, train_targets, train_lons, train_lats = run_inference(
        model, train_loader, target_transform, target_mean, target_std, device
    )

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
    print("Creating spatial maps...")
    create_spatial_maps(train_results, val_results, args.output_dir, args.buffer_distance)

    print("Creating individual distribution plots...")
    create_individual_distribution_plots(train_results, val_results, args.output_dir)

    print("Creating advanced residual analysis...")
    create_advanced_residual_analysis(train_results, val_results, args.output_dir)

    # Save detailed metrics
    save_detailed_metrics(train_results, val_results, args.output_dir)

    # Save results to pickle for further analysis
    with open(os.path.join(args.output_dir, 'complete_analysis_results.pkl'), 'wb') as f:
        pickle.dump({
            'train_results': train_results,
            'val_results': val_results,
            'model_path': args.model_path,
            'stats': stats_dict
        }, f)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print("\nSeparate distribution plots created:")
    print(f"  - {args.output_dir}/training_residuals_distribution.png")
    print(f"  - {args.output_dir}/validation_residuals_distribution.png") 
    print(f"  - {args.output_dir}/combined_residuals_distribution.png")

    # Print summary statistics
    print("\nDetailed Summary Statistics:")
    print(f"\nTraining set (n={len(train_targets):,}):")
    print(f"  R² = {train_metrics['r2']:.4f}")
    print(f"  RMSE = {train_metrics['rmse']:.4f}")
    print(f"  MAE = {train_metrics['mae']:.4f}")
    print(f"  Bias = {train_metrics['bias']:.4f}")
    print(f"  Variance = {train_metrics['variance']:.4f}")
    print(f"  Skewness = {stats.skew(train_metrics['residuals']):.4f}")
    print(f"  Kurtosis = {stats.kurtosis(train_metrics['residuals']):.4f}")

    print(f"\nValidation set (n={len(val_targets):,}):")
    print(f"  R² = {val_metrics['r2']:.4f}")
    print(f"  RMSE = {val_metrics['rmse']:.4f}")
    print(f"  MAE = {val_metrics['mae']:.4f}")
    print(f"  Bias = {val_metrics['bias']:.4f}")
    print(f"  Variance = {val_metrics['variance']:.4f}")
    print(f"  Skewness = {stats.skew(val_metrics['residuals']):.4f}")
    print(f"  Kurtosis = {stats.kurtosis(val_metrics['residuals']):.4f}")

if __name__ == "__main__":
    main()
