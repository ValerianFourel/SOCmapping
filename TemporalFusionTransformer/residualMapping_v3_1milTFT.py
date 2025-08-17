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
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib import cm

from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import separate_and_add_data
from torch.utils.data import DataLoader
from EnhancedTFT import EnhancedTFT as SimpleTFT
from config import (TIME_BEGINNING, TIME_END, MAX_OC,
                   NUM_HEADS, NUM_LAYERS, hidden_size, bands_list_order, 
                   window_size, time_before)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SimpleTFT model residuals with professional visualizations')
    parser.add_argument('--model-path', type=str, default='/home/vfourel/SOCProject/SOCmapping/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_normalize_LOSS_composite_l2_R2_0.6909.pth', help='Path to the trained model .pth file')
    parser.add_argument('--data-path', type=str, default='/home/vfourel/SOCProject/SOCmapping/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/train_val_data_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_normalize_LOSS_composite_l2.parquet', help='Path to the parquet file containing train/val data')
    parser.add_argument('--stats-path', type=str, default='/home/vfourel/SOCProject/SOCmapping/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/normalization_stats_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_normalize_LOSS_composite_l2.pkl', help='Path to normalization stats pickle file')
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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

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

            if target_transform == 'log':
                targets_transformed = torch.log(targets + 1e-10)
            elif target_transform == 'normalize':
                targets_transformed = (targets - target_mean) / (target_std + 1e-10)
            else:
                targets_transformed = targets

            outputs = model(features)

            if target_transform == 'log':
                outputs = torch.exp(outputs)
            elif target_transform == 'normalize':
                outputs = outputs * target_std + target_mean

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(original_targets)
            all_longitudes.extend(longitudes.cpu().numpy())
            all_latitudes.extend(latitudes.cpu().numpy())

    return np.array(all_outputs), np.array(all_targets), np.array(all_longitudes), np.array(all_latitudes)

def calculate_metrics(predictions, targets):
    """Calculate various performance metrics"""
    residuals = targets - predictions
    correlation_coeff = np.corrcoef(targets, predictions)[0, 1]
    r2 = correlation_coeff ** 2
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
    rpiq = iqr / rmse if rmse > 0 else float('inf')
    bias = np.mean(residuals)
    variance = np.var(residuals)
    total_error = np.var(residuals - bias)

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

def create_professional_qq_plots(train_results, val_results, output_dir):
    """Create professional Q-Q plots with standard normal distribution"""

    plt.style.use('seaborn-v0_8-whitegrid')

    # Get residuals
    train_residuals = train_results['metrics']['residuals']
    val_residuals = val_results['metrics']['residuals']

    # Calculate common scale for consistent visualization
    all_residuals = np.concatenate([train_residuals, val_residuals])
    global_min, global_max = np.percentile(all_residuals, [1, 99])

    # Ensure symmetric scale around zero
    global_extent = max(abs(global_min), abs(global_max))
    y_limits = [-global_extent * 1.1, global_extent * 1.1]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    fig.patch.set_facecolor('white')

    # Define professional colors
    colors = {'train': '#2E86AB', 'val': '#A23B72', 'reference': '#2E8B57', 'ci': '#C73E1D'}

    for ax, residuals, title_suffix, color_key in [(axes[0], train_residuals, 'Training', 'train'), 
                                                  (axes[1], val_residuals, 'Validation', 'val')]:

        # Generate Q-Q plot data properly
        residuals_clean = residuals[~np.isnan(residuals)]
        residuals_clean = residuals_clean[np.isfinite(residuals_clean)]

        if len(residuals_clean) == 0:
            print(f"Warning: No valid residuals for {title_suffix} set")
            continue

        # Generate theoretical and sample quantiles using standard normal distribution
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals_clean, dist="norm", plot=None)

        osm = np.asarray(osm).flatten()
        osr = np.asarray(osr).flatten()

        if len(osm) != len(osr):
            print(f"Warning: Quantile arrays have different lengths: {len(osm)} vs {len(osr)}")
            min_len = min(len(osm), len(osr))
            osm = osm[:min_len]
            osr = osr[:min_len]

        # Plot theoretical line (45-degree line)
        line_x = np.linspace(osm.min(), osm.max(), 100)
        line_y = line_x  # 45-degree line for standard normal distribution

        ax.plot(line_x, line_y, color=colors['reference'], linewidth=3.5, 
               label=f'Theoretical Normal Line', alpha=0.9, zorder=3)

        # Add confidence intervals
        residuals_std = np.std(residuals_clean)
        n = len(residuals_clean)

        se = residuals_std * np.sqrt(1/n + (line_x - np.mean(osm))**2 / np.sum((osm - np.mean(osm))**2))
        ci_upper = line_y + 1.96 * se
        ci_lower = line_y - 1.96 * se

        ax.fill_between(line_x, ci_lower, ci_upper, 
                       color=colors['ci'], alpha=0.15, label='95% Confidence Interval', zorder=1)

        # Plot actual data points
        ax.scatter(osm, osr, color=colors[color_key], alpha=0.6, s=25, 
                  edgecolors='none', linewidth=0, zorder=4, 
                  label=f'Observed Quantiles\n(n={len(residuals_clean):,})')

        # Styling
        ax.set_xlabel('Theoretical Quantiles (Standard Normal)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=14, fontweight='bold')

        ax.set_ylim(y_limits)

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_facecolor('#FAFAFA')

        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                          shadow=True, fontsize=11, framealpha=0.95)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')

        ax.tick_params(labelsize=12, width=1.2)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/professional_qq_plots.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/professional_qq_plots.pdf', bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()

def create_professional_histograms(train_results, val_results, output_dir):
    """Create professional publication-quality histogram plots with actual residual means"""

    plt.style.use('seaborn-v0_8-whitegrid')

    train_residuals = train_results['metrics']['residuals']
    val_residuals = val_results['metrics']['residuals']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.patch.set_facecolor('white')

    # Professional color palette
    colors = {
        'train': {'hist': '#2E86AB', 'kde': '#1F5F7A', 'gaussian': '#2E8B57'},
        'val': {'hist': '#A23B72', 'kde': '#7A2B56', 'gaussian': '#2E8B57'}
    }

    n_bins = 150

    for ax, residuals, title_suffix, color_set in [(axes[0], train_residuals, 'Training', colors['train']), 
                                                   (axes[1], val_residuals, 'Validation', colors['val'])]:

        # Calculate actual mean (NOT fitted!)
        actual_mean = np.mean(residuals)
        actual_std = np.std(residuals)

        # Create histogram with enhanced styling
        counts, bins, patches = ax.hist(residuals, bins=n_bins, alpha=0.7, 
                                       color=color_set['hist'], density=True,
                                       edgecolor='white', linewidth=0.3, rwidth=0.95)

        # Gradient effect for histogram bars
        for i, patch in enumerate(patches):
            patch.set_facecolor(color_set['hist'])
            patch.set_alpha(0.7 + 0.3 * (counts[i] / counts.max()))

        # Create smooth x-axis for curves
        x_smooth = np.linspace(residuals.min(), residuals.max(), 1000)

        # KDE for actual distribution
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        kde_curve = kde(x_smooth)

        # Theoretical Gaussian using ACTUAL mean and std
        gaussian_curve = stats.norm.pdf(x_smooth, actual_mean, actual_std)

        # Plot curves with professional styling
        ax.plot(x_smooth, kde_curve, color=color_set['kde'], linewidth=4, 
               linestyle='-', label='Kernel Density Estimate', alpha=0.9, zorder=5)

        # Reference lines
        ax.axvline(0, color='#2C5F2D', linestyle='-', linewidth=3, 
                  alpha=0.8, label='Ideal (μ=0)', zorder=3)
        ax.axvline(actual_mean, color='#C73E1D', linestyle='-', linewidth=2.5, 
                  alpha=0.8, label=f'Actual Mean (μ={actual_mean:.2f})', zorder=3)

        # Enhanced styling with better fonts
        ax.set_xlabel('Residual (Actual - Predicted)', fontsize=16, fontweight='bold', 
                      fontfamily='serif')
        ax.set_ylabel('Probability Density', fontsize=16, fontweight='bold', 
                      fontfamily='serif')

        # Professional grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_facecolor('#FAFAFA')

        # Add mean information in top left corner
        mean_text = f'Actual Mean: {actual_mean:.3f}\nStd Dev: {actual_std:.3f}'
        ax.text(0.02, 0.98, mean_text, transform=ax.transAxes, 
               fontsize=13, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.8", facecolor="white", 
                        edgecolor="gray", alpha=0.95, linewidth=1.5))

        # Professional legend
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=12, framealpha=0.95)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.2)

        # Enhanced spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#404040')

        ax.tick_params(labelsize=13, width=1.2, colors='#404040')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    plt.savefig(f'{output_dir}/professional_residual_histograms.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/professional_residual_histograms.pdf', 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_professional_spatial_maps(train_results, val_results, output_dir, buffer_distance=0.1):
    """Create professional spatial residual maps with optimized proportions and styling"""
    
    # Import the correct module for make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    plt.style.use('default')
    
    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']
    
    # Get Bavaria bounds for proper aspect ratio calculation
    bounds = bavaria.total_bounds  # [minx, miny, maxx, maxy]
    bavaria_width = bounds[2] - bounds[0]  # longitude range
    bavaria_height = bounds[3] - bounds[1]  # latitude range
    
    # Calculate proper aspect ratio (Bavaria is wider than it is tall)
    aspect_ratio = bavaria_height / bavaria_width
    
    # Enhanced color scheme for residuals
    all_residuals = np.concatenate([train_results['metrics']['residuals'], val_results['metrics']['residuals']])
    vmin, vmax = np.percentile(all_residuals, [1, 99])  # More conservative percentiles
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax
    
    # Create sophisticated colormap: Deep Blue -> White -> Deep Red
    colors = ['#1a237e', '#3949ab', '#5c6bc0', '#9fa8da', '#e8eaf6',  # Deep blue to light blue
              '#ffffff',  # Pure white center
              '#ffebee', '#ffcdd2', '#ef5350', '#d32f2f', '#b71c1c']  # Light red to deep red
    residual_cmap = LinearSegmentedColormap.from_list('professional_diverging', colors, N=256)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Optimized figure dimensions - make map longer and less wide
    base_width = 8   # Reduced width
    fig_height = base_width * aspect_ratio * 1.8  # Much taller (80% increase)
    
    # Create separate maps for train and validation sets
    for idx, (results, title_suffix) in enumerate([(train_results, 'Training'), (val_results, 'Validation')]):
        try:
            lons = results['longitudes']
            lats = results['latitudes']
            residuals = results['metrics']['residuals']
            
            # Create figure with proper proportions
            fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=300)
            fig.patch.set_facecolor('white')
            
            # Plot Bavaria boundaries first with elegant styling
            bavaria.boundary.plot(ax=ax, color='#2c3e50', linewidth=1.2, alpha=0.9, zorder=3)
            bavaria.plot(ax=ax, color='none', edgecolor='#2c3e50', linewidth=1.2, alpha=0.9, zorder=2)
            
            # Plot residual points as diamonds with sharp colors - no white edges
            point_size = max(2.0, min(10, 60000 / len(residuals)))  # Slightly larger for visibility
            
            scatter = ax.scatter(lons, lats, c=residuals, cmap=residual_cmap, norm=norm,
                               s=point_size, alpha=0.9, marker='D',  # Higher alpha, no edges
                               edgecolors='none', linewidth=0,  # Remove all edges
                               zorder=4, rasterized=True)
            
            # Set proper bounds with small buffer
            padding = buffer_distance
            ax.set_xlim(bounds[0] - padding, bounds[2] + padding)
            ax.set_ylim(bounds[1] - padding, bounds[3] + padding)
            
            # Force equal aspect ratio to prevent distortion
            ax.set_aspect('equal', adjustable='box')
            
            # Professional labels - no title
            ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold', color='#2c3e50')
            
            # Elegant grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#7f8c8d')
            ax.set_facecolor('#fafafa')
            
            # Professional spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('#34495e')
            
            ax.tick_params(labelsize=11, width=1.0, colors='#2c3e50', pad=8)
            
            # Properly sized colorbar positioned to the right - FIXED IMPORT
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.15)
            
            cbar = plt.colorbar(scatter, cax=cax, extend='both')
            cbar.set_label('Residual (Actual - Predicted)', fontsize=12, fontweight='bold', 
                          color='#2c3e50', labelpad=15)
            cbar.ax.tick_params(labelsize=10, colors='#2c3e50', width=0.8)
            
            # Add subtle ticks on colorbar
            n_ticks = 7
            tick_positions = np.linspace(vmin, vmax, n_ticks)
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels([f'{x:.2f}' for x in tick_positions])
            
            # Remove title and statistics - clean minimal design
            
            # Add subtle border around the plot area
            rect = patches.Rectangle((bounds[0] - padding, bounds[1] - padding),
                                   bavaria_width + 2*padding, bavaria_height + 2*padding,
                                   linewidth=2, edgecolor='#34495e', facecolor='none', 
                                   alpha=0.3, zorder=1)
            ax.add_patch(rect)
            
            # Add coordinate information
            coord_text = (f'Bounds: {bounds[0]:.2f}°E to {bounds[2]:.2f}°E\n'
                         f'        {bounds[1]:.2f}°N to {bounds[3]:.2f}°N')
            ax.text(0.02, 0.02, coord_text, transform=ax.transAxes,
                   fontsize=9, color='#7f8c8d', alpha=0.8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            edgecolor="none", alpha=0.8))
            
            plt.tight_layout()
            
            # Save with multiple formats
            base_filename = f"professional_spatial_residuals_{title_suffix.lower()}"
            plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), 
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none',
                       pad_inches=0.2)
            plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), 
                       bbox_inches='tight', facecolor='white', edgecolor='none',
                       pad_inches=0.2)
            
            plt.close()
            print(f"✓ Created professional spatial map for {title_suffix} set")
            
        except Exception as e:
            print(f"✗ Error processing {title_suffix} set: {e}")
            import traceback
            traceback.print_exc()
    
    # Create a combined comparison map - taller and narrower
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(base_width * 1.8, fig_height), dpi=300)  # Less wide
        fig.patch.set_facecolor('white')
        
        # Plot both datasets side by side
        datasets = [(train_results, 'Training', ax1), (val_results, 'Validation', ax2)]
        
        for results, title_suffix, ax in datasets:
            lons = results['longitudes']
            lats = results['latitudes']
            residuals = results['metrics']['residuals']
            
            # Bavaria boundaries
            bavaria.boundary.plot(ax=ax, color='#2c3e50', linewidth=1.0, alpha=0.9, zorder=3)
            
            # Residual points - sharp colors, no edges
            point_size = max(1.5, min(7, 45000 / len(residuals)))
            scatter = ax.scatter(lons, lats, c=residuals, cmap=residual_cmap, norm=norm,
                               s=point_size, alpha=0.9, marker='D',  # Higher alpha
                               edgecolors='none', linewidth=0,  # No edges for sharp colors
                               zorder=4, rasterized=True)
            
            # Styling
            ax.set_xlim(bounds[0] - buffer_distance, bounds[2] + buffer_distance)
            ax.set_ylim(bounds[1] - buffer_distance, bounds[3] + buffer_distance)
            ax.set_aspect('equal', adjustable='box')
            
            ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold', color='#2c3e50')
            
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.4, color='#7f8c8d')
            ax.set_facecolor('#fafafa')
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('#34495e')
            
            ax.tick_params(labelsize=10, width=0.8, colors='#2c3e50')
            
            # No titles - clean minimal design
        
        # Single colorbar for both plots - USING CORRECT METHOD
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax, extend='both')
        cbar.set_label('Residual (Actual - Predicted)', fontsize=12, fontweight='bold', 
                      color='#2c3e50', labelpad=15)
        cbar.ax.tick_params(labelsize=10, colors='#2c3e50')
        
        n_ticks = 7
        tick_positions = np.linspace(vmin, vmax, n_ticks)
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([f'{x:.2f}' for x in tick_positions])
        
        # Clean design - no main title
        
        plt.savefig(os.path.join(output_dir, "professional_spatial_residuals_comparison.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none',
                   pad_inches=0.2)
        plt.savefig(os.path.join(output_dir, "professional_spatial_residuals_comparison.pdf"), 
                   bbox_inches='tight', facecolor='white', edgecolor='none',
                   pad_inches=0.2)
        
        plt.close()
        print("✓ Created professional comparison spatial map")
        
    except Exception as e:
        print(f"✗ Error creating comparison map: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎨 Professional spatial maps completed!")
    print(f"📁 Maps saved to: {output_dir}")
    print(f"📊 Residual range: [{vmin:.3f}, {vmax:.3f}]")
    print(f"💎 Using diamond markers for enhanced visibility")

def save_detailed_metrics(train_results, val_results, output_dir):
    """Save detailed metrics to CSV"""
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

    # Extract transformation type
    target_transform = extract_transform_info(args.model_path)
    print(f"Detected target transform: {target_transform}")

    # Load model
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
    train_df = data_df[data_df['dataset_type'] == 'train']
    val_df = data_df[data_df['dataset_type'] == 'val']

    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples")

    # Get data paths
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

    # Run inference
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

    # Prepare results
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

    # Create professional visualizations
    print("Creating professional Q-Q plots...")
    create_professional_qq_plots(train_results, val_results, args.output_dir)

    print("Creating professional histograms...")
    create_professional_histograms(train_results, val_results, args.output_dir)

    print("Creating professional spatial maps...")
    create_professional_spatial_maps(train_results, val_results, args.output_dir, args.buffer_distance)

    # Save metrics
    save_detailed_metrics(train_results, val_results, args.output_dir)

    # Save complete results
    with open(os.path.join(args.output_dir, 'analysis_results.pkl'), 'wb') as f:
        pickle.dump({
            'train_results': train_results,
            'val_results': val_results,
            'model_path': args.model_path,
            'stats': stats_dict
        }, f)

    print(f"\nProfessional analysis complete! Results saved to {args.output_dir}")
    print("Created files:")
    print(f"  - professional_qq_plots.png/.pdf")
    print(f"  - professional_residual_histograms.png/.pdf") 
    print(f"  - professional_spatial_residuals.png/.pdf")

    # Print summary
    print(f"\nSummary Statistics:")
    print(f"Training (n={len(train_targets):,}): R²={train_metrics['r2']:.4f}, RMSE={train_metrics['rmse']:.4f}, Bias={train_metrics['bias']:.4f}")
    print(f"Validation (n={len(val_targets):,}): R²={val_metrics['r2']:.4f}, RMSE={val_metrics['rmse']:.4f}, Bias={val_metrics['bias']:.4f}")

if __name__ == "__main__":
    main()
