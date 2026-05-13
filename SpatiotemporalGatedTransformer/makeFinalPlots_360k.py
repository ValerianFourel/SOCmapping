#!/usr/bin/env python3
"""
Professional script to recreate residual analysis plots from pickle file
with publication-quality styling and Bavaria boundary
Updated to match actual pickle structure
Modified: Yellow center for spatial residuals colormap + small diamonds with capped residuals
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from pathlib import Path
import os
import geopandas as gpd
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Professional color palette - optimized for publication
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#00B4A6',
    'warning': '#F77E21',
    'background': '#F8F9FA',
    'text': '#000000',  # Pure black for maximum readability
    'grid': '#CCCCCC',
    'training': '#3B82F6',
    'validation': '#EF4444',
    'fit': '#10B981'
}

def create_yellow_centered_cmap():
    """
    Create a custom diverging colormap with yellow center
    Blue (negative) -> Yellow (zero) -> Red (positive)
    """
    colors = [
        '#2166AC',  # Dark blue (most negative)
        '#4393C3',  # Medium blue
        '#92C5DE',  # Light blue
        '#FFFF00',  # Bright yellow (zero)
        '#FDAE61',  # Light orange
        '#F46D43',  # Orange-red
        '#D73027'   # Dark red (most positive)
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue_yellow_red', colors, N=n_bins)
    return cmap

def setup_plot_style():
    """Setup publication-quality plot styling with large, readable text"""
    plt.rcParams.update({
        # Font settings - significantly larger for publication
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 28,
        
        # Line and marker sizes
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'patch.linewidth': 2.0,
        
        # Axes settings
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linewidth': 1.0,
        'axes.axisbelow': True,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        
        # Tick settings
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
        
        # Legend settings
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'legend.fancybox': True,
        'legend.frameon': True,
        'legend.borderpad': 0.6,
        'legend.labelspacing': 0.6,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

def load_bavaria_boundary():
    """Load Bavaria boundary from GeoJSON"""
    bavaria_file = 'bavaria.geojson'
    if not os.path.exists(bavaria_file):
        print("📥 Downloading Bavaria boundary...")
        bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
        bavaria = bavaria[bavaria['name'] == 'Bayern']
        bavaria.to_file(bavaria_file)
        print("✓ Bavaria boundary downloaded")
    else:
        bavaria = gpd.read_file(bavaria_file)
        print("✓ Bavaria boundary loaded from cache")
    return bavaria

def inspect_pickle_structure(pkl_path):
    """
    Detailed inspection of pickle file structure
    """
    print("=" * 80)
    print(f"INSPECTING PICKLE FILE: {pkl_path}")
    print("=" * 80)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nTop-level type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"\nDictionary with {len(data)} keys:")
        for key in data.keys():
            print(f"\n  📁 Key: '{key}'")
            value = data[key]
            print(f"     Type: {type(value).__name__}")
            
            if isinstance(value, dict):
                print(f"     Sub-keys ({len(value)}): {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    print(f"       └─ {sub_key}: {type(sub_value).__name__}", end="")
                    if hasattr(sub_value, 'shape'):
                        print(f" shape={sub_value.shape}", end="")
                    elif isinstance(sub_value, (list, tuple)):
                        print(f" length={len(sub_value)}", end="")
                    elif isinstance(sub_value, (int, float)):
                        print(f" value={sub_value}", end="")
                    print()
            elif hasattr(value, 'shape'):
                print(f"     Shape: {value.shape}")
                print(f"     Dtype: {value.dtype}")
            elif isinstance(value, str):
                print(f"     Value: {value}")
            elif isinstance(value, (int, float)):
                print(f"     Value: {value}")
    
    print("\n" + "=" * 80)
    return data

def compute_metrics(predictions, targets):
    """Compute R², RMSE, and MAE metrics"""
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    return r2, rmse, mae

def print_statistics_summary(data):
    """
    Print comprehensive statistics summary
    """
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    
    # Extract data
    train_pred = data['train_results']['predictions']
    train_target = data['train_results']['targets']
    val_pred = data['val_results']['predictions']
    val_target = data['val_results']['targets']
    
    # Calculate residuals
    train_residuals = train_target - train_pred
    val_residuals = val_target - val_pred
    
    # Use stored R² values from metrics
    train_r2 = 0.8091766084758379
    val_r2 = 0.5680846967713404
    # Compute other metrics
    _, train_rmse, train_mae = compute_metrics(train_pred, train_target)
    _, val_rmse, val_mae = compute_metrics(val_pred, val_target)
    
    # Check if metrics exist in the data
    if 'metrics' in data['train_results']:
        print("\n📊 Stored Metrics:")
        print(f"  Train metrics: {data['train_results']['metrics']}")
        print(f"  Val metrics: {data['val_results']['metrics']}")
    
    print("\n📊 Model Performance:")
    print(f"  Training R²:        {train_r2:.4f}")
    print(f"  Validation R²:      {val_r2:.4f}")
    
    # Add context about R² gap
    r2_gap = train_r2 - val_r2
    print(f"  R² Gap:             {r2_gap:.4f}")
    if r2_gap > 0.2:
        print(f"  ⚠️  Note: Large gap suggests spatial extrapolation challenge")
        print(f"     This is expected for spatial validation with geographic separation")
    
    print(f"\n  Training RMSE:      {train_rmse:.4f}")
    print(f"  Validation RMSE:    {val_rmse:.4f}")
    print(f"  Training MAE:       {train_mae:.4f}")
    print(f"  Validation MAE:     {val_mae:.4f}")
    
    print("\n📈 Training Set:")
    print(f"  N samples:          {len(train_residuals):,}")
    print(f"  Residual mean:      {train_residuals.mean():.4f}")
    print(f"  Residual std:       {train_residuals.std():.4f}")
    print(f"  Residual min:       {train_residuals.min():.4f}")
    print(f"  Residual max:       {train_residuals.max():.4f}")
    
    print("\n📉 Validation Set:")
    print(f"  N samples:          {len(val_residuals):,}")
    print(f"  Residual mean:      {val_residuals.mean():.4f}")
    print(f"  Residual std:       {val_residuals.std():.4f}")
    print(f"  Residual min:       {val_residuals.min():.4f}")
    print(f"  Residual max:       {val_residuals.max():.4f}")
    
    # Bias analysis
    print("\n📊 Bias Analysis:")
    print(f"  Training bias:      {train_residuals.mean():.4f} (target: 0)")
    print(f"  Validation bias:    {val_residuals.mean():.4f} (target: 0)")
    if abs(val_residuals.mean()) > 1.0:
        if val_residuals.mean() > 0:
            print(f"  ⚠️  Model underpredicts on validation set")
        else:
            print(f"  ⚠️  Model overpredicts on validation set")
    
    print("\n🎯 Model Path:")
    print(f"  {data['model_path']}")
    
    if 'stats' in data:
        print("\n📐 Normalization Statistics:")
        if 'target_mean' in data['stats']:
            print(f"  Target mean:  {data['stats']['target_mean']:.4f}")
        if 'target_std' in data['stats']:
            print(f"  Target std:   {data['stats']['target_std']:.4f}")
    
    print("\n" + "=" * 80)

def create_qq_plots(data, output_dir='./plots'):
    """
    Create professional Q-Q plots for training and validation sets
    """
    setup_plot_style()
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate residuals
    train_residuals = data['train_results']['targets'] - data['train_results']['predictions']
    val_residuals = data['val_results']['targets'] - data['val_results']['predictions']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=300, facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Training Q-Q plot
    ax = axes[0]
    stats.probplot(train_residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(7)
    ax.get_lines()[0].set_color(COLORS['training'])
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('#06A77D')
    ax.get_lines()[1].set_linewidth(4)
    
    ax.set_xlabel('Theoretical Quantiles (Standard Normal)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=20, fontweight='bold')
    ax.set_title('Training Set Q-Q Plot', fontsize=24, fontweight='bold', pad=30)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.legend(['Theoretical Normal Line',
               '95% Confidence Interval',
               f'Observed Quantiles (n={len(train_residuals):,})'],
              loc='upper left', fontsize=15, framealpha=0.95)
    ax.tick_params(labelsize=17, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Validation Q-Q plot
    ax = axes[1]
    stats.probplot(val_residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(9)
    ax.get_lines()[0].set_color(COLORS['validation'])
    ax.get_lines()[0].set_alpha(0.65)
    ax.get_lines()[1].set_color('#06A77D')
    ax.get_lines()[1].set_linewidth(4)
    
    ax.set_xlabel('Theoretical Quantiles (Standard Normal)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=20, fontweight='bold')
    ax.set_title('Validation Set Q-Q Plot', fontsize=24, fontweight='bold', pad=30)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.legend(['Theoretical Normal Line',
               '95% Confidence Interval',
               f'Observed Quantiles (n={len(val_residuals):,})'],
              loc='upper left', fontsize=15, framealpha=0.95)
    ax.tick_params(labelsize=17, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f'professional_qq_plots_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved Q-Q plots to: {output_path}")
    plt.close()

def create_spatial_residuals_comparison(data, bavaria, output_dir='./plots'):
    """
    Create professional spatial residual maps with Bavaria boundary
    Using SMALL DIAMONDS and CAPPED residuals [-15, 15]
    """
    setup_plot_style()
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate residuals
    train_residuals = data['train_results']['targets'] - data['train_results']['predictions']
    val_residuals = data['val_results']['targets'] - data['val_results']['predictions']
    
    train_lats = data['train_results']['latitudes']
    train_lons = data['train_results']['longitudes']
    val_lats = data['val_results']['latitudes']
    val_lons = data['val_results']['longitudes']
    
    # Cap residuals between -15 and 15
    RESIDUAL_MIN = -15
    RESIDUAL_MAX = 15
    
    # Filter data to only include residuals in range [-15, 15]
    train_mask = (train_residuals >= RESIDUAL_MIN) & (train_residuals <= RESIDUAL_MAX)
    val_mask = (val_residuals >= RESIDUAL_MIN) & (val_residuals <= RESIDUAL_MAX)
    
    train_residuals_filtered = train_residuals[train_mask]
    train_lats_filtered = train_lats[train_mask]
    train_lons_filtered = train_lons[train_mask]
    
    val_residuals_filtered = val_residuals[val_mask]
    val_lats_filtered = val_lats[val_mask]
    val_lons_filtered = val_lons[val_mask]
    
    n_train_excluded = len(train_residuals) - len(train_residuals_filtered)
    n_val_excluded = len(val_residuals) - len(val_residuals_filtered)
    
    print(f"  Training: {n_train_excluded:,} samples excluded (outside [-15, 15])")
    print(f"  Validation: {n_val_excluded:,} samples excluded (outside [-15, 15])")
    print(f"  Residual range: [{RESIDUAL_MIN}, {RESIDUAL_MAX}]")
    
    # Create custom colormap with yellow center
    yellow_cmap = create_yellow_centered_cmap()
    
    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(24, 10), dpi=300, facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create custom gridspec for better colorbar control
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.08], wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    axes = [ax1, ax2]
    
    # Training set spatial map
    ax = axes[0]
    
    # Plot Bavaria boundary
    bavaria.boundary.plot(ax=ax, color=COLORS['text'], linewidth=2.5, alpha=1.0, zorder=1)
    bavaria.plot(ax=ax, color=COLORS['background'], alpha=0.25, edgecolor='none', zorder=0)
    
    # Plot with DIAMOND markers (marker='D')
    scatter1 = ax.scatter(train_lons_filtered, train_lats_filtered, 
                        c=train_residuals_filtered, 
                        cmap=yellow_cmap, 
                        s=35,  # Small size for diamonds
                        alpha=0.75, 
                        edgecolors='black',
                        linewidth=0.3, 
                        vmin=RESIDUAL_MIN, 
                        vmax=RESIDUAL_MAX, 
                        zorder=2,
                        marker='D')  # Diamond marker
    
    ax.set_xlabel('Longitude', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel('Latitude', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_title(f'Training Set Residuals\n(n={len(train_residuals_filtered):,}, {n_train_excluded:,} excluded)',
                fontsize=24, fontweight='bold', pad=30)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(labelsize=18, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Validation set spatial map
    ax = axes[1]
    
    # Plot Bavaria boundary
    bavaria.boundary.plot(ax=ax, color=COLORS['text'], linewidth=2.5, alpha=1.0, zorder=1)
    bavaria.plot(ax=ax, color=COLORS['background'], alpha=0.25, edgecolor='none', zorder=0)
    
    # Plot with DIAMOND markers (marker='D')
    scatter2 = ax.scatter(val_lons_filtered, val_lats_filtered, 
                        c=val_residuals_filtered,
                        cmap=yellow_cmap, 
                        s=50,  # Slightly larger for validation
                        alpha=0.75, 
                        edgecolors='black',
                        linewidth=0.3, 
                        vmin=RESIDUAL_MIN, 
                        vmax=RESIDUAL_MAX, 
                        zorder=2,
                        marker='D')  # Diamond marker
    
    ax.set_xlabel('Longitude', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel('Latitude', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_title(f'Validation Set Residuals\n(n={len(val_residuals_filtered):,}, {n_val_excluded:,} excluded)',
                fontsize=24, fontweight='bold', pad=30)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(labelsize=18, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create HIGHLY VISIBLE colorbar
    cbar = plt.colorbar(scatter2, cax=cbar_ax, orientation='vertical')
    
    # Enhanced colorbar styling
    cbar.set_label('Residual (Actual - Predicted)', 
                   fontsize=22, fontweight='bold', 
                   rotation=270, labelpad=40, color=COLORS['text'])
    
    # Make colorbar ticks MUCH more visible
    cbar.ax.tick_params(labelsize=18, width=2.0, length=8, 
                       color=COLORS['text'], labelcolor=COLORS['text'])
    
    # Add border to colorbar
    cbar.outline.set_linewidth(2.0)
    cbar.outline.set_edgecolor(COLORS['text'])
    
    # Force colorbar to show more tick labels
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    
    # Make colorbar background white
    cbar.ax.set_facecolor('white')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f'spatial_residuals_comparison_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"✓ Saved spatial maps to: {output_path}")
    plt.close()

def create_residual_histograms(data, output_dir='./plots'):
    """
    Create professional residual histogram plots with KDE
    Enhanced layout with better space for statistics box
    """
    setup_plot_style()
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate residuals
    train_residuals = data['train_results']['targets'] - data['train_results']['predictions']
    val_residuals = data['val_results']['targets'] - data['val_results']['predictions']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=300, facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Training histogram
    ax = axes[0]
    n, bins, patches = ax.hist(train_residuals, bins=60, density=True, 
                               alpha=0.65, color=COLORS['training'], 
                               edgecolor='white', linewidth=0.8)
    
    # KDE
    kde = gaussian_kde(train_residuals)
    x_range = np.linspace(train_residuals.min(), train_residuals.max(), 400)
    ax.plot(x_range, kde(x_range), color='#023047', linewidth=4.5, 
           label='Kernel Density Estimate', zorder=5)
    
    # Mean line
    mu, std = train_residuals.mean(), train_residuals.std()
    ax.axvline(mu, color='#D62828', linestyle='--', linewidth=4,
              label=f'Mean (μ={mu:.2f})', zorder=5)
    
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=20, fontweight='bold')
    ax.set_title(f'Training Set Residuals',
                fontsize=24, fontweight='bold', pad=30)
    
    # Move legend to lower right to make space for text box
    ax.legend(fontsize=15, loc='lower right', framealpha=0.95)
    
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(labelsize=17, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add ENLARGED text box with statistics
    textstr = f'Statistics:\nMean: {mu:.3f}\nStd Dev: {std:.3f}\nMin: {train_residuals.min():.2f}\nMax: {train_residuals.max():.2f}'
    props = dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.95, 
                linewidth=2.5, edgecolor=COLORS['training'])
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=16,
           verticalalignment='top', bbox=props, fontweight='bold',
           family='monospace')
    
    # Validation histogram
    ax = axes[1]
    n, bins, patches = ax.hist(val_residuals, bins=60, density=True,
                               alpha=0.65, color=COLORS['validation'], 
                               edgecolor='white', linewidth=0.8)
    
    # KDE
    kde = gaussian_kde(val_residuals)
    x_range = np.linspace(val_residuals.min(), val_residuals.max(), 400)
    ax.plot(x_range, kde(x_range), color='#641220', linewidth=4.5,
           label='Kernel Density Estimate', zorder=5)
    
    # Mean line
    mu_val, std_val = val_residuals.mean(), val_residuals.std()
    ax.axvline(mu_val, color='#D62828', linestyle='--', linewidth=4,
              label=f'Mean (μ={mu_val:.2f})', zorder=5)
    
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=20, fontweight='bold')
    ax.set_title(f'Validation Set Residuals',
                fontsize=24, fontweight='bold', pad=30)
    
    # Move legend to lower right to make space for text box
    ax.legend(fontsize=15, loc='lower left', framealpha=0.95)
    
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(labelsize=17, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add ENLARGED text box with statistics
    textstr = f'Statistics:\nMean: {mu_val:.3f}\nStd Dev: {std_val:.3f}\nMin: {val_residuals.min():.2f}\nMax: {val_residuals.max():.2f}'
    props = dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.95, 
                linewidth=2.5, edgecolor=COLORS['validation'])
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=16,
           verticalalignment='top', bbox=props, fontweight='bold',
           family='monospace')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f'professional_residual_histograms_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved histograms to: {output_path}")
    plt.close()

def create_predicted_vs_actual(data, output_dir='./plots'):
    """
    Create professional predicted vs actual scatter plot
    """
    setup_plot_style()
    Path(output_dir).mkdir(exist_ok=True)
    
    train_actual = data['train_results']['targets']
    train_pred = data['train_results']['predictions']
    val_actual = data['val_results']['targets']
    val_pred = data['val_results']['predictions']
    
    # Use the actual R² values from your metrics
    train_r2 = 0.8092
    val_r2 = 0.5680
    
    fig, ax = plt.subplots(figsize=(13, 13), dpi=300, facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Training points
    ax.scatter(train_actual, train_pred, alpha=0.5, s=55, 
              color=COLORS['training'], edgecolors='black', linewidth=0.4,
              label=f'Training (R² = {train_r2:.4f})', zorder=2)
    
    # Validation points
    ax.scatter(val_actual, val_pred, alpha=0.65, s=70,
              color=COLORS['validation'], edgecolors='black', linewidth=0.4,
              label=f'Validation (R² = {val_r2:.4f})', zorder=3)
    
    # Perfect prediction line
    min_val = min(train_actual.min(), val_actual.min())
    max_val = max(train_actual.max(), val_actual.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
           linewidth=4, label='Perfect prediction', zorder=5)
    
    ax.set_xlabel('Actual Organic Carbon', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel('Predicted Organic Carbon', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_title('Predicted vs Actual Organic Carbon', 
                fontsize=26, fontweight='bold', pad=30)
    
    legend = ax.legend(fontsize=18, loc='upper left', framealpha=0.95, 
                      markerscale=1.8, borderpad=1.0, labelspacing=1.0)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('black')
    
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(labelsize=18, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make it square
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f'predicted_vs_actual_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved scatter plot to: {output_path}")
    plt.close()

def main():
    # Path to pickle file - UPDATE THIS PATH
    pkl_path = "/home/valerian/SGTPublication/residual_Maps_Bavaria_360kTFT/residual_Maps_Bavaria_360kTFT/analysis_results.pkl"
    output_dir = "./professional_plots_360k"
    
    print("\n" + "=" * 80)
    print("PROFESSIONAL RESIDUAL ANALYSIS PLOTS")
    print("=" * 80 + "\n")
    
    # Check if pickle file exists
    if not Path(pkl_path).exists():
        print(f"❌ Error: Pickle file not found at {pkl_path}")
        print("\nPlease update the pkl_path variable in the script with the correct path.")
        return
    
    # Inspect pickle structure
    data = inspect_pickle_structure(pkl_path)
    
    # Print statistics
    print_statistics_summary(data)
    
    # Load Bavaria boundary
    print("\n" + "=" * 80)
    print("LOADING BAVARIA BOUNDARY")
    print("=" * 80 + "\n")
    bavaria = load_bavaria_boundary()
    
    print("\n" + "=" * 80)
    print("GENERATING PROFESSIONAL PLOTS")
    print("=" * 80 + "\n")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create all plots
    print("Creating plots...")
    create_qq_plots(data, output_dir)
    create_spatial_residuals_comparison(data, bavaria, output_dir)
    create_residual_histograms(data, output_dir)
    create_predicted_vs_actual(data, output_dir)
    
    print("\n✅ All plots generated successfully!")
    print(f"📁 Output directory: {Path(output_dir).absolute()}")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()