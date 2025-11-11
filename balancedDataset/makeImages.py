import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from accelerate import Accelerator
from tqdm import tqdm
from scipy import stats
from scipy.stats import gaussian_kde, expon, gamma, lognorm, weibull_min, beta, chi2, invgamma, norm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from config import (
    TIME_BEGINNING, TIME_END, MAX_OC,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataframe_loader import filter_dataframe, separate_and_add_data

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")

# Custom color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#00B4A6',
    'warning': '#F77E21',
    'background': '#F8F9FA',
    'text': '#2D3436',
    'grid': '#DDD6FE',
    'training': '#3B82F6',
    'validation': '#EF4444',
    'fit': '#10B981'
}

def setup_plot_style():
    """Setup consistent plot styling"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.axisbelow': True
    })

def vectorized_haversine(lon1, lat1, lon2, lat2, device='cpu'):
    R = 6371
    if device == 'cuda' and torch.cuda.is_available():
        lon1 = torch.tensor(lon1, device='cuda', dtype=torch.float32)
        lat1 = torch.tensor(lat1, device='cuda', dtype=torch.float32)
        lon2 = torch.tensor(lon2, device='cuda', dtype=torch.float32)
        lat2 = torch.tensor(lat2, device='cuda', dtype=torch.float32)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distances = R * c
        return distances.cpu().numpy()
    else:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

def compute_min_distances(val_df, train_df, device='cpu'):
    if train_df.empty or val_df.empty:
        return np.full(len(val_df), np.inf)

    val_lons = val_df['GPS_LONG'].values.astype(float)
    val_lats = val_df['GPS_LAT'].values.astype(float)
    train_lons = train_df['GPS_LONG'].values.astype(float)
    train_lats = train_df['GPS_LAT'].values.astype(float)

    val_lons = np.expand_dims(val_lons, axis=1)
    val_lats = np.expand_dims(val_lats, axis=1)

    distances = vectorized_haversine(val_lons, val_lats, train_lons, train_lats, device)
    min_distances = np.min(distances, axis=1)

    invalid = np.isnan(val_lons.flatten()) | np.isnan(val_lats.flatten()) | np.isnan(min_distances)
    min_distances[invalid] = np.inf
    return min_distances

def fit_exponential_family(data):
    distributions = [
        (expon, "Exponential", ["loc", "scale"]),
        (gamma, "Gamma", ["a", "loc", "scale"]),
        (lognorm, "Lognormal", ["s", "loc", "scale"]),
        (weibull_min, "Weibull", ["c", "loc", "scale"]),
        (beta, "Beta", ["a", "b", "loc", "scale"]),
        (chi2, "Chi-Square", ["df", "loc", "scale"]),
        (invgamma, "Inverse Gamma", ["a", "loc", "scale"]),
        (norm, "Normal", ["loc", "scale"])
    ]

    best_dist = None
    best_params = None
    best_ks_stat = float('inf')

    # Scale data for Beta distribution (which requires [0,1] range)
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data_scaled = (data - data_min) / (data_max - data_min)
    else:
        data_scaled = data  # Avoid division by zero

    for dist, dist_name, param_names in distributions:
        try:
            # Use scaled data for Beta distribution
            fit_data = data_scaled if dist == beta else data
            params = dist.fit(fit_data)
            # Adjust parameters for Beta to map back to original scale
            if dist == beta:
                a, b, loc, scale = params
                params = (a, b, loc * (data_max - data_min) + data_min, scale * (data_max - data_min))
            ks_stat, _ = stats.ks_1samp(fit_data, dist.cdf, args=params)
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_dist = dist
                best_params = dict(zip(param_names, params))
                best_dist_name = dist_name
        except Exception as e:
            print(f"Failed to fit {dist_name}: {e}")

    print(f"Best fitting distribution: {best_dist_name}")
    print(f"Parameters: {best_params}")
    print(f"KS statistic: {best_ks_stat:.4f}")

    return best_dist, best_params, best_dist_name

def create_spatial_distribution_plot(subset_df, remaining_df, save_path, iteration=0):
    """Create a beautiful spatial distribution plot"""
    setup_plot_style()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load Bavaria boundary
    bavaria_file = 'bavaria.geojson'
    if not os.path.exists(bavaria_file):
        bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
        bavaria = bavaria[bavaria['name'] == 'Bayern']
        bavaria.to_file(bavaria_file)
    else:
        bavaria = gpd.read_file(bavaria_file)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300, facecolor='white')
    
    # Plot Bavaria boundary with style
    bavaria.boundary.plot(ax=ax, color=COLORS['text'], linewidth=2, alpha=0.8)
    bavaria.plot(ax=ax, color=COLORS['background'], alpha=0.3, edgecolor='none')
    
    # Plot points with beautiful styling
    scatter_train = ax.scatter(
        remaining_df['GPS_LONG'], remaining_df['GPS_LAT'], 
        c=COLORS['training'], 
        s=25, 
        alpha=0.7, 
        edgecolors='white',
        linewidth=0.5,
        label=f'Training Set (n={len(remaining_df)})'
    )
    
    scatter_val = ax.scatter(
        subset_df['GPS_LONG'], subset_df['GPS_LAT'], 
        c=COLORS['validation'], 
        s=35, 
        alpha=0.8,
        edgecolors='white',
        linewidth=0.8,
        label=f'Validation Set (n={len(subset_df)})',
        marker='s'
    )
    
    # Styling
    ax.set_title(f'Spatial Distribution Analysis', 
                fontsize=20, fontweight='bold', pad=20, color=COLORS['text'])
    ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='600')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='600')
    
    # Legend with custom styling
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.95, edgecolor='none')
    legend.get_frame().set_facecolor('white')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    
    plt.tight_layout()
    filename = f'01_spatial_distribution_{timestamp}_iter{iteration}.png'
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")

def create_distance_distribution_plot(subset_df, remaining_df, save_path, iteration=0, device='cpu'):
    """Create a beautiful distance distribution plot"""
    setup_plot_style()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    distances = compute_min_distances(subset_df, remaining_df, device)
    distances = distances[np.isfinite(distances)]
    
    if len(distances) == 0:
        print("No valid distances to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300, facecolor='white')
    
    # Create histogram with gradient effect
    n, bins, patches = ax.hist(distances, bins=40, density=True, alpha=0.7, 
                              color=COLORS['success'], edgecolor='white', linewidth=0.8)
    
    # Apply gradient coloring to bars
    for i, p in enumerate(patches):
        gradient_color = plt.cm.viridis(i / len(patches))
        p.set_facecolor(gradient_color)
        p.set_alpha(0.8)
    
    
    # Statistics annotation
    stats_text = f"""Statistics:
    Mean: {np.mean(distances):.2f} km
    Median: {np.median(distances):.2f} km
    Std: {np.std(distances):.2f} km
    Min: {np.min(distances):.2f} km
    Max: {np.max(distances):.2f} km"""
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor=COLORS['grid']),
            verticalalignment='top', horizontalalignment='right', fontsize=10,
            fontfamily='monospace')
    
    ax.set_title(f'Minimum Distance Distribution', 
                fontsize=20, fontweight='bold', pad=20, color=COLORS['text'])
    ax.set_xlabel('Distance (km)', fontsize=14, fontweight='600')
    ax.set_ylabel('Density', fontsize=14, fontweight='600')
    

    plt.tight_layout()
    filename = f'02_distance_distribution_{timestamp}_iter{iteration}.png'
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")

def create_oc_distribution_plot(subset_df, remaining_df, save_path, iteration=0):
    """Create a beautiful OC distribution comparison plot"""
    setup_plot_style()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300, facecolor='white',
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
    
    # Main histogram plot
    bins = np.histogram_bin_edges(np.concatenate([remaining_df['OC'], subset_df['OC']]), bins=35)
    
    ax1.hist(remaining_df['OC'].values, bins=bins, density=True, alpha=0.7, 
             color=COLORS['training'], label=f'Training Set (n={len(remaining_df)})',
             edgecolor='white', linewidth=0.5)
    ax1.hist(subset_df['OC'].values, bins=bins, density=True, alpha=0.8, 
             color=COLORS['validation'], label=f'Validation Set (n={len(subset_df)})',
             edgecolor='white', linewidth=0.5)
    
    # Add KDE overlays
    if len(remaining_df['OC']) > 1:
        kde_train = gaussian_kde(remaining_df['OC'].values)
        x_range = np.linspace(min(remaining_df['OC'].min(), subset_df['OC'].min()),
                             max(remaining_df['OC'].max(), subset_df['OC'].max()), 200)
        ax1.plot(x_range, kde_train(x_range), color=COLORS['training'], 
                linewidth=3, alpha=0.9, linestyle='--')
    
    if len(subset_df['OC']) > 1:
        kde_val = gaussian_kde(subset_df['OC'].values)
        ax1.plot(x_range, kde_val(x_range), color=COLORS['validation'], 
                linewidth=3, alpha=0.9, linestyle='--')
    
    ax1.set_title(f'Organic Carbon (OC) Distribution Comparison', 
                 fontsize=20, fontweight='bold', pad=20, color=COLORS['text'])
    ax1.set_ylabel('Density', fontsize=14, fontweight='600')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Box plot comparison
    data_to_plot = [remaining_df['OC'].values, subset_df['OC'].values]
    labels = ['Training', 'Validation']
    colors = [COLORS['training'], COLORS['validation']]
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.6))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color(COLORS['text'])
            item.set_linewidth(2)
    
    ax2.set_ylabel('OC Value', fontsize=12, fontweight='600')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Shared x-axis label
    fig.text(0.5, 0.02, 'Organic Carbon (OC) Value', ha='center', fontsize=14, fontweight='600')
    
    plt.tight_layout()
    filename = f'03_oc_distribution_{timestamp}_iter{iteration}.png'
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")

def create_oc_histogram_kde_combined_plot(subset_df, remaining_df, save_path, iteration=0):
    """Create a beautiful combined histogram and KDE plot for OC distribution"""
    setup_plot_style()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300, facecolor='white')
    
    # Determine common bins for both datasets
    bins = np.histogram_bin_edges(
        np.concatenate([remaining_df['OC'], subset_df['OC']]), 
        bins=40
    )
    
    # Plot histograms with transparency
    ax.hist(remaining_df['OC'].values, bins=bins, density=True, alpha=0.5, 
            color=COLORS['training'], label=f'Training Histogram (n={len(remaining_df)})',
            edgecolor='white', linewidth=0.8)
    
    ax.hist(subset_df['OC'].values, bins=bins, density=True, alpha=0.5, 
            color=COLORS['validation'], label=f'Validation Histogram (n={len(subset_df)})',
            edgecolor='white', linewidth=0.8)
    
    # Calculate and plot KDE overlays
    x_range = np.linspace(
        min(remaining_df['OC'].min(), subset_df['OC'].min()),
        max(remaining_df['OC'].max(), subset_df['OC'].max()), 
        300
    )
    
    if len(remaining_df['OC']) > 1:
        kde_train = gaussian_kde(remaining_df['OC'].values)
        ax.plot(x_range, kde_train(x_range), 
                color=COLORS['training'], 
                linewidth=3.5, 
                alpha=0.95,
                label=f'Training KDE',
                linestyle='-')
        # Add subtle fill under KDE
        ax.fill_between(x_range, kde_train(x_range), 
                        alpha=0.15, color=COLORS['training'])
    
    if len(subset_df['OC']) > 1:
        kde_val = gaussian_kde(subset_df['OC'].values)
        ax.plot(x_range, kde_val(x_range), 
                color=COLORS['validation'], 
                linewidth=3.5, 
                alpha=0.95,
                label=f'Validation KDE',
                linestyle='-')
        # Add subtle fill under KDE
        ax.fill_between(x_range, kde_val(x_range), 
                        alpha=0.15, color=COLORS['validation'])
    
    # Add statistics box
    train_stats = f"""Training Statistics:
    Mean: {remaining_df['OC'].mean():.3f}
    Median: {remaining_df['OC'].median():.3f}
    Std: {remaining_df['OC'].std():.3f}
    Min: {remaining_df['OC'].min():.3f}
    Max: {remaining_df['OC'].max():.3f}"""
    
    val_stats = f"""Validation Statistics:
    Mean: {subset_df['OC'].mean():.3f}
    Median: {subset_df['OC'].median():.3f}
    Std: {subset_df['OC'].std():.3f}
    Min: {subset_df['OC'].min():.3f}
    Max: {subset_df['OC'].max():.3f}"""
    
    # Position stats boxes
    ax.text(0.02, 0.98, train_stats, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['training'], 
                     alpha=0.15, edgecolor=COLORS['training'], linewidth=2),
            verticalalignment='top', horizontalalignment='left', 
            fontsize=9, fontfamily='monospace', color=COLORS['text'])
    
    ax.text(0.98, 0.98, val_stats, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['validation'], 
                     alpha=0.15, edgecolor=COLORS['validation'], linewidth=2),
            verticalalignment='top', horizontalalignment='right', 
            fontsize=9, fontfamily='monospace', color=COLORS['text'])
    
    # Styling
    ax.set_title('Organic Carbon Distribution: Histogram & KDE Analysis', 
                fontsize=20, fontweight='bold', pad=20, color=COLORS['text'])
    ax.set_xlabel('Organic Carbon (OC) Value', fontsize=14, fontweight='600')
    ax.set_ylabel('Density', fontsize=14, fontweight='600')
    
    # Legend with custom styling
    legend = ax.legend(loc='upper center', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.95, edgecolor='none',
                      ncol=2, bbox_to_anchor=(0.5, -0.08))
    legend.get_frame().set_facecolor('white')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filename = f'05_oc_histogram_kde_combined_{timestamp}_iter{iteration}.png'
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")

def create_kde_comparison_plot(subset_df, remaining_df, full_df, best_dist, best_params, dist_name, save_path, iteration=0):
    """Create a beautiful KDE vs fitted distribution comparison plot"""
    setup_plot_style()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300, facecolor='white',
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
    
    # Main comparison plot
    oc_range = np.linspace(max(0, min(full_df['OC'].min(), subset_df['OC'].min())), 
                          max(full_df['OC'].max(), subset_df['OC'].max()), 300)
    
    # KDE plots with beautiful styling
    if len(subset_df['OC']) > 1:
        kde_val = gaussian_kde(subset_df['OC'].values)
        ax1.plot(oc_range, kde_val(oc_range), color=COLORS['validation'], linewidth=3, 
                label=f'Validation KDE (n={len(subset_df)})', alpha=0.9)
        ax1.fill_between(oc_range, kde_val(oc_range), alpha=0.2, color=COLORS['validation'])

    if len(remaining_df['OC']) > 1:
        kde_train = gaussian_kde(remaining_df['OC'].values)
        ax1.plot(oc_range, kde_train(oc_range), color=COLORS['training'], linewidth=3, 
                label=f'Training KDE (n={len(remaining_df)})', alpha=0.9)
        ax1.fill_between(oc_range, kde_train(oc_range), alpha=0.2, color=COLORS['training'])

    # Plot fitted distribution
    dist_params = list(best_params.values())
    if best_dist == beta:
        # Scale oc_range for Beta distribution
        data_min, data_max = full_df['OC'].min(), full_df['OC'].max()
        scaled_range = (oc_range - data_min) / (data_max - data_min)
        dist_pdf = best_dist.pdf(scaled_range, *dist_params[:2], loc=0, scale=1) / (data_max - data_min)
    else:
        dist_pdf = best_dist.pdf(oc_range, *dist_params)
    
    ax1.plot(oc_range, dist_pdf, color=COLORS['fit'], linewidth=4, 
            label=f'{dist_name} Fit', alpha=0.9, linestyle='--')

    ax1.set_title(f'Distribution Comparison: KDE vs {dist_name} Fit', 
                 fontsize=20, fontweight='bold', pad=20, color=COLORS['text'])
    ax1.set_ylabel('Density', fontsize=14, fontweight='600')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    

    
    # Residuals plot
    if len(subset_df['OC']) > 1:
        kde_val_interp = kde_val(oc_range)
        residuals = kde_val_interp - dist_pdf
        ax2.plot(oc_range, residuals, color=COLORS['accent'], linewidth=2, alpha=0.8)
        ax2.fill_between(oc_range, residuals, alpha=0.3, color=COLORS['accent'])
        ax2.axhline(y=0, color=COLORS['text'], linestyle='-', alpha=0.5, linewidth=1)
        ax2.set_ylabel('Residuals\n(KDE - Fit)', fontsize=12, fontweight='600')
        ax2.grid(True, alpha=0.3)

    # Shared x-axis label
    fig.text(0.5, 0.02, 'Organic Carbon (OC) Value', ha='center', fontsize=14, fontweight='600')
    
    plt.tight_layout()
    filename = f'04_kde_comparison_{timestamp}_iter{iteration}.png'
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")

def create_visualizations(subset_df, remaining_df, full_df, best_dist, best_params, dist_name, save_path, iteration=0, device='cpu'):
    """Create all five separate visualization plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n🎨 Creating beautiful visualizations for iteration {iteration}...")
    
    # Create each plot separately
    create_spatial_distribution_plot(subset_df, remaining_df, save_path, iteration)
    create_distance_distribution_plot(subset_df, remaining_df, save_path, iteration, device)
    create_oc_distribution_plot(subset_df, remaining_df, save_path, iteration)
    create_kde_comparison_plot(subset_df, remaining_df, full_df, best_dist, best_params, dist_name, save_path, iteration)
    create_oc_histogram_kde_combined_plot(subset_df, remaining_df, save_path, iteration)  # NEW PLOT
    
    print(f"✨ All visualizations saved to: {save_path}")

def create_optimized_subset(df, best_dist, best_params, dist_name, target_val_ratio=0.08, output_dir='ImagesOutput', device='cpu', distance_threshold=1.4):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()

    # Fix: Convert POINTID to string to ensure consistent type for parquet writing
    if 'POINTID' in df.columns:
        df['POINTID'] = df['POINTID'].astype(str)

    df[['OC', 'GPS_LAT', 'GPS_LONG']] = df[['OC', 'GPS_LAT', 'GPS_LONG']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])

    total_samples = len(df)
    target_size = int(total_samples * target_val_ratio)
    initial_ratio = target_val_ratio + 0.02
    max_ratio = 0.995

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    while initial_ratio <= max_ratio:
        subset_size = int(total_samples * initial_ratio)

        # Sample according to the best-fitting distribution
        oc_min, oc_max = df['OC'].min(), df['OC'].max()
        dist_params = list(best_params.values())
        dist_samples = best_dist.rvs(*dist_params, size=subset_size * 2)  # Oversample
        if best_dist == beta:
            # Rescale Beta samples to OC range
            dist_samples = oc_min + (oc_max - oc_min) * dist_samples
        else:
            dist_samples = oc_min + (oc_max - oc_min) * (dist_samples - np.min(dist_samples)) / (np.max(dist_samples) - np.min(dist_samples))  # Scale to OC range

        # Assign weights based on fitted distribution's PDF
        oc_values = df['OC'].values
        if best_dist == beta:
            scaled_oc = (oc_values - oc_min) / (oc_max - oc_min)
            weights = best_dist.pdf(scaled_oc, *dist_params[:2], loc=0, scale=1)
        else:
            weights = best_dist.pdf(oc_values, *dist_params)
        weights = weights / weights.sum()  # Normalize weights

        # Sample indices based on weights
        subset_indices = np.random.choice(df.index, size=subset_size, replace=False, p=weights)

        subset_df = df.loc[subset_indices]
        remaining_df = df.drop(subset_indices)

        subset_df.to_parquet(output_dir / 'initial_subset_df.parquet')
        remaining_df.to_parquet(output_dir / 'initial_remaining_df.parquet')

        create_visualizations(subset_df, remaining_df, df, best_dist, best_params, dist_name, output_dir, iteration=0, device=device)

        min_distances = compute_min_distances(subset_df, remaining_df, device)
        validation_df = subset_df[min_distances >= distance_threshold]
        points_to_flip = subset_df[min_distances < distance_threshold]

        if not points_to_flip.empty:
            print(f"Flipping {len(points_to_flip)} points (distance < {distance_threshold} km)")
            training_df = pd.concat([remaining_df, points_to_flip])
        else:
            training_df = remaining_df

        val_size = len(validation_df)
        val_ratio = val_size / total_samples
        if val_size >= target_size:
            break
        else:
            print(f"Validation set size {val_ratio*100:.2f}% < {target_val_ratio*100}%. Increasing ratio.")
            initial_ratio += 0.02
            if initial_ratio > max_ratio:
                print(f"Max ratio {max_ratio*100}% reached. Using current validation set.")
                break

    # Ensure no empty bins in training set
    oc_bins = pd.qcut(df['OC'], q=10, duplicates='drop')
    train_bins = pd.cut(training_df['OC'], bins=oc_bins.cat.categories)
    bin_counts = train_bins.value_counts()
    empty_bins = bin_counts[bin_counts == 0].index

    if not empty_bins.empty:
        print(f"Found {len(empty_bins)} empty bins in training set. Flipping points to fill them.")
        for bin_label in empty_bins:
            val_in_bin = validation_df[pd.cut(validation_df['OC'], bins=oc_bins.cat.categories) == bin_label]
            if not val_in_bin.empty:
                flip_point = val_in_bin.sample(n=1)
                training_df = pd.concat([training_df, flip_point])
                validation_df = validation_df.drop(flip_point.index)
                print(f"Flipped 1 point from validation to training for bin {bin_label}")

    validation_df.to_parquet(output_dir / 'final_validation_df.parquet')
    training_df.to_parquet(output_dir / 'final_training_df.parquet')

    create_visualizations(validation_df, training_df, df, best_dist, best_params, dist_name, output_dir, iteration=1, device=device)

    final_min_distance = compute_min_distances(validation_df, training_df, device).min()

    print(f'\n📊 Final Results:')
    print(f'Full dataset size: {total_samples}')
    print(f'Final validation set: {len(validation_df)} ({val_ratio*100:.2f}%)')
    print(f'Final training set: {len(training_df)}')
    print(f'Minimum distance (km): {final_min_distance:.2f}')
    print(f'Validation OC distribution matched to {dist_name} with parameters: {best_params}')

    return validation_df, training_df

def main():
    parser = argparse.ArgumentParser(description='Enhanced validation set creation with beautiful visualizations')
    parser.add_argument('--output-dir', type=str, default='ImagesOutput', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold in km')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # Load and filter dataframe
    print("📂 Loading dataset...")
    df = pd.read_excel(file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)



    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)

    # Convert coordinates and OC to numeric and drop NAs
    df[['OC', 'GPS_LAT', 'GPS_LONG']] = df[['OC', 'GPS_LAT', 'GPS_LONG']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])

    print(f"✅ Dataset loaded: {len(df)} samples")

    # Fit distributions to the data
    print("📊 Fitting distributions to OC values...")
    best_dist, best_params, dist_name = fit_exponential_family(df['OC'].values)

    # Create optimized subset
    print(f"🔍 Creating optimized validation set with target ratio: {args.target_val_ratio:.2%}")
    validation_df, training_df = create_optimized_subset(
        df, 
        best_dist, 
        best_params, 
        dist_name,
        target_val_ratio=args.target_val_ratio,
        output_dir=args.output_dir,
        device=device,
        distance_threshold=args.distance_threshold
    )

    # Save results
    output_path = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_df.to_csv(output_path / f'validation_set_{timestamp}.csv', index=False)
    training_df.to_csv(output_path / f'training_set_{timestamp}.csv', index=False)

    # Generate summary
    summary = {
        'timestamp': timestamp,
        'total_samples': len(df),
        'validation_samples': len(validation_df),
        'training_samples': len(training_df),
        'validation_ratio': len(validation_df) / len(df),
        'best_distribution': dist_name,
        'distribution_parameters': best_params,
        'min_distance_km': compute_min_distances(validation_df, training_df, device).min()
    }

    # Save summary
    with open(output_path / f'summary_{timestamp}.txt', 'w') as f:
        f.write("=== Dataset Split Summary ===\n\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Total samples: {summary['total_samples']}\n")
        f.write(f"Validation samples: {summary['validation_samples']} ({summary['validation_ratio']:.2%})\n")
        f.write(f"Training samples: {summary['training_samples']}\n")
        f.write(f"Best distribution: {summary['best_distribution']}\n")
        f.write(f"Distribution parameters: {summary['distribution_parameters']}\n")
        f.write(f"Minimum distance between sets: {summary['min_distance_km']:.2f} km\n")

    print(f"\n✨ All done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()