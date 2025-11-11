import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, skew, kurtosis
from datetime import datetime
import os
from pathlib import Path

from config import (
    TIME_BEGINNING, TIME_END, MAX_OC,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataframe_loader import filter_dataframe

# Custom color palette
COLORS = {
    'primary': '#2E86AB',
    'kde_fill': '#FF6B9D',
    'histogram': '#4A90E2',
    'mean': '#E63946',
    'median': '#F77F00',
    'background': '#F8F9FA',
    'text': '#2D3436',
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
        'legend.fontsize': 10,
        'figure.titlesize': 18,
    })

def create_combined_histogram_kde_plot(data, column_name='OC', 
                                       title='SOC Distribution (Bavaria, 0-150 g/kg)',
                                       xlabel='SOC (g/kg)',
                                       output_dir='output',
                                       filename='soc_histogram_kde_combined.png',
                                       n_bins=150):
    """
    Create a combined histogram and KDE plot
    
    Parameters:
    -----------
    data : pd.DataFrame or np.array
        The data to plot
    column_name : str
        Name of the column to plot (if data is DataFrame)
    title : str
        Plot title
    xlabel : str
        X-axis label
    output_dir : str
        Directory to save the plot
    filename : str
        Output filename
    n_bins : int
        Number of bins for histogram (default: 150)
    """
    setup_plot_style()
    
    # Extract values
    if isinstance(data, pd.DataFrame):
        values = data[column_name].values
    else:
        values = data
    
    # Remove NaN values
    values = values[~np.isnan(values)]
    
    # Calculate statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    skewness = skew(values)
    kurt = kurtosis(values)
    min_val = np.min(values)
    max_val = np.max(values)
    n_samples = len(values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300, facecolor='white')
    
    # Create histogram with more bins
    n, bins, patches = ax.hist(values, bins=n_bins, density=True, alpha=0.6, 
                               color=COLORS['histogram'], edgecolor='white', 
                               linewidth=0.5, label='Histogram')
    
    # Create and plot KDE
    kde = gaussian_kde(values)
    x_range = np.linspace(min_val, max_val, 500)
    kde_values = kde(x_range)
    
    ax.plot(x_range, kde_values, color=COLORS['kde_fill'], linewidth=3.5, 
            alpha=0.95, label='KDE', linestyle='-')
    ax.fill_between(x_range, kde_values, alpha=0.2, color=COLORS['kde_fill'])
    
    # Add mean and median lines
    ax.axvline(mean_val, color=COLORS['mean'], linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_val:.1f}', alpha=0.8)
    ax.axvline(median_val, color=COLORS['median'], linestyle='--', linewidth=2.5, 
               label=f'Median: {median_val:.1f}', alpha=0.8)
    
    # Styling
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20, color=COLORS['text'])
    ax.set_xlabel(xlabel, fontsize=14, fontweight='600')
    ax.set_ylabel('Density', fontsize=14, fontweight='600')
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.95, edgecolor='none')
    legend.get_frame().set_facecolor('white')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename.replace('.png', '')}_{timestamp}.png"
    
    plt.savefig(output_path / full_filename, bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"✓ Plot saved: {output_path / full_filename}")
    print(f"\n📊 Statistics Summary:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Number of bins: {n_bins}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Std Dev: {std_val:.2f}")
    print(f"  Skewness: {skewness:.2f}")
    print(f"  Kurtosis: {kurt:.2f}")
    
    return fig

def main():
    """
    Main function - loads data using the same process as the original script
    """
    parser = argparse.ArgumentParser(description='Create combined histogram and KDE plot for SOC distribution')
    parser.add_argument('--output-dir', type=str, default='ImagesOutput', help='Output directory')
    parser.add_argument('--title', type=str, default='SOC Distribution (Bavaria, 0-150 g/kg)', 
                       help='Plot title')
    parser.add_argument('--xlabel', type=str, default='SOC (g/kg)', 
                       help='X-axis label')
    parser.add_argument('--column', type=str, default='OC', 
                       help='Column name to plot')
    parser.add_argument('--bins', type=int, default=150, 
                       help='Number of bins for histogram (default: 150)')
    args = parser.parse_args()

    # Load and filter dataframe (SAME AS ORIGINAL SCRIPT)
    print("📂 Loading dataset...")
    df = pd.read_excel(file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)
    
    print("🔍 Filtering dataset...")
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    
    # Convert coordinates and OC to numeric and drop NAs (SAME AS ORIGINAL SCRIPT)
    df[['OC', 'GPS_LAT', 'GPS_LONG']] = df[['OC', 'GPS_LAT', 'GPS_LONG']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"   OC range: {df['OC'].min():.2f} - {df['OC'].max():.2f}")
    
    # Create the combined plot
    print(f"\n🎨 Creating combined histogram and KDE plot with {args.bins} bins...")
    create_combined_histogram_kde_plot(
        data=df,
        column_name=args.column,
        title=args.title,
        xlabel=args.xlabel,
        output_dir=args.output_dir,
        filename='soc_histogram_kde_combined.png',
        n_bins=args.bins
    )
    
    print(f"\n✨ All done! Plot saved to {args.output_dir}")

if __name__ == "__main__":
    main()