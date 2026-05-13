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
    """Setup RESEARCH PAPER QUALITY plot styling with LARGE fonts"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 20,           # INCREASED from 11 to 20
        'axes.titlesize': 32,      # INCREASED from 16 to 32
        'axes.labelsize': 26,      # INCREASED from 12 to 26
        'xtick.labelsize': 22,     # INCREASED from 10 to 22
        'ytick.labelsize': 22,     # INCREASED from 10 to 22
        'legend.fontsize': 20,     # INCREASED from 10 to 20
        'figure.titlesize': 34,    # INCREASED from 18 to 34
    })

def create_combined_histogram_kde_plot(data, column_name='OC', 
                                       title='SOC Distribution (Bavaria, 0-150 g/kg)',
                                       xlabel='SOC (g/kg)',
                                       output_dir='output',
                                       filename='soc_histogram_kde_combined.png',
                                       n_bins=150):
    """
    Create a combined histogram and KDE plot with RESEARCH PAPER QUALITY formatting
    
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
    
    # Create figure - LARGER for research paper
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300, facecolor='white')  # INCREASED from (14, 9)
    
    # Create histogram with more bins
    n, bins, patches = ax.hist(values, bins=n_bins, density=True, alpha=0.6, 
                               color=COLORS['histogram'], edgecolor='white', 
                               linewidth=0.8, label='Histogram')  # Thicker edges
    
    # Create and plot KDE
    kde = gaussian_kde(values)
    x_range = np.linspace(min_val, max_val, 500)
    kde_values = kde(x_range)
    
    ax.plot(x_range, kde_values, color=COLORS['kde_fill'], linewidth=4.5,  # THICKER from 3.5 to 4.5
            alpha=0.95, label='KDE', linestyle='-')
    ax.fill_between(x_range, kde_values, alpha=0.2, color=COLORS['kde_fill'])
    
    # Add mean and median lines - THICKER
    ax.axvline(mean_val, color=COLORS['mean'], linestyle='--', linewidth=3.5,  # INCREASED from 2.5 to 3.5
               label=f'Mean: {mean_val:.1f}', alpha=0.8)
    ax.axvline(median_val, color=COLORS['median'], linestyle='--', linewidth=3.5,  # INCREASED from 2.5 to 3.5
               label=f'Median: {median_val:.1f}', alpha=0.8)
    
    # Styling - RESEARCH PAPER QUALITY with LARGER fonts
    ax.set_title(title, fontsize=32, fontweight='bold', pad=30, color=COLORS['text'])  # INCREASED from 20
    ax.set_xlabel(xlabel, fontsize=26, fontweight='bold', labelpad=15)  # INCREASED from 14, added labelpad
    ax.set_ylabel('Density', fontsize=26, fontweight='bold', labelpad=15)  # INCREASED from 14, added labelpad
    
    # Legend - LARGER and more prominent
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.98, edgecolor='black',
                      fontsize=22,  # EXPLICIT large font size
                      borderpad=1.2,  # More padding
                      labelspacing=1.0,  # More spacing between entries
                      handlelength=2.5,  # Longer lines in legend
                      handleheight=1.2)  # Taller legend markers
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(2.0)  # Thicker frame
    
    # Grid - slightly more prominent
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0, axis='y')  # Thicker grid
    ax.set_axisbelow(True)
    
    # Thicker spines for remaining axes
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Larger tick marks
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8, pad=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename.replace('.png', '')}_{timestamp}.png"
    
    # Save as PNG
    plt.savefig(output_path / full_filename, bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300, pad_inches=0.3)
    
    # ALSO save as PDF for publication
    pdf_filename = full_filename.replace('.png', '.pdf')
    plt.savefig(output_path / pdf_filename, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    
    # ALSO save as EPS for LaTeX
    eps_filename = full_filename.replace('.png', '.eps')
    plt.savefig(output_path / eps_filename, format='eps', bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    
    plt.close()
    
    print(f"✓ PNG saved: {output_path / full_filename}")
    print(f"✓ PDF saved: {output_path / pdf_filename}")
    print(f"✓ EPS saved: {output_path / eps_filename}")
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
    parser = argparse.ArgumentParser(description='Create RESEARCH PAPER QUALITY combined histogram and KDE plot')
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
    print(f"\n🎨 Creating RESEARCH PAPER QUALITY histogram and KDE plot with {args.bins} bins...")
    create_combined_histogram_kde_plot(
        data=df,
        column_name=args.column,
        title=args.title,
        xlabel=args.xlabel,
        output_dir=args.output_dir,
        filename='soc_histogram_kde_combined.png',
        n_bins=args.bins
    )
    
    print(f"\n✨ All done! RESEARCH PAPER QUALITY plots saved to {args.output_dir}")
    print(f"   Formats: PNG (300 DPI), PDF (publication), EPS (LaTeX)")

if __name__ == "__main__":
    main()