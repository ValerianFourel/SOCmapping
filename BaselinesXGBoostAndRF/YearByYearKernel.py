import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from balancedDataset import create_validation_train_sets
from scipy.stats import gaussian_kde
from matplotlib.collections import PolyCollection

class SOCRidgePlot:
    def __init__(self, output_dir='figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.full_df = None

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='SOC Ridge Plot by Year')
        parser.add_argument('--output-dir', type=str, default='figures', help='Output directory')
        parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
        parser.add_argument('--distance-threshold', type=float, default=1.4, help='Distance threshold')
        return parser.parse_args()

    def load_data(self, args):
        """Load data"""
        print("Loading data...")
        
        validation_df, training_df = create_validation_train_sets(
            df=None,
            output_dir=args.output_dir,
            target_val_ratio=args.target_val_ratio,
            use_gpu=True,
            distance_threshold=args.distance_threshold
        )
        
        self.full_df = pd.concat([validation_df, training_df], ignore_index=True)
        
        print(f"Loaded {len(self.full_df)} samples")
        print(f"Year range: {self.full_df['year'].min()} - {self.full_df['year'].max()}")
        print(f"SOC range: {self.full_df['OC'].min():.1f} - {self.full_df['OC'].max():.1f} g/kg")

    def plot_ridge(self):
        """Create beautiful 3D perspective ridge plot with density layers for each year"""
        
        # Prepare data
        data = self.full_df[['year', 'OC']].dropna()
        years = sorted(data['year'].unique())
        
        # Calculate statistics
        print("\n" + "="*70)
        print("Year    N      Mean    Median    Std      Min      Max")
        print("="*70)
        for year in years:
            year_data = data[data['year'] == year]['OC']
            print(f"{year}  {len(year_data):4d}   {year_data.mean():6.2f}  "
                  f"{year_data.median():6.2f}  {year_data.std():6.2f}  "
                  f"{year_data.min():6.2f}  {year_data.max():6.2f}")
        print("="*70)
        
        # Create figure with subtle background for 3D effect
        fig, ax = plt.subplots(figsize=(20, 14))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        
        # Calculate global range for consistent x-axis
        global_min = data['OC'].min()
        global_max = data['OC'].max()
        x_range = np.linspace(global_min - 3, global_max + 3, 500)  # Reduced from 1500
        
        # VIRIDIS colormap for SOC gradient (0-150 g/kg)
        cmap = plt.cm.viridis
        
        # Normalize SOC values from 0-150 to [0, 1] for colormap
        soc_norm = np.clip((x_range - 0) / (150 - 0), 0, 1)
        
        # INCREASED vertical spacing to prevent text overlap
        spacing = 0.75
        
        # 3D PERSPECTIVE PARAMETERS
        perspective_strength = 0.15
        
        # Plot from bottom to top (oldest to newest)
        for i, year in enumerate(years):
            year_data = data[data['year'] == year]['OC'].values
            
            # Calculate KDE with smoother bandwidth
            kde = gaussian_kde(year_data, bw_method=0.3)
            density = kde(x_range)
            
            # Normalize and scale density
            density = density / density.max() * 0.9 * spacing
            
            # PERSPECTIVE TRANSFORMATION
            depth_factor = 1.0 - (i / len(years)) * perspective_strength
            
            # Apply perspective: compress x-range from center
            x_center = (global_min + global_max) / 2
            x_range_perspective = x_center + (x_range - x_center) * depth_factor
            
            # Also scale density height slightly with perspective
            density_perspective = density * (0.85 + 0.15 * depth_factor)
            
            # Vertical offset for this year
            y_offset = i * spacing
            
            # Create 3D shadow effect
            shadow_offset = -0.02
            ax.fill_between(x_range_perspective, y_offset + shadow_offset, 
                           y_offset + density_perspective + shadow_offset, 
                           color='black', alpha=0.15, zorder=len(years)-i-0.5)
            
            # FAST GRADIENT METHOD: Use PolyCollection with facecolors
            # Create vertices for the filled area
            verts = []
            colors = []
            
            for j in range(len(x_range_perspective) - 1):
                # Create a quad (4 vertices) for each segment
                x0, x1 = x_range_perspective[j], x_range_perspective[j+1]
                y0, y1 = y_offset, y_offset + density_perspective[j]
                y2 = y_offset + density_perspective[j+1]
                
                # Quad vertices: bottom-left, bottom-right, top-right, top-left
                verts.append([(x0, y0), (x1, y0), (x1, y2), (x0, y1)])
                
                # Color based on SOC value
                color = cmap(soc_norm[j])
                colors.append(color)
            
            # Create PolyCollection for fast rendering
            poly = PolyCollection(verts, facecolors=colors, edgecolors='none', 
                                 alpha=0.85, zorder=len(years)-i)
            ax.add_collection(poly)
            
            # Add white edge on top for definition
            ax.plot(x_range_perspective, y_offset + density_perspective, 
                   color='white', linewidth=2.5, alpha=0.9, zorder=len(years)-i+0.05)
            
            # Add subtle darker edge on bottom
            ax.plot(x_range_perspective, [y_offset]*len(x_range_perspective), 
                   color='black', linewidth=1.5, alpha=0.3, zorder=len(years)-i+0.1)
            
            # Calculate statistics
            year_mean = float(year_data.mean())
            year_median = float(np.median(year_data))
            n_samples = len(year_data)
            
            # Apply perspective to mean and median positions
            mean_perspective = x_center + (year_mean - x_center) * depth_factor
            median_perspective = x_center + (year_median - x_center) * depth_factor
            
            # MEAN marker (BLUE diamond)
            ax.scatter(mean_perspective, y_offset - 0.12, s=140, marker='D', 
                      color='#2E86DE', edgecolors='white', linewidths=2, 
                      zorder=len(years)+15, alpha=0.95)
            
            # MEDIAN marker (RED circle)
            ax.scatter(median_perspective, y_offset - 0.12, s=140, marker='o', 
                      color='#EE5A6F', edgecolors='white', linewidths=2, 
                      zorder=len(years)+15, alpha=0.95)
            
            # Year label
            left_edge_perspective = x_center + (global_min - 4 - x_center) * depth_factor
            
            # Get color for year label based on median SOC value
            median_norm = np.clip((year_median - 0) / (150 - 0), 0, 1)
            year_color = cmap(median_norm)
            
            ax.text(left_edge_perspective, y_offset + 0.45*spacing, f'{year}', 
                   fontsize=16, fontweight='bold', va='center', ha='right',
                   color='#2c3e50',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=year_color, linewidth=2, alpha=0.95))
            
            # Statistics on the right - M and n ON THE SAME LINE
            right_edge_perspective = x_center + (global_max + 4 - x_center) * depth_factor
            stats_text = f'M={year_median:.1f}  n={n_samples}'
            ax.text(right_edge_perspective, y_offset + 0.45*spacing, stats_text, 
                   fontsize=13, fontweight='bold', va='center', ha='left',
                   color='#2c3e50', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                            edgecolor=year_color, linewidth=1.2, alpha=0.9))
        
        # Styling
        ax.set_xlabel('SOC (g/kg)', fontweight='bold', fontsize=18, labelpad=10, color='#2c3e50')
        ax.set_ylabel('Year', fontweight='bold', fontsize=18, labelpad=10, color='#2c3e50')
        
        # Set limits
        ax.set_xlim(global_min - 10, global_max + 10)
        ax.set_ylim(-0.45, len(years) * spacing + 0.25)
        
        # Clean up y-axis
        ax.set_yticks([])
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle=':', axis='x', zorder=0, color='gray')
        ax.grid(False, axis='y')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#2E86DE', 
                      markersize=6, markeredgewidth=1.5, markeredgecolor='white', 
                      label='Mean', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EE5A6F', 
                      markersize=6, markeredgewidth=1.5, markeredgecolor='white', 
                      label='Median', linestyle='None')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper center', fontsize=13, 
                          framealpha=0.98, edgecolor='#2c3e50', fancybox=True, 
                          shadow=True, borderpad=0.6, ncol=2,
                          bbox_to_anchor=(0.5, -0.08))
        legend.get_frame().set_linewidth(1.5)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['bottom'].set_color('#2c3e50')
        
        # Tick marks
        ax.tick_params(axis='x', labelsize=14, width=2, length=6, color='#2c3e50')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'soc_ridge_plot.png'
        plt.savefig(output_path, dpi=350, bbox_inches='tight', facecolor='#f8f9fa')
        print(f"\n✓ Saved: {output_path}")
        plt.show()
        
        # Print interpretation
        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("="*70)
        print(f"• Each layer = SOC distribution for one year")
        print(f"• Color gradient: BLUE (0 g/kg) → GREEN → YELLOW (150 g/kg)")
        print(f"• Same color scale across all years for comparison")
        print(f"• Wider areas = higher density of samples at that SOC value")
        print(f"• Blue diamonds (◆) = mean, Red circles (●) = median at bottom")
        print(f"• 3D perspective: layers recede into distance")
        print(f"• M = median, n = sample count (shown for each year)")
        print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("SOC Ridge Plot Analysis")
    print("="*70)
    
    analyzer = SOCRidgePlot()
    args = analyzer.parse_arguments()
    
    analyzer.load_data(args)
    analyzer.plot_ridge()
    
    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)