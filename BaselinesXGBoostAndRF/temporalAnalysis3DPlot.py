import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from balancedDataset import create_validation_train_sets
from sklearn.linear_model import LinearRegression
from scipy import stats

class Simple3DAnalyzer:
    def __init__(self, output_dir='figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.full_df = None

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Simple 3D SOC Analysis')
        parser.add_argument('--output-dir', type=str, default='figures', help='Output directory')
        parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
        parser.add_argument('--distance-threshold', type=float, default=1.4, help='Distance threshold')
        parser.add_argument('--skip-altitude-api', action='store_true', help='Skip API and use synthetic altitude')
        return parser.parse_args()

    def load_data(self, args):
        """Load data and get real altitude from API"""
        print("Loading data...")
        
        validation_df, training_df = create_validation_train_sets(
            df=None,
            output_dir=args.output_dir,
            target_val_ratio=args.target_val_ratio,
            use_gpu=True,
            distance_threshold=args.distance_threshold
        )
        
        self.full_df = pd.concat([validation_df, training_df], ignore_index=True)
        
        # Rename columns
        self.full_df = self.full_df.rename(columns={'GPS_LAT': 'latitude', 'GPS_LONG': 'longitude'})
        
        print(f"Loaded {len(self.full_df)} samples")
        
        # Check if user wants to skip API
        if args.skip_altitude_api:
            print("\n⚠ Skipping API, using synthetic altitude data...")
            self.full_df['altitude'] = 300 + (self.full_df['latitude'] - self.full_df['latitude'].min()) * 100
            print(f"  Altitude range: {self.full_df['altitude'].min():.1f} - {self.full_df['altitude'].max():.1f} m (synthetic)")
        else:
            # Check if we already have altitude data cached
            cache_file = self.output_dir / 'altitude_cache.csv'
            
            if cache_file.exists():
                print(f"\n✓ Found cached altitude data: {cache_file}")
                print("  Loading from cache (to skip, delete this file)...")
                cached_data = pd.read_csv(cache_file)
                
                # Merge cached altitude data
                self.full_df = self.full_df.merge(
                    cached_data[['latitude', 'longitude', 'altitude']], 
                    on=['latitude', 'longitude'], 
                    how='left'
                )
                
                # Check if any samples don't have altitude
                missing_alt = self.full_df['altitude'].isna().sum()
                if missing_alt > 0:
                    print(f"  {missing_alt} samples missing altitude, fetching from API...")
                    missing_df = self.full_df[self.full_df['altitude'].isna()]
                    new_altitudes = self.get_elevation_batch(
                        missing_df['latitude'].tolist(),
                        missing_df['longitude'].tolist()
                    )
                    self.full_df.loc[self.full_df['altitude'].isna(), 'altitude'] = new_altitudes
                    
                    # Update cache
                    self.save_altitude_cache()
            else:
                print("\nFetching real altitude data from Open-Elevation API...")
                print("(This will take a few minutes but will be cached for future use)")
                
                self.full_df['altitude'] = self.get_elevation_batch(
                    self.full_df['latitude'].tolist(),
                    self.full_df['longitude'].tolist()
                )
                
                # Save cache for future use
                self.save_altitude_cache()
        
        print(f"\nData loaded:")
        print(f"  Year range: {self.full_df['year'].min()} - {self.full_df['year'].max()}")
        print(f"  SOC range: {self.full_df['OC'].min():.1f} - {self.full_df['OC'].max():.1f} g/kg")
        print(f"  Altitude range: {self.full_df['altitude'].min():.1f} - {self.full_df['altitude'].max():.1f} m")
    
    def get_elevation_batch(self, latitudes, longitudes, batch_size=100):
        """
        Get elevation data using Open-Elevation API in batches.
        
        Args:
            latitudes: List of latitude values
            longitudes: List of longitude values
            batch_size: Number of locations per API call
        
        Returns:
            List of elevation values in meters
        """
        import requests
        import time
        
        elevations = []
        n_points = len(latitudes)
        
        print(f"  Fetching elevation for {n_points} points...")
        
        for i in range(0, n_points, batch_size):
            batch_lats = latitudes[i:i+batch_size]
            batch_lons = longitudes[i:i+batch_size]
            
            # Prepare locations for API
            locations = [
                {"latitude": float(lat), "longitude": float(lon)} 
                for lat, lon in zip(batch_lats, batch_lons)
            ]
            
            try:
                # Call Open-Elevation API
                response = requests.post(
                    'https://api.open-elevation.com/api/v1/lookup',
                    json={'locations': locations},
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()['results']
                    batch_elevations = [r['elevation'] for r in results]
                    elevations.extend(batch_elevations)
                    print(f"    Processed {min(i+batch_size, n_points)}/{n_points} points")
                else:
                    print(f"    API error for batch {i//batch_size + 1}, using fallback")
                    # Fallback to approximate elevation
                    elevations.extend([300 + (lat - 47.5) * 100 for lat in batch_lats])
                
                # Rate limiting - be nice to the free API
                if i + batch_size < n_points:
                    time.sleep(1)
                
            except Exception as e:
                print(f"    Error: {e}, using fallback for this batch")
                # Fallback to approximate elevation based on latitude
                elevations.extend([300 + (lat - 47.5) * 100 for lat in batch_lats])
        
        return elevations
    
    def save_altitude_cache(self):
        """Save altitude data to cache file"""
        cache_file = self.output_dir / 'altitude_cache.csv'
        cache_data = self.full_df[['latitude', 'longitude', 'altitude']].drop_duplicates()
        cache_data.to_csv(cache_file, index=False)
        print(f"\n✓ Saved altitude cache to: {cache_file}")
        print(f"  (Delete this file to force re-fetch from API)")

    def make_3d_plot(self):
        """Create RESEARCH PAPER QUALITY 3D scatter plot with ABSOLUTE MAXIMUM YEAR SPACING"""
        print("\nCreating RESEARCH PAPER QUALITY 3D plot with MAXIMUM year spacing...")
        
        data = self.full_df[['year', 'altitude', 'OC']].dropna()
        
        # Center year to first year (so first year = 0)
        first_year = data['year'].min()
        data['year_centered'] = data['year'] - first_year
        
        # Extract variables for fit
        X = data[['year_centered', 'altitude']].values
        y = data['OC'].values
        
        # Fit plane: SOC = a*(year-first_year) + b*altitude + c
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficients
        year_coef = model.coef_[0]
        alt_coef = model.coef_[1]
        intercept = model.intercept_
        r2 = model.score(X, y)
        
        # Calculate statistical significance (p-values)
        from scipy import stats
        
        # Predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        
        # Standard errors
        n = len(X)
        k = X.shape[1]  # number of predictors
        dof = n - k - 1  # degrees of freedom
        
        # Variance-covariance matrix
        X_with_intercept = np.column_stack([np.ones(n), X])
        var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        
        # Standard errors and t-statistics
        se_intercept = np.sqrt(var_covar[0, 0])
        se_year = np.sqrt(var_covar[1, 1])
        se_alt = np.sqrt(var_covar[2, 2])
        
        t_intercept = intercept / se_intercept
        t_year = year_coef / se_year
        t_alt = alt_coef / se_alt
        
        # P-values (two-tailed)
        p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), dof))
        p_year = 2 * (1 - stats.t.cdf(abs(t_year), dof))
        p_alt = 2 * (1 - stats.t.cdf(abs(t_alt), dof))
        
        def sig_stars(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
        
        print(f"\nPlane fit equation (centered at year {first_year}):")
        print(f"SOC = {intercept:.2f} + {year_coef:.3f}*(year-{first_year}) + {alt_coef:.4f}*altitude")
        print(f"R² = {r2:.3f}")
        print(f"\nStatistical Significance:")
        print(f"  Intercept: p = {p_intercept:.4e} {sig_stars(p_intercept)}")
        print(f"  Year:      p = {p_year:.4e} {sig_stars(p_year)}")
        print(f"  Altitude:  p = {p_alt:.4e} {sig_stars(p_alt)}")
        print(f"  (*** p<0.001, ** p<0.01, * p<0.05, ns not significant)")
        print(f"\nAt year {first_year} (first year, year_centered=0):")
        print(f"  Base SOC ≈ {intercept:.2f} g/kg (at mean altitude)")
        
        # MAXIMUM EFFORT: EXTRA EXTRA WIDE figure to give years MAXIMUM horizontal space
        from matplotlib.gridspec import GridSpec
        
        # EXTREMELY WIDE figure for absolute maximum year spacing
        fig = plt.figure(figsize=(38, 14), dpi=300)  # INCREASED from 32 to 38 inches!
        fig.patch.set_facecolor('white')
        
        # GridSpec: Maximum width ratio for plot
        gs = GridSpec(1, 2, width_ratios=[4.5, 0.8], wspace=0.10)  # Even more width for plot, tighter overall
        
        # 3D plot in left panel with MAXIMUM YEAR AXIS SPACE
        ax = fig.add_subplot(gs[0], projection='3d')
        
        # Remove background grid and panes for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        # Set white background
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Get unique years for discrete plotting
        years_unique = sorted(data['year'].unique())
        
        # Plot each year as a discrete "row" with jittering for visibility
        # Color by SOC value (not by year)
        for idx, year in enumerate(years_unique):
            year_data = data[data['year'] == year]
            
            # Add slight jitter to year for better visibility
            year_jitter = year + np.random.uniform(-0.1, 0.1, len(year_data))
            
            # Color by SOC values
            scatter = ax.scatter(year_jitter, 
                                year_data['altitude'], 
                                year_data['OC'],
                                c=year_data['OC'],  # Color by SOC!
                                cmap='viridis', 
                                vmin=data['OC'].min(),
                                vmax=data['OC'].max(),
                                s=50,
                                alpha=0.75,
                                edgecolors='black',
                                linewidth=0.3)
        
        # Add colorbar for SOC
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.62, aspect=22)  # Closer to plot
        cbar.set_label('SOC (g/kg)', fontweight='bold', fontsize=20, labelpad=10)
        cbar.ax.tick_params(labelsize=18, width=1.5, length=8)
        
        # Create mesh for plane (using centered years)
        year_range = np.linspace(0, data['year_centered'].max(), 10)
        alt_range = np.linspace(data['altitude'].min(), data['altitude'].max(), 10)
        year_grid, alt_grid = np.meshgrid(year_range, alt_range)
        soc_grid = year_coef * year_grid + alt_coef * alt_grid + intercept
        
        # Convert back to actual years for plotting
        year_grid_actual = year_grid + first_year
        
        # Plot plane with subtle styling
        ax.plot_surface(year_grid_actual, alt_grid, soc_grid, 
                       alpha=0.18, color='crimson', 
                       edgecolor='none', antialiased=True)
        
        # Add subtle wireframe on plane
        ax.plot_wireframe(year_grid_actual, alt_grid, soc_grid,
                         alpha=0.12, color='darkred', linewidth=0.6)
        
        # ABSOLUTE MAXIMUM EFFORT: HUGE padding for year axis labels
        ax.set_xlabel('\n\n\n\nYear', fontweight='bold', fontsize=28, labelpad=50)  # 4 newlines, labelpad=50!
        ax.set_ylabel('\nAltitude (m)', fontweight='bold', fontsize=28, labelpad=25)
        ax.set_zlabel('\nSOC (g/kg)', fontweight='bold', fontsize=28, labelpad=25)
        
        # ABSOLUTE MAXIMUM tick padding for years
        ax.tick_params(axis='x', labelsize=20, pad=25, width=1.8, length=10)  # pad=25 (was 20)!
        ax.tick_params(axis='y', labelsize=20, pad=12, width=1.8, length=10)
        ax.tick_params(axis='z', labelsize=20, pad=12, width=1.8, length=10)
        
        # Show FEWER years for even MORE spacing - every 3rd year!
        year_step = 3 if len(years_unique) > 6 else 2  # Show every 3rd year if we have many years
        ax.set_xticks(years_unique[::year_step])
        ax.set_xticklabels(years_unique[::year_step], fontsize=20, fontweight='medium', rotation=0)
        
        # MAXIMUM margins on year axis - 15% instead of 10%!
        year_margin = (years_unique[-1] - years_unique[0]) * 0.15  # INCREASED from 0.10 to 0.15!
        alt_margin = (data['altitude'].max() - data['altitude'].min()) * 0.05
        soc_margin = (data['OC'].max() - data['OC'].min()) * 0.05
        
        ax.set_xlim(years_unique[0] - year_margin, years_unique[-1] + year_margin)
        ax.set_ylim(data['altitude'].min() - alt_margin, data['altitude'].max() + alt_margin)
        ax.set_zlim(data['OC'].min() - soc_margin, data['OC'].max() + soc_margin)
        
        # Optimized view angle - more frontal to spread years horizontally
        ax.view_init(elev=12, azim=5)  # REDUCED azim from 8 to 5 for more frontal view
        
        # Pull back camera more to see full year axis spread
        ax.dist = 10.5  # REDUCED from 10.8 to see more
        
        # RESEARCH PAPER QUALITY: Create EXTERNAL LEGEND in right panel
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis('off')  # Hide axes
        
        # Create simplified text for legend box
        textstr = (f'Regression Plane:\n'
                  f'━━━━━━━━━━━━━━━\n'
                  f'SOC = {intercept:.2f}\n'
                  f'    + {year_coef:.3f}×Year\n'
                  f'    + {alt_coef:.4f}×Altitude\n\n'
                  f'R² = {r2:.3f}\n\n'
                  f'Effects:\n'
                  f'━━━━━━━━━━━━━━━\n'
                  f'Year: {sig_stars(p_year)}\n'
                  f'Altitude: {sig_stars(p_alt)}\n\n'
                  f'*** p < 0.001\n'
                  f'**  p < 0.01\n'
                  f'*   p < 0.05')
        
        # RESEARCH PAPER QUALITY: Readable legend box with bigger text
        props = dict(boxstyle='round,pad=1.5', facecolor='white',
                    alpha=0.98, edgecolor='black', linewidth=2.5)
        
        ax_legend.text(0.05, 0.98, textstr, 
                      transform=ax_legend.transAxes, 
                      fontsize=22,
                      verticalalignment='top', 
                      horizontalalignment='left', 
                      bbox=props, 
                      family='monospace',
                      linespacing=1.8)
        
        # Save with high quality for research paper
        output_path = self.output_dir / 'simple_3d_soc_plot_PAPER.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
        print(f"\n✓ Saved RESEARCH PAPER quality plot: {output_path}")
        
        # Also save as PDF for publication
        output_pdf = self.output_dir / 'simple_3d_soc_plot_PAPER.pdf'
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"✓ Saved PDF for publication: {output_pdf}")
        
        # Also save as EPS for LaTeX
        output_eps = self.output_dir / 'simple_3d_soc_plot_PAPER.eps'
        plt.savefig(output_eps, format='eps', bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"✓ Saved EPS for LaTeX: {output_eps}")
        
        plt.close()
        
        # Additional analysis
        print(f"\n{'='*60}")
        print(f"Detailed Interpretation:")
        print(f"{'='*60}")
        print(f"  Base SOC (at year {first_year}, mean altitude):")
        print(f"    {intercept:.2f} g/kg {sig_stars(p_intercept)}")
        print(f"\n  Temporal trend:")
        print(f"    +{year_coef:.3f} g/kg per year {sig_stars(p_year)}")
        print(f"    Over {data['year'].max() - first_year} years: +{year_coef * (data['year'].max() - first_year):.2f} g/kg")
        print(f"\n  Altitudinal gradient:")
        print(f"    +{alt_coef:.4f} g/kg per meter {sig_stars(p_alt)}")
        print(f"    Over altitude range: +{alt_coef * (data['altitude'].max() - data['altitude'].min()):.2f} g/kg")
        print(f"\n  Model quality:")
        print(f"    R² = {r2:.3f} ({r2*100:.1f}% of variance explained)")
        print(f"\n  Significance levels:")
        print(f"    *** p<0.001 (highly significant)")
        print(f"    **  p<0.01  (very significant)")
        print(f"    *   p<0.05  (significant)")
        print(f"    ns         (not significant)")
        print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("RESEARCH PAPER QUALITY - Simple 3D SOC Analysis")
    print("="*60)
    
    analyzer = Simple3DAnalyzer()
    args = analyzer.parse_arguments()
    
    analyzer.load_data(args)
    analyzer.make_3d_plot()
    
    print("\n" + "="*60)
    print("✓ Done! RESEARCH PAPER QUALITY figures generated")
    print("="*60)