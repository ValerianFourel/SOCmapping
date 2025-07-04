import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

def create_spatial_distribution_plot(save_path, iteration=0):
    """Create a spatial distribution plot with points colored by SOC using a deep green to dark brown colormap"""
    # Define base path and file path for the DataFrame
    base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"
    
    # Load the DataFrame
    try:
        df = pd.read_excel(file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return
    
    # Create output directory
    output_dir = os.path.join(save_path, 'socmappingSamplesMap')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load Bavaria boundary
    bavaria_file = 'bavaria.geojson'
    if not os.path.exists(bavaria_file):
        bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
        bavaria = bavaria[bavaria['name'] == 'Bayern']
        bavaria.to_file(bavaria_file)
    else:
        bavaria = gpd.read_file(bavaria_file)

    # Use provided column names
    lon_col = 'GPS_LONG'
    lat_col = 'GPS_LAT'
    soc_col = 'OC'

    # Verify required columns exist
    required_cols = [lon_col, lat_col, soc_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in DataFrame: {missing_cols}")
        return

    # Determine the SOC range, truncating at 150 g/kg
    global_min = 0
    global_max = min(df[soc_col].max(), 150)  # Use max OC value or 150, whichever is smaller
    if np.isnan(global_max):
        print("Error: OC column contains only NaN values")
        return

    # Define a custom deep green to dark brown colormap
    n_colors = 256
    colors = [
        (0.0, 0.4, 0.0),  # Deep green for 0 g/kg
        (0.2, 0.5, 0.1),  # Olive green
        (0.4, 0.4, 0.2),  # Greenish-brown
        (0.5, 0.3, 0.1),  # Mid-brown
        (0.2, 0.1, 0.0),  # Dark brown
        (0.15, 0.08, 0.0)  # Very dark brown (almost black) for max OC (≤ 150 g/kg)
    ]
    positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Normalized positions for color stops
    custom_earthy = LinearSegmentedColormap.from_list('custom_earthy', list(zip(positions, colors)), N=n_colors)

    # Create figure and axes
    fig = plt.figure(figsize=(14, 10), dpi=300, facecolor='white')
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # Adjusted for colorbar [left, bottom, width, height]

    # Plot Bavaria boundary
    bavaria.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.8)
    bavaria.plot(ax=ax, color='#f5f5f5', alpha=0.3, edgecolor='none')

    # Clip SOC values to the display range
    soc_values = np.clip(df[soc_col].values, global_min, global_max)

    # Plot all points
    if not df.empty:
        ax.scatter(
            df[lon_col], df[lat_col],
            c=soc_values,
            cmap=custom_earthy,
            s=35,  # Larger points for visibility
            alpha=0.5,  # Transparent to reduce overlap
            edgecolors='white',
            linewidth=0.5,
            label=f'SOC Samples (n={len(df)})',
            vmin=global_min,
            vmax=global_max
        )

    # Styling
    ax.set_title('Spatial Distribution of Soil Organic Carbon', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)

    # Add colorbar with ticks every 10 g/kg up to global_max
    cax = fig.add_axes([0.8, 0.1, 0.05, 0.8])  # Colorbar axes [left, bottom, width, height]
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap=custom_earthy),
        cax=cax
    )
    cbar.set_label('SOC (g/kg)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(np.linspace(global_min, global_max, int(global_max / 10) + 1))  # Ticks every 10 g/kg

    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # Save figure
    filename = f'spatial_distribution_SOC_{timestamp}_iter{iteration}.png'
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, filename)}")


# Execute the mapping function
if __name__ == "__main__":
    create_spatial_distribution_plot('samplesSOCPointCloudMap')