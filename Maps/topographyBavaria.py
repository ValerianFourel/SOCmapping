import ee
import geemap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import geopandas as gpd
from datetime import datetime
from pyproj import Transformer
from shapely.geometry import shape

# Authenticate and Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='sgtmodel')

def load_bavaria_boundary():
    """Load Bavaria boundary from GeoJSON."""
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']
    return bavaria

def get_boundary_coords(ee_geometry):
    """Extract coordinates from EE geometry for matplotlib plotting"""
    geojson = ee_geometry.getInfo()
    geom = shape(geojson)
    
    # Handle MultiPolygon and Polygon
    coords_list = []
    if geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:
            coords_list.append(np.array(polygon.exterior.coords))
    elif geom.geom_type == 'Polygon':
        coords_list.append(np.array(geom.exterior.coords))
    
    return coords_list

def create_slope_colormap():
    """Create a colormap for slope visualization from flat to steep terrain."""
    colors = [
        # Flat terrain (0-5 degrees) - blues/greens
        '#f7fcf5',
        '#e5f5e0',
        '#c7e9c0',
        '#a1d99b',
        '#74c476',
        
        # Gentle slopes (5-15 degrees) - light greens to yellows
        '#41ab5d',
        '#238b45',
        '#006d2c',
        '#31a354',
        '#addd8e',
        
        # Moderate slopes (15-25 degrees) - yellows to oranges
        '#ffff99',
        '#ffed6f',
        '#ffdb45',
        '#ffc91b',
        '#ffb700',
        
        # Steep slopes (25-35 degrees) - oranges to reds
        '#ff9500',
        '#ff7300',
        '#ff5100',
        '#ff2f00',
        '#ff0d00',
        
        # Very steep slopes (35-45 degrees) - reds to dark reds
        '#eb0000',
        '#d70000',
        '#c30000',
        '#af0000',
        '#9b0000',
        
        # Extremely steep slopes (45+ degrees) - dark reds to purples
        '#870000',
        '#730000',
        '#5f0000',
        '#4b0000',
        '#370000',
        
        # Near-vertical (60+ degrees) - purples to black
        '#3d0037',
        '#2d0027',
        '#1d0017',
        '#0d0007',
        '#000000',
    ]
    return LinearSegmentedColormap.from_list('slope_cmap', colors, N=512)

def create_slope_map(output_dir='./bavaria_slope', scale=250):
    """Create slope map of Bavaria with boundary overlay."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🗺️  Loading Bavaria boundary...")
    bavaria_gdf = load_bavaria_boundary()
    
    # Get Bavaria boundary from Earth Engine
    germany = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM1_NAME', 'Bayern'))
    bavaria_boundary = germany.geometry()
    
    # Define ROI as rectangle around Bavaria
    roi = ee.Geometry.Rectangle([8.9, 47.2, 13.9, 50.6])  # [west, south, east, north]
    
    # Get boundary coordinates for overlay
    print("📍 Extracting Bavaria boundary coordinates...")
    bavaria_coords = get_boundary_coords(bavaria_boundary)
    print(f"✓ Boundary loaded with {len(bavaria_coords)} polygon(s)")
    
    # Set projection
    projection = 'EPSG:32632'  # UTM Zone 32N for Bavaria
    
    print("🛰️  Loading SRTM elevation data and calculating slope...")
    # Load SRTM elevation data
    elevation = ee.Image("USGS/SRTMGL1_003").select('elevation')
    
    # Calculate slope (in degrees)
    slope = ee.Terrain.slope(elevation)
    
    # Clip to ROI
    slope = slope.clip(roi)
    
    # Export as GeoTIFF
    geotiff_path = os.path.join(output_dir, f'bavaria_slope_{timestamp}.tif')
    print(f"💾 Exporting slope data to GeoTIFF...")
    
    geemap.ee_export_image(
        slope,
        filename=geotiff_path,
        scale=scale,
        region=roi,
        crs=projection,
        file_per_band=False
    )
    
    print(f"✓ GeoTIFF saved: {geotiff_path}")
    
    # Create custom colormap
    custom_cmap = create_slope_colormap()
    
    # Create visualizations
    print("🎨 Creating visualizations...")
    
    # 1. Full slope visualization
    create_slope_visualization(geotiff_path, bavaria_coords, projection, output_dir, timestamp, custom_cmap)
    
    # 2. Slope with contours
    create_slope_contour_visualization(geotiff_path, bavaria_coords, projection, output_dir, timestamp, custom_cmap)
    
    # 3. Classified slope categories
    create_classified_slope_visualization(geotiff_path, bavaria_coords, projection, output_dir, timestamp)
    
    print(f"\n✅ All maps created successfully!")
    print(f"📁 Results saved in: {output_dir}")

def create_slope_visualization(geotiff_path, boundary_coords, projection, output_dir, timestamp, cmap):
    """Create slope visualization with boundary overlay."""
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        bounds = src.bounds
        height, width = data.shape
        data_aspect = width / height
        
        # Create figure with proper aspect ratio
        fig_width = 10
        fig_height = fig_width / data_aspect + 1.2
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.05, 0.12, 0.9, map_height * 0.95])
        
        # Plot slope data with vmin/vmax to enhance visualization
        im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='bilinear',
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       vmin=0, vmax=60)  # Cap at 60 degrees for better visualization
        
        # Plot Bavaria boundary
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            ax.plot(x_coords, y_coords, color='white', linewidth=0.8, linestyle='-', 
                   zorder=10, alpha=0.9)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove all axes borders and frames
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_title('Slope Map - Bavaria', fontsize=14, pad=10, fontweight='bold')
        
        # Colorbar
        cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.012])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Slope (degrees)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        output_path = os.path.join(output_dir, f'bavaria_slope_continuous_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Continuous slope map saved: {output_path}")

def create_slope_contour_visualization(geotiff_path, boundary_coords, projection, output_dir, timestamp, cmap):
    """Create slope contour visualization with boundary overlay."""
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        bounds = src.bounds
        height, width = data.shape
        data_aspect = width / height
        
        # Create figure
        fig_width = 10
        fig_height = fig_width / data_aspect + 1.2
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.05, 0.12, 0.9, map_height * 0.95])
        
        # Create meshgrid for contours
        x = np.linspace(bounds.left, bounds.right, width)
        y = np.linspace(bounds.bottom, bounds.top, height)
        X, Y = np.meshgrid(x, y)
        
        # Plot filled contours
        contourf = ax.contourf(X, Y, data, levels=20, cmap=cmap, alpha=0.9, vmin=0, vmax=60)
        
        # Add contour lines at specific slope angles
        contour_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # Key slope thresholds
        contours = ax.contour(X, Y, data, levels=contour_levels, 
                             colors='white', linewidths=0.4, alpha=0.7)
        ax.clabel(contours, contour_levels, inline=True, fontsize=7, fmt='%d°')
        
        # Plot Bavaria boundary
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            ax.plot(x_coords, y_coords, color='white', linewidth=1.0, linestyle='-',
                   zorder=100, alpha=1.0)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove all axes borders and frames
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_title('Slope Contour Map - Bavaria', fontsize=14, pad=10, fontweight='bold')
        
        # Colorbar
        cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.012])
        cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Slope (degrees)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        output_path = os.path.join(output_dir, f'bavaria_slope_contours_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Slope contour map saved: {output_path}")

def create_classified_slope_visualization(geotiff_path, boundary_coords, projection, output_dir, timestamp):
    """Create classified slope visualization with discrete categories."""
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        bounds = src.bounds
        height, width = data.shape
        data_aspect = width / height
        
        # Create figure
        fig_width = 10
        fig_height = fig_width / data_aspect + 1.2
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.05, 0.12, 0.9, map_height * 0.95])
        
        # Define slope classes
        slope_classes = [0, 3, 8, 15, 25, 35, 45, 90]
        slope_labels = [
            'Flat (0-3°)',
            'Gentle (3-8°)',
            'Moderate (8-15°)',
            'Moderately Steep (15-25°)',
            'Steep (25-35°)',
            'Very Steep (35-45°)',
            'Extremely Steep (45°+)'
        ]
        
        # Create discrete colormap
        colors = ['#f7fcf5', '#74c476', '#ffff99', '#ff9500', '#ff0d00', '#9b0000', '#1d0017']
        n_classes = len(colors)
        cmap_discrete = LinearSegmentedColormap.from_list('slope_classes', colors, N=n_classes)
        
        # Classify the data
        classified_data = np.digitize(data, slope_classes[1:])
        
        # Plot classified slope
        im = ax.imshow(classified_data, cmap=cmap_discrete, aspect='equal', 
                       interpolation='nearest',
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       vmin=0, vmax=n_classes-1)
        
        # Plot Bavaria boundary
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            ax.plot(x_coords, y_coords, color='black', linewidth=0.8, linestyle='-',
                   zorder=100, alpha=1.0)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove all axes borders and frames
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_title('Classified Slope Map - Bavaria', fontsize=14, pad=10, fontweight='bold')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=slope_labels[i]) 
                          for i in range(n_classes)]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                 framealpha=0.9, title='Slope Classes')
        
        output_path = os.path.join(output_dir, f'bavaria_slope_classified_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Classified slope map saved: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create slope maps of Bavaria with boundary overlay')
    parser.add_argument('--output-dir', type=str, default='./bavaria_slope',
                        help='Output directory (default: ./bavaria_slope)')
    parser.add_argument('--scale', type=int, default=250,
                        help='Resolution in meters (default: 250)')
    
    args = parser.parse_args()
    
    create_slope_map(output_dir=args.output_dir, scale=args.scale)

if __name__ == "__main__":
    main()