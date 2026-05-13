import ee
import geemap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon
from matplotlib.patches import Patch
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

def add_north_arrow(ax, x=0.92, y=0.08, arrow_length=0.045, fontsize=16):
    """Add a modern compass rose to the map."""
    # Outer circle (white background)
    circle_radius = 0.032
    circle_outer = plt.Circle((x, y), circle_radius, transform=ax.transAxes,
                             facecolor='white', zorder=1001, edgecolor='#2c3e50', linewidth=1.8)
    ax.add_patch(circle_outer)
    
    # North arrow (red, pointing up)
    arrow_props_n = dict(
        arrowstyle='->,head_width=0.35,head_length=0.6',
        color='#e74c3c',
        linewidth=2.5,
        zorder=1002
    )
    arrow_n = FancyArrowPatch(
        (x, y), (x, y + arrow_length),
        transform=ax.transAxes,
        **arrow_props_n
    )
    ax.add_patch(arrow_n)
    
    # South arrow (gray, pointing down)
    arrow_props_s = dict(
        arrowstyle='->,head_width=0.25,head_length=0.4',
        color='#7f8c8d',
        linewidth=1.5,
        zorder=1002
    )
    arrow_s = FancyArrowPatch(
        (x, y), (x, y - arrow_length * 0.7),
        transform=ax.transAxes,
        **arrow_props_s
    )
    ax.add_patch(arrow_s)
    
    # East arrow (gray, pointing right)
    arrow_e = FancyArrowPatch(
        (x, y), (x + arrow_length * 0.7, y),
        transform=ax.transAxes,
        **arrow_props_s
    )
    ax.add_patch(arrow_e)
    
    # West arrow (gray, pointing left)
    arrow_w = FancyArrowPatch(
        (x, y), (x - arrow_length * 0.7, y),
        transform=ax.transAxes,
        **arrow_props_s
    )
    ax.add_patch(arrow_w)
    
    # Add cardinal direction labels
    label_distance = arrow_length + 0.012
    
    # N label (bold, larger)
    ax.text(x, y + label_distance, 'N', transform=ax.transAxes,
            ha='center', va='center', fontsize=fontsize + 2, fontweight='bold',
            color='#2c3e50', zorder=1003)
    
    # S label
    ax.text(x, y - label_distance * 0.85, 'S', transform=ax.transAxes,
            ha='center', va='center', fontsize=fontsize, fontweight='600',
            color='#2c3e50', zorder=1003)
    
    # E label
    ax.text(x + label_distance * 0.85, y, 'E', transform=ax.transAxes,
            ha='center', va='center', fontsize=fontsize, fontweight='600',
            color='#2c3e50', zorder=1003)
    
    # W label
    ax.text(x - label_distance * 0.85, y, 'W', transform=ax.transAxes,
            ha='center', va='center', fontsize=fontsize, fontweight='600',
            color='#2c3e50', zorder=1003)

def add_scale_bar(ax, bounds, projection, length_m=50000, location='lower left', fontsize=15):
    """
    Add a modern scale bar to the map for UTM projection.
    
    Parameters:
    -----------
    ax : matplotlib axes
    bounds : rasterio bounds object
    projection : str, coordinate reference system
    length_m : int, desired length of scale bar in meters
    location : str, location of scale bar
    fontsize : int, font size for labels
    """
    # Calculate position based on location - MOVED EVEN MORE LEFT
    if location == 'lower left':
        x_start, y_pos = 0.025, 0.05  # Moved from 0.04 to 0.025
    elif location == 'lower right':
        x_start, y_pos = 0.72, 0.05
    elif location == 'upper left':
        x_start, y_pos = 0.025, 0.88
    elif location == 'upper right':
        x_start, y_pos = 0.72, 0.88
    else:
        x_start, y_pos = 0.025, 0.05
    
    bar_width_axes = 0.14  # Bar width
    bar_height = 0.011  # Bar height
    
    # White background with rounded corners - LARGER background rectangle
    bg_rect = FancyBboxPatch(
        (x_start - 0.010, y_pos - 0.008), 
        bar_width_axes + 0.035, 0.058,  # Increased width and height
        transform=ax.transAxes,
        boxstyle="round,pad=0.005",
        facecolor='white', 
        edgecolor='#34495e', 
        linewidth=1.5,
        zorder=998,
        alpha=0.95
    )
    ax.add_patch(bg_rect)
    
    # Black and white alternating scale bar - positioned higher in rectangle
    n_segments = 4
    segment_width = bar_width_axes / n_segments
    bar_y_position = y_pos + 0.012  # MOVED UP inside rectangle
    
    for i in range(n_segments):
        color = '#2c3e50' if i % 2 == 0 else 'white'
        rect = Rectangle(
            (x_start + i * segment_width, bar_y_position), 
            segment_width, bar_height,
            transform=ax.transAxes,
            facecolor=color, 
            edgecolor='#34495e', 
            linewidth=0.8, 
            zorder=999
        )
        ax.add_patch(rect)
    
    # Add scale labels BELOW the bar, centered in rectangle
    label_y = bar_y_position - 0.003  # Just below the bar
    ax.text(x_start, label_y, '0', transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, fontweight='600',
            color='#2c3e50', zorder=1000)
    
    if length_m >= 1000:
        label_text = f'{length_m//1000} km'
    else:
        label_text = f'{length_m} m'
    
    # Position the right label slightly more to the left to stay inside box
    ax.text(x_start + bar_width_axes - 0.003, label_y, label_text,
            transform=ax.transAxes, ha='center', va='top', 
            fontsize=fontsize, fontweight='600', color='#2c3e50', zorder=1000)

def clean_axes(ax):
    """
    Clean up axes by removing ticks and labels.
    
    Parameters:
    -----------
    ax : matplotlib axes
    """
    # Remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def create_modern_colormap():
    """Create a modern, vibrant colormap for slope visualization."""
    colors = [
        # Flat terrain - soft greens and teals
        '#f0fdf4',  # very light green
        '#dcfce7',
        '#bbf7d0',
        '#86efac',
        '#4ade80',
        '#22c55e',
        '#16a34a',
        
        # Gentle slopes - yellows and lime
        '#84cc16',
        '#a3e635',
        '#d9f99d',
        '#fef08a',
        '#fde047',
        '#facc15',
        
        # Moderate slopes - oranges
        '#fb923c',
        '#f97316',
        '#ea580c',
        '#dc2626',
        
        # Steep slopes - reds
        '#dc2626',
        '#b91c1c',
        '#991b1b',
        '#7f1d1d',
        
        # Very steep - dark reds to purples
        '#881337',
        '#701a75',
        '#581c87',
        '#4c1d95',
        '#3b0764',
        
        # Extreme - deep purples to near black
        '#2e1065',
        '#1e1b4b',
        '#0f172a',
    ]
    return LinearSegmentedColormap.from_list('modern_slope', colors, N=512)

def create_slope_colormap():
    """Create a colormap for slope visualization from flat to steep terrain."""
    return create_modern_colormap()

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
    """Create modern slope visualization with boundary overlay."""
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        bounds = src.bounds
        height, width = data.shape
        data_aspect = width / height
        
        # Create figure with modern styling
        fig_width = 12
        fig_height = fig_width / data_aspect + 1.5
        
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.08, 0.14, 0.88, map_height * 0.92])
        
        # Plot slope data with enhanced visualization
        im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='bilinear',
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       vmin=0, vmax=50)
        
        # Plot Bavaria boundary with glow effect
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            # Outer glow
            ax.plot(x_coords, y_coords, color='white', linewidth=3.5, 
                   linestyle='-', zorder=9, alpha=0.6)
            # Main boundary
            ax.plot(x_coords, y_coords, color='#2c3e50', linewidth=1.8, 
                   linestyle='-', zorder=10, alpha=0.95)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        
        # Clean axes - no grid or labels
        clean_axes(ax)
        
        # Modern frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#34495e')
        
        # Simple title without box - INCREASED FONT SIZE
        ax.set_title('Slope Map of Bavaria', fontsize=24, pad=15, 
                    fontweight='700', color='#1a1a1a')
        
        # Add scale bar and north arrow
        add_scale_bar(ax, bounds, projection, length_m=50000, location='lower left')
        add_north_arrow(ax, x=0.92, y=0.08, arrow_length=0.045)
        
        # Modern colorbar - INCREASED FONT SIZES
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.018])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Slope Angle (degrees)', fontsize=16, fontweight='600', color='#2c3e50')
        cbar.ax.tick_params(labelsize=14, colors='#2c3e50', width=1)
        cbar.outline.set_edgecolor('#34495e')
        cbar.outline.set_linewidth(1.2)
        
        # Add subtle attribution - INCREASED FONT SIZE
        fig.text(0.99, 0.01, 'Data: SRTM | UTM Zone 32N', 
                ha='right', va='bottom', fontsize=10, color='#7f8c8d', style='italic')
        
        output_path = os.path.join(output_dir, f'bavaria_slope_continuous_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Continuous slope map saved: {output_path}")

def create_slope_contour_visualization(geotiff_path, boundary_coords, projection, output_dir, timestamp, cmap):
    """Create modern slope contour visualization with boundary overlay."""
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        bounds = src.bounds
        height, width = data.shape
        data_aspect = width / height
        
        # Create figure
        fig_width = 12
        fig_height = fig_width / data_aspect + 1.5
        
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.08, 0.14, 0.88, map_height * 0.92])
        
        # Create meshgrid for contours
        x = np.linspace(bounds.left, bounds.right, width)
        y = np.linspace(bounds.bottom, bounds.top, height)
        X, Y = np.meshgrid(x, y)
        
        # Plot filled contours
        contourf = ax.contourf(X, Y, data, levels=25, cmap=cmap, alpha=0.95, vmin=0, vmax=50)
        
        # Add contour lines at specific slope angles
        contour_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45]
        contours = ax.contour(X, Y, data, levels=contour_levels, 
                             colors='white', linewidths=0.6, alpha=0.8)
        # Add labels with better styling - INCREASED FONT SIZE
        ax.clabel(contours, contour_levels, inline=True, fontsize=12, fmt='%d°',
                 colors='white')
        
        # Plot Bavaria boundary with glow effect
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            # Outer glow
            ax.plot(x_coords, y_coords, color='white', linewidth=3.5, 
                   linestyle='-', zorder=99, alpha=0.6)
            # Main boundary
            ax.plot(x_coords, y_coords, color='#2c3e50', linewidth=1.8,
                   linestyle='-', zorder=100, alpha=0.95)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        
        # Clean axes - no grid or labels
        clean_axes(ax)
        
        # Modern frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#34495e')
        
        # Simple title without box - INCREASED FONT SIZE
        ax.set_title('Slope Contour Map of Bavaria', fontsize=24, pad=15,
                    fontweight='700', color='#1a1a1a')
        
        # Add scale bar and north arrow
        add_scale_bar(ax, bounds, projection, length_m=50000, location='lower left')
        add_north_arrow(ax, x=0.92, y=0.08, arrow_length=0.045)
        
        # Modern colorbar - INCREASED FONT SIZES
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.018])
        cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Slope Angle (degrees)', fontsize=16, fontweight='600', color='#2c3e50')
        cbar.ax.tick_params(labelsize=14, colors='#2c3e50', width=1)
        cbar.outline.set_edgecolor('#34495e')
        cbar.outline.set_linewidth(1.2)
        
        # Add subtle attribution - INCREASED FONT SIZE
        fig.text(0.99, 0.01, 'Data: SRTM | UTM Zone 32N', 
                ha='right', va='bottom', fontsize=10, color='#7f8c8d', style='italic')
        
        output_path = os.path.join(output_dir, f'bavaria_slope_contours_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Slope contour map saved: {output_path}")

def create_classified_slope_visualization(geotiff_path, boundary_coords, projection, output_dir, timestamp):
    """Create modern classified slope visualization with discrete categories."""
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        bounds = src.bounds
        height, width = data.shape
        data_aspect = width / height
        
        # Create figure
        fig_width = 12
        fig_height = fig_width / data_aspect + 1.5
        
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.08, 0.14, 0.88, map_height * 0.92])
        
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
        
        # Modern, vibrant discrete colors
        colors = [
            '#d1f4e0',  # light mint green
            '#4ade80',  # green
            '#facc15',  # yellow
            '#fb923c',  # orange
            '#ef4444',  # red
            '#991b1b',  # dark red
            '#4c1d95',  # purple
        ]
        n_classes = len(colors)
        cmap_discrete = LinearSegmentedColormap.from_list('slope_classes', colors, N=n_classes)
        
        # Classify the data
        classified_data = np.digitize(data, slope_classes[1:])
        
        # Plot classified slope
        im = ax.imshow(classified_data, cmap=cmap_discrete, aspect='equal', 
                       interpolation='nearest',
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       vmin=0, vmax=n_classes-1)
        
        # Plot Bavaria boundary with glow effect
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            # Outer glow
            ax.plot(x_coords, y_coords, color='white', linewidth=3.5,
                   linestyle='-', zorder=99, alpha=0.6)
            # Main boundary
            ax.plot(x_coords, y_coords, color='#2c3e50', linewidth=1.8,
                   linestyle='-', zorder=100, alpha=0.95)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        
        # Clean axes - no grid or labels
        clean_axes(ax)
        
        # Modern frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#34495e')
        
        # Simple title without box - INCREASED FONT SIZE
        ax.set_title('Classified Slope Map of Bavaria', fontsize=24, pad=15,
                    fontweight='700', color='#1a1a1a')
        
        # Add scale bar and north arrow
        add_scale_bar(ax, bounds, projection, length_m=50000, location='lower left')
        add_north_arrow(ax, x=0.92, y=0.08, arrow_length=0.045)
        
        # Create modern legend with better styling - INCREASED FONT SIZES
        legend_elements = [
            Patch(facecolor=colors[i], label=slope_labels[i], 
                  edgecolor='#34495e', linewidth=0.8) 
            for i in range(n_classes)
        ]
        
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                          fontsize=13, framealpha=0.95, title='Slope Classes',
                          title_fontsize=15, edgecolor='#34495e', 
                          fancybox=True, shadow=True)
        legend.get_title().set_fontweight('700')
        legend.get_title().set_color('#2c3e50')
        
        # Add subtle attribution - INCREASED FONT SIZE
        fig.text(0.99, 0.01, 'Data: SRTM | UTM Zone 32N', 
                ha='right', va='bottom', fontsize=10, color='#7f8c8d', style='italic')
        
        output_path = os.path.join(output_dir, f'bavaria_slope_classified_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Classified slope map saved: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create modern slope maps of Bavaria with boundary overlay')
    parser.add_argument('--output-dir', type=str, default='./bavaria_slope',
                        help='Output directory (default: ./bavaria_slope)')
    parser.add_argument('--scale', type=int, default=250,
                        help='Resolution in meters (default: 250)')
    
    args = parser.parse_args()
    
    create_slope_map(output_dir=args.output_dir, scale=args.scale)

if __name__ == "__main__":
    main()