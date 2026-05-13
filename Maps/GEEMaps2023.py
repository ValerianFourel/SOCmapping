import ee
import geemap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib_scalebar.scalebar import ScaleBar
import rasterio
import requests
from shapely.geometry import shape
import json
from PIL import Image


ee.Authenticate()
# Initialize Earth Engine
ee.Initialize(project='sgtmodel')

# Get Bavaria's administrative boundary for border overlay only
germany = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM1_NAME', 'Bayern'))
bavaria_boundary = germany.geometry()

# Define ROI as rectangle (data will fill this entire area)
roi = ee.Geometry.Rectangle([8.9, 47.2, 13.9, 50.6])  # [west, south, east, north]

# Set projection and scale
# Using UTM Zone 32N which is appropriate for Bavaria
projection = 'EPSG:32632'  # UTM Zone 32N
scale = 1000  # 1km resolution in meters

# Output directory
output_dir = 'Bavaria_Bands_2023'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the bands with their colormaps
collections = [
    {'name': 'Elevation', 'id': 'USGS/SRTMGL1_003', 'type': 'Image', 'bands': ['elevation'], 'colormap': 'terrain'},
    {'name': 'Net_Primary_Production', 'id': 'MODIS/061/MOD17A3HGF', 'type': 'ImageCollection', 'bands': ['Npp'],
     'custom_colors': ['#FFE4B5', '#F4A460', '#228B22', '#006400', '#004225']},
    {'name': 'Soil_Evaporation', 'id': 'CAS/IGSNRR/PML/V2_v018', 'type': 'ImageCollection', 'bands': ['Es'],
     'custom_colors': ['#F5DEB3', '#87CEEB', '#4682B4', '#191970', '#000080']},
    {'name': 'Leaf_Area_Index', 'id': 'MODIS/061/MOD15A2H', 'type': 'ImageCollection', 'bands': ['Lai_500m'],
     'custom_colors': ['#F0E68C', '#ADFF2F', '#32CD32', '#228B22', '#006400']},
    {'name': 'Evapotranspiration', 'id': 'MODIS/061/MOD16A2', 'type': 'ImageCollection', 'bands': ['ET'],
     'custom_colors': ['#FFFACD', '#B0E0E6', '#87CEEB', '#4169E1', '#00008B']},
    {'name': 'Land_Surface_Temperature', 'id': 'MODIS/061/MOD11A2', 'type': 'ImageCollection', 'bands': ['LST_Day_1km'],
     'custom_colors': ['#4B0082', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8C00', '#FF0000']}
]

# Time range for 2023
start_date = '2023-01-01'
end_date = '2023-12-31'

# Function to extract boundary coordinates from Earth Engine geometry
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

# Get Bavaria boundary coordinates once
print("Fetching Bavaria boundary...")
bavaria_coords = get_boundary_coords(bavaria_boundary)
print(f"✓ Boundary loaded with {len(bavaria_coords)} polygon(s)")

# Function to get scale factor and units from the catalog
def get_band_info(collection_id, band_name):
    """Try to get band info from catalog, return defaults if not found"""
    return 1, 'unknown'

# Function to create a smooth colormap
def create_custom_colormap(colors):
    return LinearSegmentedColormap.from_list('custom', colors, N=256)

# Function to add beautiful north arrow
def add_north_arrow(ax, x=0.96, y=0.08, arrow_length=0.05, fontsize=24):
    """Add an elegant north arrow with shadow to the map - bottom right position"""
    # Shadow arrow (slightly offset)
    shadow = FancyArrowPatch(
        (x-0.002, y+0.002), (x-0.002, y + arrow_length+0.002),
        transform=ax.transAxes,
        arrowstyle='->,head_width=0.40,head_length=0.8',
        color='gray',
        linewidth=2.5,
        alpha=0.3,
        zorder=98
    )
    ax.add_patch(shadow)
    
    # Main arrow with white background
    bg_circle = plt.Circle((x, y + arrow_length/2), 0.030, transform=ax.transAxes,
                           color='white', alpha=0.9, zorder=99)
    ax.add_patch(bg_circle)
    
    # Main arrow - pointing up from bottom
    arrow = FancyArrowPatch(
        (x, y), (x, y + arrow_length),
        transform=ax.transAxes,
        arrowstyle='->,head_width=0.40,head_length=0.8',
        color='#2C3E50',
        linewidth=2.5,
        zorder=100
    )
    ax.add_patch(arrow)
    
    # N label with shadow
    ax.text(x-0.002, y + arrow_length + 0.012-0.002, 'N', transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
            color='gray', alpha=0.3, zorder=98)
    ax.text(x, y + arrow_length + 0.012, 'N', transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
            color='#2C3E50', zorder=100)

# Function to export GeoTIFF
def export_geotiff(image, scale_factor, output_path):
    """Export image as GeoTIFF with proper parameters"""
    image = image.multiply(scale_factor)
    
    # Use rectangular ROI so data extends beyond Bavaria
    geemap.ee_export_image(
        image, 
        filename=output_path, 
        scale=scale, 
        region=roi, 
        crs=projection, 
        file_per_band=False
    )

# Function to visualize with enhanced styling and professional cartographic elements
def visualize(geotiff_path, collection_info, boundary_coords, output_path):
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)
        
        # Get the bounds and actual data shape
        bounds = src.bounds
        height, width = data.shape
        
        # Calculate aspect ratio from actual data dimensions
        data_aspect = width / height
        
        # Create figure with aspect ratio matching the data - larger for better quality
        # Increased bottom margin to accommodate colorbar and label without clashing
        fig_width = 10
        fig_height = fig_width / data_aspect + 2.5  # Increased from 2 to 2.5 for more space
        
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
        
        # Calculate axes position to fit data properly with margins
        # Adjusted to provide more space at bottom for colorbar
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.08, 0.15, 0.84, map_height * 0.85])  # Increased bottom margin from 0.12 to 0.15

        # Use better colormap or custom colors
        cmap = plt.get_cmap(collection_info['colormap']) if 'colormap' in collection_info else create_custom_colormap(collection_info['custom_colors'])
        im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='bilinear', 
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

        # Transform Bavaria boundary from lat/lon to UTM
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            x_coords, y_coords = transformer.transform(lons, lats)
            # Double-line boundary with white outline for contrast
            ax.plot(x_coords, y_coords, color='white', linewidth=2.5, linestyle='-', zorder=9, alpha=0.9)
            ax.plot(x_coords, y_coords, color='#2C3E50', linewidth=1.2, linestyle='-', zorder=10, alpha=0.95)

        # No grid - clean canvas look
        ax.grid(False)
        
        # Set up the axes with proper labels
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        
        # Add styled axis labels for UTM coordinates - LARGER FONT
        ax.set_xlabel('Easting (m)', fontsize=23, fontweight='medium', color='#2C3E50', labelpad=10)
        ax.set_ylabel('Northing (m)', fontsize=23, fontweight='medium', color='#2C3E50', labelpad=10)
        
        # Configure tick labels with better formatting - LARGER FONT
        ax.tick_params(axis='both', which='major', labelsize=16, length=6, width=1.0, 
                       colors='#2C3E50', direction='out')
        
        # Format ticks for UTM (in meters) - show in thousands
        from matplotlib.ticker import FuncFormatter
        def format_coord(x, pos):
            return f'{x/1000:.0f}k'
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_coord))
        ax.yaxis.set_major_formatter(FuncFormatter(format_coord))
        
        # Styled axes borders - thicker frame for canvas effect
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('#2C3E50')
            spine.set_alpha(0.9)
        
        # NO TITLE - removed for clean individual maps

        # Scale bar with professional styling - LARGER FONT
        scalebar = ScaleBar(
            dx=1,  # 1 meter per map unit in UTM
            units='m',
            location='lower left',
            length_fraction=0.15,
            width_fraction=0.015,
            box_alpha=0.85,
            color='#2C3E50',
            box_color='white',
            font_properties={'size': 13, 'weight': 'medium'},
            sep=5,
            frameon=True,
            pad=0.5,
            border_pad=0.5
        )
        ax.add_artist(scalebar)
        
        # North arrow in BOTTOM RIGHT - LARGER FONT
        add_north_arrow(ax, x=0.96, y=0.08, arrow_length=0.05, fontsize=24)

        # Elegant colorbar with better positioning and increased spacing
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.020])  # Moved down slightly from 0.06 to 0.05
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(collection_info['units'], fontsize=26, fontweight='medium',
                      color='#2C3E50', labelpad=15)  # Increased labelpad from 10 to 15
        cbar.ax.tick_params(labelsize=16, colors='#2C3E50', length=5, width=1.0)
        cbar.outline.set_linewidth(1.2)
        cbar.outline.set_edgecolor('#2C3E50')
        cbar.outline.set_alpha(0.8)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.close()

# Function to create collage
def create_collage(image_paths, output_path):
    """Create a 3x2 collage of all maps"""
    print("\nCreating 3x2 collage...")
    
    # Load all images
    images = [Image.open(path) for path in image_paths]
    
    # Get dimensions (assuming all images are same size after bbox_inches='tight')
    widths, heights = zip(*(img.size for img in images))
    
    # Create 3 columns x 2 rows layout
    n_cols = 3
    n_rows = 2
    
    # Calculate collage dimensions with some padding
    max_width = max(widths)
    max_height = max(heights)
    
    padding = 20  # pixels between images
    
    collage_width = max_width * n_cols + padding * (n_cols + 1)
    collage_height = max_height * n_rows + padding * (n_rows + 1)
    
    # Create new image with light gray background
    collage = Image.new('RGB', (collage_width, collage_height), '#F8F9FA')
    
    # Paste images in 3x2 grid with padding
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        x = padding + col * (max_width + padding) + (max_width - img.width) // 2
        y = padding + row * (max_height + padding) + (max_height - img.height) // 2
        collage.paste(img, (x, y))
    
    # Save collage
    collage.save(output_path, dpi=(300, 300), quality=95)
    print(f"✓ Collage saved: {output_path}")

# Process each band
png_paths = []
for collection in collections:
    name = collection['name']
    print(f"\nProcessing {name}...")

    try:
        # Get scale factor and units
        scale_factor, units = get_band_info(collection['id'], collection['bands'][0])

        # Apply specific unit adjustments (manual overrides)
        if name == 'Soil_Evaporation':
            scale_factor = 0.1 / 8  # Convert mm/8d to mm/d
            units = 'mm/d'
        elif name == 'Leaf_Area_Index':
            scale_factor = 0.1  # Scale factor for LAI
            units = 'Area fraction'
        elif name == 'Elevation':
            scale_factor = 1
            units = 'meters'
        elif name == 'Land_Surface_Temperature':
            scale_factor = 0.02  # Scale factor for LST
            units = 'K'
        elif name == 'Net_Primary_Production':
            scale_factor = 0.0001  # Scale factor for NPP
            units = 'kg*C/m²'
        elif name == 'Evapotranspiration':
            scale_factor = 0.1  # Scale factor for ET
            units = 'kg/m²/8day'

        collection['scale_factor'] = scale_factor
        collection['units'] = units

        # Load the data
        if collection['type'] == 'Image':
            image = ee.Image(collection['id']).select(collection['bands'])
        else:
            collection_data = ee.ImageCollection(collection['id']).select(collection['bands']).filterDate(start_date, end_date)
            if name == 'Net_Primary_Production':
                image = collection_data.first()  # Annual product
            else:
                image = collection_data.mean()  # Average over year for rates

        # Clip to rectangular ROI (not Bavaria boundary)
        image = image.clip(roi)

        # Export as GeoTIFF
        geotiff_path = os.path.join(output_dir, f'{name}.tif')
        export_geotiff(image, scale_factor, geotiff_path)
        print(f"✓ GeoTIFF saved: {geotiff_path}")

        # Create the visualization with boundary
        png_path = os.path.join(output_dir, f'{name}.png')
        visualize(geotiff_path, collection, bavaria_coords, png_path)
        print(f"✓ Visualization saved: {png_path}")
        png_paths.append(png_path)

    except Exception as e:
        print(f"✗ Error with {name}: {e}")
        import traceback
        traceback.print_exc()

# Create collage of all maps
if len(png_paths) == 6:
    collage_path = os.path.join(output_dir, 'Bavaria_All_Variables_Collage.png')
    create_collage(png_paths, collage_path)

print(f"\nDone! Check your files in {output_dir}")