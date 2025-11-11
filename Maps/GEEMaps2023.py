import ee
import geemap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

# Function to visualize with boundary overlay
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
        
        # Create figure with aspect ratio matching the data
        fig_width = 8
        fig_height = fig_width / data_aspect + 1.5  # Add space for colorbar
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Calculate axes position to fit data properly
        map_height = (fig_width / data_aspect) / fig_height
        ax = plt.axes([0.05, 0.15, 0.9, map_height * 0.95])

        cmap = plt.get_cmap(collection_info['colormap']) if 'colormap' in collection_info else create_custom_colormap(collection_info['custom_colors'])
        im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='bilinear', 
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

        # Plot Bavaria boundary - THINNER line
        for coords in boundary_coords:
            lons, lats = coords[:, 0], coords[:, 1]
            
            # Transform coordinates from lat/lon to the projection
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
            x_coords, y_coords = transformer.transform(lons, lats)
            
            ax.plot(x_coords, y_coords, color='black', linewidth=0.6, linestyle='-', zorder=10, alpha=0.8)

        # Remove all axes borders and frames
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{collection_info['name'].replace('_', ' ')} - Bavaria Region 2023", 
                     fontsize=14, pad=10)

        # Thin colorbar just below the map
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.012])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(collection_info['units'], fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    
    # Calculate collage dimensions
    max_width = max(widths)
    max_height = max(heights)
    
    collage_width = max_width * n_cols
    collage_height = max_height * n_rows
    
    # Create new image
    collage = Image.new('RGB', (collage_width, collage_height), 'white')
    
    # Paste images in 3x2 grid
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * max_width + (max_width - img.width) // 2  # Center horizontally
        y = row * max_height + (max_height - img.height) // 2  # Center vertically
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