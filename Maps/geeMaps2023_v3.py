import ee
import geemap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import requests

# Initialize Earth Engine
ee.Initialize()

# Define your ROI (Bavaria)
roi = ee.Geometry.Rectangle([9.5, 47.2, 13.9, 50.6])  # [west, south, east, north]

# Set projection and scale
projection = 'EPSG:4326'  # WGS84 lat/lon
scale = 1000  # ~1km resolution in meters (approximate at this latitude)
output_dimensions = 800  # Square output in pixels (800x800)

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
    {'name': 'Evapotranspiration', 'id': 'MODIS/006/MOD16A2', 'type': 'ImageCollection', 'bands': ['ET'],
     'custom_colors': ['#FFFACD', '#B0E0E6', '#87CEEB', '#4169E1', '#00008B']},
    {'name': 'Land_Surface_Temperature', 'id': 'MODIS/061/MOD11A2', 'type': 'ImageCollection', 'bands': ['LST_Day_1km'],
     'custom_colors': ['#4B0082', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8C00', '#FF0000']}
]

# Time range for 2023
start_date = '2023-01-01'
end_date = '2023-12-31'

# Function to get scale factor and units from the catalog
def get_band_info(collection_id, band_name):
    catalog_id = collection_id.replace('/', '_')
    url = f"https://developers.google.com/earth-engine/datasets/catalog/{catalog_id}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        for band in data.get('bands', []):
            if band['id'] == band_name:
                return band.get('gee:scale', 1), band.get('units', 'unknown')
        print(f"Band {band_name} not found for {collection_id}")
        return 1, 'unknown'
    except Exception as e:
        print(f"Error fetching {collection_id}: {e}")
        return 1, 'unknown'

# Function to create a smooth colormap
def create_custom_colormap(colors):
    return LinearSegmentedColormap.from_list('custom', colors, N=256)

# Function to export GeoTIFF
def export_geotiff(image, scale_factor, output_path):
    image = image.multiply(scale_factor)
    geemap.ee_export_image(image, filename=output_path, scale=scale, region=roi, crs=projection, 
                           file_per_band=False, dimensions=output_dimensions)

# Function to visualize
def visualize(geotiff_path, collection_info, output_path):
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_array(data, data == nodata)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes([0.1, 0.15, 0.8, 0.75])

        cmap = plt.get_cmap(collection_info['colormap']) if 'colormap' in collection_info else create_custom_colormap(collection_info['custom_colors'])
        im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation='bilinear')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{collection_info['name'].replace('_', ' ')} - Bavaria 2023", fontsize=18, pad=20, fontweight='bold')

        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(collection_info['units'], fontsize=14)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

# Process each band
for collection in collections:
    name = collection['name']
    print(f"\nProcessing {name}...")

    try:
        # Get scale factor and units
        scale_factor, units = get_band_info(collection['id'], collection['bands'][0])

        # Apply specific unit adjustments
        if name == 'Soil_Evaporation':
            scale_factor /= 8  # Convert mm/8d to mm/d
            units = 'mm/d'
        elif name == 'Leaf_Area_Index':
            scale_factor = 1  # Keep raw pixel values (0-100)
            units = 'Area fraction'
        elif name == 'Elevation':
            units = 'meters'
        elif name == 'Land_Surface_Temperature':
            units = 'K'
        elif name == 'Net_Primary_Production':
            units = 'kg*C/m²'
        elif name == 'Evapotranspiration':
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

        image = image.clip(roi)

        # Export as GeoTIFF
        geotiff_path = os.path.join(output_dir, f'{name}.tif')
        export_geotiff(image, scale_factor, geotiff_path)
        print(f"✓ GeoTIFF saved: {geotiff_path}")

        # Create the visualization
        png_path = os.path.join(output_dir, f'{name}.png')
        visualize(geotiff_path, collection, png_path)
        print(f"✓ Visualization saved: {png_path}")

    except Exception as e:
        print(f"✗ Error with {name}: {e}")

print(f"\nDone! Check your files in {output_dir}")