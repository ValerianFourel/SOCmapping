import ee
import geemap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Initialize Earth Engine
ee.Initialize()

# Define the region of interest (ROI) over Bavaria - make it square
roi = ee.Geometry.Rectangle([9.5, 47.2, 13.9, 50.6])  # [west, south, east, north]

# Define a common projection (UTM Zone 32N for Bavaria)
projection = 'EPSG:32632'
scale = 1000  # 1km resolution

# Define output parameters
output_size = 800  # Square output in pixels
output_dir = 'Bavaria_EO_Maps_2023'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the collections with appropriate continuous color scales
collections = [
    {
        'name': 'Elevation',
        'id': 'USGS/SRTMGL1_003',
        'type': 'Image',
        'bands': ['elevation'],
        'scale_factor': 1,
        'units': 'meters',
        'min': 100,
        'max': 2000,
        'colormap': 'terrain',  # matplotlib colormap
        'custom_colors': None
    },
    {
        'name': 'Net_Primary_Production',
        'id': 'MODIS/061/MOD17A3HGF',
        'type': 'ImageCollection',
        'bands': ['Npp'],
        'scale_factor': 0.0001,
        'units': 'kgC/m²/year',
        'min': 0,
        'max': 1.5,
        'colormap': None,
        'custom_colors': ['#FFE4B5', '#F4A460', '#228B22', '#006400', '#004225']  # Sand to dark green
    },
    {
        'name': 'Soil_Evaporation',
        'id': 'CAS/IGSNRR/PML/V2_v018',
        'type': 'ImageCollection',
        'bands': ['Es'],
        'scale_factor': 1,
        'units': 'mm/year',
        'min': 0,
        'max': 400,
        'colormap': None,
        'custom_colors': ['#F5DEB3', '#87CEEB', '#4682B4', '#191970', '#000080']  # Wheat to navy blue
    },
    {
        'name': 'Leaf_Area_Index',
        'id': 'MODIS/061/MOD15A2H',
        'type': 'ImageCollection',
        'bands': ['Lai_500m'],
        'scale_factor': 0.1,
        'units': 'm²/m²',
        'min': 0,
        'max': 6,
        'colormap': None,
        'custom_colors': ['#F0E68C', '#ADFF2F', '#32CD32', '#228B22', '#006400']  # Khaki to dark green
    },
    {
        'name': 'Evapotranspiration',
        'id': 'MODIS/006/MOD16A2',
        'type': 'ImageCollection',
        'bands': ['ET'],
        'scale_factor': 0.1,
        'units': 'mm/year',
        'min': 0,
        'max': 1000,
        'colormap': None,
        'custom_colors': ['#FFFACD', '#B0E0E6', '#87CEEB', '#4169E1', '#00008B']  # Lemon to dark blue
    },
    {
        'name': 'Land_Surface_Temperature',
        'id': 'MODIS/061/MOD11A2',
        'type': 'ImageCollection',
        'bands': ['LST_Day_1km'],
        'scale_factor': 0.02,
        'offset': -273.15,
        'units': '°C',
        'min': 0,
        'max': 30,
        'colormap': None,
        'custom_colors': ['#4B0082', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8C00', '#FF0000']  # Cold to hot
    }
]

# Time range for 2023
start_date = '2023-01-01'
end_date = '2023-12-31'

def create_custom_colormap(colors, name='custom'):
    """Create a smooth continuous colormap from color list"""
    return LinearSegmentedColormap.from_list(name, colors, N=256)

def process_and_export_geotiff(image, collection_info, output_path):
    """Export Earth Engine image as GeoTIFF with consistent projection"""
    # Apply scale factor and offset
    scale_factor = collection_info.get('scale_factor', 1)
    offset = collection_info.get('offset', 0)

    if scale_factor != 1:
        image = image.multiply(scale_factor)
    if offset != 0:
        image = image.add(offset)

    # Export with consistent parameters
    geemap.ee_export_image(
        image,
        filename=output_path,
        scale=scale,
        region=roi,
        crs=projection,
        file_per_band=False
    )
    return image

def create_uniform_visualization(geotiff_path, collection_info, output_path):
    """Create uniform square visualization from GeoTIFF"""
    with rasterio.open(geotiff_path) as src:
        # Read the data
        data = src.read(1)

        # Handle nodata values
        nodata = src.nodata
        if nodata is not None:
            mask = data == nodata
            data = np.ma.masked_array(data, mask)

        # Create figure with exact square dimensions
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes([0.1, 0.15, 0.8, 0.75])  # [left, bottom, width, height]

        # Get colormap
        if collection_info['colormap']:
            cmap = plt.get_cmap(collection_info['colormap'])
        else:
            cmap = create_custom_colormap(collection_info['custom_colors'])

        # Plot the data
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=collection_info['min'],
            vmax=collection_info['max'],
            aspect='equal',
            interpolation='bilinear'
        )

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)

        # Add title
        title = f"{collection_info['name'].replace('_', ' ')} - Bavaria 2023"
        ax.set_title(title, fontsize=18, pad=20, fontweight='bold')

        # Add colorbar with consistent size and position
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # Horizontal colorbar
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f"{collection_info['units']}", fontsize=14)

        # Format colorbar ticks
        if collection_info['max'] - collection_info['min'] > 100:
            cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        # Save with high DPI for quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()

def get_image_statistics(image, roi, name):
    """Calculate and print image statistics"""
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.minMax(), '', True
        ).combine(
            ee.Reducer.stdDev(), '', True
        ),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    )

    stats_dict = stats.getInfo()
    print(f"\n{name} Statistics:")
    for key, value in stats_dict.items():
        if value is not None and not np.isnan(value):
            print(f"  {key}: {value:.3f}")

# Process each collection
for collection_info in collections:
    name = collection_info['name']
    print(f"\nProcessing {name}...")

    try:
        # Load the data
        if collection_info['type'] == 'Image':
            image = ee.Image(collection_info['id']).select(collection_info['bands'])
        else:
            collection = ee.ImageCollection(collection_info['id']).select(collection_info['bands'])
            collection = collection.filterDate(start_date, end_date)

            # Aggregate based on variable type
            if name in ['Evapotranspiration', 'Soil_Evaporation']:
                # Sum for water fluxes (convert 8-day to annual)
                count = collection.size()
                image = collection.sum()
                # Approximate annual sum (46 8-day periods)
                if name == 'Evapotranspiration':
                    image = image.multiply(365.0/8.0/count.getInfo())
            else:
                # Mean for other variables
                image = collection.mean()

        # Clip to ROI
        image = image.clip(roi)

        # Get statistics before export
        stats_image = image.multiply(collection_info.get('scale_factor', 1))
        if 'offset' in collection_info:
            stats_image = stats_image.add(collection_info['offset'])
        get_image_statistics(stats_image, roi, name)

        # Export GeoTIFF
        geotiff_path = os.path.join(output_dir, f'{name}.tif')
        process_and_export_geotiff(image, collection_info, geotiff_path)
        print(f"✓ Exported GeoTIFF: {geotiff_path}")

        # Create visualization
        png_path = os.path.join(output_dir, f'{name}.png')
        create_uniform_visualization(geotiff_path, collection_info, png_path)
        print(f"✓ Created visualization: {png_path}")

        # Also create a composite thumbnail using Earth Engine
        vis_image = image.multiply(collection_info.get('scale_factor', 1))
        if 'offset' in collection_info:
            vis_image = vis_image.add(collection_info['offset'])

        # Get thumbnail URL for quick preview
        colors = collection_info.get('custom_colors', ['black', 'white'])
        thumb_params = {
            'dimensions': output_size,
            'region': roi,
            'format': 'png',
            'min': collection_info['min'],
            'max': collection_info['max']
        }

        if colors:
            thumb_params['palette'] = [c.replace('#', '') for c in colors]

        thumb_url = vis_image.getThumbURL(thumb_params)
        print(f"  Preview URL: {thumb_url}")

    except Exception as e:
        print(f"✗ Error processing {name}: {str(e)}")

# Create a composite overview figure
print("\nCreating composite overview...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, collection_info in enumerate(collections):
    ax = axes[idx]
    geotiff_path = os.path.join(output_dir, f"{collection_info['name']}.tif")

    if os.path.exists(geotiff_path):
        try:
            with rasterio.open(geotiff_path) as src:
                data = src.read(1)

                # Handle nodata
                if src.nodata is not None:
                    data = np.ma.masked_array(data, data == src.nodata)

                # Get colormap
                if collection_info['colormap']:
                    cmap = plt.get_cmap(collection_info['colormap'])
                else:
                    cmap = create_custom_colormap(collection_info['custom_colors'])

                # Plot
                im = ax.imshow(
                    data,
                    cmap=cmap,
                    vmin=collection_info['min'],
                    vmax=collection_info['max'],
                    aspect='equal'
                )

                ax.set_title(collection_info['name'].replace('_', ' '), fontsize=12)
                ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{collection_info['name']}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

plt.suptitle('Bavaria Environmental Variables - 2023', fontsize=16, fontweight='bold')
plt.tight_layout()
composite_path = os.path.join(output_dir, 'Bavaria_All_Variables_2023.png')
plt.savefig(composite_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Created composite overview: {composite_path}")

print(f"\n✓ All processing complete! Files saved in: {output_dir}")
