import ee

# Initialize the Earth Engine API
ee.Initialize()

# Load ESA WorldCover 2020 image
worldcover = ee.ImageCollection("ESA/WorldCover/v100") \
    .filterDate('2020-01-01', '2021-01-01') \
    .first() \
    .select('Map')

# Dictionary of class values and their descriptions
class_dict = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen'
}

def get_landcover_category(lat, lon):
    """Returns the land cover class value and description at the given lat/lon."""
    point = ee.Geometry.Point([lon, lat])
    
    # Sample the image at this point
    sample = worldcover.sample(region=point, scale=10, numPixels=1).first()
    
    # Get the result and print
    result = sample.get('Map').getInfo()
    
    if result is not None:
        description = class_dict.get(result, 'Unknown class')
        print(f'Class Value: {result}')
        print(f'Description: {description}')
        return result, description
    else:
        print('No data available at this location.')
        return None, None

# Example usage
get_landcover_category(48.8566, 2.3522)  # Example: Paris
