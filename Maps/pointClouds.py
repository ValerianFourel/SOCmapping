import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from scipy.interpolate import griddata
from matplotlib import cm

# Paths to data files
excel_path = "/home/vfourel/SOCProject/SOCmapping/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"
landcover_path = "/lustre/home/vfourel/SOCProject/SOCmapping/Maps/landcover_results/landcover_values_merged.npy"
landcover_coordinates_path = "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy"
geojson_url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json"

# Load and filter Excel data
df = pd.read_excel(excel_path)
df = df[df['OC'] <= 150]  # Filter points with OC <= 150 g/kg
lons = df['GPS_LONG'].values
lats = df['GPS_LAT'].values
oc_values = df['OC'].values

# Load Bavaria boundaries
bavaria = gpd.read_file(geojson_url)
bavaria = bavaria[bavaria['name'] == 'Bayern']
minx, miny, maxx, maxy = bavaria.total_bounds  # Get Bavaria's bounding box

# Load landcover data
landcover_values = np.load(landcover_path, allow_pickle=True)
landcover_coordinates = np.load(landcover_coordinates_path)

# Create grid based on Bavaria's bounds
grid_x = np.linspace(minx, maxx, 300)
grid_y = np.linspace(miny, maxy, 300)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

# Interpolate landcover data
valid_mask = np.array([v is not None for v in landcover_values])
valid_coords = landcover_coordinates[valid_mask]
valid_values = np.array([float(v) for v in landcover_values[valid_mask]])
landcover_grid = griddata(valid_coords, valid_values, (grid_x, grid_y), method='nearest')

# Create RGBA image for landcover
landcover_image = np.zeros((grid_y.shape[0], grid_x.shape[1], 4))
# Built-up areas (class 50) in black
built_up_mask = landcover_grid == 50
landcover_image[built_up_mask, 0] = 0  # R
landcover_image[built_up_mask, 1] = 0  # G
landcover_image[built_up_mask, 2] = 0  # B
landcover_image[built_up_mask, 3] = 1  # Alpha
# Water bodies (class 80) in deep blue (#000080)
water_mask = landcover_grid == 80
landcover_image[water_mask, 0] = 0    # R
landcover_image[water_mask, 1] = 0    # G
landcover_image[water_mask, 2] = 0.5  # B (128/255 ≈ 0.5)
landcover_image[water_mask, 3] = 1    # Alpha

# Get the full terrain colormap
terrain = cm.get_cmap('terrain', 256)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
# Plot landcover image
ax.imshow(landcover_image, extent=[minx, maxx, miny, maxy], origin='lower', zorder=0)
# Plot Bavaria boundaries
bavaria.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=1)
# Plot point cloud with terrain colormap and transparency
scatter = ax.scatter(lons, lats, c=oc_values, cmap=terrain, s=20, alpha=0.6, vmin=0, vmax=150, zorder=2)
# Add grid
ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
# Add colorbar
cbar = fig.colorbar(scatter, ax=ax, label='Soil Organic Carbon (g/kg)', shrink=0.8)
cbar.set_ticks(np.linspace(0, 150, 6))  # More ticks for clarity
cbar.outline.set_linewidth(0.5)
# Set labels and title
ax.set_title('Soil Organic Carbon Point Cloud in Bavaria', fontsize=16, pad=10)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Save the figure
plt.savefig('point_cloud_bavaria_pretty.png', bbox_inches='tight')
plt.close()

# Calculate approximate area of the region of interest
min_lon = lons.min()
max_lon = lons.max()
min_lat = lats.min()
max_lat = lats.max()
delta_lon = max_lon - min_lon
delta_lat = max_lat - min_lat
lat_avg = (min_lat + max_lat) / 2
lon_length = 111.32 * np.cos(np.radians(lat_avg))  # km per degree of longitude
lat_length = 111.32  # km per degree of latitude
area = (delta_lat * lat_length) * (delta_lon * lon_length)
print(f"Approximate area of the region of interest: {area:.2f} square kilometers")