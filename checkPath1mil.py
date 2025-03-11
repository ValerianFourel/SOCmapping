import numpy as np
import pandas as pd
from pathlib import Path

# File paths
file_path_elevation_coords = "/lustre/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/StaticValue/Elevation/coordinates.npy"
file_path_raster_elevation_coords = "/lustre/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/RasterBandsData/StaticValue/Elevation/coordinates.npy"

# Load the NumPy files
print("Loading first NumPy file (Elevation)...")
elevation_data = np.load(file_path_elevation_coords)
print(f"Elevation coordinates loaded. Shape: {elevation_data.shape}")

print("Loading second NumPy file (RasterBandsData/Elevation)...")
raster_elevation_data = np.load(file_path_raster_elevation_coords)
print(f"RasterBandsData/Elevation coordinates loaded. Shape: {raster_elevation_data.shape}")

# Convert NumPy arrays to DataFrames for easier manipulation
# Assuming first column is longitude, second is latitude
elevation_df = pd.DataFrame(elevation_data, columns=['longitude', 'latitude'])
raster_elevation_df = pd.DataFrame(raster_elevation_data, columns=['longitude', 'latitude'])

# Round coordinates to a reasonable precision (e.g., 6 decimal places) to handle floating-point differences
precision = 6
elevation_df['longitude'] = elevation_df['longitude'].round(precision)
elevation_df['latitude'] = elevation_df['latitude'].round(precision)
raster_elevation_df['longitude'] = raster_elevation_df['longitude'].round(precision)
raster_elevation_df['latitude'] = raster_elevation_df['latitude'].round(precision)

# Merge the DataFrames to find matches
merged = pd.merge(
    elevation_df[['longitude', 'latitude']],
    raster_elevation_df[['longitude', 'latitude']],
    how='outer',
    on=['longitude', 'latitude']
)

# Count matches (entries present in both files)
match_count = len(merged.dropna())
print(f"Number of matches: {match_count}")

# Find non-matching entries from Elevation
non_matching_elevation = merged[merged.duplicated(['longitude', 'latitude'], keep='last') | merged.isnull().any(axis=1)]
non_matching_elevation_count = len(non_matching_elevation) - match_count  # Subtract matches to avoid double-counting
print(f"Number of non-matching entries in Elevation: {non_matching_elevation_count}")

# Find non-matching entries from RasterBandsData/Elevation
non_matching_raster = merged[merged.duplicated(['longitude', 'latitude'], keep='first') | merged.isnull().any(axis=1)]
non_matching_raster_count = len(non_matching_raster) - match_count  # Subtract matches to avoid double-counting
print(f"Number of non-matching entries in RasterBandsData/Elevation: {non_matching_raster_count}")

# Save non-matching entries to files for inspection (optional)
output_dir = Path.cwd() / "non_matching_outputs"
output_dir.mkdir(exist_ok=True)

non_matching_elevation.to_csv(output_dir / "non_matching_elevation.csv", index=False)
non_matching_raster.to_csv(output_dir / "non_matching_raster_elevation.csv", index=False)
print(f"Non-matching entries saved to {output_dir}")

# Summary
print("\nSummary:")
print(f"Total Elevation entries: {len(elevation_df)}")
print(f"Total RasterBandsData/Elevation entries: {len(raster_elevation_df)}")
print(f"Matches: {match_count}")
print(f"Non-matching Elevation entries: {non_matching_elevation_count}")
print(f"Non-matching RasterBandsData/Elevation entries: {non_matching_raster_count}")

# Small resume of differences
print("\nResume of Differences:")
print(f"1. Size Difference: Elevation has {len(elevation_df)} entries, while RasterBandsData/Elevation has {len(raster_elevation_df)}.")
print(f"2. Overlap: {match_count} coordinates are identical between the two files.")
print(f"3. Unique to Elevation: {non_matching_elevation_count} coordinates are present only in Elevation.")
print(f"4. Unique to RasterBandsData/Elevation: {non_matching_raster_count} coordinates are present only in RasterBandsData/Elevation.")
if elevation_data.shape[1] != raster_elevation_data.shape[1]:
    print(f"5. Structural Difference: Elevation has {elevation_data.shape[1]} columns, while RasterBandsData/Elevation has {raster_elevation_data.shape[1]}.")
else:
    print("5. Structural Similarity: Both files have the same number of columns.")
