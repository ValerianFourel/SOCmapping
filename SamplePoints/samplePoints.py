import os.path
import zipfile
import os # we import hte necessary packages

# Import necessary modules
import numpy as np
# here we want to plot the data we received
import matplotlib.pyplot as plt
import copy
import rasterio
import re
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import pandas as pd


import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os

# Load the dataframe
df_sample = pd.read_csv('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv')

path = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data'

MODIS_NPP_Path_yearly = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/MODIS_NPP'
Elevation_Path_static = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/StaticValue/Elevation'
LST_Path_yearly = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/LST'
LAI_Path_yearly = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/LAI'
SoilEvaporation_Path_yearly = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/SoilEvaporation'
TotalEvaporation_Path_yearly = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/TotalEvapotranspiration'
LAI_Path_Season = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/LAI'
LST_Path_Season = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/LST'
SoilEvaporation_Path_Season = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/SoilEvaporation'
TotalEvaporation_Path_Season = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/TotalEvapotranspiration'

data_paths = [
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/MODIS_NPP',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/StaticValue/Elevation',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/LST',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/LAI',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/SoilEvaporation',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/YearlyValue/TotalEvapotranspiration',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/LAI',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/LST',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/SoilEvaporation',
    '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/SeasonalValue/TotalEvapotranspiration'
]

def build_kdtree_from_npy_file(file_path):
    # Load data from .npy file
    data = np.load(file_path)

    # Extract longitude and latitude columns
    longitude = data[:, 1]
    latitude = data[:, 2]

    # Build KD-tree
    tree = cKDTree(np.column_stack((longitude, latitude)))

    return tree

def find_closest_indices(tree, query_point, k=4):
    # Query the tree for k nearest neighbors
    distances, indices = tree.query(query_point, k=k)

    return indices


def generate_filename_dict(folder_path):
    filename_dict = {}

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Extract the ID number from the filename
        match = re.search(r'ID(\d+)N', filename)
        if match:
            id_number = int(match.group(1))

            # Store the filename in the dictionary with the ID number as the key
            filename_dict[id_number] = file_path

    return filename_dict

file_dict = {}
crs_dict = {}

for path in data_paths:
  for root, dirs, files in os.walk(path):
        for dir in dirs:
            dir_path = os.path.join(root, path)
            for file in os.listdir(dir_path):
                if file.endswith('.npy') and dir in file:

                    subfolder = os.path.join(path,dir)
                                        # Get a list of all files in the subfolder
                    files = os.listdir(subfolder)

                    # Sort the list of files
                    files.sort()

                    # Get the first file
                    first_file = files[0]

                    # If you need the full path to the file
                    first_file_path = os.path.join(subfolder, first_file)
                    src_dataset = rasterio.open(first_file_path)
                    # Deep copy the source CRS
                    src_crs = copy.deepcopy(src_dataset.crs)
                    # Close the source dataset
                    src_dataset.close()
                        # Iterate over the files in the folder
                    transform_dict = {}
                    for filename in os.listdir(subfolder):
                        file_path = os.path.join(subfolder, filename)

                        # Extract the ID number from the filename
                        match = re.search(r'ID(\d+)N', filename)
                        if match:
                            id_number = int(match.group(1))

                            # Read the rasterio src.transform
                            with rasterio.open(file_path) as src:
                                src_transform_copy = copy.deepcopy(src.transform)

                            # Store the transform in the dictionary with the ID number as the key
                            transform_dict[id_number] = src_transform_copy
                    file_dict[subfolder] = [ build_kdtree_from_npy_file(os.path.join(dir_path,file)), os.path.join(path,file),generate_filename_dict(subfolder),src_crs,transform_dict]



#print(file_dict)
# print(crs_dict)

subfolder = Elevation_Path_static
# Get a list of all files in the subfolder
files = os.listdir(subfolder)

                    # Sort the list of files
files.sort()

                    # Get the first file
first_file = files[0]

                    # If you need the full path to the file
first_file_path = os.path.join(subfolder, first_file)
src_dataset = rasterio.open(first_file_path)
                    # Deep copy the source CRS
src_crs = copy.deepcopy(src_dataset.crs)
                    # Close the source dataset
src_dataset.close()
                        # Iterate over the files in the folder
transform_dict = {}
for filename in os.listdir(subfolder):
  file_path = os.path.join(subfolder, filename)
                   # Extract the ID number from the filename
  match = re.search(r'ID(\d+)N', filename)
  if match:
    id_number = int(match.group(1))

                              # Read the rasterio src.transform
  with rasterio.open(file_path) as src:
    src_transform_copy = copy.deepcopy(src.transform)

                              # Store the transform in the dictionary with the ID number as the key
    transform_dict[id_number] = src_transform_copy
file_dict[subfolder] = [ build_kdtree_from_npy_file(os.path.join('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/StaticValue/','bounds_array_Elevation.npy')), os.path.join('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/RasterBandsData/StaticValue/','bounds_array_Elevation.npy'),generate_filename_dict(subfolder),src_crs,transform_dict]


def generate_filename_dict(folder_path):
    filename_dict = {}

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Extract the ID number from the filename
        match = re.search(r'ID(\d+)N', filename)
        if match:
            id_number = int(match.group(1))

            # Store the filename in the dictionary with the ID number as the key
            filename_dict[id_number] = file_path

    return filename_dict

def transform_coordinates(lon, lat, crs):
    # Coordinates in the target CRS
    x, y = transform({'init': 'EPSG:4326'}, crs, [lon], [lat])
    return x[0], y[0]


def get_tif_ID( latitude, longitude,kdTree,DataBoundsArray,DictFilenames,src_crs):
    x, y = transform_coordinates(longitude, latitude,src_crs)
    index_list = find_closest_indices(kdTree,[y,x])
    DataBoundsArray = np.load(DataBoundsArray)


    id_list = DataBoundsArray[index_list, 0]

    for id in id_list:
        filename = DictFilenames[id]
        # Replace underscores with periods
        filenameTransformed = filename.replace('_', '.')
        # Open the raster file using rasterio
        #     # Read the raster data (assuming only one band)
        #                     # Check if all data points have the same value
        #     if (raster_data == raster_data[0]).all():
        #       print('All the same value for ',filename)
        #     else:
              # Extract the four floats using regular expressions
        coordinates = re.findall(r'[NSWE]-?\d+\.\d+|[NSWE]0', filenameTransformed)
              # Convert the strings to floats
        coordinates = [float(coord[1:]) for coord in coordinates]
              # coordinates = [0.0 if coord.endswith('0') else float(coord[1:]) if coord != 'E0' else 0.0 for coord in coordinates]

              # Check if the sampled coordinates match the filename
        if coordinates[3] >= x and coordinates[2]  <= x and coordinates[0] >= y and coordinates[1] <= y:
          return x,y, id
        # else:
        #   return 0,0,0
           # print(coordinates,x,y,id)

    #print('f')
    #print(latitude, longitude)
    return 0,0,0

def get_tif_ArrayPosition( latitude, longitude,kdTree,DataBoundsArray,DictFilenames,src_crs,DictTransform):

  x_original,y_original, id = get_tif_ID( latitude, longitude,kdTree,DataBoundsArray,DictFilenames,src_crs)
  x, y = rasterio.transform.rowcol(DictTransform[id], x_original, y_original)
  if x < 0:
    print(x,y,id)
  if y < 0:
    print(y,x,id)
  return latitude, longitude , id, x , y

# Specify the path
path = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/Coordinates1Mil'

# Assume file_dict.keys() contains a list of paths
start_key = 'SeasonalValue/LAI/2003_autumn'
done = 0
start_processing = True

# for subfolder in file_dict.keys():
#   # If the start key is found, enable processing
#   # Extract the last 3 parts of the path
#   last_three_parts = '/'.join(subfolder.split(os.sep)[-3:])
#   print(subfolder)
#   #if last_three_parts == start_key:
#   #    start_processing = True
#   #    print(done)
#   #    continue  # Skip the start key itself
#   # Only process keys after the start key
#   if start_processing:
#     samplingOn = np.array(list(map(lambda coord: get_tif_ArrayPosition(float(coord[1]), float(coord[0]), file_dict[subfolder][0] , file_dict[subfolder][1] , file_dict[subfolder][2] , file_dict[subfolder][3] , file_dict[subfolder][4]), df_sample.to_numpy())))
#     # Get the last three subfolders in the path
#     subfolders = subfolder.split(os.sep)[-3:]

#     # Create subfolders within the path
#     new_path = os.path.join(path, *subfolders)

#     # Check if the new path exists, if not, create it
#     if not os.path.exists(new_path):
#       os.makedirs(new_path)
#     print(new_path)
#     # Specify the file name
#     filename = 'coordinates.npy'

#     # Specify the full path to the file
#     file_path = os.path.join(new_path, filename)

#     # Save the numpy matrix
#     np.save(file_path, samplingOn)
#     print('file_path  ',file_path)
#   done += 1


def process_subfolder(subfolder_data):
    subfolder, file_dict, df_sample, path = subfolder_data

    try:
        # Extract the last 3 parts of the path
        last_three_parts = '/'.join(subfolder.split(os.sep)[-3:])

        samplingOn = np.array([
            get_tif_ArrayPosition(
                float(coord[1]), 
                float(coord[0]), 
                file_dict[subfolder][0],
                file_dict[subfolder][1],
                file_dict[subfolder][2],
                file_dict[subfolder][3],
                file_dict[subfolder][4]
            ) 
            for coord in tqdm(
                df_sample.to_numpy(),
                desc=f"Processing coordinates for {subfolder.split(os.sep)[-1]}",
                unit="coord"
            )
        ])


        # Get the last three subfolders in the path
        subfolders = subfolder.split(os.sep)[-3:]

        # Create new path
        new_path = os.path.join(path, *subfolders)

        # Create directory if it doesn't exist
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        print(new_path)

        # Save the numpy matrix
        file_path = os.path.join(new_path, 'coordinates.npy')
        np.save(file_path, samplingOn)

        return file_path

    except Exception as e:
        print(f"Error processing {subfolder}: {str(e)}")
        return None

def process_all_subfolders(file_dict, df_sample, path, start_key=None):
    # Initialize processing flag
    start_processing = True if start_key is None else False

    # Filter subfolders based on start_key if provided
    subfolders_to_process = []
    for subfolder in file_dict.keys():
        last_three_parts = '/'.join(subfolder.split(os.sep)[-3:])

        if start_key and last_three_parts == start_key:
            start_processing = True
            continue

        if start_processing:
            subfolders_to_process.append(subfolder)

    # Prepare data for parallel processing
    subfolder_data = [(subfolder, file_dict, df_sample, path) 
                     for subfolder in subfolders_to_process]

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor for parallel processing
    processed_count = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Process subfolders with progress bar
        for result in tqdm(
            executor.map(process_subfolder, subfolder_data),
            total=len(subfolder_data),
            desc="Processing subfolders"
        ):
            if result is not None:
                processed_count += 1

    return processed_count
# Usage
done = process_all_subfolders(file_dict, df_sample, path)
print(f"Total processed: {done}")

import pandas as pd

# Specify the path to the Excel file
filename = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx'

# Read the 'GPS_LONG', 'GPS_LAT', and 'year' columns
df = pd.read_excel(filename, usecols=['GPS_LONG', 'GPS_LAT', 'year'])

# print(df)

pathLUCAS = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx'

LUCAScoordinates = "/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/OC_LUCAS_LFU_LfL_Coordinates"


# for subfolder in file_dict.keys():
#     # Get the last subfolder in the path
#     last_subfolder = subfolder.split(os.sep)[-1]
#     print(subfolder)
#     subfolders = subfolder.split(os.sep)[-3:]

#     # Try to convert the last subfolder to an integer (assuming the year is represented as an integer)
#     try:
#         year = int(last_subfolder)
#     except ValueError:
#         continue  # If it can't be converted to an integer, skip this iteration of the loop

#     # Filter the DataFrame based on the year
#     df_year = df[df['year'] == year]
#     # Remove rows where 'GPS_LONG' or 'GPS_LAT' are not finite numbers
#     df_year = df_year[np.isfinite(df_year['GPS_LONG']) & np.isfinite(df_year['GPS_LAT'])]
#     # If the DataFrame is empty, skip this iteration of the loop
#     if df_year.empty:
#         continue
#     samplingOn = np.array(list(map(lambda coord: get_tif_ArrayPosition(float(coord[1]), float(coord[0]), file_dict[subfolder][0] , file_dict[subfolder][1] , file_dict[subfolder][2] , file_dict[subfolder][3] , file_dict[subfolder][4]), df_year.to_numpy())))
#     # Get the last three subfolders in the path

#     # Create subfolders within the path
#     new_path = os.path.join(LUCAScoordinates, *subfolders)

#     #  Check if the new path exists, if not, create it
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#     print(new_path)
#     # Specify the file name
#     filename = 'coordinates.npy'

#     # Specify the full path to the file
#     file_path = os.path.join(new_path, filename)

#     # Save the numpy matrix
#     np.save(file_path, samplingOn)




def process_subfolder_OC_yearly(subfolder_data):
    subfolder, file_dict, df, LUCAScoordinates = subfolder_data

    try:
        # Get the last subfolder in the path
        last_subfolder = subfolder.split(os.sep)[-1]
        print(subfolder)
        subfolders = subfolder.split(os.sep)[-3:]

        # Try to convert the last subfolder to an integer
        year = int(last_subfolder)

        # Filter the DataFrame based on the year
        df_year = df[df['year'] == year]
        # Remove rows where 'GPS_LONG' or 'GPS_LAT' are not finite numbers
        df_year = df_year[np.isfinite(df_year['GPS_LONG']) & np.isfinite(df_year['GPS_LAT'])]

        if df_year.empty:
            return None

        samplingOn = np.array([
            get_tif_ArrayPosition(
                float(coord[1]), 
                float(coord[0]), 
                file_dict[subfolder][0],
                file_dict[subfolder][1],
                file_dict[subfolder][2],
                file_dict[subfolder][3],
                file_dict[subfolder][4]
            ) 
            for coord in df_year.to_numpy()
        ])

        # Create new path
        new_path = os.path.join(LUCAScoordinates, *subfolders)

        # Create directory if it doesn't exist
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        print(new_path)

        # Save the numpy matrix
        file_path = os.path.join(new_path, 'coordinates.npy')
        np.save(file_path, samplingOn)

        return new_path

    except ValueError:
        return None
    except Exception as e:
        print(f"Error processing {subfolder}: {str(e)}")
        return None

def process_all_subfolders_OC_yearly(file_dict, df, LUCAScoordinates):
    # Prepare data for parallel processing
    subfolder_data = [(subfolder, file_dict, df, LUCAScoordinates) 
                     for subfolder in file_dict.keys()]

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Process subfolders with progress bar
        results = list(tqdm(
            executor.map(process_subfolder_OC_yearly, subfolder_data),
            total=len(subfolder_data),
            desc="Processing subfolders"
        ))

    # Filter out None results and print successful paths
    successful_paths = [path for path in results if path is not None]
    print(f"Successfully processed {len(successful_paths)} out of {len(subfolder_data)} subfolders")

    return successful_paths

# Usage
results = process_all_subfolders_OC_yearly(file_dict, df, LUCAScoordinates)


######################### SEASON ##############################################################

pathSeason = "/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/LUCAS_LFU_Bavaria_OC_joint_data_modified.xlsx"

import pandas as pd

# Specify the path to the Excel file
filename = pathSeason
# Read the 'GPS_LONG', 'GPS_LAT', and 'survey_date' columns
df = pd.read_excel(filename, usecols=['GPS_LONG', 'GPS_LAT', 'survey_date'])

# Convert 'survey_date' to datetime format
df['survey_date'] = pd.to_datetime(df['survey_date'])

# Extract the year and month from 'survey_date'
df['year'] = df['survey_date'].dt.year
df['month'] = df['survey_date'].dt.month

# Define the months for each season
seasons = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11]
}

# Adjust the logic for assigning seasons:
def get_season_and_adjusted_year(month, year):
    if month == 12:  # December should be part of next year's winter
        return 'winter', year + 1
    elif month in [1, 2]:
        return 'winter', year
    elif month in [3, 4, 5]:
        return 'spring', year
    elif month in [6, 7, 8]:
        return 'summer', year
    elif month in [9, 10, 11]:
        return 'autumn', year

# Apply the function to get 'season' and adjusted 'year'
df['season_year'] = df.apply(lambda row: get_season_and_adjusted_year(row['month'], row['year']), axis=1)

# Separate the tuple into 'season' and 'year'
df['season'] = df['season_year'].apply(lambda x: x[0])
df['adjusted_year'] = df['season_year'].apply(lambda x: x[1])

# Create a 'year_season' column using the adjusted year and the season
df['year_season'] = df['adjusted_year'].astype(str) + '_' + df['season']

# Create a list of years (2000 to 2023)
years = list(range(2000, 2024))  # This will create a list from 2000 to 2023

# Filter the DataFrame based on the year and season
df_filtered = df[df['adjusted_year'].isin(years) & df['season'].isin(seasons.keys())]


def process_subfolder_OC_seasons(subfolder_data):
    subfolder, file_dict, df = subfolder_data
    # 
    # Get the last subfolder in the path
    last_subfolder = subfolder.split(os.sep)[-1]
    subfolders = subfolder.split(os.sep)[-3:]

    try:
        year_season = last_subfolder

        # Filter the DataFrame based on the year
        df_year_season = df[df['year_season'] == year_season]
        # Remove rows where 'GPS_LONG' or 'GPS_LAT' are not finite numbers
        df_year_season = df_year_season[np.isfinite(df_year_season['GPS_LONG']) & 
                                      np.isfinite(df_year_season['GPS_LAT'])]

        if df_year_season.empty:
            return None

        samplingOn = np.array([
            get_tif_ArrayPosition(
                float(coord[1]), 
                float(coord[0]), 
                file_dict[subfolder][0],
                file_dict[subfolder][1],
                file_dict[subfolder][2],
                file_dict[subfolder][3],
                file_dict[subfolder][4]
            ) 
            for coord in df_year_season.to_numpy()
        ])

        # Create new path and save results
        new_path = os.path.join('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/OC_LUCAS_LFU_LfL_Coordinates', *subfolders)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        file_path = os.path.join(new_path, 'coordinates.npy')
        np.save(file_path, samplingOn)

        return new_path

    except ValueError:
        return None

def process_all_subfolders_OC_seasons(file_dict, df):
    # Prepare data for parallel processing
    subfolder_data = [(subfolder, file_dict, df) for subfolder in file_dict.keys()]

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_subfolder_OC_seasons, subfolder_data))

    # Filter out None results and print successful paths
    successful_paths = [path for path in results if path is not None]
    for path in successful_paths:
        print(path)

# Usage
process_all_subfolders_OC_seasons(file_dict, df)

###################################################################

pathOC = '/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx'

df_OC = pd.read_excel(pathOC, usecols=['GPS_LONG', 'GPS_LAT', 'year'])
df_OC = df_OC[np.isfinite(df_OC['GPS_LONG']) & 
                                      np.isfinite(df_OC['GPS_LAT'])]

def process_subfolder_OC_Elevation():

    try:

        print(file_dict[Elevation_Path_static])
        print(file_dict[Elevation_Path_static][0])
        
        samplingOn = np.array([
            get_tif_ArrayPosition(
                float(coord[1]), 
                float(coord[0]), 
                file_dict[Elevation_Path_static][0],
                file_dict[Elevation_Path_static][1],
                file_dict[Elevation_Path_static][2],
                file_dict[Elevation_Path_static][3],
                file_dict[Elevation_Path_static][4]
            ) 
            for coord in df_OC.to_numpy()
        ])

        # Create new path and save results
        new_path = os.path.join('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/OC_LUCAS_LFU_LfL_Coordinates', 'StaticValue/Elevation')

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        file_path = os.path.join(new_path, 'coordinates.npy')
        np.save(file_path, samplingOn)

        return new_path

    except ValueError:
        return None

process_subfolder_OC_Elevation()

df_sample = pd.read_csv('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv')


def process_subfolder_Elevation():

    try:

        print(file_dict[Elevation_Path_static])
        print(file_dict[Elevation_Path_static][0])
        
        samplingOn = np.array([
            get_tif_ArrayPosition(
                float(coord[1]), 
                float(coord[0]), 
                file_dict[Elevation_Path_static][0],
                file_dict[Elevation_Path_static][1],
                file_dict[Elevation_Path_static][2],
                file_dict[Elevation_Path_static][3],
                file_dict[Elevation_Path_static][4]
            ) 
            for coord in df_sample.to_numpy()
        ])

        # Create new path and save results
        new_path = os.path.join('/home/vfourel/MasterThesis/Data_Dec2024/DataVersion1Mil/Data/Coordinates1Mil', 'StaticValue/Elevation')

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        file_path = os.path.join(new_path, 'coordinates.npy')
        np.save(file_path, samplingOn)

        return new_path

    except ValueError:
        return None

process_subfolder_Elevation()