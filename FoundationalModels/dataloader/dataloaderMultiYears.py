

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
import pandas as pd
from config import bands_list_order , bands_dict ,  time_before, LOADING_TIME_BEGINNING , window_size


import torch


import numpy as np

def consolidate_tensors(data_dict, band_to_index):
    # First, collect all tensors and their corresponding indices
    tensors = []
    indices = []

    # Iterate through the dictionary in a sorted manner to maintain consistency
    for band in sorted(data_dict.keys()):
        for year in sorted(data_dict[band].keys()):
            # Get the tensor
            tensor = data_dict[band][year]

            # Store the tensor
            tensors.append(tensor)

            # Store the corresponding indices
            indices.append([band_to_index[band], year])

    # Convert tensors list to a single 3D tensor (n x 98 x 98)
    tensors_combined = np.stack(tensors, axis=0)

    # Convert indices to a 2D tensor (n x 2)
    indices_tensor = np.array(indices)

    return tensors_combined, indices_tensor

def get_metadata(self, id_num):
    """Get metadata from filename"""
    if id_num not in self.id_to_file:
        raise ValueError(f"ID {id_num} not found")

    filename = self.id_to_file[id_num].name
    pattern = r'ID(\d+)N(\d+\.\d+)S(\d+\.\d+)W(\d+\.\d+)E(\d+\.\d+)'
    match = re.search(pattern, filename)

    if match:
        return {
            'id': int(match.group(1)),
            'north': float(match.group(2)),
            'south': float(match.group(3)),
            'west': float(match.group(4)),
            'east': float(match.group(5))
        }
    return None

def get_available_ids(self):
    """Return list of available IDs"""
    return list(self.id_to_file.keys())

class RasterTensorDataset(Dataset):
    def __init__(self, base_path):
        self.folder_path = base_path
        self.bands_list_order = ['Elevation', 'LAI', 'LST', 'MODIS_NPP', 
                                'SoilEvaporation', 'TotalEvapotranspiration']
        self.type = self._extract_type_from_path()
        if self.type is None:
            raise ValueError(f"Path must contain one of these types: {self.bands_list_order}")

        self.id_to_file = self._create_id_mapping()
        self.data_cache = {}
        for id_num, filepath in self.id_to_file.items():
            self.data_cache[id_num] = np.load(filepath)

    def _extract_type_from_path(self):
        for band_type in self.bands_list_order:
            if band_type in self.folder_path:
                return band_type
        return None

    def _create_id_mapping(self):
        id_to_file = {}
        for file_path in glob.glob(os.path.join(self.folder_path, "*.npy")):
            match = re.search(r'ID(\d+)N', file_path)
            if match:
                id_num = int(match.group(1))
                id_to_file[id_num] = file_path
        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=window_size):
        """
        Get a processed window around the specified x,y coordinates

        Parameters:
        id_num: int, ID number from filename
        x: int, x coordinate
        y: int, y coordinate
        window_size: int, size of the square window (default 17)

        Returns:
        torch.Tensor: processed window_size x window_size tensor
        """
        if id_num not in self.id_to_file:
            raise ValueError(f"ID {id_num} not found in dataset")

        # Get the data array
        if id_num in self.data_cache:
            data = self.data_cache[id_num]
        else:
            data = np.load(self.id_to_file[id_num])

        # Adjust window size based on type
        if self.type == 'LST':
            adjusted_window_size = window_size // 4
        elif self.type != 'Elevation':  # All other types except Elevation
            adjusted_window_size = window_size // 2
        else:  # Elevation
            adjusted_window_size = window_size

        # Calculate window boundaries with adjusted size
        half_window = adjusted_window_size // 2
        x_start = int(max(0, x - half_window))
        x_end = int(min(data.shape[0], x + half_window) )
        y_start = int(max(0, y - half_window))
        y_end = int(min(data.shape[1], y + half_window))

        # Extract window
        window = data[x_start:x_end, y_start:y_end]
        # Pad if necessary
        if window.shape != (adjusted_window_size, adjusted_window_size):
            padded_window = np.zeros((adjusted_window_size, adjusted_window_size))
            x_offset = half_window - (x - x_start)
            y_offset = half_window - (y - y_start)
            padded_window[
                x_offset:x_offset+window.shape[0],
                y_offset:y_offset+window.shape[1]
            ] = window
            window = padded_window

        # Stretch the window back to original size based on type
        if self.type == 'LST':
            window = np.repeat(np.repeat(window, 4, axis=0), 4, axis=1)
        elif self.type != 'Elevation':  # All other types except Elevation
            window = np.repeat(np.repeat(window, 2, axis=0), 2, axis=1)
        tensor , type_of_band =  torch.from_numpy(window).float() , self.type
        return tensor , type_of_band

    def __len__(self):
        return len(self.id_to_file)

    def __getitem__(self, idx):
        id_num = list(self.id_to_file.keys())[idx]
        return self.data_cache[id_num]

    # def get_last_subfolder_if_number(self,path):
    #     # Split the path and get the last non-empty part
    #     parts = [p for p in path.split('/') if p]
    #     if not parts:
    #         return None

    #     # Get the last part
    #     last_part = parts[-1]

    #     # Try to convert to number, return None if not possible
    #     try:
    #         return int(last_part)
    #     except ValueError:
    #         return None


# Example usage:
"""
# Initialize the dataset
base_path = "/content/drive/MyDrive/Colab Notebooks/MappingSOC/Data/RasterTensorData"
dataset = RasterTensorDataset(base_path, "Elevation")

# Get the dictionary mapping IDs to filenames
id_mapping = dataset.id_to_file
print("ID to filename mapping:", id_mapping)

# Get a 17x17 window for a specific location
id_num = 10  # example ID
x, y = 100, 100  # example coordinates
window = dataset.get_tensor_by_location(id_num, x, y)
print("Window shape:", window.shape)

# Create a DataLoader if needed
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
"""


class MultiRasterDatasetMultiYears(Dataset):
    def __init__(self, samples_coordinates_array_subfolders ,  data_array_subfolders , dataframe, time_before = time_before):
        """
        Parameters:
        subfolders: list of str, names of subfolders to include
        dataframe: pandas.DataFrame, contains columns GPS_LONG, GPS_LAT, and OC (target variable)
        """
        self.data_array_subfolders = data_array_subfolders
        self.seasonalityBased = self.check_seasonality(data_array_subfolders)
        self.time_before = time_before
        self.samples_coordinates_array_subfolders = samples_coordinates_array_subfolders
        self.dataframe = dataframe
        self.datasets = {
            self.get_last_three_folders(subfolder): RasterTensorDataset(subfolder)
            for subfolder in  self.data_array_subfolders
        }
        
        self.coordinates = {
            self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")#[np.isfinite(np.load(f"{subfolder}/coordinates.npy"))]
            for subfolder in self.samples_coordinates_array_subfolders
        }

    def check_seasonality(self,data_array_subfolders):
        seasons = ['winter', 'spring', 'summer', 'autumn']

        # Check if any subfolder contains a season name
        is_seasonal = any(
            any(season in subfolder.lower() for season in seasons)
            for subfolder in data_array_subfolders
        )

        return 1 if is_seasonal else 0


    def get_last_three_folders(self,path):
        # Split the path into components
        parts = path.rstrip('/').split('/')
        # Return last 3 components, or all if less than 3
        return '/'.join(parts[-2:])

    def find_coordinates_index(self, subfolder, longitude, latitude):
        """
        Finds the index of the matching coordinates in the subfolder's coordinates.npy file.

        Parameters:
        subfolder: str, name of the subfolder
        longitude: float, longitude to match
        latitude: float, latitude to match

        Returns:
        tuple: (id_num, x, y) if match is found, otherwise raises an error
        """

        coords = self.coordinates[subfolder]
        # Assuming the first two columns of `coordinates.npy` are longitude and latitude
        match = np.where((coords[:, 1] == longitude) & (coords[:, 0] == latitude))[0]
        if match.size == 0:
            raise ValueError(f"{coords} Coordinates ({longitude}, {latitude}) not found in {subfolder}")

        # Return id_num, x, y from the same row
        return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

    def __getitem__(self, index):
        """
        Retrieve tensor and target value for a given index.

        Parameters:
        index: int, index of the row in the dataframe

        Returns:
        tuple: (longitude, latitude, data_dict, oc), where data_dict contains organized tensors and metadata
        """
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]

        filtered_array = self.filter_by_season_or_year(row['season'], row['year'], self.seasonalityBased)
        # Initialize dictionary to store data for each band
        data_dict = {band: {} for band in bands_list_order}

        # Handle Elevation separately since it's constant
        elevation_processed = False

        for subfolder in filtered_array:
            subfolder = self.get_last_three_folders(subfolder)

            if subfolder.split(os.path.sep)[-1] == 'Elevation' and not elevation_processed:
                id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
                tensor, type_of_band = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

                if tensor is not None:
                    data_dict['Elevation'][0] = tensor
                    #data_dict['Elevation']['years'].append(0)
                    elevation_processed = True

            else:
                # Get the year from the last subfolder
                year = int(subfolder.split(os.path.sep)[-1])
                band = subfolder.split(os.path.sep)[-2]
                
                # Process each year in the time window
                for decrement in range(self.time_before):
                    current_year = year - decrement
                    decremented_subfolder = os.path.sep.join(
                        subfolder.split(os.path.sep)[:-1] + [str(current_year)]
                    )
                    id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                    tensor, type_of_band = self.datasets[decremented_subfolder].get_tensor_by_location(
                        id_num, x, y
                    )

                    if tensor is not None:
                        data_dict[band][current_year] = tensor
                        #data_dict[band]['years'].append(current_year)
                        
        band_to_index = {v: k for k, v in bands_dict.items()} 
        tensors_combined, indices_tensor = consolidate_tensors(data_dict, band_to_index)
        # stacked_tensor, metadata = encode_data_dict(data_dict)
        return longitude, latitude, tensors_combined, indices_tensor, oc
    # def __getitem__(self, index):
    #     """
    #     Retrieve tensor and target value for a given index with nested structure by bands and years.

    #     Parameters:
    #     index: int, index of the row in the dataframe

    #     Returns:
    #     tuple: (longitude, latitude, data_dict, oc), where data_dict contains organized tensors by bands and years
    #     """
    #     row = self.dataframe.iloc[index]
    #     print(row['year'])
    #     longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]

    #     filtered_array = self.filter_by_season_or_year(row['season'], row['year'], self.seasonalityBased)
    #     # Initialize nested dictionary structure: band -> year -> tensor
    #     data_dict = {band: {} for band in bands_list_order}

    #     # Handle Elevation separately since it's constant
    #     elevation_processed = False

    #     for subfolder in filtered_array:
    #         subfolder = self.get_last_three_folders(subfolder)

    #         if subfolder.split(os.path.sep)[-1] == 'Elevation' and not elevation_processed:
    #             id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
    #             tensor, _ = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

    #             if tensor is not None:
    #                 # For elevation, use year 0
    #                 data_dict['Elevation'][0] = tensor
    #                 elevation_processed = True

    #         else:
    #             # Get the year from the last subfolder
    #             year = int(subfolder.split(os.path.sep)[-1])
    #             band = subfolder.split(os.path.sep)[-2]

    #             # Process each year in the time window
    #             for decrement in range(self.time_before):
    #                 current_year = year - decrement
    #                 decremented_subfolder = os.path.sep.join(
    #                     subfolder.split(os.path.sep)[:-1] + [str(current_year)]
    #                 )
    #                 print(decremented_subfolder)
    #                 id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
    #                 tensor, _ = self.datasets[decremented_subfolder].get_tensor_by_location(
    #                     id_num, x, y
    #                 )
                    
    #                 if tensor is not None:
    #                     # Store only one tensor per band-year pair
    #                     if current_year not in data_dict[band]:
    #                         data_dict[band][current_year] = tensor

    #     return longitude, latitude, data_dict, oc


    def filter_by_season_or_year(self, season,year,Season_or_year):

        if Season_or_year:
            filtered_array = [
                path for path in self.samples_coordinates_array_subfolders
                if ('Elevation' in path) or
                ('MODIS_NPP' in path and path.endswith(str(year))) or
                (not 'Elevation' in path and not 'MODIS_NPP' in path and path.endswith(season))
            ]
            
        else:
            filtered_array = [
                            path for path in self.samples_coordinates_array_subfolders
                            if ('Elevation' in path) or
                            (not 'Elevation' in path and path.endswith(str(year)))
                        ]
        return filtered_array

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        """Get tensor from specific subfolder dataset"""
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

