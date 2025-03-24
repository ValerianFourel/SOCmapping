

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
import pandas as pd
from config import bands_list_order , time_before, LOADING_TIME_BEGINNING , window_size


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
        """
        Initialize the dataset

        Parameters:
        base_path: str, base path to RasterTensorData directory
        subfolder: str, name of the subfolder (e.g., 'Elevation')
        """
        
        self.folder_path = base_path

        # Create ID to filename mapping
        self.id_to_file = self._create_id_mapping()

        # Load all numpy arrays into memory (optional, can be modified to load on demand)
        self.data_cache = {}
        for id_num, filepath in self.id_to_file.items():
            self.data_cache[id_num] = np.load(filepath)

    def _create_id_mapping(self):
        """Create a dictionary mapping IDs to their corresponding file paths"""
        id_to_file = {}
        for file_path in glob.glob(os.path.join(self.folder_path, "*.npy")):
            # Extract ID number from filename
            match = re.search(r'ID(\d+)N', file_path)
            if match:
                id_num = int(match.group(1))
                id_to_file[id_num] = file_path

        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=window_size):
        """
        Get a window_size x window_size square around the specified x,y coordinates

        Parameters:
        id_num: int, ID number from filename
        x: int, x coordinate
        y: int, y coordinate
        window_size: int, size of the square window (default 17)

        Returns:
        torch.Tensor: window_size x window_size tensor
        """
        if id_num not in self.id_to_file:
            raise ValueError(f"ID {id_num} not found in dataset")

        # Get the data array
        if id_num in self.data_cache:
            data = self.data_cache[id_num]
        else:
            data = np.load(self.id_to_file[id_num])

        # Calculate window boundaries
        half_window = window_size // 2
        x_start = int(max(0, x - half_window))
        x_end = int(min(data.shape[0], x + half_window + 1))
        y_start = int(max(0, y - half_window))
        y_end = int(min(data.shape[1], y + half_window + 1))

        # Extract window
        window = data[x_start:x_end, y_start:y_end]

        # Pad if necessary
        if window.shape != (window_size, window_size):
            padded_window = np.zeros((window_size, window_size))
            x_offset = half_window - (x - x_start)
            y_offset = half_window - (y - y_start)
            padded_window[
                x_offset:x_offset+window.shape[0],
                y_offset:y_offset+window.shape[1]
            ] = window
            window = padded_window
        
        #window = np.asarray(window)
        return torch.from_numpy(window).float()

    def __len__(self):
        return len(self.id_to_file)

    def __getitem__(self, idx):
        # This is a placeholder implementation
        # Modify according to your specific needs
        id_num = list(self.id_to_file.keys())[idx]
        return self.data_cache[id_num]

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
        tuple: (tensor, OC), where tensor is the data and OC is the target variable
        """
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]
        tensors = []
        
        filtered_array = self.filter_by_season_or_year(row['season'],row['year'],self.seasonalityBased)
                # Initialize a dictionary to hold the tensors for each band
        band_tensors = {band: [] for band in bands_list_order}

        for subfolder in filtered_array:
            subfolder = self.get_last_three_folders(subfolder)

            # Check if the forelast subfolder is 'Elevation'
            if subfolder.split(os.path.sep)[-1] == 'Elevation':
                # Get the tensor for 'Elevation'
                id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
                elevation_tensor = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

                if elevation_tensor is not None:
                    # Repeat the 'Elevation' tensor self.time_before times
                    for _ in range(self.time_before):
                        band_tensors['Elevation'].append(elevation_tensor)
            else:
                # Get the year from the last subfolder
                year = int(subfolder.split(os.path.sep)[-1])

                # Decrement the year by self.time_before
                for decrement in range(self.time_before):
                    current_year = year - decrement

                    # Construct the subfolder with the decremented year
                    decremented_subfolder = os.path.sep.join(subfolder.split(os.path.sep)[:-1] + [str(current_year)])

                    id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                    tensor = self.datasets[decremented_subfolder].get_tensor_by_location(id_num, x, y)

                    if tensor is not None:
                        # Append the tensor to the corresponding band in the dictionary
                        band = subfolder.split(os.path.sep)[-2]
                        band_tensors[band].append(tensor)

        # Stack the tensors for each band
        stacked_tensors = []
        for band in bands_list_order:
            if band_tensors[band]:
                # Stack the tensors for the current band
                stacked_tensor = torch.stack(band_tensors[band])
                stacked_tensors.append(stacked_tensor)

        # Concatenate all stacked tensors along the band dimension
        final_tensor = torch.stack(stacked_tensors)
        final_tensor = final_tensor.permute(0, 2, 3, 1)

        return longitude, latitude, final_tensor, oc
       

    
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