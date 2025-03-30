

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
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, time_before=time_before):
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
            for subfolder in self.data_array_subfolders
        }
        
        self.coordinates = {
            self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")
            for subfolder in self.samples_coordinates_array_subfolders
        }

    def check_seasonality(self, data_array_subfolders):
        seasons = ['winter', 'spring', 'summer', 'autumn']
        is_seasonal = any(
            any(season in subfolder.lower() for season in seasons)
            for subfolder in data_array_subfolders
        )
        return 1 if is_seasonal else 0

    def get_last_three_folders(self, path):
        parts = path.rstrip('/').split('/')
        return '/'.join(parts[-2:])

    def find_coordinates_index(self, subfolder, longitude, latitude):
        coords = self.coordinates[subfolder]
        match = np.where((coords[:, 1] == longitude) & (coords[:, 0] == latitude))[0]
        if match.size == 0:
            raise ValueError(f"Coordinates ({longitude}, {latitude}) not found in {subfolder}")
        return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

    def __getitem__(self, index):
        """
        Retrieve tensor, encoding tensor, and target value for a given index.

        Returns:
        tuple: (longitude, latitude, final_tensor, encoding_tensor, oc)
        where final_tensor is the data, encoding_tensor tracks valid data, and oc is the target variable
        """
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]
        band_tensors = {band: [] for band in bands_list_order}
        encodings = {band: [] for band in bands_list_order}  # To store encoding for each timestep
        
        filtered_array = self.filter_by_season_or_year(row['season'], row['year'], self.seasonalityBased)

        for subfolder in filtered_array:
            subfolder_key = self.get_last_three_folders(subfolder)
            band = subfolder_key.split(os.path.sep)[-2]  # Extract band name

            if band == 'Elevation':
                id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                elevation_tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                if elevation_tensor is not None:
                    for _ in range(self.time_before):
                        band_tensors['Elevation'].append(elevation_tensor)
                        encodings['Elevation'].append(torch.tensor(1.0))  # 1.0 for valid data
            else:
                year = int(subfolder_key.split(os.path.sep)[-1])
                for decrement in range(self.time_before):
                    current_year = year - decrement
                    decremented_subfolder = os.path.sep.join(subfolder_key.split(os.path.sep)[:-1] + [str(current_year)])
                    
                    try:
                        id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                        tensor = self.datasets[decremented_subfolder].get_tensor_by_location(id_num, x, y)
                        encoding = torch.tensor(1.0) if tensor is not None else torch.tensor(0.0)
                        tensor = tensor if tensor is not None else torch.zeros(window_size, window_size)
                    except (ValueError, KeyError):
                        tensor = torch.zeros(window_size, window_size)
                        encoding = torch.tensor(0.0)
                    
                    band_tensors[band].append(tensor)
                    encodings[band].append(encoding)

        # Stack tensors and encodings for each band
        stacked_tensors = []
        stacked_encodings = []
        for band in bands_list_order:
            if band_tensors[band]:
                stacked_tensor = torch.stack(band_tensors[band])  # Shape: (time_steps, window_size, window_size)
                stacked_tensors.append(stacked_tensor)
                stacked_encodings.append(torch.stack(encodings[band]))  # Shape: (time_steps)

        final_tensor = torch.stack(stacked_tensors)  # Shape: (bands, time_steps, window_size, window_size)
        encoding_tensor = torch.stack(stacked_encodings)  # Shape: (bands, time_steps)
        final_tensor = final_tensor.permute(0, 2, 3, 1)  # Shape: (bands, window_size, window_size, time_steps)

        return longitude, latitude, final_tensor, encoding_tensor, oc

    def filter_by_season_or_year(self, season, year, Season_or_year):
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
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)


class NormalizedMultiRasterDatasetMultiYears(MultiRasterDatasetMultiYears):
    """Wrapper around MultiRasterDatasetMultiYears that adds feature normalization"""
    def __init__(self, samples_coordinates_array_path, data_array_path, df):
        super().__init__(samples_coordinates_array_path, data_array_path, df)
        self.compute_statistics()
        
    def compute_statistics(self):
        """Compute mean and std across all features for normalization"""
        features_list = []
        for i in range(len(self)):
            _, _, features, _, _ = super().__getitem__(i)
            features_list.append(features.numpy())
            
        features_array = np.stack(features_list)
        self.feature_means = torch.tensor(np.mean(features_array, axis=(0, 2, 3)), dtype=torch.float32)
        self.feature_stds = torch.tensor(np.std(features_array, axis=(0, 2, 3)), dtype=torch.float32)
        self.feature_stds = torch.clamp(self.feature_stds, min=1e-8)
        
    def __getitem__(self, idx):
        longitude, latitude, features, encoding_tensor, target = super().__getitem__(idx)
        features = (features - self.feature_means[:, None, None]) / self.feature_stds[:, None, None]
        return longitude, latitude, features, encoding_tensor, target
    
    def getStatistics(self):
        return self.feature_means, self.feature_stds