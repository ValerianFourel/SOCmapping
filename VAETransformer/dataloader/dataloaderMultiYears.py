import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
import pandas as pd
from config import bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size

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
        """
        self.folder_path = base_path
        self.id_to_file = self._create_id_mapping()
        self.data_cache = {}
        for id_num, filepath in self.id_to_file.items():
            self.data_cache[id_num] = np.load(filepath)

    def _create_id_mapping(self):
        """Create a dictionary mapping IDs to their corresponding file paths"""
        id_to_file = {}
        for file_path in glob.glob(os.path.join(self.folder_path, "*.npy")):
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
        if id_num in self.data_cache:
            data = self.data_cache[id_num]
        else:
            data = np.load(self.id_to_file[id_num])
        half_window = window_size // 2
        x_start = int(max(0, x - half_window))
        x_end = int(min(data.shape[0], x + half_window + 1))
        y_start = int(max(0, y - half_window))
        y_end = int(min(data.shape[1], y + half_window + 1))
        window = data[x_start:x_end, y_start:y_end]
        if window.shape != (window_size, window_size):
            padded_window = np.zeros((window_size, window_size))
            x_offset = half_window - (x - x_start)
            y_offset = half_window - (y - y_start)
            padded_window[x_offset:x_offset+window.shape[0], y_offset:y_offset+window.shape[1]] = window
            window = padded_window
        return torch.from_numpy(window).float()

    def __len__(self):
        return len(self.id_to_file)

    def __getitem__(self, idx):
        id_num = list(self.id_to_file.keys())[idx]
        return self.data_cache[id_num]


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
        self.datasets = {self.get_last_three_folders(subfolder): RasterTensorDataset(subfolder)
                         for subfolder in self.data_array_subfolders}
        self.coordinates = {self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")
                            for subfolder in self.samples_coordinates_array_subfolders}
        # Compute normalization statistics
        self.compute_statistics()

    def check_seasonality(self, data_array_subfolders):
        seasons = ['winter', 'spring', 'summer', 'autumn']
        is_seasonal = any(any(season in subfolder.lower() for season in seasons) for subfolder in data_array_subfolders)
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

    def filter_by_season_or_year(self, season, year, season_or_year):
        if season_or_year:
            filtered_array = [path for path in self.samples_coordinates_array_subfolders
                              if ('Elevation' in path) or
                                 ('MODIS_NPP' in path and path.endswith(str(year))) or
                                 (not 'Elevation' in path and not 'MODIS_NPP' in path and path.endswith(season))]
        else:
            filtered_array = [path for path in self.samples_coordinates_array_subfolders
                              if ('Elevation' in path) or
                                 (not 'Elevation' in path and path.endswith(str(year)))]
        return filtered_array

    def compute_statistics(self):
        """Compute mean and std across all features for the full dataset"""
        features_list = []
        for i in range(len(self)):
            longitude, latitude, final_tensor, _, _ = self._get_unnormalized_item(i)
            features_list.append(final_tensor.numpy())
        
        features_array = np.stack(features_list)  # Shape: (samples, bands, window_size, window_size, time_before)
        self.feature_means = torch.tensor(np.mean(features_array, axis=(0, 2, 3)), dtype=torch.float32)  # Shape: (bands, time_before)
        self.feature_stds = torch.tensor(np.std(features_array, axis=(0, 2, 3)), dtype=torch.float32)    # Shape: (bands, time_before)
        self.feature_stds = torch.clamp(self.feature_stds, min=1e-8)  # Avoid division by zero

    def _get_unnormalized_item(self, index):
        """Helper method to get unnormalized data and encoding for statistics computation"""
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]
        filtered_array = self.filter_by_season_or_year(row['season'], row['year'], self.seasonalityBased)

        band_tensors = {band: [None] * self.time_before for band in bands_list_order}
        encoding = torch.full((len(bands_list_order), self.time_before), -1, dtype=torch.long)

        for subfolder in filtered_array:
            subfolder_key = self.get_last_three_folders(subfolder)
            if subfolder_key.split(os.path.sep)[-1] == 'Elevation':
                id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                elevation_tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                if elevation_tensor is not None:
                    for t in range(self.time_before):
                        band_tensors['Elevation'][t] = elevation_tensor
                        encoding[bands_list_order.index('Elevation'), t] = 0  # Elevation has no year
            else:
                year = int(subfolder_key.split(os.path.sep)[-1])
                for decrement in range(self.time_before):
                    current_year = year - decrement
                    decremented_subfolder = os.path.sep.join(subfolder_key.split(os.path.sep)[:-1] + [str(current_year)])
                    if decremented_subfolder in self.datasets:
                        id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                        tensor = self.datasets[decremented_subfolder].get_tensor_by_location(id_num, x, y)
                        if tensor is not None:
                            band = subfolder_key.split(os.path.sep)[-2]
                            band_idx = bands_list_order.index(band)
                            band_tensors[band][decrement] = tensor
                            encoding[band_idx, decrement] = current_year

        stacked_tensors = []
        for band in bands_list_order:
            for t in range(self.time_before):
                if band_tensors[band][t] is None:
                    band_tensors[band][t] = torch.zeros(window_size, window_size)
            stacked_tensor = torch.stack(band_tensors[band])
            stacked_tensors.append(stacked_tensor)

        final_tensor = torch.stack(stacked_tensors)
        final_tensor = final_tensor.permute(0, 2, 3, 1)

        return longitude, latitude, final_tensor, oc, encoding

    def __getitem__(self, index):
        """
        Retrieve normalized tensor, target value, and encoding for a given index.
        Returns:
        tuple: (longitude, latitude, tensor, OC, encoding)
        """
        longitude, latitude, final_tensor, oc, encoding = self._get_unnormalized_item(index)
        # Normalize the final tensor
        final_tensor = (final_tensor - self.feature_means[:, None, None]) / self.feature_stds[:, None, None]
        return longitude, latitude, final_tensor, oc, encoding

    def __len__(self):
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)
