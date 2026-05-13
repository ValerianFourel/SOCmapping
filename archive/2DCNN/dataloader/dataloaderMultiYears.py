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
                id_to_file[id_num] = Path(file_path)

        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=window_size):
        """
        Get a window_size x window_size square around the specified x,y coordinates

        Parameters:
        id_num: int, ID number from filename
        x: int, x coordinate
        y: int, y coordinate
        window_size: int, size of the square window (default from config)

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

        return torch.from_numpy(window).float()

    def __len__(self):
        return len(self.id_to_file)

    def __getitem__(self, idx):
        # Placeholder implementation
        id_num = list(self.id_to_file.keys())[idx]
        return self.data_cache[id_num]


class MultiRasterDatasetMultiYears(Dataset):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, time_before=time_before):
        """
        Parameters:
        samples_coordinates_array_subfolders: list of str, paths to subfolders containing coordinates
        data_array_subfolders: list of str, paths to subfolders containing data
        dataframe: pandas.DataFrame, contains columns GPS_LONG, GPS_LAT, and OC (target variable)
        time_before: int, number of previous time steps to consider for non-elevation bands
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

        # Ensure the total number of channels matches the expected 26
        self.expected_channels = 26
        self.non_elevation_bands = [band for band in bands_list_order if band != 'Elevation']
        self.channels_per_band = time_before  # Each non-elevation band contributes time_before channels
        if len(self.non_elevation_bands) * self.channels_per_band + 1 != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} channels, but got "
                f"{len(self.non_elevation_bands) * self.channels_per_band + 1} "
                f"(1 for Elevation + {len(self.non_elevation_bands)} bands * {self.channels_per_band} time steps)"
            )

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
        match = np.where((coords[:, 1] == longitude) & (coords[:, 0] == latitude))[0]
        if match.size == 0:
            raise ValueError(f"Coordinates ({longitude}, {latitude}) not found in {subfolder}")

        return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

    def __getitem__(self, index):
        """
        Retrieve tensor and target value for a given index.

        Parameters:
        index: int, index of the row in the dataframe

        Returns:
        tuple: (longitude, latitude, tensor, OC), where tensor is of shape (26, 5, 5)
        """
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]
        
        filtered_array = self.filter_by_season_or_year(row['season'], row['year'], self.seasonalityBased)
        
        # Initialize a list to hold tensors for all channels (26 in total)
        channel_tensors = []

        # Handle Elevation (1 channel, added once)
        elevation_subfolder = next(
            (subfolder for subfolder in filtered_array if 'Elevation' in subfolder), None
        )
        if elevation_subfolder:
            subfolder_key = self.get_last_three_folders(elevation_subfolder)
            id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
            elevation_tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
            if elevation_tensor is not None:
                channel_tensors.append(elevation_tensor)
            else:
                channel_tensors.append(torch.zeros(window_size, window_size))
        else:
            channel_tensors.append(torch.zeros(window_size, window_size))

        # Handle non-elevation bands (each band contributes time_before channels)
        for band in self.non_elevation_bands:
            # Get subfolders for the current band and specified year/season
            band_subfolders = [
                subfolder for subfolder in filtered_array
                if band in subfolder and not 'Elevation' in subfolder
            ]
            if not band_subfolders:
                # If no data for the band, add zero tensors for all time steps
                for _ in range(self.time_before):
                    channel_tensors.append(torch.zeros(window_size, window_size))
                continue

            # Get the base year from the first matching subfolder
            base_subfolder = band_subfolders[0]
            base_subfolder_key = self.get_last_three_folders(base_subfolder)
            try:
                base_year = int(base_subfolder_key.split(os.path.sep)[-1])
            except ValueError:
                base_year = row['year']  # Fallback to row year if parsing fails

            # Collect tensors for the current band across time_before years
            for decrement in range(self.time_before):
                current_year = base_year - decrement
                decremented_subfolder = os.path.sep.join(
                    base_subfolder_key.split(os.path.sep)[:-1] + [str(current_year)]
                )
                
                # Check if the decremented subfolder exists in datasets
                if decremented_subfolder in self.datasets:
                    try:
                        id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                        tensor = self.datasets[decremented_subfolder].get_tensor_by_location(id_num, x, y)
                        if tensor is not None:
                            channel_tensors.append(tensor)
                        else:
                            channel_tensors.append(torch.zeros(window_size, window_size))
                    except ValueError:
                        channel_tensors.append(torch.zeros(window_size, window_size))
                else:
                    channel_tensors.append(torch.zeros(window_size, window_size))

        # Stack all tensors to form the final tensor of shape (26, 5, 5)
        if len(channel_tensors) != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} channels, but got {len(channel_tensors)}"
            )
        
        final_tensor = torch.stack(channel_tensors)

        return longitude, latitude, final_tensor, oc

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
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        """Get tensor from specific subfolder dataset"""
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
            _, _, features, _ = super().__getitem__(i)
            features_list.append(features.numpy())

        features_array = np.stack(features_list)
        self._feature_means = torch.tensor(np.mean(features_array, axis=(0, 2, 3)), dtype=torch.float32)
        self._feature_stds = torch.tensor(np.std(features_array, axis=(0, 2, 3)), dtype=torch.float32)
        self._feature_stds = torch.clamp(self._feature_stds, min=1e-8)

    def __getitem__(self, idx):
        longitude, latitude, features, target = super().__getitem__(idx)
        features = (features - self._feature_means[:, None, None]) / self._feature_stds[:, None, None]
        return longitude, latitude, features, target

    def get_statistics(self):
        """Getter for feature means and standard deviations"""
        return self._feature_means, self._feature_stds

    def get_feature_means(self):
        """Getter for feature means"""
        return self._feature_means

    def get_feature_stds(self):
        """Getter for feature standard deviations"""
        return self._feature_stds

    def set_feature_means(self, means):
        """Setter for feature means"""
        self._feature_means = means

    def set_feature_stds(self, stds):
        """Setter for feature standard deviations"""
        self._feature_stds = stds
