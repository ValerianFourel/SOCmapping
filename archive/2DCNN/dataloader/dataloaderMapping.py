import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from config import bands_list_order, time_before, window_size
import re
from accelerate import Accelerator
from torch.utils.data import DataLoader

class RasterTensorDataset1Mil(Dataset):
    def __init__(self, base_path):
        self.folder_path = base_path
        self.id_to_file = self._create_id_mapping()
        self.data_cache = {id_num: np.load(filepath) for id_num, filepath in self.id_to_file.items()}

    def _create_id_mapping(self):
        id_to_file = {}
        for file_path in glob.glob(os.path.join(self.folder_path, "*.npy")):
            match = re.search(r'ID(\d+)N', file_path)
            if match:
                id_num = int(match.group(1))
                id_to_file[id_num] = file_path
        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=window_size):
        if id_num not in self.id_to_file:
            raise ValueError(f"ID {id_num} not found in dataset")
        data = self.data_cache.get(id_num, np.load(self.id_to_file[id_num]))
        half_window = window_size // 2
        x_start, x_end = int(max(0, x - half_window)), int(min(data.shape[0], x + half_window + 1))
        y_start, y_end = int(max(0, y - half_window)), int(min(data.shape[1], y + half_window + 1))
        window = data[x_start:x_end, y_start:y_end]
        if window.shape != (window_size, window_size):
            padded_window = np.zeros((window_size, window_size))
            x_offset = half_window - (x - x_start)
            y_offset = half_window - (y - y_start)
            padded_window[x_offset:x_offset + window.shape[0], y_offset:y_offset + window.shape[1]] = window
            window = padded_window
        return torch.from_numpy(window).float()

    def __len__(self):
        return len(self.id_to_file)

    def __getitem__(self, idx):
        id_num = list(self.id_to_file.keys())[idx]
        return self.data_cache[id_num]


class MultiRasterDataset1MilMultiYears(Dataset):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, time_before=time_before):
        def flatten_list(lst):
            return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

        self.data_array_subfolders = flatten_list(data_array_subfolders)
        self.seasonalityBased = self.check_seasonality(self.data_array_subfolders)
        self.time_before = time_before
        self.samples_coordinates_array_subfolders = flatten_list(samples_coordinates_array_subfolders)
        self.dataframe = dataframe
        self.datasets = {
            self.get_last_three_folders(subfolder): RasterTensorDataset1Mil(subfolder)
            for subfolder in self.data_array_subfolders
        }
        self.coordinates = {
            self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")
            for subfolder in self.samples_coordinates_array_subfolders
        }

    def check_seasonality(self, data_array_subfolders):
        seasons = ['winter', 'spring', 'summer', 'autumn']
        return any(any(season in subfolder.lower() for season in seasons) for subfolder in data_array_subfolders)

    def get_last_three_folders(self, path):
        parts = path.rstrip('/').split('/')
        return '/'.join(parts[-2:])

    def find_coordinates_index(self, subfolder, longitude, latitude):
        coords = self.coordinates[subfolder]
        match = np.where((coords[:, 1] == longitude) & (coords[:, 0] == latitude))[0]
        if match.size == 0:
            raise ValueError(f"Coordinates ({longitude}, {latitude}) not found in {subfolder}")
        return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

    def filter_by_season_or_year(self, season, year, seasonality_based):
        if seasonality_based:
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

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        longitude, latitude = row["longitude"], row["latitude"]
        filtered_array = self.filter_by_season_or_year(row.get('season', ''), row.get('year', ''), self.seasonalityBased)


        
        # Initialize a list to hold tensors for all channels (26 in total)
        channel_tensors = []
        self.expected_channels = 26
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
        self.non_elevation_bands = [band for band in bands_list_order if band != 'Elevation']
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

        return longitude, latitude, final_tensor
    def __len__(self):
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

class NormalizedMultiRasterDataset1MilMultiYears(MultiRasterDataset1MilMultiYears):
    """Wrapper around MultiRasterDatasetMultiYears that adds feature normalization"""
    def __init__(self, samples_coordinates_array_path, data_array_path, df,feature_means,feature_stds,time_before):
        super().__init__(samples_coordinates_array_path, data_array_path, df,time_before)
        self.feature_means=feature_means
        self.feature_stds=feature_stds
        time_before=time_before
        

    def __getitem__(self, idx):
        longitude, latitude, features = super().__getitem__(idx)
        features = (features - self.feature_means[:, None, None]) / self.feature_stds[:, None, None]
        return longitude, latitude, features
