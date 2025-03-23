import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
import re
from config import bands_list_order, bands_dict, time_before, window_size

def consolidate_tensors(data_dict, band_to_index):
    """Consolidate tensors and indices from data_dict into a combined tensor and metadata."""
    tensors = []
    indices = []

    for band in sorted(data_dict.keys()):
        for year in sorted(data_dict[band].keys()):
            tensor = data_dict[band][year]
            tensors.append(tensor)
            indices.append([band_to_index[band], year])

    if not tensors:
        raise ValueError("No valid tensors found in data_dict to consolidate.")
    
    tensors_combined = np.stack(tensors, axis=0)  # Shape: (n, window_size, window_size)
    indices_tensor = np.array(indices)            # Shape: (n, 2)
    return tensors_combined, indices_tensor

class RasterTensorDataset1Mil(Dataset):
    def __init__(self, base_path):
        self.folder_path = base_path
        self.bands_list_order = ['Elevation', 'LAI', 'LST', 'MODIS_NPP', 'SoilEvaporation', 'TotalEvapotranspiration']
        self.type = self._extract_type_from_path()
        if self.type is None:
            raise ValueError(f"Path must contain one of these types: {self.bands_list_order}")
        self.id_to_file = self._create_id_mapping()
        self.data_cache = {id_num: np.load(filepath) for id_num, filepath in self.id_to_file.items()}

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
        if id_num not in self.id_to_file:
            raise ValueError(f"ID {id_num} not found in dataset")
        data = self.data_cache.get(id_num, np.load(self.id_to_file[id_num]))

        if self.type == 'LST':
            base_window_size = window_size // 4
        elif self.type != 'Elevation':
            base_window_size = window_size // 2
        else:
            base_window_size = window_size

        half_window = base_window_size // 2
        x, y = float(x), float(y)
        x_start = int(max(0, x - half_window))
        x_end = int(min(data.shape[0], x + half_window))
        y_start = int(max(0, y - half_window))
        y_end = int(min(data.shape[1], y + half_window))

        window = data[x_start:x_end, y_start:y_end]
        padded_window = np.zeros((base_window_size, base_window_size))
        x_offset = int(half_window - (x - x_start))
        y_offset = int(half_window - (y - y_start))
        x_offset = max(0, min(x_offset, base_window_size - window.shape[0]))
        y_offset = max(0, min(y_offset, base_window_size - window.shape[1]))
        x_slice_end = min(x_offset + window.shape[0], base_window_size)
        y_slice_end = min(y_offset + window.shape[1], base_window_size)
        window_height = x_slice_end - x_offset
        window_width = y_slice_end - y_offset
        padded_window[x_offset:x_slice_end, y_offset:y_slice_end] = window[:window_height, :window_width]

        if self.type == 'LST':
            window = np.repeat(np.repeat(padded_window, 4, axis=0), 4, axis=1)
        elif self.type != 'Elevation':
            window = np.repeat(np.repeat(padded_window, 2, axis=0), 2, axis=1)
        else:
            window = padded_window

        if window.shape != (window_size, window_size):
            final_window = np.zeros((window_size, window_size))
            final_window[:min(window.shape[0], window_size), :min(window.shape[1], window_size)] = \
                window[:min(window.shape[0], window_size), :min(window.shape[1], window_size)]
            window = final_window

        tensor = torch.from_numpy(window).float()
        return tensor, self.type

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
        """
        Retrieve tensor and metadata for a given index (inference mode).
        Parameters:
            index: int, index of the row in the dataframe
        Returns:
            tuple: (longitude, latitude, tensors_combined, indices_tensor)
        """
        row = self.dataframe.iloc[index]
        longitude, latitude = row["longitude"], row["latitude"]
        filtered_array = self.filter_by_season_or_year(row.get('season', ''), row.get('year', ''), self.seasonalityBased)

        data_dict = {band: {} for band in bands_list_order}
        elevation_processed = False

        # Process Elevation (1 step)
        for subfolder in filtered_array:
            subfolder_key = self.get_last_three_folders(subfolder)
            if subfolder_key.split(os.path.sep)[-1] == 'Elevation' and not elevation_processed:
                id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                tensor, type_of_band = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                if tensor is not None:
                    data_dict['Elevation'][0] = tensor
                else:
                    data_dict['Elevation'][0] = torch.zeros((window_size, window_size))  # Pad if missing
                elevation_processed = True

        # Process non-Elevation bands (ensure time_before steps per band)
        num_channels = len(bands_list_order) - 1  # Exclude Elevation
        for subfolder in filtered_array:
            subfolder_key = self.get_last_three_folders(subfolder)
            if subfolder_key.split(os.path.sep)[-1] != 'Elevation':
                year = int(subfolder_key.split(os.path.sep)[-1])
                band = subfolder_key.split(os.path.sep)[-2]
                for decrement in range(self.time_before):
                    current_year = year - decrement
                    decremented_subfolder = os.path.sep.join(subfolder_key.split(os.path.sep)[:-1] + [str(current_year)])
                    if decremented_subfolder in self.datasets:
                        id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                        tensor, type_of_band = self.datasets[decremented_subfolder].get_tensor_by_location(id_num, x, y)
                        if tensor is not None:
                            data_dict[band][current_year] = tensor
                        else:
                            data_dict[band][current_year] = torch.zeros((window_size, window_size))  # Pad if tensor is None
                    else:
                        data_dict[band][current_year] = torch.zeros((window_size, window_size))  # Pad if year missing

        # Ensure exactly time_before tensors per non-Elevation band
        for band in bands_list_order:
            if band != 'Elevation':
                available_years = sorted(data_dict[band].keys(), reverse=True)
                required_years = [year - i for i in range(self.time_before)]
                for req_year in required_years:
                    if req_year not in data_dict[band]:
                        data_dict[band][req_year] = torch.zeros((window_size, window_size))  # Pad missing years

        band_to_index = {v: k for k, v in bands_dict.items()}
        tensors_combined, indices_tensor = consolidate_tensors(data_dict, band_to_index)

        # Validate shape
        expected_steps = 1 + num_channels * self.time_before
        if tensors_combined.shape[0] != expected_steps:
            raise ValueError(f"Output tensor has {tensors_combined.shape[0]} steps, expected {expected_steps}")

        return longitude, latitude, torch.from_numpy(tensors_combined).float(), torch.from_numpy(indices_tensor).long()
    def __len__(self):
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)