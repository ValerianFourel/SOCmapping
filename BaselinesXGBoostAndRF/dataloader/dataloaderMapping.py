
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# from pathlib import Path
# import re
# import glob
# from config import window_size

# class RasterTensorDatasetMapping(Dataset):
#     def __init__(self, base_path):
#         """
#         Initialize the dataset

#         Parameters:
#         base_path: str, base path to RasterTensorData directory
#         subfolder: str, name of the subfolder (e.g., 'Elevation')
#         """
#         # Replace "OC_LUCAS_LFU_LfL_Coordinates" with "RasterTensorData" in the base path
#         self.base_path = Path(base_path.replace("Coordinates1Mil", "RasterTensorData"))
#         self.folder_path = self.base_path

#         # Create ID to filename mapping
#         self.id_to_file = self._create_id_mapping()

#         # Load all numpy arrays into memory (optional, can be modified to load on demand)
#         self.data_cache = {}
#         for id_num, filepath in self.id_to_file.items():
#             self.data_cache[id_num] = np.load(filepath)

#     def _create_id_mapping(self):
#         """Create a dictionary mapping IDs to their corresponding file paths"""
#         id_to_file = {}

#         for file_path in self.folder_path.glob("*.npy"):
#             # Extract ID number from filename
#             match = re.search(r'ID(\d+)N', file_path.name)
#             if match:
#                 id_num = int(match.group(1))
#                 id_to_file[id_num] = file_path

#         return id_to_file

#     def get_tensor_by_location(self, id_num, x, y, window_size=window_size):
#         """
#         Get a window_size x window_size square around the specified x,y coordinates

#         Parameters:
#         id_num: int, ID number from filename
#         x: int, x coordinate
#         y: int, y coordinate
#         window_size: int, size of the square window (default 17)

#         Returns:
#         torch.Tensor: window_size x window_size tensor
#         """
#         if id_num not in self.id_to_file:
#             raise ValueError(f"ID {id_num} not found in dataset")

#         # Get the data array
#         if id_num in self.data_cache:
#             data = self.data_cache[id_num]
#         else:
#             data = np.load(self.id_to_file[id_num])

#         # Calculate window boundaries
#         half_window = window_size // 2
#         x_start = int(max(0, x - half_window))
#         x_end = int(min(data.shape[0], x + half_window + 1))
#         y_start = int(max(0, y - half_window))
#         y_end = int(min(data.shape[1], y + half_window + 1))

#         # Extract window
#         window = data[x_start:x_end, y_start:y_end]

#         # Pad if necessary
#         if window.shape != (window_size, window_size):
#             padded_window = np.zeros((window_size, window_size))
#             x_offset = half_window - (x - x_start)
#             y_offset = half_window - (y - y_start)
#             padded_window[
#                 x_offset:x_offset+window.shape[0],
#                 y_offset:y_offset+window.shape[1]
#             ] = window
#             window = padded_window

#         return torch.from_numpy(window).float()

#     def __len__(self):
#         return len(self.id_to_file)

#     def __getitem__(self, idx):
#         # This is a placeholder implementation
#         # Modify according to your specific needs
#         id_num = list(self.id_to_file.keys())[idx]
#         return self.data_cache[id_num]

# # Example usage:
# """
# # Initialize the dataset
# base_path = "/content/drive/MyDrive/Colab Notebooks/MappingSOC/Data/RasterTensorData"
# dataset = RasterTensorDataset(base_path, "Elevation")

# # Get the dictionary mapping IDs to filenames
# id_mapping = dataset.id_to_file
# print("ID to filename mapping:", id_mapping)

# # Get a 17x17 window for a specific location
# id_num = 10  # example ID
# x, y = 100, 100  # example coordinates
# window = dataset.get_tensor_by_location(id_num, x, y)
# print("Window shape:", window.shape)

# # Create a DataLoader if needed
# from torch.utils.data import DataLoader
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# """


# class MultiRasterDatasetMapping(Dataset):
#     def __init__(self, subfolders, dataframe):
#         """
#         Parameters:
#         subfolders: list of str, names of subfolders to include
#         dataframe: pandas.DataFrame, contains columns GPS_LONG, GPS_LAT, and OC (target variable)
#         """
#         self.subfolders = subfolders
#         self.dataframe = dataframe
#         self.datasets = {
#             subfolder: RasterTensorDatasetMapping(subfolder)
#             for subfolder in subfolders
#         }
#         self.coordinates = {
#             subfolder: np.load(f"{subfolder}/coordinates.npy")
#             for subfolder in subfolders
#         }

#     def find_coordinates_index(self, subfolder, longitude, latitude):
#         """
#         Finds the index of the matching coordinates in the subfolder's coordinates.npy file.

#         Parameters:
#         subfolder: str, name of the subfolder
#         longitude: float, longitude to match
#         latitude: float, latitude to match

#         Returns:
#         tuple: (id_num, x, y) if match is found, otherwise raises an error
#         """
#         coords = self.coordinates[subfolder]
#         # Assuming the first two columns of `coordinates.npy` are longitude and latitude
#         match = np.where((coords[:, 1] == longitude) & (coords[:, 0] == latitude))[0]
#         if match.size == 0:
#             raise ValueError(f"Coordinates ({longitude}, {latitude}) not found in {subfolder}")

#         # Return id_num, x, y from the same row
#         return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

#     def __getitem__(self, index):
#         """
#         Retrieve tensor and target value for a given index.

#         Parameters:
#         index: int, index of the row in the dataframe

#         Returns:
#         tuple: (tensor, OC), where tensor is the data and OC is the target variable
#         """
#         row = self.dataframe.iloc[index]
#         longitude, latitude = row["longitude"], row["latitude"]

#         tensors = {}
#         for subfolder in self.subfolders:
#             id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
#             tensors[subfolder] = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

#         return longitude, latitude, tensors

#     def __len__(self):
#         """
#         Return the number of samples in the dataset.
#         """
#         return len(self.dataframe)

#     def get_tensor_by_location(self, subfolder, id_num, x, y):
#         """Get tensor from specific subfolder dataset"""
#         return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from config import bands_list_order, time_before, window_size
#from configElevationOnlyExperiment import bands_list_order, time_before, window_size

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


class MultiRasterDatasetMapping(Dataset):
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

        band_tensors = {band: [] for band in bands_list_order}

        for subfolder in filtered_array:
            subfolder_key = self.get_last_three_folders(subfolder)
            if subfolder_key.split(os.path.sep)[-1] == 'Elevation':
                id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                elevation_tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                if elevation_tensor is not None:
                    for _ in range(self.time_before):
                        band_tensors['Elevation'].append(elevation_tensor)
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
                            if band in band_tensors:
                                band_tensors[band].append(tensor)

        stacked_tensors = []
        for band in bands_list_order:
            if not band_tensors[band]:
                band_tensors[band] = [torch.zeros(window_size, window_size) for _ in range(self.time_before)]
            elif len(band_tensors[band]) < self.time_before:
                while len(band_tensors[band]) < self.time_before:
                    band_tensors[band].append(torch.zeros(window_size, window_size))
            elif len(band_tensors[band]) > self.time_before:
                band_tensors[band] = band_tensors[band][:self.time_before]
            stacked_tensor = torch.stack(band_tensors[band])
            stacked_tensors.append(stacked_tensor)

        if len(stacked_tensors) != len(bands_list_order):
            raise ValueError(f"Expected {len(bands_list_order)} bands, but got {len(stacked_tensors)}")

        final_tensor = torch.stack(stacked_tensors)
        final_tensor = final_tensor.permute(0, 2, 3, 1)
        return longitude, latitude, final_tensor

    def __len__(self):
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

