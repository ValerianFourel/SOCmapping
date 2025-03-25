import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
import pandas as pd
from config import bands_list_order, TIME_BEGINNING, TIME_END, window_size
from accelerate import Accelerator
import logging

# Convert TIME_BEGINNING and TIME_END to integers at the file level
TIME_BEGINNING = int(TIME_BEGINNING)
TIME_END = int(TIME_END)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RasterTensorDataset(Dataset):
    def __init__(self, base_path, preload=True):
        self.folder_path = base_path
        self.preload = preload
        self.id_to_file = self._create_id_mapping()
        if preload:
            self.data_cache = {id_num: np.load(filepath) for id_num, filepath in self.id_to_file.items()}
        else:
            self.data_cache = {}

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
        if self.preload:
            data = self.data_cache[id_num]
        else:
            data = np.load(self.id_to_file[id_num])
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
        if self.preload:
            return self.data_cache[id_num]
        return np.load(self.id_to_file[id_num])

class BandSpecificRasterDataset(Dataset):
    def __init__(self, band, subfolders, coordinates_subfolders, dataframe, preload=True):
        self.band = band
        self.dataframe = dataframe
        self.preload = preload
        self.subfolders = subfolders
        self.datasets = {self.get_last_three_folders(subfolder): RasterTensorDataset(subfolder, preload=preload)
                         for subfolder in subfolders}
        self.coordinates = {self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")
                            for subfolder in coordinates_subfolders}
        
        if self.band == 'Elevation' and not self.subfolders:
            logger.warning(f"No subfolders found for Elevation band. Using zero tensor as fallback.")
        
        # Determine time steps for this band using integer TIME_BEGINNING and TIME_END
        self.time_steps = 1 if self.band == 'Elevation' else (TIME_END - TIME_BEGINNING + 1)

    def get_last_three_folders(self, path):
        parts = path.rstrip('/').split('/')
        return '/'.join(parts[-2:])

    def find_coordinates_index(self, subfolder, longitude, latitude):
        coords = self.coordinates[subfolder]
        match = np.where((coords[:, 1] == longitude) & (coords[:, 0] == latitude))[0]
        if match.size == 0:
            raise ValueError(f"Coordinates ({longitude}, {latitude}) not found in {subfolder}")
        return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

    def filter_by_year(self, year):
        if self.band == 'Elevation':
            return [path for path in self.subfolders if 'Elevation' in path]
        else:
            return [path for path in self.subfolders if path.endswith(str(year))]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        longitude, latitude = row["longitude"], row["latitude"]

        band_tensors = [None] * self.time_steps
        encoding = torch.full((self.time_steps,), -1, dtype=torch.long)

        if self.band == 'Elevation':
            if self.subfolders:  # Check if there are any subfolders
                subfolder_key = self.get_last_three_folders(self.subfolders[0])
                try:
                    id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                    elevation_tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                    if elevation_tensor is not None:
                        band_tensors[0] = elevation_tensor
                        encoding[0] = 0
                except ValueError as e:
                    logger.warning(f"Elevation data fetch failed for ({longitude}, {latitude}): {e}. Using zero tensor.")
            if band_tensors[0] is None:  # Fallback if no data or fetch fails
                band_tensors[0] = torch.zeros(window_size, window_size)
                encoding[0] = -1
        else:
            for t, year in enumerate(range(TIME_BEGINNING, TIME_END + 1)):
                filtered_array = self.filter_by_year(year)
                for subfolder in filtered_array:
                    subfolder_key = self.get_last_three_folders(subfolder)
                    if subfolder_key in self.datasets:
                        try:
                            id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                            tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                            if tensor is not None:
                                band_tensors[t] = tensor
                                encoding[t] = year
                        except ValueError as e:
                            logger.warning(f"Data fetch failed for {self.band}, year {year} at ({longitude}, {latitude}): {e}")

        for t in range(self.time_steps):
            if band_tensors[t] is None:
                band_tensors[t] = torch.zeros(window_size, window_size)

        stacked_tensor = torch.stack(band_tensors)  # Shape: (time_steps, window_size, window_size)
        return longitude, latitude, stacked_tensor, encoding

    def __len__(self):
        return len(self.dataframe)

class MultiRasterDataset1MilMultiYears:
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, preload=True):
        def flatten_list(lst):
            return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

        self.dataframe = dataframe
        self.coordinates_subfolders = flatten_list(samples_coordinates_array_subfolders)
        self.data_subfolders = flatten_list(data_array_subfolders)

        self.band_datasets = {}
        for band in bands_list_order:
            band_subfolders = [sf for sf in self.data_subfolders if band in sf.split(os.path.sep)[-2]]
            self.band_datasets[band] = BandSpecificRasterDataset(
                band, band_subfolders, self.coordinates_subfolders, dataframe, preload
            )
            logger.info(f"Initialized {band} with {len(band_subfolders)} subfolders: {band_subfolders}")

    def get_band_dataset(self, band):
        if band not in self.band_datasets:
            raise ValueError(f"Band {band} not found in available datasets")
        return self.band_datasets[band]

    def __getitem__(self, index):
        longitude, latitude = None, None
        band_tensors = []
        encodings = []

        for band in bands_list_order:
            lon, lat, tensor, encoding = self.band_datasets[band][index]
            if longitude is None:
                longitude, latitude = lon, lat
            band_tensors.append(tensor)
            encodings.append(encoding)

        final_tensor = torch.stack(band_tensors)  # Shape: (bands, time_steps, window_size, window_size)
        encoding_tensor = torch.stack(encodings)  # Shape: (bands, time_steps)
        final_tensor = final_tensor.permute(0, 2, 3, 1)  # Shape: (bands, window_size, window_size, time_steps)
        return longitude, latitude, final_tensor, encoding_tensor

    def __len__(self):
        return len(self.dataframe)

class NormalizedMultiRasterDataset1MilMultiYears(MultiRasterDataset1MilMultiYears):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, accelerator=None, batch_size=64, num_workers=4, preload=True):
        super().__init__(samples_coordinates_array_subfolders, data_array_subfolders, dataframe, preload)
        self.accelerator = accelerator if accelerator else Accelerator()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __getitem__(self, idx):
        longitude, latitude, final_tensor, encoding = super().__getitem__(idx)
        return longitude, latitude, final_tensor, encoding
