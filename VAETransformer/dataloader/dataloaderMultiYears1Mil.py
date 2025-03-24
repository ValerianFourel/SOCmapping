import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
import pandas as pd
from config import bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size
from accelerate import Accelerator

class RasterTensorDataset(Dataset):
    def __init__(self, base_path, preload=True):
        """
        Args:
            base_path (str): Directory containing .npy files.
            preload (bool): If True, load all data into memory; if False, load on demand.
        """
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

class MultiRasterDataset1MilMultiYears(Dataset):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, time_before=time_before, preload=True):
        def flatten_list(lst):
            return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]
        
        self.data_array_subfolders = flatten_list(data_array_subfolders)
        self.seasonalityBased = self.check_seasonality(self.data_array_subfolders)
        self.time_before = time_before
        self.samples_coordinates_array_subfolders = flatten_list(samples_coordinates_array_subfolders)
        self.dataframe = dataframe
        self.datasets = {self.get_last_three_folders(subfolder): RasterTensorDataset(subfolder, preload=preload)
                         for subfolder in self.data_array_subfolders}
        self.coordinates = {self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")
                            for subfolder in self.samples_coordinates_array_subfolders}

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
            filtered_array = [path for path in self.samples_coordinates_array_subfolders
                              if ('Elevation' in path) or
                                 ('MODIS_NPP' in path and path.endswith(str(year))) or
                                 (not 'Elevation' in path and not 'MODIS_NPP' in path and path.endswith(season))]
        else:
            filtered_array = [path for path in self.samples_coordinates_array_subfolders
                              if ('Elevation' in path) or
                                 (not 'Elevation' in path and path.endswith(str(year)))]
        return filtered_array

    def _get_unnormalized_item(self, index):
        row = self.dataframe.iloc[index]
        longitude, latitude = row["longitude"], row["latitude"]
        filtered_array = self.filter_by_season_or_year(row.get('season', ''), row.get('year', ''), self.seasonalityBased)

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
                        encoding[bands_list_order.index('Elevation'), t] = 0
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
        final_tensor = final_tensor.permute(0, 2, 3, 1)  # Shape: (bands, window_size, window_size, time_before)

        return longitude, latitude, final_tensor, encoding

    def __getitem__(self, index):
        return self._get_unnormalized_item(index)

    def __len__(self):
        return len(self.dataframe)

class NormalizedMultiRasterDataset1MilMultiYears(MultiRasterDataset1MilMultiYears):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, accelerator=None, time_before=time_before, batch_size=64, num_workers=4, preload=True):
        super().__init__(samples_coordinates_array_subfolders, data_array_subfolders, dataframe, time_before, preload=preload)
        self.accelerator = accelerator if accelerator else Accelerator()
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def compute_statistics(self):
    #     """Compute mean and std per band and time step, excluding missing data (encoding == -1)"""
    #     class StatsDataset(Dataset):
    #         def __init__(self, base_dataset):
    #             self.base_dataset = base_dataset

    #         def __len__(self):
    #             return len(self.base_dataset)

    #         def __getitem__(self, idx):
    #             longitude, latitude, final_tensor, encoding = self.base_dataset[idx]
    #             return final_tensor, encoding

    #     stats_dataset = StatsDataset(self)
    #     dataloader = DataLoader(stats_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    #     dataloader = self.accelerator.prepare(dataloader)

    #     # Accumulators for each band and time step
    #     total_sum = torch.zeros((len(bands_list_order), self.time_before), device=self.accelerator.device)
    #     total_sum_squares = torch.zeros((len(bands_list_order), self.time_before), device=self.accelerator.device)
    #     total_count = torch.zeros((len(bands_list_order), self.time_before), device=self.accelerator.device)

    #     for batch in dataloader:
    #         features, encoding = batch
    #         features = features.to(self.accelerator.device)  # (batch_size, bands, window_size, window_size, time_before)
    #         encoding = encoding.to(self.accelerator.device)  # (batch_size, bands, time_before)
    #         B, bands, H, W, T = features.shape

    #         for band_idx in range(bands):
    #             for t in range(T):
    #                 # Mask where data is present (encoding != -1)
    #                 mask = (encoding[:, band_idx, t] != -1).float()  # (batch_size,)
    #                 batch_sum = (features[:, band_idx, :, :, t] * mask.view(-1, 1, 1)).sum(dim=(0, 1, 2))
    #                 batch_sum_squares = ((features[:, band_idx, :, :, t] ** 2) * mask.view(-1, 1, 1)).sum(dim=(0, 1, 2))
    #                 batch_count = mask.sum() * H * W
    #                 total_sum[band_idx, t] += batch_sum
    #                 total_sum_squares[band_idx, t] += batch_sum_squares
    #                 total_count[band_idx, t] += batch_count

    #     # Aggregate across processes
    #     total_sum = self.accelerator.reduce(total_sum, reduction='sum')
    #     total_sum_squares = self.accelerator.reduce(total_sum_squares, reduction='sum')
    #     total_count = self.accelerator.reduce(total_count, reduction='sum')

    #     if self.accelerator.is_main_process:
    #         total_sum = total_sum.cpu()
    #         total_sum_squares = total_sum_squares.cpu()
    #         total_count = total_count.cpu()

    #         self.feature_means = total_sum / torch.clamp(total_count, min=1)  # Avoid div by zero
    #         self.feature_stds = torch.sqrt(torch.clamp((total_sum_squares / torch.clamp(total_count, min=1)) - (self.feature_means ** 2), min=0))
    #         self.feature_stds = torch.clamp(self.feature_stds, min=1e-8)

    #     self.feature_means = self.accelerator.broadcast(self.feature_means.to(self.accelerator.device))
    #     self.feature_stds = self.accelerator.broadcast(self.feature_stds.to(self.accelerator.device))

    def __getitem__(self, idx):
        longitude, latitude, final_tensor, encoding = super().__getitem__(idx)
        # final_tensor = (final_tensor - self.feature_means[:, None, None]) / self.feature_stds[:, None, None]
        return longitude, latitude, final_tensor, encoding