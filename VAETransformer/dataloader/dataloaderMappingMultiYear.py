import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import re
import glob
from config import bands_list_order, time_before, window_size

class RasterTensorDatasetMapping(Dataset):
    def __init__(self, base_path):
        self.base_path = Path(base_path.replace("Coordinates1Mil", "RasterTensorData"))
        self.folder_path = self.base_path
        self.id_to_file = self._create_id_mapping()
        self.data_cache = {}
        for id_num, filepath in self.id_to_file.items():
            self.data_cache[id_num] = np.load(filepath)

    def _create_id_mapping(self):
        id_to_file = {}
        for file_path in self.folder_path.glob("*.npy"):
            match = re.search(r'ID(\d+)N', file_path.name)
            if match:
                id_num = int(match.group(1))
                id_to_file[id_num] = file_path
        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=17):
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
    
from config import bands_list_order, time_before, window_size
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class MultiRasterDatasetMappingMultiYears(Dataset):
    def __init__(self, subfolders, dataframe, time_before=time_before):
        self.subfolders = subfolders
        self.dataframe = dataframe
        self.time_before = time_before
        self.seasonalityBased = self.check_seasonality(subfolders)
        self.datasets = {
            self.get_last_three_folders(subfolder): RasterTensorDatasetMapping(subfolder)
            for subfolder in subfolders
        }
        self.coordinates = {}
        for subfolder in subfolders:
            key = self.get_last_three_folders(subfolder)
            coord_path = f"{subfolder}/coordinates.npy"
            if os.path.exists(coord_path):
                self.coordinates[key] = np.load(coord_path)
            else:
                raise FileNotFoundError(f"coordinates.npy not found at {coord_path}")

    def check_seasonality(self, subfolders):
        seasons = ['winter', 'spring', 'summer', 'autumn']
        return any(any(season in subfolder.lower() for season in seasons) for subfolder in subfolders)

    def get_last_three_folders(self, path):
        parts = path.rstrip('/').split('/')
        return '/'.join(parts[-2:])

    def find_coordinates_index(self, subfolder, longitude, latitude, tolerance=1e-5):
        coords = self.coordinates[subfolder]
        # Debugging: Print available coordinates
        if not coords.size:
            raise ValueError(f"No coordinates loaded for {subfolder}")
        
        # Use tolerance for floating-point comparison
        match = np.where(
            (np.abs(coords[:, 1] - longitude) < tolerance) & 
            (np.abs(coords[:, 0] - latitude) < tolerance)
        )[0]
        
        if match.size == 0:
            # More detailed error message
            min_dist = np.min(np.sqrt((coords[:, 1] - longitude)**2 + (coords[:, 0] - latitude)**2))
            raise ValueError(
                f"Coordinates ({longitude}, {latitude}) not found in {subfolder}. "
                f"Closest match distance: {min_dist}. Sample coords: {coords[:5]}"
            )
        
        return int(coords[match[0], 2]), int(coords[match[0], 3]), int(coords[match[0], 4])

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        longitude, latitude = row["GPS_LONG"], row["GPS_LAT"]  # Adjusted column names

        tensors = {}
        band_tensors = {band: [] for band in bands_list_order}

        reference_year = None
        for subfolder in self.subfolders:
            if 'Elevation' not in subfolder and 'MODIS_NPP' not in subfolder:
                reference_year = int(subfolder.split(os.path.sep)[-1][:4])
                break
        if reference_year is None:
            reference_year = int(INFERENCE_TIME[:4])

        for subfolder in self.subfolders:
            subfolder_key = self.get_last_three_folders(subfolder)
            last_folder = subfolder.split(os.path.sep)[-1]

            if last_folder == 'Elevation':
                id_num, x, y = self.find_coordinates_index(subfolder_key, longitude, latitude)
                elevation_tensor = self.datasets[subfolder_key].get_tensor_by_location(id_num, x, y)
                if elevation_tensor is not None:
                    for _ in range(self.time_before):
                        band_tensors['Elevation'].append(elevation_tensor)
            else:
                year = int(last_folder[:4]) if last_folder.isdigit() else reference_year
                for decrement in range(self.time_before):
                    current_year = year - decrement
                    decremented_subfolder = os.path.sep.join(subfolder.split(os.path.sep)[:-1] + [str(current_year)])
                    decremented_key = self.get_last_three_folders(decremented_subfolder)

                    if decremented_key in self.datasets:
                        try:
                            id_num, x, y = self.find_coordinates_index(decremented_key, longitude, latitude)
                            tensor = self.datasets[decremented_key].get_tensor_by_location(id_num, x, y)
                            if tensor is not None:
                                band = subfolder.split(os.path.sep)[-2]
                                band_tensors[band].append(tensor)
                        except ValueError as e:
                            if accelerator.is_main_process:
                                print(f"Skipping {decremented_key} due to: {e}")
                            continue

        stacked_tensors = []
        for band in bands_list_order:
            if band_tensors[band]:
                stacked_tensor = torch.stack(band_tensors[band])
                stacked_tensors.append(stacked_tensor)

        if not stacked_tensors:
            raise ValueError(f"No valid tensors found for index {index}, coords ({longitude}, {latitude})")

        final_tensor = torch.stack(stacked_tensors)
        final_tensor = final_tensor.permute(0, 2, 3, 1)

        return longitude, latitude, final_tensor

    def __len__(self):
        return len(self.dataframe)