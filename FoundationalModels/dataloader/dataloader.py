

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
from config import seasons, years_padded , YEARS_BACK , window_size


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
    def __init__(self, base_path, resolution='250m', window_size=window_size):
        """
        Initialize the dataset

        Parameters:
        base_path: str, base path to RasterTensorData directory
        resolution: str, pixel resolution ('250m', '500m', or '1km')
        window_size: int, base window size for 1km resolution
        """
        self.folder_path = base_path
        self.resolution = resolution
        self.window_size = window_size  # This is for 1km resolution
        self.target_size = window_size * 4  # Final tensor size (250m resolution)

        # Calculate expansion factor based on resolution
        self.expansion_factor = {
            '250m': 4,  # Already at target resolution
            '500m': 2,  # Each pixel becomes 2x2
            '1km': 1    # Each pixel becomes 4x4
        }.get(resolution, 4)

        # Adjust input window size based on resolution
        self.input_window_size = self.window_size * self.expansion_factor

        # Create ID to filename mapping
        self.id_to_file = self._create_id_mapping()

        # Load all numpy arrays into memory
        self.data_cache = {}
        for id_num, filepath in self.id_to_file.items():
            self.data_cache[id_num] = np.load(filepath)

    def get_tensor_by_location(self, id_num, x, y):
        """
        Get a window around the specified x,y coordinates and expand it to match 250m resolution

        Parameters:
        id_num: int, ID number from filename
        x: int, x coordinate
        y: int, y coordinate

        Returns:
        torch.Tensor: target_size x target_size tensor (window_size*4 x window_size*4)
        """
        if id_num not in self.id_to_file:
            raise ValueError(f"ID {id_num} not found in dataset")

        # Scale coordinates based on resolution
        scaled_x = int(x / self.expansion_factor)
        scaled_y = int(y / self.expansion_factor)

        # Get the data array
        data = self.data_cache.get(id_num, np.load(self.id_to_file[id_num]))

        # Calculate window boundaries for input resolution
        half_window = self.input_window_size // 2
        x_start = int(max(0, scaled_x - half_window))
        x_end = int(min(data.shape[0], scaled_x + half_window + 1))
        y_start = int(max(0, scaled_y - half_window))
        y_end = int(min(data.shape[1], scaled_y + half_window + 1))

        # Extract window
        window = data[x_start:x_end, y_start:y_end]

        # Pad if necessary
        if window.shape != (self.input_window_size, self.input_window_size):
            padded_window = np.zeros((self.input_window_size, self.input_window_size))
            x_offset = int(half_window - (scaled_x - x_start))
            y_offset = int(half_window - (scaled_y - y_start))

            # Ensure all indices are within bounds
            x_start_pad = max(0, x_offset)
            y_start_pad = max(0, y_offset)
            x_end_pad = min(self.input_window_size, x_offset + window.shape[0])
            y_end_pad = min(self.input_window_size, y_offset + window.shape[1])

            # Calculate corresponding window slices
            x_start_win = max(0, -x_offset)
            y_start_win = max(0, -y_offset)
            x_end_win = min(window.shape[0], self.input_window_size - x_offset)
            y_end_win = min(window.shape[1], self.input_window_size - y_offset)

            padded_window[
                x_start_pad:x_end_pad,
                y_start_pad:y_end_pad
            ] = window[
                x_start_win:x_end_win,
                y_start_win:y_end_win
            ]
            window = padded_window

        # Convert to tensor
        window_tensor = torch.from_numpy(window).float()

        # Expand to target size based on resolution
        if self.resolution != '250m':
            window_tensor = window_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Use nearest neighbor for pixel expansion
            resized_tensor = torch.nn.functional.interpolate(
                window_tensor,
                size=(self.target_size, self.target_size),
                mode='nearest'
            )
            window_tensor = resized_tensor.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

        return window_tensor

    def _create_id_mapping(self):
        """Create a dictionary mapping IDs to their corresponding file paths"""
        id_to_file = {}
        for file_path in glob.glob(os.path.join(self.folder_path, "*.npy")):
            match = re.search(r'ID(\d+)N', file_path)
            if match:
                id_num = int(match.group(1))
                id_to_file[id_num] = file_path
        return id_to_file

    def __len__(self):
        return len(self.id_to_file)

    def __getitem__(self, idx):
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


class MultiRasterDataset(Dataset):
    def __init__(self, samples_coordinates_array_subfolders ,  data_array_subfolders , dataframe,YEARS_BACK= YEARS_BACK,seasons = seasons,years_padded = years_padded):
        """
        Parameters:
        subfolders: list of str, names of subfolders to include
        dataframe: pandas.DataFrame, contains columns GPS_LONG, GPS_LAT, and OC (target variable)
        """
        self.data_array_subfolders = data_array_subfolders
        self.seasonalityBased = self.check_seasonality(data_array_subfolders)

        self.samples_coordinates_array_subfolders = samples_coordinates_array_subfolders
        self.dataframe = dataframe

        self.seasons = seasons  # The pre-generated seasonal list
        self.years_padded = years_padded  # The pre-generated years list
        self.YEARS_BACK = YEARS_BACK  # Number of years to look back
        self.datasets = {
            self.get_last_three_folders(subfolder): RasterTensorDataset(subfolder,self._get_resolution_from_path(subfolder))
            for subfolder in  self.data_array_subfolders
        }
        
        self.coordinates = {
            self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")#[np.isfinite(np.load(f"{subfolder}/coordinates.npy"))]
            for subfolder in self.samples_coordinates_array_subfolders
        }
    def _get_resolution_from_path(self, path):
        """
        Determine resolution based on folder name
        """
        if 'LAI' in path:
            return '500m'
        elif 'LST' in path:
            return '1km'
        elif 'SoilEvaporation' in path:
            return '500m'
        elif 'MODIS_NPP' in path:
            return '500m'
        elif 'Elevation' in path:
            return '250m'
        elif 'TotalEvapotranspiration' in path:
            return '500m'
        else:
            return '250m'  # default resolution

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


    def extend_filtered_array(self, filtered_array, years_back):
        """
        Extend filtered_array with historical paths based on YEARS_BACK.

        Parameters:
        filtered_array: list of str, original paths
        years_back: int, number of years to look back

        Returns:
        list: Extended list of paths including historical data
        """
        extended_array = []
        base_path = '/home/vfourel/SOCProject/SOCmapping/Data/OC_LUCAS_LFU_LfL_Coordinates_v2'

        for path in filtered_array:
            if 'StaticValue/Elevation' in path:
                # Elevation appears only once
                extended_array.append(path)
                continue
            if self.seasonalityBased:
                if 'YearlyValue/MODIS_NPP' in path:
                    # Get the current year from the path
                    current_year = int(path.split('/')[-1])
                    # Add paths for previous years
                    for year in range(current_year, current_year - years_back, -1):
                        new_path = f"{base_path}/YearlyValue/MODIS_NPP/{year}"
                        extended_array.append(new_path)
                    continue

                if 'SeasonalValue' in path:
                    # Extract band type (LAI, LST, etc.) and current year_season
                    band_type = path.split('SeasonalValue/')[1].split('/')[0]
                    current_year_season = path.split('/')[-1]  # e.g., "2009_summer"
                    #current_year = int(current_year_season.split('_')[0])
                    #current_season = current_year_season.split('_')[1]
                    #print(current_year_season)

                    # Find index of current season in seasons list
                    season_idx = self.seasons.index(current_year_season)

                    # Get years_back * 4 seasons going backwards
                    for i in range(season_idx - (years_back * 4)+1,season_idx+1):
                        if i < len(self.seasons):  # Ensure we don't go out of bounds
                            new_path = f"{base_path}/SeasonalValue/{band_type}/{self.seasons[i]}"
                            extended_array.append(new_path)
            else:
                if 'YearlyValue' in path:
                    # Extract band type (LAI, LST, etc.) and current year_season
                    band_type = path.split('YearlyValue/')[1].split('/')[0]
                    current_year_season = path.split('/')[-1]  # e.g., "2009_summer"
                    #current_year = int(current_year_season.split('_')[0])
                    #current_season = current_year_season.split('_')[1]
                    # print(current_year_season)

                    # Find index of current season in seasons list
                    years_idx = self.years_padded.index(current_year_season)

                    # Get years_back * 4 seasons going backwards
                    for i in range(years_idx - (years_back)+1,years_idx+1):
                        
                        if i < len(self.years_padded):  # Ensure we don't go out of bounds
                            new_path = f"{base_path}/YearlyValue/{band_type}/{self.years_padded[i]}"
                            extended_array.append(new_path)
            

            

        return extended_array

    def __getitem__(self, index):
        """
        Retrieve tensor and target value for a given index.

        Parameters:
        index: int, index of the row in the dataframe

        Returns:
        tuple: (longitude, latitude, tensor, OC)
        """
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]

        filtered_array = self.filter_by_season_or_year(row['season'], row['year'], self.seasonalityBased)
        extended_array = self.extend_filtered_array(filtered_array,self.YEARS_BACK)

        tensors = []
        # print(' filtered_array   ',filtered_array)
        for path in extended_array:
            last_three = self.get_last_three_folders(path)
            id_num, x, y = self.find_coordinates_index(last_three, longitude, latitude)
            tensor = self.datasets[last_three].get_tensor_by_location(id_num, x, y)
            if tensor is not None:
                tensors.append(tensor)

        return longitude, latitude, torch.stack(tensors), oc
    
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

