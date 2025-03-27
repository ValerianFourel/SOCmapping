
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob


class RasterTensorDatasetMapping(Dataset):
    def __init__(self, base_path):
        """
        Initialize the dataset

        Parameters:
        base_path: str, base path to RasterTensorData directory
        subfolder: str, name of the subfolder (e.g., 'Elevation')
        """
        # Replace "OC_LUCAS_LFU_LfL_Coordinates" with "RasterTensorData" in the base path
        self.base_path = Path(base_path.replace("Coordinates1Mil", "RasterTensorData"))
        self.folder_path = self.base_path

        # Create ID to filename mapping
        self.id_to_file = self._create_id_mapping()

        # Load all numpy arrays into memory (optional, can be modified to load on demand)
        self.data_cache = {}
        for id_num, filepath in self.id_to_file.items():
            self.data_cache[id_num] = np.load(filepath)

    def _create_id_mapping(self):
        """Create a dictionary mapping IDs to their corresponding file paths"""
        id_to_file = {}

        for file_path in self.folder_path.glob("*.npy"):
            # Extract ID number from filename
            match = re.search(r'ID(\d+)N', file_path.name)
            if match:
                id_num = int(match.group(1))
                id_to_file[id_num] = file_path

        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=17):
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


from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
import os

class MultiRasterDatasetMapping(Dataset):
    def __init__(self, subfolders, dataframe, bands_list_order=None):
        """
        Parameters:
        subfolders: list of str, paths to directories containing raster data and coordinates.npy (e.g., /path/to/LAI/2015)
        dataframe: pandas.DataFrame, contains columns 'longitude' and 'latitude'
        bands_list_order: list of str, order of bands (e.g., ['Elevation', 'LAI', ...])
        """
        self.subfolders = subfolders
        self.dataframe = dataframe
        self.bands_list_order = bands_list_order if bands_list_order is not None else []
        self.datasets = {}
        self.coordinates = {}

        # Initialize datasets and coordinates with error checking
        for subfolder in subfolders:
            coord_file = f"{subfolder}/coordinates.npy"
            if not Path(coord_file).exists():
                raise FileNotFoundError(f"Coordinates file not found: {coord_file}")
            self.coordinates[subfolder] = np.load(coord_file)
            self.datasets[subfolder] = RasterTensorDatasetMapping(subfolder)  # Assuming this class exists

    def find_coordinates_index(self, subfolder, longitude, latitude):
        """
        Finds the index of the matching coordinates in the subfolder's coordinates.npy file.

        Parameters:
        subfolder: str, path to the subfolder (e.g., /path/to/LAI/2015)
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

    def filter_by_year(self, inference_year):
        """
        Filter subfolders for a single inference year.

        Parameters:
        inference_year: str, the year for inference (e.g., '2015')

        Returns:
        dict: {band: subfolder path for the specified year}
        """
        filtered_paths = {band: None for band in self.bands_list_order}
        for subfolder in self.subfolders:
            parts = subfolder.split('/')
            band = parts[-2] if 'Elevation' not in subfolder else 'Elevation'
            subfolder_year = parts[-1] if 'Elevation' not in subfolder else None

            if band in self.bands_list_order:
                if band == 'Elevation':
                    filtered_paths['Elevation'] = subfolder
                elif subfolder_year == inference_year:
                    filtered_paths[band] = subfolder
        return filtered_paths

    def __getitem__(self, index):
        """
        Retrieve tensor and coordinates for a given index for a single year.

        Parameters:
        index: int, index of the row in the dataframe

        Returns:
        tuple: (tensor, coordinates), where tensor is (num_bands, height, width)
               and coordinates are (longitude, latitude)
        """
        row = self.dataframe.iloc[index]
        longitude, latitude = row["longitude"], row["latitude"]
        inference_year = "2015"  # Hardcoded to INFERENCE_TIME; adjust if needed in dataframe

        # Filter subfolders by year
        filtered_paths = self.filter_by_year(inference_year)

        # Initialize band tensors
        band_tensors = []
        window_size = 33  # From your config

        # Collect tensors for each band for the single year
        for band in self.bands_list_order:
            subfolder = filtered_paths[band]
            if subfolder:
                id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
                tensor = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)
                band_tensors.append(tensor if tensor is not None else torch.zeros(window_size, window_size))
            else:
                band_tensors.append(torch.zeros(window_size, window_size))

        if len(band_tensors) != len(self.bands_list_order):
            raise ValueError(f"Expected {len(self.bands_list_order)} bands, but got {len(band_tensors)}")

        final_tensor = torch.stack(band_tensors)  # Shape: (num_bands, height, width)
        return final_tensor, torch.tensor([longitude, latitude], dtype=torch.float32)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        """Get tensor from specific subfolder dataset"""
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)