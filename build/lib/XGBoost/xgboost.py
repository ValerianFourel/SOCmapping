
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np



class MultiRasterDataset(Dataset):
    def __init__(self, subfolders, dataframe):
        """
        Parameters:
        subfolders: list of str, names of subfolders to include
        dataframe: pandas.DataFrame, contains columns GPS_LONG, GPS_LAT, and OC (target variable)
        """
        self.subfolders = subfolders
        self.dataframe = dataframe
        self.datasets = {
            subfolder: RasterTensorDataset(subfolder)
            for subfolder in subfolders
        }
        self.coordinates = {
            subfolder: np.load(f"{subfolder}/coordinates.npy")
            for subfolder in subfolders
        }

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
            raise ValueError(f"Coordinates ({longitude}, {latitude}) not found in {subfolder}")

        # Return id_num, x, y from the same row
        return coords[match[0], 2], coords[match[0], 3], coords[match[0], 4]

    def __getitem__(self, index):
        """
        Retrieve tensor and target value for a given index.

        Parameters:
        index: int, index of the row in the dataframe

        Returns:
        tuple: (tensor, OC), where tensor is the data and OC is the target variable
        """
        row = self.dataframe.iloc[index]
        longitude, latitude, oc = row["GPS_LONG"], row["GPS_LAT"], row["OC"]

        tensors = {}
        for subfolder in self.subfolders:
            id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
            tensors[subfolder] = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

        return longitude, latitude, tensors, oc

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataframe)

    def get_tensor_by_location(self, subfolder, id_num, x, y):
        """Get tensor from specific subfolder dataset"""
        return self.datasets[subfolder].get_tensor_by_location(id_num, x, y)
