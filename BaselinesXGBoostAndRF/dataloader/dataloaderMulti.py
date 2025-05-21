

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import re
import glob
import pandas as pd
#from config import bands_list_order , time_before, LOADING_TIME_BEGINNING , window_size , LOADING_TIME_BEGINNING, TIME_BEGINNING ,TIME_END , INFERENCE_TIME, LOADING_TIME_BEGINNING_INFERENCE, seasons, years_padded  , SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally ,file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC 
from configElevationOnlyExperiment import bands_list_order , time_before, LOADING_TIME_BEGINNING , window_size , LOADING_TIME_BEGINNING, TIME_BEGINNING ,TIME_END , INFERENCE_TIME, LOADING_TIME_BEGINNING_INFERENCE, seasons, years_padded  , SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally ,file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC 

import pandas as pd
import numpy as np

def get_time_range(TIME_BEGINNING= TIME_BEGINNING, TIME_END=TIME_END, seasons=seasons, years_padded=years_padded):
    # Define seasons list for matching
    seasons_list = ['winter', 'spring', 'summer', 'autumn']

    # Check if TIME_BEGINNING is a season
    is_season = any(season in TIME_BEGINNING.lower() for season in seasons_list)

    if is_season:
        # Handle seasons case
        start_idx = next(i for i, season in enumerate(seasons) 
                        if TIME_BEGINNING.lower() in season.lower())
        end_idx = next(i for i, season in enumerate(seasons) 
                      if TIME_END.lower() in season.lower())

        # Get the subset including both start and end
        return seasons[start_idx:end_idx + 1]
    else:
        # Handle years case
        start_idx = years_padded.index(TIME_BEGINNING)
        end_idx = years_padded.index(TIME_END)

        # Get the subset including both start and end
        return years_padded[start_idx:end_idx + 1]


def process_paths_yearly(path, year, seen_years):
    if 'Elevation' in path:
        return path
    elif 'MODIS_NPP' in path:
        paths = []
        # Add current year
        if year not in seen_years:
            seen_years.add(year)
            paths.append(f"{path}/{year}")
        # Add previous year
        prev_year = str(int(year) - 1)
        if prev_year not in seen_years:
            seen_years.add(prev_year)
            paths.append(f"{path}/{prev_year}")
        return paths
    else:
        return f"{path}/{year}"

def create_path_arrays_yearly(SamplesCoordinates_Yearly, DataYearly, selected_years):
    seen_years_samples = set()
    seen_years_data = set()

    samples_coordinates_array_path = [
        processed_path
        for idx, base_path in enumerate(SamplesCoordinates_Yearly)
        for year in selected_years
        if idx < len(SamplesCoordinates_Yearly)
        if (processed_path := process_paths_yearly(base_path, year, seen_years_samples)) is not None
    ]

    data_yearly_array_path = [
        processed_path
        for idx, base_path in enumerate(DataYearly)
        for year in selected_years
        if idx < len(DataYearly)
        if (processed_path := process_paths_yearly(base_path, year, seen_years_data)) is not None
    ]

    return samples_coordinates_array_path, data_yearly_array_path


def process_paths(path, season, seen_years):
    if 'Elevation' in path:
        return path
    elif 'MODIS_NPP' in path:
        year = season.split('_')[0][:4]  # Get year from season
        paths = []
        # Add current year
        if year not in seen_years:
            seen_years.add(year)
            paths.append(f"{path}/{year}")
        # Add previous year
        prev_year = str(int(year) - 1)
        if prev_year not in seen_years:
            seen_years.add(prev_year)
            paths.append(f"{path}/{prev_year}")
        return paths
    else:
        return f"{path}/{season}"

def create_path_arrays(SamplesCoordinates_Seasonally, DataSeasonally, selected_seasons):
    seen_years_samples = set()
    seen_years_data = set()

    samples_coordinates_array_path = [
        processed_path
        for idx, base_path in enumerate(SamplesCoordinates_Seasonally)
        for season in selected_seasons
        if idx < len(SamplesCoordinates_Seasonally)
        if (processed_path := process_paths(base_path, season, seen_years_samples)) is not None
    ]

    data_seasons_array_path = [
        processed_path
        for idx, base_path in enumerate(DataSeasonally)
        for season in selected_seasons
        if idx < len(DataSeasonally)
        if (processed_path := process_paths(base_path, season, seen_years_data)) is not None
    ]
    
    return samples_coordinates_array_path, data_seasons_array_path



def separate_and_add_data(LOADING_TIME_BEGINNING=LOADING_TIME_BEGINNING, TIME_END=TIME_END, seasons=seasons, years_padded=years_padded, 
                         SamplesCoordinates_Yearly=SamplesCoordinates_Yearly, DataYearly=DataYearly,
                         SamplesCoordinates_Seasonally=SamplesCoordinates_Seasonally, DataSeasonally=DataSeasonally):

    # Define seasons list for matching
    seasons_list = ['winter', 'spring', 'summer', 'autumn']

    # Check if LOADING_TIME_BEGINNING is a season
    is_season = any(season in LOADING_TIME_BEGINNING.lower() for season in seasons_list)

    if is_season:
        # Handle seasons case
        start_idx = next(i for i, season in enumerate(seasons) 
                        if LOADING_TIME_BEGINNING.lower() in season.lower())
        end_idx = next(i for i, season in enumerate(seasons) 
                      if TIME_END.lower() in season.lower())

        # Get the seasonal range
        selected_seasons = seasons[start_idx:end_idx + 1]


        # Add seasonal data pairs
        return create_path_arrays(SamplesCoordinates_Seasonally, DataSeasonally, selected_seasons)
    else:
        start_idx = years_padded.index(LOADING_TIME_BEGINNING)
        end_idx = years_padded.index(TIME_END)
        selected_years = years_padded[start_idx:end_idx + 1]
        return create_path_arrays_yearly(SamplesCoordinates_Yearly, DataYearly, selected_years)

def add_season_column(dataframe):
    seasons_months = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }

    month_to_season = {
        month: season
        for season, months in seasons_months.items()
        for month in months
    }

    dataframe['survey_date'] = pd.to_datetime(dataframe['survey_date'])

    def get_season_year(row):
        if pd.isna(row['survey_date']):
            return None

        month = row['survey_date'].month
        year = row['survey_date'].year

        if month == 12:
            year += 1

        season = month_to_season.get(month)
        if season:
            return f"{year}_{season}"
        return None

    valid_dates_mask = dataframe['survey_date'] >= '2000-01-01'
    dataframe['season'] = None
    dataframe.loc[valid_dates_mask, 'season'] = (
        dataframe[valid_dates_mask].apply(get_season_year, axis=1)
    )

    return dataframe

def filter_dataframe(time_beginning, time_end, max_oc=150):
    # Read and prepare data
    df = pd.read_excel(file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)
    df = add_season_column(df)

    # Convert columns to numeric
    df['GPS_LONG'] = pd.to_numeric(df['GPS_LONG'], errors='coerce')
    df['GPS_LAT'] = pd.to_numeric(df['GPS_LAT'], errors='coerce')
    df['OC'] = pd.to_numeric(df['OC'], errors='coerce')

    # Basic data quality mask
    quality_mask = (
        (df['OC'] <= max_oc) &
        df['GPS_LONG'].notna() &
        df['GPS_LAT'].notna() &
        df['OC'].notna()
    )

    # Check if time_beginning contains a season
    seasons = ['winter', 'spring', 'summer', 'autumn']
    is_season = any(season in time_beginning.lower() for season in seasons)

    if is_season:
        # Create a list of all valid seasons between time_beginning and time_end
        start_year, start_season = time_beginning.split('_')
        end_year, end_season = time_end.split('_')
        start_year = int(start_year)
        end_year = int(end_year)

        valid_seasons = []
        current_year = start_year
        season_order = ['winter', 'spring', 'summer', 'autumn']
        start_idx = season_order.index(start_season)
        end_idx = season_order.index(end_season)

        while current_year <= end_year:
            if current_year == start_year:
                season_start = start_idx
            else:
                season_start = 0

            if current_year == end_year:
                season_end = end_idx
            else:
                season_end = len(season_order) - 1

            for season in season_order[season_start:season_end + 1]:
                valid_seasons.append(f"{current_year}_{season}")

            current_year += 1

        # Filter using the valid seasons list
        filtered_df = df[
            df['season'].isin(valid_seasons) &
            quality_mask
        ]
    else:
        # Filter by year range
        start_year = int(time_beginning)
        end_year = int(time_end)
        filtered_df = df[
            (df['year'].between(start_year, end_year, inclusive='both')) &
            quality_mask
        ]

    print(f"Initial shape: {df.shape}")
    print(f"Final filtered shape: {filtered_df.shape}")

    if filtered_df.empty:
        print("\nDebug information:")
        print("NaN counts:", df[['GPS_LONG', 'GPS_LAT', 'OC', 'survey_date']].isna().sum())
        print(f"OC range: {df['OC'].min()} to {df['OC'].max()}")

    return filtered_df


def separate_and_add_data_1mil_inference(LOADING_TIME_BEGINNING=LOADING_TIME_BEGINNING_INFERENCE, TIME_END=INFERENCE_TIME, seasons=seasons, years_padded=years_padded, 
                         SamplesCoordinates_Yearly=MatrixCoordinates_1mil_Yearly, DataYearly=DataYearly,
                         SamplesCoordinates_Seasonally=MatrixCoordinates_1mil_Seasonally, DataSeasonally=DataSeasonally):

    # Define seasons list for matching
    seasons_list = ['winter', 'spring', 'summer', 'autumn']

    # Check if LOADING_TIME_BEGINNING is a season
    is_season = any(season in LOADING_TIME_BEGINNING.lower() for season in seasons_list)

    if is_season:
        # Handle seasons case
        start_idx = next(i for i, season in enumerate(seasons) 
                        if LOADING_TIME_BEGINNING.lower() in season.lower())
        end_idx = next(i for i, season in enumerate(seasons) 
                      if TIME_END.lower() in season.lower())

        # Get the seasonal range
        selected_seasons = seasons[start_idx:end_idx + 1]


        # Add seasonal data pairs
        return create_path_arrays(SamplesCoordinates_Seasonally, DataSeasonally, selected_seasons)
    else:
        start_idx = years_padded.index(LOADING_TIME_BEGINNING)
        end_idx = years_padded.index(TIME_END)
        selected_years = years_padded[start_idx:end_idx + 1]
        return create_path_arrays_yearly(SamplesCoordinates_Yearly, DataYearly, selected_years)

def add_season_column(dataframe):
    seasons_months = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }

    month_to_season = {
        month: season
        for season, months in seasons_months.items()
        for month in months
    }

    dataframe['survey_date'] = pd.to_datetime(dataframe['survey_date'])

    def get_season_year(row):
        if pd.isna(row['survey_date']):
            return None

        month = row['survey_date'].month
        year = row['survey_date'].year

        if month == 12:
            year += 1

        season = month_to_season.get(month)
        if season:
            return f"{year}_{season}"
        return None

    valid_dates_mask = dataframe['survey_date'] >= '2000-01-01'
    dataframe['season'] = None
    dataframe.loc[valid_dates_mask, 'season'] = (
        dataframe[valid_dates_mask].apply(get_season_year, axis=1)
    )

    return dataframe


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
        subfolder: str, name of the subfolder (e.g., 'Elevation')
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
                id_to_file[id_num] = file_path

        return id_to_file

    def get_tensor_by_location(self, id_num, x, y, window_size=window_size):
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
        
        #window = np.asarray(window)
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


class MultiRasterDatasetMultiYears(Dataset):
    def __init__(self, samples_coordinates_array_subfolders ,  data_array_subfolders , dataframe, time_before = time_before):
        """
        Parameters:
        subfolders: list of str, names of subfolders to include
        dataframe: pandas.DataFrame, contains columns GPS_LONG, GPS_LAT, and OC (target variable)
        """
        self.data_array_subfolders = data_array_subfolders
        self.seasonalityBased = self.check_seasonality(data_array_subfolders)
        self.time_before = time_before
        self.samples_coordinates_array_subfolders = samples_coordinates_array_subfolders
        self.dataframe = dataframe
        self.datasets = {
            self.get_last_three_folders(subfolder): RasterTensorDataset(subfolder)
            for subfolder in  self.data_array_subfolders
        }
        
        self.coordinates = {
            self.get_last_three_folders(subfolder): np.load(f"{subfolder}/coordinates.npy")#[np.isfinite(np.load(f"{subfolder}/coordinates.npy"))]
            for subfolder in self.samples_coordinates_array_subfolders
        }

    def check_seasonality(self,data_array_subfolders):
        seasons = ['winter', 'spring', 'summer', 'autumn']

        # Check if any subfolder contains a season name
        is_seasonal = any(
            any(season in str(subfolder).lower() for season in seasons)
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
        tensors = []
        
        filtered_array = self.filter_by_season_or_year(row['season'],row['year'],self.seasonalityBased)
                # Initialize a dictionary to hold the tensors for each band
        band_tensors = {band: [] for band in bands_list_order}

        for subfolder in filtered_array:
            subfolder = self.get_last_three_folders(subfolder)

            # Check if the forelast subfolder is 'Elevation'
            if subfolder.split(os.path.sep)[-1] == 'Elevation':
                # Get the tensor for 'Elevation'
                id_num, x, y = self.find_coordinates_index(subfolder, longitude, latitude)
                elevation_tensor = self.datasets[subfolder].get_tensor_by_location(id_num, x, y)

                if elevation_tensor is not None:
                    # Repeat the 'Elevation' tensor self.time_before times
                    for _ in range(self.time_before):
                        band_tensors['Elevation'].append(elevation_tensor)
            else:
                # Get the year from the last subfolder
                year = int(subfolder.split(os.path.sep)[-1])

                # Decrement the year by self.time_before
                for decrement in range(self.time_before):
                    current_year = year - decrement

                    # Construct the subfolder with the decremented year
                    decremented_subfolder = os.path.sep.join(subfolder.split(os.path.sep)[:-1] + [str(current_year)])

                    id_num, x, y = self.find_coordinates_index(decremented_subfolder, longitude, latitude)
                    tensor = self.datasets[decremented_subfolder].get_tensor_by_location(id_num, x, y)

                    if tensor is not None:
                        # Append the tensor to the corresponding band in the dictionary
                        band = subfolder.split(os.path.sep)[-2]
                        band_tensors[band].append(tensor)

        # Stack the tensors for each band
        stacked_tensors = []
        for band in bands_list_order:
            if band_tensors[band]:
                # Stack the tensors for the current band
                stacked_tensor = torch.stack(band_tensors[band])
                stacked_tensors.append(stacked_tensor)

        # Concatenate all stacked tensors along the band dimension
        final_tensor = torch.stack(stacked_tensors)
        final_tensor = final_tensor.permute(0, 2, 3, 1)

        return longitude, latitude, final_tensor, oc
       

    
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
