import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe , separate_and_add_data
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import   TIME_BEGINNING ,TIME_END , seasons, years_padded  , SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally ,file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC 
from XGBoost_map.mapping import  create_prediction_visualizations , parallel_predict

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def modify_matrix_coordinates(MatrixCoordinates_1mil_Yearly=MatrixCoordinates_1mil_Yearly, MatrixCoordinates_1mil_Seasonally = MatrixCoordinates_1mil_Seasonally, TIME_END= TIME_END):
    # For MatrixCoordinates_1mil_Seasonally
    for path in MatrixCoordinates_1mil_Seasonally:
        folders = path.split('/')
        last_folder = folders[-1]

        if last_folder == 'Elevation':
            continue  # Skip Elevation folders
        elif last_folder == 'MODIS_NPP':
            # Add just the year (first 4 characters of TIME_END)
            new_path = f"{path}/{TIME_END[:4]}"
        else:
            # Add full TIME_END
            new_path = f"{path}/{TIME_END}"

        folders = path.split('/')
        folders.append(TIME_END if last_folder != 'MODIS_NPP' else TIME_END[:4])
        new_path = '/'.join(folders)
        MatrixCoordinates_1mil_Seasonally[MatrixCoordinates_1mil_Seasonally.index(path)] = new_path

    # For MatrixCoordinates_1mil_Yearly
    for path in MatrixCoordinates_1mil_Yearly:
        folders = path.split('/')
        last_folder = folders[-1]

        if last_folder == 'Elevation':
            continue  # Skip Elevation folders
        else:
            # Add just the year (first 4 characters of TIME_END)
            folders.append(TIME_END[:4])
            new_path = '/'.join(folders)
            MatrixCoordinates_1mil_Yearly[MatrixCoordinates_1mil_Yearly.index(path)] = new_path

    return MatrixCoordinates_1mil_Yearly, MatrixCoordinates_1mil_Seasonally



##################################################################

# Drawing the mapping

file_path = file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC

def get_top_sampling_years(file_path, top_n=3):
    """
    Read the Excel file and return the top n years with the most samples

    Parameters:
    file_path: str, path to the Excel file
    top_n: int, number of top years to return (default=3)
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Count samples per year and sort in descending order
        year_counts = df['year'].value_counts()
        top_years = year_counts.head(top_n)

        print(f"\nTop {top_n} years with the most samples:")
        for year, count in top_years.items():
            print(f"Year {year}: {count} samples")

        return df, top_years

    except Exception as e:
        print(f"Error reading file: {str(e)}")


df = filter_dataframe(TIME_BEGINNING,TIME_END,150)



# Loop to update variables dynamically
samples_coordinates_array_path ,  data_array_path = separate_and_add_data()
def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

samples_coordinates_array_path = flatten_paths(samples_coordinates_array_path)
data_array_path = flatten_paths(data_array_path)

# Remove duplicates
samples_coordinates_array_path = list(dict.fromkeys(samples_coordinates_array_path))
data_array_path = list(dict.fromkeys(data_array_path))


# Create dataset and dataloader
dataset = MultiRasterDataset(samples_coordinates_array_path ,  data_array_path , df)

print("Dataset length:", len(df))
# If using a custom dataset, verify the data is loaded correctly

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#############################


# Prepare data for XGBoost
X_train, y_train = [], []
coordinates = []  # To store longitude and latitude

for longitudes, latitudes, batch_features, batch_targets in dataloader:
    # Convert to numpy arrays for easier handling
    longs = longitudes.numpy()
    lats = latitudes.numpy()

    # Create mask for valid coordinates (not NaN)
    valid_mask = ~(np.isnan(longs) | np.isnan(lats))

    # Skip if all entries in batch are invalid
    if not np.any(valid_mask):
        continue

    # Filter coordinates and store only valid ones
    coordinates.append(np.column_stack((longs[valid_mask], lats[valid_mask])))

    # Handle batch_features (shape: [batch_size, 6, 17, 17])
    features_np = batch_features.numpy()  # Convert to numpy
    # Reshape to (batch_size, 6*17*17)
    flattened_features = features_np.reshape(features_np.shape[0], -1)
    # Apply mask to filter invalid coordinates
    filtered_features = flattened_features[valid_mask]

    # Filter targets
    filtered_targets = batch_targets.numpy()[valid_mask]

    X_train.extend(filtered_features)
    y_train.extend(filtered_targets)

# Convert lists to numpy arrays at the end
X_train = np.array(X_train)
y_train = np.array(y_train)
coordinates = np.vstack(coordinates)  # Stack all coordinates

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
print("XGBoost model trained successfully!")

# Make predictions
predictions = xgb_model.predict(X_train)

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1],
                     c=predictions,
                     cmap='viridis',  # You can change the colormap
                     alpha=0.6)
plt.colorbar(scatter, label='Predicted Values')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Values on Geographic Coordinates')
plt.grid(True)
plt.show()

save_path = '/home/vfourel/SOCProject/SOCmapping/plots'
# create_prediction_map(coordinates, predictions, save_path, filename='bavaria_predictions.png')

print('0')

# Example usage
# Define the file path
file_path = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"

# Load the CSV file into a DataFrame
try:
    df_full = pd.read_csv(file_path)
    df_full.head()  # Display the first few rows of the DataFrame
except Exception as e:
    print(e)

# Display the first few rows
print(df_full.head())

# df_full = df_full.iloc[::4]


BandsYearly_1milPoints, _ = modify_matrix_coordinates()


# Call the parallel prediction function
coordinates, predictions = parallel_predict(
    df_full=df_full,
    xgb_model=xgb_model,
    bands_yearly=BandsYearly_1milPoints,
    batch_size=8,
    num_threads=120
)

###########################################################
# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1],
                     c=predictions,
                     cmap='viridis',  # You can change the colormap
                     alpha=0.6)
plt.colorbar(scatter, label='Predicted Values')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Values on Geographic Coordinates')
plt.grid(True)
plt.show()

save_path = '/home/vfourel/SOCProject/SOCmapping/predictions_plots'
create_prediction_visualizations(TIME_END, coordinates, predictions, save_path)
