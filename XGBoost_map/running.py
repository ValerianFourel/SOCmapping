import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset , MultiRasterDatasetMapping
from XGBoost_map.mapping import filter_dataframe , BandsYearly , create_prediction_visualizations , file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC , BandsYearly_1milPoints
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

##################################################################

YEAR = 2003
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

# Use the function
print(file_path)
df_original, top_years = get_top_sampling_years(file_path)

# Read the Excel file
df = pd.read_excel(file_path)

        # Get and display column names
columns = df.columns.tolist()

print("\nColumns in the dataset:")
for i, col in enumerate(columns, 1):
  print(f"{i}. {col}")

        # Print total number of columns
print(f"\nTotal number of columns: {len(columns)}")

df = filter_dataframe(YEAR)

# Loop to update variables dynamically
for i in range(len(BandsYearly)):
    BandsYearly[i] = BandsYearly[i] + f'/{YEAR}'

# Create dataset and dataloader
dataset = MultiRasterDataset(BandsYearly, df)
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

    # Concatenate all values in the batch_features dictionary
    concatenated_features = np.concatenate([value.numpy() for value in batch_features.values()], axis=1)
    # Flatten the features and filter
    flattened_features = concatenated_features.reshape(concatenated_features.shape[0], -1)[valid_mask]

    # Filter targets
    filtered_targets = batch_targets.numpy()[valid_mask]

    X_train.extend(flattened_features)
    y_train.extend(filtered_targets)


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


##################################

###########################################################
# Make this parallel
###########################################################
# for longitudes, latitudes, batch_features in dataloader:
#     # Store coordinates for plotting
#     coordinates.append(np.column_stack((longitudes.numpy(), latitudes.numpy())))

#     # Concatenate all values in the batch_features dictionary
#     concatenated_features = np.concatenate([value.numpy() for value in batch_features.values()], axis=1)
#     # Flatten the features
#     flattened_features = concatenated_features.reshape(concatenated_features.shape[0], -1)
#     X_train.extend(flattened_features)

# X_train = np.array(X_train)
# coordinates = np.vstack(coordinates)  # Stack all coordinates

# # Make predictions
# predictions = xgb_model.predict(X_train)

print('0')
import numpy as np
import concurrent.futures
from sklearn.utils import shuffle
import copy
import pandas as pd
from torch.utils.data import DataLoader, Subset
import torch

# Define the worker function
def process_batch(df_chunk, model_copy, bands_yearly, batch_size):
    # Create dataset and dataloader for this chunk
    chunk_dataset = MultiRasterDatasetMapping(bands_yearly, df_chunk)
    chunk_dataloader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

    chunk_coordinates = []
    chunk_features = []

    for longitudes, latitudes, batch_features in tqdm(
        chunk_dataloader,
        desc=f"Processing chunk of size {len(df_chunk)}",
        leave=False
    ):
        # Store coordinates for plotting
        chunk_coordinates.append(np.column_stack((longitudes.numpy(), latitudes.numpy())))

        # Concatenate all values in the batch_features dictionary
        concatenated_features = np.concatenate([value.numpy() for value in batch_features.values()], axis=1)
        # Flatten the features
        flattened_features = concatenated_features.reshape(concatenated_features.shape[0], -1)
        chunk_features.extend(flattened_features)

    # Convert to arrays
    chunk_features = np.array(chunk_features)
    chunk_coordinates = np.vstack(chunk_coordinates)

    # Make predictions using the model copy
    chunk_predictions = model_copy.predict(chunk_features)

    return chunk_coordinates, chunk_predictions

# Main function
def parallel_predict(df_full, xgb_model, bands_yearly, batch_size=4, num_threads=4):
    # Shuffle the DataFrame
    print('1')
    df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
    print('2')

    # Split DataFrame into chunks for each thread
    chunk_size = len(df_shuffled) // num_threads
    df_chunks = [df_shuffled[i:i + chunk_size] for i in range(0, len(df_shuffled), chunk_size)]

    # Ensure that predictions and coordinates match
    all_coordinates = []
    all_predictions = []

    # Use ThreadPoolExecutor for multithreading
    print(num_threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_batch, 
                chunk, 
                copy.deepcopy(xgb_model),
                bands_yearly,
                batch_size
            ) for chunk in df_chunks
        ]

        # Visualize progress with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Batches"):
            coordinates, predictions = future.result()
            all_coordinates.append(coordinates)
            all_predictions.append(predictions)

    # Combine results from all threads
    all_coordinates = np.vstack(all_coordinates)
    all_predictions = np.concatenate(all_predictions)

    return all_coordinates, all_predictions

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

# Loop to update variables dynamically
for i in range(len(BandsYearly_1milPoints)):
    BandsYearly_1milPoints[i] = BandsYearly_1milPoints[i] + f'/{YEAR}'

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
create_prediction_visualizations(YEAR, coordinates, predictions, save_path)
