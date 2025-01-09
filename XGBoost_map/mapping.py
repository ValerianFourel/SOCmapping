import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from config import base_path_data , file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from sklearn.utils import shuffle
import copy
from torch.utils.data import DataLoader, Subset
import torch
import tqdm
##################################################################

# Loading the Data



##################################################################


def create_prediction_visualizations(year,coordinates, predictions, save_path):
    """
    Create and save three separate map visualizations of predictions in Bavaria plus a triptych,
    with timestamps in filenames.

    Parameters:
    coordinates (numpy.array): Array of coordinates (longitude, latitude)
    predictions (numpy.array): Array of prediction values
    save_path (str): Directory path where the images should be saved
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import geopandas as gpd
    from datetime import datetime

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    individual_path = os.path.join(save_path, 'individual')
    os.makedirs(individual_path, exist_ok=True)

    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']

    # Create interpolation grid with higher resolution
    grid_x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 300)
    grid_y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 300)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate values with cubic interpolation
    grid_z = griddata(coordinates, predictions, (grid_x, grid_y), method='linear')

    # Common plotting parameters
    plot_params = {
        'figsize': (12, 10),
        'dpi': 300
    }

    # Function to set common elements for all plots
    def set_common_elements(ax, title):
        bavaria.boundary.plot(ax=ax, color='black', linewidth=1)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

    # Function to generate filename with timestamp
    def get_filename(base_name):
        return f"{base_name}_{timestamp}.png"

    # 1. Interpolated surface
    fig_interp, ax_interp = plt.subplots(**plot_params)
    contour = ax_interp.contourf(grid_x, grid_y, grid_z,
                                levels=50,
                                cmap='viridis',
                                alpha=0.8)
    set_common_elements(ax_interp, 'Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax_interp, label='Predicted Values')
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_interpolated')), 
                bbox_inches='tight')
    plt.close()

    # 2. Scatter plot
    fig_scatter, ax_scatter = plt.subplots(**plot_params)
    scatter = ax_scatter.scatter(coordinates[:, 0], coordinates[:, 1],
                               c=predictions,
                               cmap='viridis',
                               alpha=0.6,
                               s=50)
    set_common_elements(ax_scatter, 'Scatter Plot of Predicted Values')
    plt.colorbar(scatter, ax=ax_scatter, label='Predicted Values')
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_scatter')), 
                bbox_inches='tight')
    plt.close()

    # 3. Discrete points
    fig_discrete, ax_discrete = plt.subplots(**plot_params)
    discrete = ax_discrete.scatter(coordinates[:, 0], coordinates[:, 1],
                                 c=predictions,
                                 cmap='viridis',
                                 alpha=1.0,
                                 s=20)
    set_common_elements(ax_discrete, 'Discrete Points of Predicted Values')
    plt.colorbar(discrete, ax=ax_discrete, label='Predicted Values')
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_discrete')), 
                bbox_inches='tight')
    plt.close()

    # Create triptych
    fig_triptych = plt.figure(figsize=(30, 10))

    # Interpolated plot
    ax1 = plt.subplot(131)
    contour = ax1.contourf(grid_x, grid_y, grid_z,
                          levels=50,
                          cmap='viridis',
                          alpha=0.8)
    set_common_elements(ax1, 'Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax1, label='Predicted Values')

    # Scatter plot
    ax2 = plt.subplot(132)
    scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=predictions,
                         cmap='viridis',
                         alpha=0.6,
                         s=50)
    set_common_elements(ax2, 'Scatter Plot of Predicted Values')
    plt.colorbar(scatter, ax=ax2, label='Predicted Values')

    # Discrete points
    ax3 = plt.subplot(133)
    discrete = ax3.scatter(coordinates[:, 0], coordinates[:, 1],
                          c=predictions,
                          cmap='viridis',
                          alpha=1.0,
                          s=20)
    set_common_elements(ax3, 'Discrete Points of Predicted Values')
    plt.colorbar(discrete, ax=ax3, label='Predicted Values')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, get_filename(f'{year}_bavaria_triptych')), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Example usage:
# create_prediction_map(bavaria, coordinates, predictions, save_path='output/maps')




# Define the worker function
def process_batch(df_chunk, model_copy, bands_yearly, batch_size):
    # Create dataset and dataloader for this chunk
    chunk_dataset = MultiRasterDatasetMapping(bands_yearly, df_chunk)
    chunk_dataloader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

    chunk_coordinates = []
    chunk_features = []

    for longitudes, latitudes, batch_features in chunk_dataloader:
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

def parallel_predict(df_full, xgb_model, bands_yearly, batch_size=4, num_threads=4):
    # Shuffle the DataFrame
    df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

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
        for future in futures:
                coordinates, predictions = future.result()
                all_coordinates.append(coordinates)
                all_predictions.append(predictions)
    # Combine results from all threads
    all_coordinates = np.vstack(all_coordinates)
    all_predictions = np.concatenate(all_predictions)

    return all_coordinates, all_predictions