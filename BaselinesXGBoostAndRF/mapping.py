import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from config import (base_path_data, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, MAX_OC,
 TIME_BEGINNING, TIME_END, INFERENCE_TIME)
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
from sklearn.utils import shuffle
import copy
from torch.utils.data import DataLoader, Subset
import torch
import tqdm
import time
import threading
from collections import defaultdict
##################################################################

# Loading the Data

##################################################################

# Thread performance tracking
thread_stats = defaultdict(lambda: {'processed_items': 0, 'start_time': 0, 'end_time': 0})
thread_lock = threading.Lock()

def create_prediction_visualizations(year, coordinates, predictions, save_path):
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
        return f"{base_name}_MAX_OC_{str(MAX_OC)}_Beginning_{TIME_BEGINNING}_End_{TIME_END}__InferenceTime{INFERENCE_TIME}_{timestamp}.png"

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

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def process_batch(samples_coords_1mil, data_1mil, df_chunk, model_copy, batch_size, thread_id):
    # Start timing for this thread
    with thread_lock:
        thread_stats[thread_id]['start_time'] = time.time()
        thread_stats[thread_id]['processed_items'] = 0

    # Create dataset and dataloader for this chunk
    samples_coords_1mil = list(dict.fromkeys(flatten_paths(samples_coords_1mil)))
    data_1mil = list(dict.fromkeys(flatten_paths(data_1mil)))
    chunk_dataset = MultiRasterDatasetMapping(samples_coords_1mil, data_1mil, df_chunk)
    chunk_dataloader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

    chunk_coordinates = []
    chunk_features = []

    # Create progress bar for this thread
    batch_pbar = tqdm.tqdm(
        total=len(chunk_dataloader), 
        desc=f"Thread {thread_id}", 
        position=thread_id,
        leave=False
    )

    for longitudes, latitudes, features in chunk_dataloader:
        # Store coordinates for plotting
        chunk_coordinates.append(np.column_stack((longitudes.numpy(), latitudes.numpy())))

        # features is a tensor of shape (batch_size, num_bands, window_size, window_size, time_steps)
        # Convert to NumPy array
        features_np = features.numpy()

        # Flatten each sample's feature tensor
        # From (batch_size, num_bands, window_size, window_size, time_steps)
        # to (batch_size, num_bands * window_size * window_size * time_steps)
        flattened_features = features_np.reshape(features_np.shape[0], -1)
        chunk_features.append(flattened_features)

        # Update progress and stats
        batch_pbar.update(1)
        with thread_lock:
            thread_stats[thread_id]['processed_items'] += len(longitudes)

    # Convert to arrays
    chunk_features = np.vstack(chunk_features)
    chunk_coordinates = np.vstack(chunk_coordinates)

    # Make predictions using the model copy
    prediction_start = time.time()
    batch_pbar.set_description(f"Thread {thread_id}: Predicting")
    chunk_predictions = model_copy.predict(chunk_features)
    prediction_time = time.time() - prediction_start

    # Update final stats
    with thread_lock:
        thread_stats[thread_id]['end_time'] = time.time()
        thread_stats[thread_id]['prediction_time'] = prediction_time

    batch_pbar.close()
    return chunk_coordinates, chunk_predictions


def parallel_predict(df_full, model, batch_size=4, num_threads=4):
    # Start overall timing
    overall_start = time.time()

    print(f"Starting parallel prediction with {num_threads} threads and batch size {batch_size}")
    print(f"Total samples to process: {len(df_full)}")

    # Shuffle the DataFrame
    samples_coords_1mil, data_1mil = separate_and_add_data_1mil_inference()
    df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split DataFrame into chunks for each thread
    chunk_size = len(df_shuffled) // num_threads
    df_chunks = [df_shuffled[i:i + chunk_size] for i in range(0, len(df_shuffled), chunk_size)]

    print(f"Split data into {len(df_chunks)} chunks of approximately {chunk_size} samples each")

    # Ensure that predictions and coordinates match
    all_coordinates = []
    all_predictions = []

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create progress bar for overall tracking
        main_pbar = tqdm.tqdm(total=num_threads, desc="Overall Progress", position=0)

        # Submit tasks
        futures = []
        for i, chunk in enumerate(df_chunks):
            future = executor.submit(
                process_batch, 
                samples_coords_1mil, 
                data_1mil,
                chunk, 
                copy.deepcopy(model),
                batch_size,
                i  # Thread ID
            )
            futures.append(future)

        # Process results as they complete
        for future in as_completed(futures):
            coordinates, predictions = future.result()
            all_coordinates.append(coordinates)
            all_predictions.append(predictions)
            main_pbar.update(1)

        main_pbar.close()

    # Combine results from all threads
    all_coordinates = np.vstack(all_coordinates)
    all_predictions = np.concatenate(all_predictions)

    # Calculate overall time
    overall_time = time.time() - overall_start

    # Print performance statistics
    print("\n" + "="*60)
    print("PARALLEL PREDICTION PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total execution time: {overall_time:.2f} seconds")
    print(f"Total samples processed: {len(all_predictions)}")
    print(f"Processing rate: {len(all_predictions)/overall_time:.2f} samples/second")
    print("\nPer-Thread Statistics:")
    print("-"*60)
    print(f"{'Thread':^10}|{'Samples':^10}|{'Time (s)':^12}|{'Rate (samples/s)':^18}|{'Prediction Time (s)':^20}")
    print("-"*60)

    for thread_id, stats in thread_stats.items():
        thread_time = stats['end_time'] - stats['start_time']
        samples = stats['processed_items']
        rate = samples / thread_time if thread_time > 0 else 0
        pred_time = stats.get('prediction_time', 0)
        print(f"{thread_id:^10}|{samples:^10}|{thread_time:^12.2f}|{rate:^18.2f}|{pred_time:^20.2f}")

    print("="*60)

    return all_coordinates, all_predictions
