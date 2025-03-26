import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

import os

from datetime import datetime
from scipy.interpolate import griddata
import geopandas as gpd
from config import (base_path_data , file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC , MAX_OC , PICTURE_VERSION,
                TIME_BEGINNING , TIME_END , INFERENCE_TIME)
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataloader.dataloaderMapping import MultiRasterDataset1MilMultiYears
from sklearn.utils import shuffle
import copy
from torch.utils.data import DataLoader, Subset
import torch
from accelerate import Accelerator

from tqdm import tqdm

##################################################################

# Loading the Data

VMIN = 0
VMAX = 190

##################################################################



def create_prediction_visualizations(year, coordinates, predictions, save_path, vmin=VMIN, vmax=VMAX):
    """
    Create and save three separate map visualizations of predictions in Bavaria plus a triptych,
    with timestamps in filenames and unified scales.

    Parameters:
    year (int): Year of the predictions
    coordinates (numpy.array): Array of coordinates (longitude, latitude)
    predictions (numpy.array): Array of prediction values
    save_path (str): Directory path where the images should be saved
    vmin (float, optional): Minimum value for the color scale (default: min of predictions)
    vmax (float, optional): Maximum value for the color scale (default: max of predictions or MAX_OC)
    """
    # Set global min/max if not provided
    if vmin is None:
        vmin = np.min(predictions)  # Or set a fixed value, e.g., 0
    if vmax is None:
        vmax = min(np.max(predictions), MAX_OC)  # Cap at MAX_OC from config or use a fixed value

    # Define fixed levels for contour plot (e.g., 50 levels between vmin and vmax)
    levels = np.linspace(vmin, vmax, 51)  # 51 to get 50 intervals

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

    # Interpolate values with linear interpolation
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def get_filename(base_name):
        return f"{PICTURE_VERSION}_{base_name}_MAX_OC_{str(MAX_OC)}_Beginning_{TIME_BEGINNING}_End_{TIME_END}__InferenceTime{INFERENCE_TIME}_{timestamp}.png"

    # Define common colormap
    cmap = 'viridis'

    # 1. Interpolated surface
    fig_interp, ax_interp = plt.subplots(**plot_params)
    contour = ax_interp.contourf(grid_x, grid_y, grid_z,
                                 levels=levels,
                                 cmap=cmap,
                                 alpha=0.8,
                                 vmin=vmin,
                                 vmax=vmax)
    set_common_elements(ax_interp, 'Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax_interp, label='Predicted Values', 
                 ticks=np.linspace(vmin, vmax, 11))  # Consistent ticks
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_interpolated_{PICTURE_VERSION}')), 
                bbox_inches='tight')
    plt.close()

    # 2. Scatter plot
    fig_scatter, ax_scatter = plt.subplots(**plot_params)
    scatter = ax_scatter.scatter(coordinates[:, 0], coordinates[:, 1],
                                 c=predictions,
                                 cmap=cmap,
                                 alpha=0.6,
                                 s=50,
                                 vmin=vmin,
                                 vmax=vmax)
    set_common_elements(ax_scatter, 'Scatter Plot of Predicted Values')
    plt.colorbar(scatter, ax=ax_scatter, label='Predicted Values', 
                 ticks=np.linspace(vmin, vmax, 11))
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_scatter_{PICTURE_VERSION}')), 
                bbox_inches='tight')
    plt.close()

    # 3. Discrete points
    fig_discrete, ax_discrete = plt.subplots(**plot_params)
    discrete = ax_discrete.scatter(coordinates[:, 0], coordinates[:, 1],
                                   c=predictions,
                                   cmap=cmap,
                                   alpha=1.0,
                                   s=20,
                                   vmin=vmin,
                                   vmax=vmax)
    set_common_elements(ax_discrete, 'Discrete Points of Predicted Values')
    plt.colorbar(discrete, ax=ax_discrete, label='Predicted Values', 
                 ticks=np.linspace(vmin, vmax, 11))
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_discrete_{PICTURE_VERSION}')), 
                bbox_inches='tight')
    plt.close()

    # Create triptych
    fig_triptych = plt.figure(figsize=(30, 10))

    # Interpolated plot
    ax1 = plt.subplot(131)
    contour = ax1.contourf(grid_x, grid_y, grid_z,
                           levels=levels,
                           cmap=cmap,
                           alpha=0.8,
                           vmin=vmin,
                           vmax=vmax)
    set_common_elements(ax1, 'Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax1, label='Predicted Values', 
                 ticks=np.linspace(vmin, vmax, 11))

    # Scatter plot
    ax2 = plt.subplot(132)
    scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 1],
                          c=predictions,
                          cmap=cmap,
                          alpha=0.6,
                          s=50,
                          vmin=vmin,
                          vmax=vmax)
    set_common_elements(ax2, 'Scatter Plot of Predicted Values')
    plt.colorbar(scatter, ax=ax2, label='Predicted Values', 
                 ticks=np.linspace(vmin, vmax, 11))

    # Discrete points
    ax3 = plt.subplot(133)
    discrete = ax3.scatter(coordinates[:, 0], coordinates[:, 1],
                           c=predictions,
                           cmap=cmap,
                           alpha=1.0,
                           s=20,
                           vmin=vmin,
                           vmax=vmax)
    set_common_elements(ax3, 'Discrete Points of Predicted Values')
    plt.colorbar(discrete, ax=ax3, label='Predicted Values', 
                 ticks=np.linspace(vmin, vmax, 11))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, get_filename(f'{year}_bavaria_triptych')), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def process_batch(df_chunk, model, bands_yearly, batch_size, device):
    model = model.to(device)
    chunk_dataset = MultiRasterDataset1MilMultiYears(bands_yearly, df_chunk)
    chunk_dataloader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)
    
    chunk_coordinates = []
    chunk_predictions = []
    
    model.eval()
    with torch.no_grad():
        for longitudes, latitudes, batch_features in chunk_dataloader:
            # Store coordinates
            chunk_coordinates.append(np.column_stack((longitudes.numpy(), latitudes.numpy())))
            
            # Move features to GPU
            # features_gpu = {k: v.to(device) for k, v in batch_features.items()}
            
            # Run inference
            predictions = model(batch_features).cpu().numpy()
            chunk_predictions.extend(predictions)
    
    return np.vstack(chunk_coordinates), np.array(chunk_predictions)

def parallel_predict(df_full, cnn_model, bands_yearly, batch_size=256):
    accelerator = Accelerator()
    print(bands_yearly)
    dataset = MultiRasterDataset1MilMultiYears(bands_yearly, df_full)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cnn_model, dataloader = accelerator.prepare(cnn_model, dataloader)
    cnn_model.eval()

    coordinates, predictions = [], []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Predicting', leave=True)
        for longitudes, latitudes, features in progress_bar:
            coordinates.append(np.column_stack((longitudes.cpu().numpy(), latitudes.cpu().numpy())))
            # Convert dictionary of features into stacked tensor
            features_stacked = torch.stack(list(features.values()), dim=1)
            features_stacked = features_stacked.to(accelerator.device)
            batch_preds = cnn_model(features_stacked).cpu().numpy()
            predictions.extend(batch_preds)

    return np.vstack(coordinates), np.array(predictions)
