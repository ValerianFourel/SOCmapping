import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import MultiRasterDataset1MilMultiYears
from mapping import create_prediction_visualizations, parallel_predict
from models import RefittedCovLSTM
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator


# Load CNN-LSTM model with Accelerator
def load_cnn_model(model_path="/home/vfourel/SOCProject/SOCmapping/CNNLSTM/cnnlstm_model_MAX_OC_160_TIME_BEGINNING_2007_TIME_END_2023.pth"):
    accelerator = Accelerator()  # Initialize Accelerator
    device = accelerator.device  # Get the device (GPU or CPU) assigned to this process
    
    model = RefittedCovLSTM(
        num_channels=len(bands_list_order),  # Must match training (e.g., 6)
        lstm_input_size=128,  # Fixed by CNN output, must match training
        lstm_hidden_size=128,  # Must match training
        num_layers=2,         # Must match training
        dropout=0.25          # Must match training
    )
    
    # Load the state dict and strip 'module.' prefix if present
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        # Check if keys have 'module.' prefix and strip it
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading model weights: {e}")
        raise
    
    model.eval()
    
    # Prepare model for distributed inference
    model = accelerator.prepare(model)
    
    return model, device, accelerator

import os
from pathlib import Path

def run_inference(model, dataloader, accelerator, temp_dir="temp_results"):
    model.eval()
    all_coordinates = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitudes, latitudes, tensors = batch
            tensors = tensors.to(accelerator.device)  # Shape: (batch_size, bands, window_size, window_size, time)

            # Run model inference
            outputs = model(tensors)  # Assuming model outputs a single value per sample
            predictions = outputs
            coords = torch.stack([longitudes, latitudes], dim=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    # Concatenate locally on each process as tensors
    all_coordinates = torch.cat(all_coordinates, dim=0) if all_coordinates else torch.tensor([], device=accelerator.device)
    all_predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([], device=accelerator.device)

    # Convert to numpy on each process
    local_coordinates = all_coordinates.cpu().numpy()
    local_predictions = all_predictions.cpu().numpy()

    # Create temporary directory if it doesn't exist
    temp_dir_path = Path(temp_dir)
    if accelerator.is_local_main_process:
        temp_dir_path.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()  # Ensure directory is created before all processes proceed

    # Each process saves its results to a unique file
    process_rank = accelerator.process_index
    coord_file = temp_dir_path / f"coordinates_rank{process_rank}.npy"
    pred_file = temp_dir_path / f"predictions_rank{process_rank}.npy"
    np.save(coord_file, local_coordinates)
    np.save(pred_file, local_predictions)

    # Synchronize all processes to ensure all files are written
    accelerator.wait_for_everyone()

    # Only main process collects and concatenates results
    if accelerator.is_local_main_process:
        total_coordinates = []
        total_predictions = []
        num_processes = accelerator.num_processes

        # Read results from each process
        for rank in range(num_processes):
            coord_file = temp_dir_path / f"coordinates_rank{rank}.npy"
            pred_file = temp_dir_path / f"predictions_rank{rank}.npy"
            coords = np.load(coord_file)
            preds = np.load(pred_file)
            total_coordinates.append(coords)
            total_predictions.append(preds)

        # Concatenate all results
        total_coordinates = np.concatenate(total_coordinates, axis=0) if total_coordinates else np.array([])
        total_predictions = np.concatenate(total_predictions, axis=0) if total_predictions else np.array([])

        # Clean up temporary files
        for rank in range(num_processes):
            os.remove(temp_dir_path / f"coordinates_rank{rank}.npy")
            os.remove(temp_dir_path / f"predictions_rank{rank}.npy")
        os.rmdir(temp_dir_path)

        return total_coordinates, total_predictions
    else:
        return np.array([]), np.array([])  # Non-main processes return empty arrays

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened


def main():
    # Initialize Accelerator at the start of main
    accelerator = Accelerator()

    # Load the full inference dataframe independently on each process
    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print("Loaded inference dataframe:")
            print(df_full.head())
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    # Prepare data paths (deterministic, so can be done on all processes)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()
    samples_coordinates_array_path_1mil = flatten_paths(samples_coordinates_array_path_1mil)
    data_array_path_1mil = flatten_paths(data_array_path_1mil)

    samples_coordinates_array_path_1mil = list(dict.fromkeys(samples_coordinates_array_path_1mil))
    data_array_path_1mil = list(dict.fromkeys(data_array_path_1mil))
# Main code
# Initialize dataset
    inference_dataset = MultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=df_full,
        time_before=time_before
    )

    # Create DataLoader and prepare it for distributed inference
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    inference_loader = accelerator.prepare(inference_loader)

    # Load the CNN-LSTM model
    cnn_model, device, accelerator = load_cnn_model()
    if accelerator.is_local_main_process:
        print("Loaded CNN-LSTM model:")
        print(cnn_model)

    # Run inference
    coordinates, predictions = run_inference(cnn_model, inference_loader, accelerator)
    np.save("coordinates_1mil.npy", coordinates)
    np.save("predictions_1mil.npy", predictions)
    # Only the main process handles printing and visualization
    if accelerator.is_local_main_process:
        print(f"Inference completed. Coordinates shape: {coordinates.shape}, Predictions shape: {predictions.shape}")
        create_prediction_visualizations(
            INFERENCE_TIME,
            coordinates,
            predictions,
            save_path_predictions_plots
        )
        print(f"Visualizations saved to {save_path_predictions_plots}")

if __name__ == "__main__":
    main()