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
from modelCNNMultiYear import Small3DCNN
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator


# Load CNN model with Accelerator
def load_cnn_model(model_path="/home/vfourel/SOCProject/SOCmapping/3DCNN/cnn_model_MAX_OC_160_TIME_BEGINNING_2007_TIME_END_2023_4epochs.pth"):
    accelerator = Accelerator()  # Initialize Accelerator
    device = accelerator.device  # Get the device (GPU or CPU) assigned to this process
    
    model = Small3DCNN(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Prepare model for distributed inference
    model = accelerator.prepare(model)
    
    return model, device, accelerator


def run_inference(model, dataloader, accelerator):
    model.eval()
    all_coordinates = []
    all_predictions = []

    with torch.no_grad():
        # Use accelerator.print for synchronized printing across processes
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitudes, latitudes, tensors = batch
            tensors = tensors.to(accelerator.device)  # Move to the assigned device

            # Run model inference
            outputs = model(tensors)  # Assuming model outputs a single value per sample
            predictions = outputs.cpu().numpy()

            # Store coordinates and predictions
            coords = np.stack([longitudes.numpy(), latitudes.numpy()], axis=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    # Gather results from all GPUs
    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

    # Use accelerator.gather to collect results from all processes
    all_coordinates = accelerator.gather(torch.tensor(all_coordinates, device=accelerator.device)).cpu().numpy()
    all_predictions = accelerator.gather(torch.tensor(all_predictions, device=accelerator.device)).cpu().numpy()

    return all_coordinates, all_predictions


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
    
    # Load the full inference dataframe (only on main process to avoid duplication)
    if accelerator.is_local_main_process:
        try:
            df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
            print("Loaded inference dataframe:")
            print(df_full.head())
        except Exception as e:
            print(f"Error loading inference dataframe: {e}")
            return
    else:
        df_full = None
    
    # Synchronize processes to ensure df_full is broadcasted or handled correctly
    df_full = accelerator.broadcast_object(df_full) if df_full is not None else pd.read_csv(file_path_coordinates_Bavaria_1mil)

    # Prepare data paths (can be done on all processes since it’s deterministic)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()
    samples_coordinates_array_path_1mil = flatten_paths(samples_coordinates_array_path_1mil)
    data_array_path_1mil = flatten_paths(data_array_path_1mil)

    samples_coordinates_array_path_1mil = list(dict.fromkeys(samples_coordinates_array_path_1mil))
    data_array_path_1mil = list(dict.fromkeys(data_array_path_1mil))

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

    # Load the CNN model
    cnn_model, device, accelerator = load_cnn_model()
    if accelerator.is_local_main_process:
        print("Loaded CNN model:")
        print(cnn_model)

    # Run inference
    coordinates, predictions = run_inference(cnn_model, inference_loader, accelerator)
    
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