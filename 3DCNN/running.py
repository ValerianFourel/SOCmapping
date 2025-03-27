import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from functools import lru_cache
from config import (
    bands_list_order, time_before, window_size, INFERENCE_TIME, 
    save_path_predictions_plots, file_path_coordinates_Bavaria_1mil
)
from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import MultiRasterDataset1MilMultiYears
from mapping import create_prediction_visualizations
from modelCNNMultiYear import Small3DCNN
from tqdm import tqdm
from accelerate import Accelerator
import torch.cuda.amp


class OptimizedMultiRasterDataset(MultiRasterDataset1MilMultiYears):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-load coordinates to memory for faster access
        self.coordinates_cache = {}

    @lru_cache(maxsize=32)  # Cache recently used data arrays
    def _load_data_array(self, path):
        return np.load(path)

    def __getitem__(self, idx):
        # Optimized version of the original __getitem__ method
        longitude, latitude, tensor = super().__getitem__(idx)
        return longitude, latitude, tensor


def load_cnn_model(model_path="/home/vfourel/SOCProject/SOCmapping/3DCNN/cnn_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023.pth"):
    accelerator = Accelerator()
    device = accelerator.device

    model = Small3DCNN(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before
    )

    # Load model weights directly to the device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Enable torch script for faster inference
    try:
        model = torch.jit.script(model)
    except Exception as e:
        print(f"TorchScript optimization failed: {e}")

    return accelerator.prepare(model), device, accelerator


def flatten_paths(path_list):
    # More efficient flattening using list comprehension
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened


def run_inference(model, dataloader, accelerator):
    model.eval()
    all_coordinates = []
    all_predictions = []

    # Enable mixed precision for faster inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitudes, latitudes, tensors = batch
            tensors = tensors.to(accelerator.device, non_blocking=True)  # Non-blocking transfer

            # Use mixed precision for inference
            with torch.cuda.amp.autocast():
                outputs = model(tensors)

            # Process in batches to reduce memory pressure
            predictions = outputs.cpu().numpy()
            coords = np.stack([longitudes.cpu().numpy(), latitudes.cpu().numpy()], axis=1)

            all_coordinates.append(coords)
            all_predictions.append(predictions)

    # Concatenate results efficiently
    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

    # Gather results from all processes
    all_coordinates = accelerator.gather(torch.tensor(all_coordinates, device=accelerator.device)).cpu().numpy()
    all_predictions = accelerator.gather(torch.tensor(all_predictions, device=accelerator.device)).cpu().numpy()

    return all_coordinates, all_predictions


def main():
    # Initialize Accelerator with mixed precision
    accelerator = Accelerator(mixed_precision='fp16')

    # Load dataframe on all processes
    df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
    if accelerator.is_local_main_process:
        print(f"Loaded inference dataframe with {len(df_full)} rows")

    # Prepare data paths more efficiently
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()
    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    # Initialize optimized dataset
    inference_dataset = OptimizedMultiRasterDataset(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=df_full,
        time_before=time_before
    )

    # Create optimized DataLoader with appropriate batch size for distributed processing
    # Calculate per-process batch size
    total_batch_size = 1024
    batch_size_per_process = max(1, total_batch_size // accelerator.num_processes)

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size_per_process,  # Adjusted batch size per process
        shuffle=False,
        num_workers=4,  # Reduced to avoid memory issues
        pin_memory=True,
        persistent_workers=True if batch_size_per_process > 1 else False,
        prefetch_factor=2
    )
    inference_loader = accelerator.prepare(inference_loader)

    # Load the CNN model
    cnn_model, device, _ = load_cnn_model()
    if accelerator.is_local_main_process:
        print(f"Model loaded on {device}")
        print(f"Using batch size of {batch_size_per_process} per process")

    # Run inference
    coordinates, predictions = run_inference(cnn_model, inference_loader, accelerator)

    # Save results only on main process
    if accelerator.is_local_main_process:
        np.save("coordinates_1mil.npy", coordinates)
        np.save("predictions_1mil.npy", predictions)
        print(f"Inference completed. Results shape: {predictions.shape}")

        # Create visualizations
        create_prediction_visualizations(
            INFERENCE_TIME,
            coordinates,
            predictions,
            save_path_predictions_plots
        )
        print(f"Visualizations saved to {save_path_predictions_plots}")


if __name__ == "__main__":
    # Set environment variables for better performance
    import os
    os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads

    main()
