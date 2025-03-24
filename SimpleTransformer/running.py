import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded, NUM_HEADS, NUM_LAYERS,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import NormalizedMultiRasterDataset1MilMultiYears, RasterTensorDataset1Mil , MultiRasterDataset1MilMultiYears
from mapping import create_prediction_visualizations, parallel_predict
from modelSimpleTransformer import SimpleTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
import json
from datetime import datetime
import argparse

def load_cnn_model(model_path="/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/transformer_model_MAX_OC_160_TIME_BEGINNING_2007_TIME_END_2023_R2_0_5578.pth", accelerator=None):
    device = accelerator.device
    model = SimpleTransformer(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=0.3
    )
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Remove 'module.' prefix if present
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    
    # Load the adjusted state dictionary into the model
    model.load_state_dict(state_dict)
    model.eval()
    # Prepare the model for multi-GPU usage with Accelerate
    model = accelerator.prepare(model)
    return model, device

def compute_statistics(accelerator, samples_coordinates_array_path_1mil, data_array_path_1mil, dataframe, batch_size=128, num_workers=4, sample_fraction=0.05):
    """Compute mean and std across a sampled fraction of features using MultiRasterDataset1MilMultiYears."""
    def features_only_collate(batch):
        features = [item[2] for item in batch]  # Extract features from (longitude, latitude, features)
        return torch.stack(features)

    # Sample a fraction of the dataframe
    sample_size = max(1, int(len(dataframe) * sample_fraction))
    sampled_df = dataframe.sample(n=sample_size, random_state=42)  # Reproducible sampling
    if accelerator.is_local_main_process:
        print(f"Computing statistics on {sample_fraction*100}% of data ({sample_size} samples)")

    # Use the non-normalized dataset for statistics
    stats_dataset = MultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=sampled_df,
        time_before=time_before
    )
    
    stats_loader = DataLoader(
        stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=features_only_collate
    )
    stats_loader = accelerator.prepare(stats_loader)

    # Initialize accumulators on GPU for efficiency
    total_sum = torch.zeros((len(bands_list_order), time_before), device=accelerator.device)
    total_sum_squares = torch.zeros((len(bands_list_order), time_before), device=accelerator.device)
    total_count = 0

    # Accumulate statistics
    for batch in tqdm(stats_loader, desc="Computing Statistics", disable=not accelerator.is_local_main_process):
        features = batch.to(accelerator.device)
        B, bands, H, W, T = features.shape  # [batch, bands, height, width, time]
        batch_sum = features.sum(dim=(0, 2, 3))  # Sum over batch, height, width
        batch_sum_squares = (features ** 2).sum(dim=(0, 2, 3))
        total_sum += batch_sum
        total_sum_squares += batch_sum_squares
        total_count += B * H * W

    # Synchronize across processes
    total_sum = accelerator.reduce(total_sum, reduction='sum')
    total_sum_squares = accelerator.reduce(total_sum_squares, reduction='sum')
    total_count = accelerator.reduce(torch.tensor(total_count, device=accelerator.device), reduction='sum').item()

    # Compute mean and std on GPU, then move to CPU
    feature_means = total_sum / total_count
    feature_stds = torch.sqrt(torch.clamp((total_sum_squares / total_count) - (feature_means ** 2), min=0))
    feature_stds = torch.clamp(feature_stds, min=1e-8)  # Avoid division by zero

    # Move to CPU for use in dataset
    feature_means = feature_means.cpu()
    feature_stds = feature_stds.cpu()

    # Save statistics on main process
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = "stats_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"stats_{timestamp}.pt")
        metadata = {
            "timestamp": timestamp,
            "sample_fraction": sample_fraction,
            "sample_size": sample_size,
            "total_size": len(dataframe),
            "feature_means": feature_means.tolist(),
            "feature_stds": feature_stds.tolist()
        }
        torch.save({"metadata": metadata, "feature_means": feature_means, "feature_stds": feature_stds}, cache_file)
        print(f"Saved statistics to {cache_file}")

    return feature_means, feature_stds

def load_statistics(accelerator, stats_file):
    """Load statistics from a previous run."""
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Stats file {stats_file} not found")
    
    data = torch.load(stats_file)
    feature_means = data["feature_means"]
    feature_stds = data["feature_stds"]
    metadata = data["metadata"]
    
    if accelerator.is_local_main_process:
        print(f"Loaded statistics from {stats_file}:")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Sample fraction: {metadata['sample_fraction']*100}% ({metadata['sample_size']}/{metadata['total_size']} samples)")
    
    return feature_means, feature_stds

def run_inference(model, dataloader, accelerator):
    model.eval()
    all_coordinates = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitudes, latitudes, tensors = batch
            tensors = tensors.to(accelerator.device)
            outputs = model(tensors)
            predictions = outputs.cpu().numpy()
            coords = np.stack([longitudes.cpu().numpy(), latitudes.cpu().numpy()], axis=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])
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
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Run inference with optional precomputed statistics")
    parser.add_argument("--use-previous-stats", type=str, default=None, help="Path to previous stats file (e.g., stats_cache/stats_20230324_123456.pt)")
    args = parser.parse_args()

    accelerator = Accelerator()

    # Load inference dataframe
    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print("Loaded inference dataframe:")
            print(df_full.head())
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    # Prepare data paths
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()
    samples_coordinates_array_path_1mil = flatten_paths(samples_coordinates_array_path_1mil)
    data_array_path_1mil = flatten_paths(data_array_path_1mil)
    samples_coordinates_array_path_1mil = list(dict.fromkeys(samples_coordinates_array_path_1mil))
    data_array_path_1mil = list(dict.fromkeys(data_array_path_1mil))
    ###################
    # TMP: 
    #
    #samples_coordinates_array_path_1mil = samples_coordinates_array_path_1mil[:200]
    #data_array_path_1mil = data_array_path_1mil[:200]
    #df_full = df_full[:200]
    #########################
    if accelerator.is_local_main_process:
        print('Finished getting the 200 paths')

    # Load or compute statistics
    if args.use_previous_stats:
        try:
            feature_means, feature_stds = load_statistics(accelerator, args.use_previous_stats)
        except Exception as e:
            if accelerator.is_local_main_process:
                print(f"Failed to load previous stats: {e}. Computing new statistics.")
            feature_means, feature_stds = compute_statistics(
                accelerator,
                samples_coordinates_array_path_1mil,
                data_array_path_1mil,
                df_full,
                sample_fraction=0.01  # 1% of the dataset
            )
    else:
        feature_means, feature_stds = compute_statistics(
            accelerator,
            samples_coordinates_array_path_1mil,
            data_array_path_1mil,
            df_full,
            sample_fraction=0.05  # 5% of the dataset
        )
    print(feature_means, feature_stds)

    # Initialize normalized dataset
    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path=samples_coordinates_array_path_1mil,
        data_array_path=data_array_path_1mil,
        df=df_full,
        feature_means=feature_means,
        feature_stds=feature_stds,
        time_before=time_before
    )

    # Create DataLoader
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    inference_loader = accelerator.prepare(inference_loader)

    # Load the CNN model
    cnn_model, device = load_cnn_model(accelerator=accelerator)
    if accelerator.is_local_main_process:
        print("Loaded CNN model:")
        print(cnn_model)

    # Run inference
    coordinates, predictions = run_inference(cnn_model, inference_loader, accelerator)

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