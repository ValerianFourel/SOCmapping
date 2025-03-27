import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size,MAX_OC,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded, NUM_HEADS, NUM_LAYERS,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import NormalizedMultiRasterDataset1MilMultiYears, RasterTensorDataset1Mil , MultiRasterDataset1MilMultiYears
from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from mapping import create_prediction_visualizations, parallel_predict
from modelSimpleTransformer import SimpleTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
import json
from datetime import datetime
import argparse
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse


def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
    """
    Create a balanced dataset by binning OC values and resampling.
    If use_validation is True, splits into training and validation sets.
    If use_validation is False, returns only a balanced training set.
    """
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    training_dfs = []
    
    if use_validation:
        validation_indices = []
        
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) >= 4:
                val_samples = bin_data.sample(n=min(8, len(bin_data)))
                validation_indices.extend(val_samples.index)
                train_samples = bin_data.drop(val_samples.index)
                if len(train_samples) > 0:
                    if len(train_samples) < min_samples:
                        resampled = train_samples.sample(n=min_samples, replace=True)
                        training_dfs.append(resampled)
                    else:
                        training_dfs.append(train_samples)
        
        if not training_dfs or not validation_indices:
            raise ValueError("No training or validation data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        validation_df = df.loc[validation_indices].drop('bin', axis=1)
        return training_df, validation_df
    
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        
        if not training_dfs:
            raise ValueError("No training data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        return training_df


def load_SimpleTransformer_model(model_path="/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1_0000.pth", accelerator=None):
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



def compute_training_statistics():
    """Compute feature statistics from training dataset before Accelerate initialization"""
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    train_coords, train_data = separate_and_add_data()
    train_coords = list(dict.fromkeys(flatten_paths(train_coords)))
    train_data = list(dict.fromkeys(flatten_paths(train_data)))
    train_df = create_balanced_dataset(df_train)
    
    # Initialize training dataset and compute statistics
    train_dataset = NormalizedMultiRasterDatasetMultiYears(train_coords, train_data, train_df)
    feature_means, feature_stds = train_dataset.getStatistics()
    
    print(f"Computed training statistics - Means: {feature_means}, Stds: {feature_stds}")
    
    # Verify channel consistency
    expected_channels = len(bands_list_order)
    if feature_means.shape[0] != expected_channels:
        raise ValueError(f"Training feature_means has {feature_means.shape[0]} channels, but expected {expected_channels}")
    
    return feature_means, feature_stds

def main():
    # Compute training statistics before Accelerate
    feature_means, feature_stds = compute_training_statistics()
    
    # Initialize Accelerator after statistics computation
    accelerator = Accelerator()
    
    parser = argparse.ArgumentParser(description="Accelerated inference script with multi-GPU support")
    parser.add_argument("--model-path", type=str, 
                       default="/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1_0000.pth", 
                       help="Path to the trained model")
    args = parser.parse_args()

    # Load inference dataframe only on main process initially
    if accelerator.is_main_process:
        try:
            df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
            print(f"Loaded inference dataframe with {len(df_full)} samples")
        except Exception as e:
            print(f"Error loading inference dataframe: {e}")
            return
    
    # Wait for all processes to sync
    accelerator.wait_for_everyone()

    # Prepare inference data paths (done on all processes)
    samples_coords_1mil, data_1mil = separate_and_add_data_1mil_inference()
    samples_coords_1mil = list(dict.fromkeys(flatten_paths(samples_coords_1mil)))
    data_1mil = list(dict.fromkeys(flatten_paths(data_1mil)))

    # Load df_full on other processes after sync
    if not accelerator.is_main_process:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)

    # Initialize inference dataset with pre-computed statistics
    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path=samples_coords_1mil,
        data_array_path=data_1mil,
        df=df_full,
        feature_means=feature_means,
        feature_stds=feature_stds
    )

    # Create DataLoader with DistributedSampler
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            inference_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index
        )
    )

    # Load and prepare model
    model = load_SimpleTransformer_model(args.model_path)
    model = accelerator.prepare(model)
    inference_loader = accelerator.prepare(inference_loader)
    
    print(f"Loaded model on {accelerator.device} (Process {accelerator.process_index})")

    # Run inference
    coordinates, predictions = run_inference(model, inference_loader)
    
    # Gather results from all processes
    all_coordinates = accelerator.gather(coordinates)
    all_predictions = accelerator.gather(predictions)
    
    # Save results and create visualizations only on main process
    if accelerator.is_main_process:
        # Convert to numpy arrays if needed
        all_coordinates = np.concatenate(all_coordinates, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        np.save("coordinates_1mil.npy", all_coordinates)
        np.save("predictions_1mil.npy", all_predictions)
        print(f"Inference completed: Coordinates shape: {all_coordinates.shape}, Predictions shape: {all_predictions.shape}")
        
        create_prediction_visualizations(INFERENCE_TIME, all_coordinates, all_predictions, save_path_predictions_plots)
        print(f"Visualizations saved to {save_path_predictions_plots}")

if __name__ == "__main__":
    main()  # Call main directly; use 'accelerate launch' from command line