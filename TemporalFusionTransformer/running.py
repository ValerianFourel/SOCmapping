import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size, MAX_OC,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded, NUM_HEADS, NUM_LAYERS,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,hidden_size,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import NormalizedMultiRasterDataset1MilMultiYears, RasterTensorDataset1Mil, MultiRasterDataset1MilMultiYears
from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from mapping import create_prediction_visualizations, parallel_predict
from SimpleTFT import SimpleTFT
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
import json
from datetime import datetime
import argparse

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
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

def load_SimpleTFT_model(model_path="/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/TFT_model_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_normalize_LOSS_l1.pth", accelerator=None):
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTFT(
        input_channels=len(bands_list_order),
        height=window_size,
        width=window_size,
        time_steps=time_before,
        d_model=hidden_size  # Adjust this based on the saved model
    )
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
        
        if accelerator.is_local_main_process:
            print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} parameters from {model_path}")
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading model weights: {e}")
        raise
    model.eval()
    if accelerator:
        model = accelerator.prepare(model)
    return model, device, accelerator

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def compute_training_statistics_oc():
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)

                
    
        # Calculate target statistics from balanced dataset
    target_mean = df_train['OC'].mean()
    target_std = df_train['OC'].std()

    
    return target_mean, target_std

def compute_training_statistics():
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    train_coords, train_data = separate_and_add_data()
    train_coords = list(dict.fromkeys(flatten_paths(train_coords)))
    train_data = list(dict.fromkeys(flatten_paths(train_data)))
    
    if not isinstance(df_train, pd.DataFrame):
        raise TypeError(f"Expected train_df to be a pandas DataFrame, got {type(df_train)}")
    
    train_dataset = NormalizedMultiRasterDatasetMultiYears(train_coords, train_data, df_train)

    feature_means, feature_stds = train_dataset.get_statistics()
    target_mean, target_std = compute_training_statistics_oc()
    print(f"Computed training statistics - Means: {feature_means}, Stds: {feature_stds}")
    
    expected_channels = len(bands_list_order)
    if feature_means.shape[0] != expected_channels:
        raise ValueError(f"Training feature_means has {feature_means.shape[0]} channels, but expected {expected_channels}")
    
    return target_mean, target_std , feature_means, feature_stds

def run_inference(model, dataloader, accelerator):
    model.eval()
    all_coordinates = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitudes, latitudes, tensors = batch
            tensors = tensors.to(accelerator.device)  # Move to the assigned device

            # Run model inference
            outputs = model(tensors)  # Assuming model outputs a single value per sample
            predictions = outputs.cpu().numpy()
            accelerator.print(predictions)

            # Store coordinates and predictions, ensuring tensors are moved to CPU
            coords = np.stack([longitudes.cpu().numpy(), latitudes.cpu().numpy()], axis=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    # Concatenate local results
    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

    # Gather results from all GPUs
    all_coordinates = accelerator.gather(torch.tensor(all_coordinates, device=accelerator.device)).cpu().numpy()
    all_predictions = accelerator.gather(torch.tensor(all_predictions, device=accelerator.device)).cpu().numpy()

    return all_coordinates, all_predictions

def apply_inverse_transform(predictions, target_transform, target_mean=None, target_std=None):
    """
    Apply inverse transformation to convert predictions and targets back to the original scale.

    Args:
        predictions (np.array): Model predictions in transformed scale.
        target_transform (str): Type of transformation applied ('log', 'normalize', or None).
        target_mean (float, optional): Mean used for normalization. Required if target_transform is 'normalize'.
        target_std (float, optional): Standard deviation used for normalization. Required if target_transform is 'normalize'.

    Returns:
        tuple: (original_val_outputs, ) - Predictions in original scale.
    """
    if target_transform == 'log':
        original_val_outputs = np.exp(predictions)
    elif target_transform == 'normalize':
        original_val_outputs = predictions * target_std + target_mean
    else:
        original_val_outputs = predictions

    return original_val_outputs

def main(target_transform):
    print(' FULL DATASET  ')
    target_mean, target_std ,feature_means, feature_stds = compute_training_statistics()
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Accelerated inference script with multi-GPU support")
    parser.add_argument("--model-path", type=str, 
                        default="/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/TFT_model_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_normalize_LOSS_l1.pth", 
                        help="Path to the trained model")
    args = parser.parse_args()
    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print(f"Loaded inference dataframe with {len(df_full)} samples")
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    samples_coords_1mil, data_1mil = separate_and_add_data_1mil_inference()
    samples_coords_1mil = list(dict.fromkeys(flatten_paths(samples_coords_1mil)))
    data_1mil = list(dict.fromkeys(flatten_paths(data_1mil)))
    
    third_size = len(df_full) // 4
    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path=samples_coords_1mil,
        data_array_path=data_1mil,
        df=df_full[:120000],
        feature_means=feature_means,
        feature_stds=feature_stds,
        time_before=time_before
    )
   # iference_dataset = MultiRasterDataset1MilMultiYears(samples_coordinates_array_subfolders=samples_coords_1mil, data_array_subfolders=data_1mil, dataframe=df_full[:2000])
    # Main code
    dataset_len = len(inference_dataset)
    if accelerator.is_local_main_process:
        print(f"Dataset length: {dataset_len}")
        if dataset_len != 2000:
            print(f"Warning: Dataset length is {dataset_len}, expected 2000. Check NormalizedMultiRasterDataset1MilMultiYears implementation.")

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    inference_loader = accelerator.prepare(inference_loader)
    model_path = "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/TFT_model_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_normalize_LOSS_l1.pth"
    model, device, accelerator = load_SimpleTFT_model(model_path, accelerator)
    if accelerator.is_local_main_process:
        print(f"Loaded SimpleTransformer model on {device}")

    coordinates, predictions = run_inference(model, inference_loader, accelerator)
    predictions = apply_inverse_transform(predictions, target_transform, target_mean=target_mean, target_std=target_std)
    np.save("coordinates_0k_to_120k.npy", coordinates)
    np.save("predictions_0k_to_120k.npy", predictions)
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
    target_transform = "normalize"
    main(target_transform)
