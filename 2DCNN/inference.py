# -*- coding: utf-8 -*-
"""
Script for performing inference with a trained ResNet2DCNN model for Soil Organic Carbon mapping.
Uses PyTorch, Accelerate for distributed inference, and WandB for logging.
Saves results and creates visualizations on the main process using NormalizedMultiRasterDataset1MilMultiYears.
"""

import argparse
import os
from typing import List, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# --- Configuration Imports (Assuming these exist) ---
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
    seasons, years_padded, bands_list_order, window_size, time_before,
    file_path_coordinates_Bavaria_1mil
)
from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from dataloader.dataloaderMapping import NormalizedMultiRasterDataset1MilMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference
from model2DCNN import ResNet2DCNN
from balancedDataset import create_balanced_dataset
from mapping import create_prediction_visualizations

# --- Constants ---
BATCH_SIZE = 1024  # Increased for inference efficiency
NUM_WORKERS = 10
OUTPUT_DIR = "/home/vfourel/SOCProject/SOCmapping/2DCNN/output"
VISUALIZATION_DIR = "/home/vfourel/SOCProject/SOCmapping/2DCNN/output/prediction_plots"

# --- Helper Functions ---

def flatten_paths(path_list: List[Union[str, List]]) -> List[str]:
    """Flattens a nested list of paths."""
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        elif item is not None:
            flattened.append(item)
    return flattened

def inverse_transform_target(outputs: np.ndarray, transform_type: str, mean: Optional[float] = None, std: Optional[float] = None) -> np.ndarray:
    """Applies inverse transformation to model outputs (numpy array)."""
    if transform_type == 'log':
        outputs_clipped = np.clip(outputs, -50, 50)
        return np.exp(outputs_clipped)
    elif transform_type == 'normalize':
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for inverse normalization.")
        return outputs * std + mean
    elif transform_type == 'none':
        return outputs
    else:
        raise ValueError(f"Unknown target transformation: {transform_type}")

def compute_training_statistics_oc():
    """Computes mean and std of target 'OC' from balanced training dataset."""
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    train_coords, train_data = separate_and_add_data()
    train_coords = list(dict.fromkeys(flatten_paths(train_coords)))
    train_data = list(dict.fromkeys(flatten_paths(train_data)))
    train_df, _ = create_balanced_dataset(
        df_train,
        n_bins=128,
        min_ratio=0.75,
        use_validation=False
    )
    target_mean = train_df['OC'].mean()
    target_std = train_df['OC'].std()
    return target_mean, target_std

# --- Inference Function ---

def perform_inference(
    args: argparse.Namespace,
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None
) -> Dict[str, Any]:
    """Performs inference and returns predictions with coordinates using mixed precision."""
    model.eval()
    all_predictions = []
    all_longitudes = []
    all_latitudes = []
    inference_loop = tqdm(data_loader, desc="Inference", disable=not accelerator.is_main_process)

    scaler = torch.cuda.amp.GradScaler() if accelerator.device.type == 'cuda' else None

    with torch.no_grad():
        for longitudes, latitudes, features in inference_loop:
            features = features.float().to(accelerator.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=accelerator.device.type == 'cuda'):
                outputs = model(features)
            print(outputs)
            gathered_outputs = accelerator.gather(outputs).cpu().numpy()
            gathered_longitudes = accelerator.gather(longitudes).cpu().numpy()
            gathered_latitudes = accelerator.gather(latitudes).cpu().numpy()
            all_predictions.append(gathered_outputs)
            all_longitudes.append(gathered_longitudes)
            all_latitudes.append(gathered_latitudes)

    # Concatenate results
    predictions = np.concatenate(all_predictions).flatten() if all_predictions else np.array([])
    longitudes = np.concatenate(all_longitudes).flatten() if all_longitudes else np.array([])
    latitudes = np.concatenate(all_latitudes).flatten() if all_latitudes else np.array([])

    # Apply inverse transformation to predictions
    predictions_transformed = inverse_transform_target(
        predictions, args.target_transform, target_mean, target_std
    )

    # Clip predictions to reasonable range [0, MAX_OC]
    predictions_transformed = np.clip(predictions_transformed, 0, MAX_OC)

    # Scale predictions to [2.0, 150.0]
    if accelerator.is_local_main_process:
        pred_min = np.min(predictions_transformed)
        pred_max = np.max(predictions_transformed)
        print(f"Prediction min: {pred_min}, max: {pred_max}")
        if pred_max > pred_min:  # Avoid division by zero
            predictions_transformed = 2.0 + (predictions_transformed - pred_min) * (150.0 - 2.0) / (pred_max - pred_min)
        else:
            predictions_transformed = np.full_like(predictions_transformed, 2.0)

    coordinates = np.stack((longitudes, latitudes), axis=1) if longitudes.size > 0 else np.array([])

    return {
        'coordinates': coordinates,
        'predictions': predictions_transformed
    }

# --- Save Predictions and Visualizations ---

def save_results_and_visualize(
    accelerator: Accelerator,
    coordinates: np.ndarray,
    predictions: np.ndarray,
    save_path_predictions_plots: str
) -> None:
    """Saves coordinates and predictions as NumPy arrays and creates visualizations on the main process."""
    if accelerator.is_local_main_process:
        np.save("coordinates_1mil.npy", coordinates)
        np.save("predictions_1mil.npy", predictions)
        print(f"Inference completed. Results shape: {predictions.shape}")

        create_prediction_visualizations(
            INFERENCE_TIME,
            coordinates,
            predictions,
            save_path_predictions_plots
        )
        print(f"Visualizations saved to {save_path_predictions_plots}")

# --- Argument Parsing ---

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for inference."""
    parser = argparse.ArgumentParser(description='Perform inference with ResNet2DCNN model.')
    parser.add_argument('--model-path', type=str, 
                        default="/home/vfourel/SOCProject/SOCmapping/2DCNN/resnet2dcnn_OC_150_T_2007-2023_R2_no_val_TRANS_none_LOSS_mse_run_1.pth",
                        help='Path to the trained model .pth file.')
    parser.add_argument('--target-transform', type=str, default='none', choices=['none', 'log', 'normalize'],
                        help='Transformation applied to target during training.')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Directory to save inference results and visualizations.')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage (uses CPU).')
    parser.add_argument('--project-name', type=str, default='socmapping-2DCNN-inference', help='WandB project name.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size per process for inference.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS, help='Number of data loader workers.')
    return parser.parse_args()

# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualization_dir = VISUALIZATION_DIR
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # Initialize Accelerator with mixed precision
    accelerator = Accelerator(mixed_precision='fp16', cpu=args.no_gpu)

    # Initialize WandB
    if accelerator.is_main_process:
        try:
            wandb.init(project=args.project_name, name=f"inference_{os.path.basename(args.model_path)}")
            wandb.config.update(vars(args))
        except Exception as e:
            print(f"Error initializing WandB: {e}. Continuing without WandB logging.")

    try:
        # Compute training statistics for target transformation
        if accelerator.is_main_process:
            print("Computing training statistics for target...")
        target_mean, target_std = compute_training_statistics_oc()
        if accelerator.is_main_process:
            print(f"Target stats: mean={target_mean:.4f}, std={target_std:.4f}")

        # Load inference dataframe
        if accelerator.is_main_process:
            print("Loading inference dataframe...")
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        df_full = df_full[:300000]
        if df_full.empty:
            raise ValueError("Inference dataframe is empty.")
        if accelerator.is_main_process:
            print(f"Loaded inference dataframe with {len(df_full)} rows")

        # Prepare data paths for inference
        if accelerator.is_main_process:
            print("Separating inference data paths...")
        samples_coords_paths_1mil, data_paths_1mil = separate_and_add_data_1mil_inference()
        samples_coords_paths_1mil_flat = list(dict.fromkeys(flatten_paths(samples_coords_paths_1mil)))
        data_paths_1mil_flat = list(dict.fromkeys(flatten_paths(data_paths_1mil)))

        if not samples_coords_paths_1mil_flat or not data_paths_1mil_flat:
            raise ValueError("No valid inference data paths found after flattening.")

        # Prepare training data for feature normalization
        if accelerator.is_main_process:
            print("Preparing training data for feature normalization...")
        df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        train_coords, train_data = separate_and_add_data()
        train_coords_flat = list(dict.fromkeys(flatten_paths(train_coords)))
        train_data_flat = list(dict.fromkeys(flatten_paths(train_data)))
        train_df, _ = create_balanced_dataset(df_train, n_bins=128, min_ratio=0.75, use_validation=False)

        # Create training dataset to get feature statistics
        train_dataset = NormalizedMultiRasterDatasetMultiYears(train_coords_flat, train_data_flat, train_df)
        feature_means, feature_stds = train_dataset.get_statistics()

        # Create inference dataset
        if accelerator.is_main_process:
            print("Creating inference dataset...")
        inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
            samples_coordinates_array_path=samples_coords_paths_1mil_flat,
            data_array_path=data_paths_1mil_flat,
            df=df_full,
            time_before=time_before,
            feature_means=feature_means,
            feature_stds=feature_stds
        )

        # Create DataLoader with optimized settings
        batch_size_per_process = max(1, args.batch_size // accelerator.num_processes)
        if accelerator.is_main_process:
            print(f"Using batch size of {batch_size_per_process} per process")
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=batch_size_per_process,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        inference_loader = accelerator.prepare(inference_loader)

        # Initialize model
        if accelerator.is_main_process:
            print("Loading model...")
        input_channels = (len(bands_list_order) - 1) * time_before + 1
        model = ResNet2DCNN(
            input_channels=input_channels,
            input_height=window_size,
            input_width=window_size,
        )

        # Load trained model weights
        model.load_state_dict(torch.load(args.model_path, map_location=accelerator.device))
        model.eval()

        # Attempt TorchScript optimization
        if accelerator.is_main_process:
            try:
                model = torch.jit.script(model)
                print("Model optimized with TorchScript")
            except Exception as e:
                print(f"TorchScript optimization failed: {e}")

        model = accelerator.prepare(model)

        # Perform inference
        if accelerator.is_main_process:
            print("Starting inference...")
        results = perform_inference(
            args=args,
            model=model,
            data_loader=inference_loader,
            accelerator=accelerator,
            target_mean=target_mean if args.target_transform == 'normalize' else None,
            target_std=target_std if args.target_transform == 'normalize' else None
        )

        # Save results and create visualizations
        if accelerator.is_main_process:
            print("Saving results and generating visualizations...")
        save_results_and_visualize(
            accelerator=accelerator,
            coordinates=results['coordinates'],
            predictions=results['predictions'],
            save_path_predictions_plots=visualization_dir
        )

        # Optionally save results as CSV
        if accelerator.is_main_process:
            output_filename = f"{output_dir}/predictions_{os.path.basename(args.model_path).replace('.pth', '')}.csv"
            df_results = pd.DataFrame({
                'longitude': results['coordinates'][:, 0],
                'latitude': results['coordinates'][:, 1],
                'predicted_OC': results['predictions']
            })
            df_results.to_csv(output_filename, index=False)
            print(f"CSV results saved to: {output_filename}")
            if wandb.run:
                wandb.save(output_filename)
                wandb.save("coordinates_1mil.npy")
                wandb.save("predictions_1mil.npy")

    except Exception as e:
        accelerator.print(f"Error during inference: {e}")
        import traceback
        accelerator.print(traceback.format_exc())
        raise

    finally:
        if accelerator.is_main_process and wandb.run:
            wandb.finish()

    accelerator.print("Inference completed.")