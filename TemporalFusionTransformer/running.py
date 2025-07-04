import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
import pickle
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size, MAX_OC,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded, NUM_HEADS, NUM_LAYERS,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, hidden_size,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import NormalizedMultiRasterDataset1MilMultiYears, RasterTensorDataset1Mil, MultiRasterDataset1MilMultiYears
from dataloader.dataloaderMultiYears import NormalizedMultiRasterDatasetMultiYears
from mapping import create_prediction_visualizations, parallel_predict
from EnhancedTFT import EnhancedTFT as SimpleTFT
from tqdm import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
import json
from datetime import datetime
import argparse

def load_experiment_config(experiment_dir):
    """Load experiment configuration from JSON file."""
    config_file = os.path.join(experiment_dir, "experiment_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def load_normalization_stats_from_experiment(experiment_dir):
    """Load normalization statistics from experiment data folder."""
    data_dir = os.path.join(experiment_dir, "data")
    if os.path.exists(data_dir):
        # Look for normalization stats files
        stats_files = glob.glob(os.path.join(data_dir, "normalization_stats_run_*.pkl"))
        if stats_files:
            # Use the first available stats file (they should be consistent across runs)
            with open(stats_files[0], 'rb') as f:
                return pickle.load(f)
    return None

def load_model_with_metadata(model_path, accelerator=None):
    """Load model and extract metadata including normalization statistics."""
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the saved checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Check if it's a model with metadata (from your training script)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            metadata = {
                'best_run_number': checkpoint.get('best_run_number'),
                'best_metrics': checkpoint.get('best_metrics'),
                'model_config': checkpoint.get('model_config'),
                'normalization_stats': checkpoint.get('normalization_stats'),
                'experiment_info': checkpoint.get('experiment_info'),
                'training_config': checkpoint.get('training_config'),
                'training_args': checkpoint.get('training_args')
            }
            if accelerator and accelerator.is_local_main_process:
                run_num = metadata.get('best_run_number', metadata.get('run_number', 'unknown'))
                if metadata.get('best_metrics'):
                    r2 = metadata['best_metrics'].get('r_squared', 'unknown')
                    print(f"Loaded model from run {run_num} with R²: {r2:.4f}")
                else:
                    print(f"Loaded model from run {run_num}")
        else:
            # Fallback: assume it's just a state dict
            model_state_dict = checkpoint
            metadata = None
            if accelerator and accelerator.is_local_main_process:
                print("Loaded model state dict without metadata")

        # Initialize model with config from metadata or defaults
        if metadata and metadata.get('model_config'):
            config = metadata['model_config']
            model = SimpleTFT(
                input_channels=config['input_channels'],
                height=config['height'],
                width=config['width'],
                time_steps=config['time_steps'],
                d_model=config['d_model']
            )
        else:
            model = SimpleTFT(
                input_channels=len(bands_list_order),
                height=window_size,
                width=window_size,
                time_steps=time_before,
                d_model=hidden_size
            )

        # Load state dict - handle both wrapped and unwrapped models
        if any(k.startswith('module.') for k in model_state_dict.keys()):
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

        model.load_state_dict(model_state_dict)
        model.eval()

        if accelerator:
            model = accelerator.prepare(model)

        return model, metadata, device

    except Exception as e:
        if accelerator and accelerator.is_local_main_process:
            print(f"Error loading model: {e}")
        raise

def extract_transform_from_path(model_path):
    """Extract target transform type from model or experiment path."""
    path_str = str(model_path)
    if 'transform_log' in path_str:
        return 'log'
    elif 'transform_normalize' in path_str:
        return 'normalize'
    elif 'transform_none' in path_str:
        return 'none'
    else:
        print(f"Warning: Could not determine transform from path: {path_str}")
        return 'none'

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def compute_training_statistics_oc():
    """Fallback function to compute training statistics if not available."""
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    target_mean = df_train['OC'].mean()
    target_std = df_train['OC'].std()
    return target_mean, target_std

def run_inference(model, dataloader, accelerator):
    model.eval()
    all_coordinates = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitudes, latitudes, tensors = batch
            tensors = tensors.to(accelerator.device)

            outputs = model(tensors)
            predictions = outputs.cpu().numpy().squeeze()  # Remove extra dimensions

            coords = np.stack([longitudes.cpu().numpy(), latitudes.cpu().numpy()], axis=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    # Concatenate local results
    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

    # Gather results from all GPUs
    if len(all_coordinates) > 0:
        all_coordinates = accelerator.gather(torch.tensor(all_coordinates, device=accelerator.device)).cpu().numpy()
        all_predictions = accelerator.gather(torch.tensor(all_predictions, device=accelerator.device)).cpu().numpy()

    return all_coordinates, all_predictions

def apply_inverse_transform(predictions, target_transform, accelerator, target_mean=None, target_std=None):
    """Apply inverse transformation to convert predictions back to original scale."""
    if target_transform == 'log':
        original_predictions = np.exp(predictions)
    elif target_transform == 'normalize':
        if target_mean is None or target_std is None:
            raise ValueError("target_mean and target_std required for 'normalize' transform")
        original_predictions = predictions * target_std + target_mean
    else:  # 'none'
        original_predictions = predictions

    if accelerator.is_local_main_process:
        print(f"Applied inverse transform '{target_transform}'")
        print(f"Prediction stats - Min: {np.min(original_predictions):.4f}, Max: {np.max(original_predictions):.4f}, Mean: {np.mean(original_predictions):.4f}")

    return original_predictions

def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Inference script with automatic model configuration")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Path to experiment directory (auto-detected if not provided)")
    parser.add_argument("--start-idx", type=int, default=480000, help="Start index for inference dataset")
    parser.add_argument("--end-idx", type=int, default=500000, help="End index for inference dataset")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--output-dir", type=str, default="./inference_output", help="Output directory for results")
    args = parser.parse_args()

    # Auto-detect experiment directory if not provided
    if args.experiment_dir is None:
        model_dir = os.path.dirname(args.model_path)
        if os.path.basename(model_dir) == "models":
            args.experiment_dir = os.path.dirname(model_dir)
        else:
            args.experiment_dir = model_dir

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_local_main_process:
        print(f"Experiment directory: {args.experiment_dir}")
        print(f"Model path: {args.model_path}")
        print(f"Output directory: {args.output_dir}")

    # Load experiment configuration
    experiment_config = load_experiment_config(args.experiment_dir)
    if accelerator.is_local_main_process:
        if experiment_config:
            print("Loaded experiment configuration")
        else:
            print("No experiment configuration found, using defaults")

    # Load model and metadata
    model, metadata, device = load_model_with_metadata(args.model_path, accelerator)
    if accelerator.is_local_main_process:
        print(f"Model loaded on {device}")

    # Determine target transform
    target_transform = 'normalize'  # Default from your experiment name

    if metadata and metadata.get('training_config'):
        target_transform = metadata['training_config'].get('target_transform', target_transform)
    elif experiment_config and experiment_config.get('args'):
        target_transform = experiment_config['args'].get('target_transform', target_transform)
    else:
        target_transform = extract_transform_from_path(args.model_path)

    if accelerator.is_local_main_process:
        print(f"Using target transform: {target_transform}")

    # Load normalization statistics
    target_mean, target_std, feature_means, feature_stds = None, None, None, None

    if metadata and metadata.get('normalization_stats'):
        norm_stats = metadata['normalization_stats']
        target_mean = norm_stats.get('target_mean')
        target_std = norm_stats.get('target_std')
        feature_means = norm_stats.get('feature_means')
        feature_stds = norm_stats.get('feature_stds')
        if accelerator.is_local_main_process:
            print("Using normalization stats from model metadata")

    if target_mean is None:
        norm_stats = load_normalization_stats_from_experiment(args.experiment_dir)
        if norm_stats:
            target_mean = norm_stats.get('target_mean')
            target_std = norm_stats.get('target_std')
            feature_means = norm_stats.get('feature_means')
            feature_stds = norm_stats.get('feature_stds')
            if accelerator.is_local_main_process:
                print("Using normalization stats from experiment data folder")

    if target_mean is None:
        if accelerator.is_local_main_process:
            print("Computing normalization stats from training data...")
        target_mean, target_std = compute_training_statistics_oc()

        df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        train_coords, train_data = separate_and_add_data()
        train_coords = list(dict.fromkeys(flatten_paths(train_coords)))
        train_data = list(dict.fromkeys(flatten_paths(train_data)))
        temp_dataset = NormalizedMultiRasterDatasetMultiYears(train_coords, train_data, df_train)
        feature_means = temp_dataset.get_feature_means()
        feature_stds = temp_dataset.get_feature_stds()

    if accelerator.is_local_main_process:
        print(f"Normalization stats - Target mean: {target_mean:.4f}, Target std: {target_std:.4f}")

    # **Fix: Ensure feature_means and feature_stds are on CPU**
    if isinstance(feature_means, torch.Tensor):
        feature_means = feature_means.cpu()
    else:
        feature_means = torch.tensor(feature_means).float().cpu()

    if isinstance(feature_stds, torch.Tensor):
        feature_stds = feature_stds.cpu()
    else:
        feature_stds = torch.tensor(feature_stds).float().cpu()

    # Load inference data
    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print(f"Loaded inference dataframe with {len(df_full)} samples")
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    # Prepare dataset
    samples_coords_1mil, data_1mil = separate_and_add_data_1mil_inference()
    samples_coords_1mil = list(dict.fromkeys(flatten_paths(samples_coords_1mil)))
    data_1mil = list(dict.fromkeys(flatten_paths(data_1mil)))

    inference_subset = df_full[args.start_idx:args.end_idx]
    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path=samples_coords_1mil,
        data_array_path=data_1mil,
        df=inference_subset,
        feature_means=feature_means,
        feature_stds=feature_stds,
        time_before=time_before
    )

    if accelerator.is_local_main_process:
        print(f"Inference dataset size: {len(inference_dataset)}")

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    inference_loader = accelerator.prepare(inference_loader)

    # Run inference
    if accelerator.is_local_main_process:
        print("Starting inference...")

    coordinates, predictions = run_inference(model, inference_loader, accelerator)

    # Apply inverse transform
    original_predictions = apply_inverse_transform(
        predictions, target_transform, accelerator, target_mean, target_std
    )

    # Save results
    if accelerator.is_local_main_process:
        output_coords_file = os.path.join(args.output_dir, f"coordinates_{args.start_idx}_to_{args.end_idx}.npy")
        output_preds_file = os.path.join(args.output_dir, f"predictions_{args.start_idx}_to_{args.end_idx}.npy")

        np.save(output_coords_file, coordinates)
        np.save(output_preds_file, original_predictions)

        print(f"Results saved:")
        print(f"  Coordinates: {output_coords_file} (shape: {coordinates.shape})")
        print(f"  Predictions: {output_preds_file} (shape: {original_predictions.shape})")

        # Save metadata
        metadata_file = os.path.join(args.output_dir, f"inference_metadata_{args.start_idx}_to_{args.end_idx}.json")
        inference_metadata = {
            "model_path": args.model_path,
            "experiment_dir": args.experiment_dir,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "batch_size": args.batch_size,
            "target_transform": target_transform,
            "normalization_stats": {
                "target_mean": float(target_mean),
                "target_std": float(target_std)
            },
            "prediction_stats": {
                "min": float(np.min(original_predictions)),
                "max": float(np.max(original_predictions)),
                "mean": float(np.mean(original_predictions)),
                "std": float(np.std(original_predictions))
            },
            "timestamp": datetime.now().isoformat()
        }

        with open(metadata_file, 'w') as f:
            json.dump(inference_metadata, f, indent=2)
        print(f"  Metadata: {metadata_file}")

        # Create visualizations
        try:
            create_prediction_visualizations(
                INFERENCE_TIME,
                coordinates,
                original_predictions,
                args.output_dir
            )
            print(f"Visualizations saved to {args.output_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main()