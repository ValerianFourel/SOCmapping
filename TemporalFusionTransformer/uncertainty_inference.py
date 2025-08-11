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

def load_kfold_models(experiment_dir, accelerator=None):
    """Load all k-fold models from experiment directory"""
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.path.join(experiment_dir, 'models')

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    fold_models = []

    # Load models in fold order (1-5)
    for fold in range(1, 6):
        fold_files = [f for f in os.listdir(models_dir) if f.startswith(f'TFT_fold_{fold}_')]
        if not fold_files:
            if accelerator and accelerator.is_local_main_process:
                print(f"Warning: No model found for fold {fold}")
            continue

        model_path = os.path.join(models_dir, fold_files[0])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Initialize model with same architecture
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = SimpleTFT(
                input_channels=config['input_channels'],
                height=config['height'],
                width=config['width'],
                time_steps=config['time_steps'],
                d_model=config['d_model']
            )
        else:
            # Fallback to default config
            model = SimpleTFT(
                input_channels=len(bands_list_order),
                height=window_size,
                width=window_size,
                time_steps=time_before,
                d_model=hidden_size
            )

        # Load state dict - handle wrapped models
        model_state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in model_state_dict.keys()):
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        if accelerator:
            model = accelerator.prepare(model)

        fold_models.append({
            'model': model,
            'fold_number': checkpoint.get('fold_number', fold),
            'normalization_stats': checkpoint.get('normalization_stats'),
            'model_path': model_path,
            'training_config': checkpoint.get('training_config', {})
        })

    if accelerator and accelerator.is_local_main_process:
        print(f"Loaded {len(fold_models)} fold models")

    return fold_models

def load_full_dataset_model(model_path, accelerator=None):
    """Load the single model trained on all samples"""
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model configuration
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        model_config = checkpoint.get('model_config')
        normalization_stats = checkpoint.get('normalization_stats')
        training_config = checkpoint.get('training_config', {})
    else:
        model_state_dict = checkpoint
        model_config = None
        normalization_stats = None
        training_config = {}

    # Initialize model
    if model_config:
        model = SimpleTFT(
            input_channels=model_config['input_channels'],
            height=model_config['height'],
            width=model_config['width'],
            time_steps=model_config['time_steps'],
            d_model=model_config['d_model']
        )
    else:
        model = SimpleTFT(
            input_channels=len(bands_list_order),
            height=window_size,
            width=window_size,
            time_steps=time_before,
            d_model=hidden_size
        )

    # Load state dict - handle wrapped models
    if any(k.startswith('module.') for k in model_state_dict.keys()):
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    if accelerator:
        model = accelerator.prepare(model)

    return model, normalization_stats, training_config

def normalize_features_with_stats(features, feature_means, feature_stds):
    """Apply normalization using provided statistics"""
    if isinstance(feature_means, torch.Tensor):
        means = feature_means.to(features.device)
    else:
        means = torch.tensor(feature_means).float().to(features.device)

    if isinstance(feature_stds, torch.Tensor):
        stds = feature_stds.to(features.device)
    else:
        stds = torch.tensor(feature_stds).float().to(features.device)

    # Ensure proper broadcasting
    while len(means.shape) < len(features.shape):
        means = means.unsqueeze(0)
    while len(stds.shape) < len(features.shape):
        stds = stds.unsqueeze(0)

    return (features - means) / (stds + 1e-10)

def denormalize_targets(predictions, target_mean, target_std):
    """Convert normalized predictions back to original scale"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    return predictions * target_std + target_mean

def apply_inverse_transform_uncertainty(predictions, target_transform, target_mean=None, target_std=None):
    """Apply inverse transformation to convert predictions back to original scale"""
    if target_transform == 'log':
        return np.exp(predictions)
    elif target_transform == 'normalize':
        if target_mean is None or target_std is None:
            raise ValueError("target_mean and target_std required for 'normalize' transform")
        return predictions * target_std + target_mean
    else:  # 'none'
        return predictions

def predict_with_uncertainty(fold_models, full_model, full_model_stats, full_model_config,
                           dataloader, accelerator, target_transform='normalize'):
    """
    Generate predictions and uncertainty estimates

    Returns:
    - ensemble_predictions: Mean prediction across k-fold models
    - prediction_uncertainty: Standard deviation across k-fold models  
    - full_dataset_predictions: Predictions from model trained on all samples
    - epistemic_uncertainty: Difference between ensemble and full-dataset predictions
    """

    ensemble_predictions = []
    all_fold_predictions = []  # Store predictions from each fold
    full_dataset_predictions = []
    all_coordinates = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Uncertainty Inference", 
                                             disable=not accelerator.is_local_main_process)):
            longitudes, latitudes, features = batch
            features = features.to(accelerator.device)
            batch_size_actual = features.shape[0]

            # Store coordinates
            coords = torch.stack([longitudes, latitudes], dim=1)
            all_coordinates.append(coords.cpu())

            # Predictions from k-fold models
            fold_predictions = []
            for fold_data in fold_models:
                model = fold_data['model']
                norm_stats = fold_data['normalization_stats']
                training_config = fold_data['training_config']

                # Apply fold-specific normalization if available
                if norm_stats and 'feature_means' in norm_stats and 'feature_stds' in norm_stats:
                    features_norm = normalize_features_with_stats(
                        features, 
                        norm_stats['feature_means'], 
                        norm_stats['feature_stds']
                    )
                else:
                    features_norm = features

                pred = model(features_norm)

                # Apply inverse transform if needed
                pred_np = pred.cpu().numpy().squeeze()
                if target_transform == 'normalize' and norm_stats and 'target_mean' in norm_stats:
                    pred_np = apply_inverse_transform_uncertainty(
                        pred_np, target_transform,
                        norm_stats['target_mean'], norm_stats['target_std']
                    )
                elif target_transform == 'log':
                    pred_np = apply_inverse_transform_uncertainty(pred_np, target_transform)

                fold_predictions.append(pred_np)

            # Stack fold predictions: shape (n_folds, batch_size)
            if fold_predictions:
                fold_predictions = np.stack(fold_predictions, axis=0)
                all_fold_predictions.append(fold_predictions)

                # Calculate ensemble statistics
                batch_ensemble_mean = np.mean(fold_predictions, axis=0)
                ensemble_predictions.extend(batch_ensemble_mean)

            # Prediction from full dataset model
            if full_model_stats and 'feature_means' in full_model_stats and 'feature_stds' in full_model_stats:
                features_norm_full = normalize_features_with_stats(
                    features,
                    full_model_stats['feature_means'],
                    full_model_stats['feature_stds']
                )
            else:
                features_norm_full = features

            full_pred = full_model(features_norm_full)
            full_pred_np = full_pred.cpu().numpy().squeeze()

            # Apply inverse transform for full model
            if target_transform == 'normalize' and full_model_stats and 'target_mean' in full_model_stats:
                full_pred_np = apply_inverse_transform_uncertainty(
                    full_pred_np, target_transform,
                    full_model_stats['target_mean'], full_model_stats['target_std']
                )
            elif target_transform == 'log':
                full_pred_np = apply_inverse_transform_uncertainty(full_pred_np, target_transform)

            full_dataset_predictions.extend(full_pred_np)

    # Gather results from all GPUs
    all_coordinates = torch.cat(all_coordinates, dim=0)
    if accelerator.num_processes > 1:
        all_coordinates = accelerator.gather(all_coordinates).cpu().numpy()
    else:
        all_coordinates = all_coordinates.cpu().numpy()

    # Calculate uncertainties
    if all_fold_predictions:
        all_fold_predictions = np.concatenate(all_fold_predictions, axis=1)  # Shape: (n_folds, n_samples)
        prediction_uncertainty = np.std(all_fold_predictions, axis=0)  # Uncertainty from model disagreement
    else:
        all_fold_predictions = None
        prediction_uncertainty = None

    ensemble_predictions = np.array(ensemble_predictions)
    full_dataset_predictions = np.array(full_dataset_predictions)

    # Gather predictions from all GPUs
    if accelerator.num_processes > 1:
        if len(ensemble_predictions) > 0:
            ensemble_predictions = accelerator.gather(torch.tensor(ensemble_predictions, device=accelerator.device)).cpu().numpy()
        if len(full_dataset_predictions) > 0:
            full_dataset_predictions = accelerator.gather(torch.tensor(full_dataset_predictions, device=accelerator.device)).cpu().numpy()
        if prediction_uncertainty is not None:
            prediction_uncertainty = accelerator.gather(torch.tensor(prediction_uncertainty, device=accelerator.device)).cpu().numpy()

    # Calculate epistemic uncertainty
    if len(ensemble_predictions) > 0 and len(full_dataset_predictions) > 0:
        epistemic_uncertainty = np.abs(ensemble_predictions - full_dataset_predictions)
    else:
        epistemic_uncertainty = None

    return {
        'coordinates': all_coordinates,
        'ensemble_predictions': ensemble_predictions,
        'prediction_uncertainty': prediction_uncertainty,
        'full_dataset_predictions': full_dataset_predictions,  
        'epistemic_uncertainty': epistemic_uncertainty,
        'individual_fold_predictions': all_fold_predictions
    }

def analyze_uncertainty_quality(results, accelerator):
    """Analyze the quality and patterns of uncertainty estimates"""
    if not accelerator.is_local_main_process:
        return

    uncertainty = results['prediction_uncertainty']
    epistemic_unc = results['epistemic_uncertainty']
    ensemble_pred = results['ensemble_predictions']
    full_pred = results['full_dataset_predictions']

    print("\n" + "="*60)
    print("UNCERTAINTY ANALYSIS")
    print("="*60)

    if uncertainty is not None:
        print(f"Prediction Uncertainty (model disagreement):")
        print(f"  Mean: {np.mean(uncertainty):.4f}")
        print(f"  Std: {np.std(uncertainty):.4f}")
        print(f"  Range: {np.min(uncertainty):.4f} - {np.max(uncertainty):.4f}")
        print(f"  Median: {np.median(uncertainty):.4f}")
        print(f"  95th percentile: {np.percentile(uncertainty, 95):.4f}")
    else:
        print("Prediction Uncertainty: Not available")

    if epistemic_unc is not None:
        print(f"\nEpistemic Uncertainty (ensemble vs full-data model):")
        print(f"  Mean: {np.mean(epistemic_unc):.4f}")
        print(f"  Std: {np.std(epistemic_unc):.4f}")
        print(f"  Range: {np.min(epistemic_unc):.4f} - {np.max(epistemic_unc):.4f}")
        print(f"  Median: {np.median(epistemic_unc):.4f}")
        print(f"  95th percentile: {np.percentile(epistemic_unc, 95):.4f}")
    else:
        print("Epistemic Uncertainty: Not available")

    if len(ensemble_pred) > 0 and len(full_pred) > 0:
        print(f"\nPrediction Comparison:")
        print(f"  Ensemble mean: {np.mean(ensemble_pred):.4f} ± {np.std(ensemble_pred):.4f}")
        print(f"  Full-data mean: {np.mean(full_pred):.4f} ± {np.std(full_pred):.4f}")
        print(f"  Prediction correlation: {np.corrcoef(ensemble_pred, full_pred)[0,1]:.4f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(ensemble_pred - full_pred)):.4f}")

    # Uncertainty-prediction relationship
    if uncertainty is not None:
        high_uncertainty_mask = uncertainty > np.percentile(uncertainty, 90)
        print(f"\nHigh Uncertainty Areas (top 10%):")
        print(f"  Number of samples: {np.sum(high_uncertainty_mask)}")
        print(f"  Average prediction uncertainty: {np.mean(uncertainty[high_uncertainty_mask]):.4f}")
        if epistemic_unc is not None:
            print(f"  Average epistemic uncertainty: {np.mean(epistemic_unc[high_uncertainty_mask]):.4f}")

def create_uncertainty_maps(results, output_dir, accelerator):
    """Create spatial maps of predictions and uncertainty"""
    if not accelerator.is_local_main_process:
        return

    coordinates = results['coordinates']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save prediction and uncertainty arrays
    np.save(os.path.join(output_dir, 'coordinates.npy'), coordinates)

    if results['ensemble_predictions'] is not None and len(results['ensemble_predictions']) > 0:
        np.save(os.path.join(output_dir, 'ensemble_predictions.npy'), results['ensemble_predictions'])

    if results['prediction_uncertainty'] is not None:
        np.save(os.path.join(output_dir, 'prediction_uncertainty.npy'), results['prediction_uncertainty'])

    if results['full_dataset_predictions'] is not None and len(results['full_dataset_predictions']) > 0:
        np.save(os.path.join(output_dir, 'full_dataset_predictions.npy'), results['full_dataset_predictions'])

    if results['epistemic_uncertainty'] is not None:
        np.save(os.path.join(output_dir, 'epistemic_uncertainty.npy'), results['epistemic_uncertainty'])

    if results['individual_fold_predictions'] is not None:
        np.save(os.path.join(output_dir, 'individual_fold_predictions.npy'), results['individual_fold_predictions'])

    # Create spatial dataframe for mapping
    spatial_data = {
        'longitude': coordinates[:, 0],
        'latitude': coordinates[:, 1]
    }

    if results['ensemble_predictions'] is not None and len(results['ensemble_predictions']) > 0:
        spatial_data['ensemble_prediction'] = results['ensemble_predictions']

    if results['prediction_uncertainty'] is not None:
        spatial_data['prediction_uncertainty'] = results['prediction_uncertainty']

    if results['full_dataset_predictions'] is not None and len(results['full_dataset_predictions']) > 0:
        spatial_data['full_dataset_prediction'] = results['full_dataset_predictions']

    if results['epistemic_uncertainty'] is not None:
        spatial_data['epistemic_uncertainty'] = results['epistemic_uncertainty']

    spatial_results = pd.DataFrame(spatial_data)
    spatial_results.to_parquet(os.path.join(output_dir, 'spatial_predictions_with_uncertainty.parquet'))

    print(f"Uncertainty maps and data saved to: {output_dir}")
    return spatial_results

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description='Uncertainty inference using k-fold models')
    parser.add_argument('--kfold-experiment-dir', required=True, 
                       help='Directory containing k-fold models (e.g., output/TFT_KFOLD_UNCERTAINTY_20240110_120000)')
    parser.add_argument('--full-model-path', required=True, 
                       help='Path to model trained on all samples')
    parser.add_argument('--start-idx', type=int, default=480000, help='Start index for inference dataset')
    parser.add_argument('--end-idx', type=int, default=500000, help='End index for inference dataset')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--output-dir', default='uncertainty_results', help='Output directory for results')
    parser.add_argument('--target-transform', default='normalize', choices=['none', 'log', 'normalize'],
                       help='Target transformation type used during training')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_local_main_process:
        print(f"K-fold experiment directory: {args.kfold_experiment_dir}")
        print(f"Full model path: {args.full_model_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Inference range: {args.start_idx} to {args.end_idx}")

    # Load models
    if accelerator.is_local_main_process:
        print("Loading k-fold models...")
    fold_models = load_kfold_models(args.kfold_experiment_dir, accelerator)

    if accelerator.is_local_main_process:
        print("Loading full-dataset model...")
    full_model, full_model_stats, full_model_config = load_full_dataset_model(args.full_model_path, accelerator)

    # Load inference data
    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print(f"Loaded inference dataframe with {len(df_full)} samples")
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    # Prepare dataset - get normalization stats from the first fold model or full model
    feature_means, feature_stds = None, None

    # Try to get normalization stats from fold models first
    for fold_data in fold_models:
        if fold_data['normalization_stats'] and 'feature_means' in fold_data['normalization_stats']:
            norm_stats = fold_data['normalization_stats']
            feature_means = norm_stats['feature_means']
            feature_stds = norm_stats['feature_stds']
            break

    # Fallback to full model stats
    if feature_means is None and full_model_stats and 'feature_means' in full_model_stats:
        feature_means = full_model_stats['feature_means']
        feature_stds = full_model_stats['feature_stds']

    # Final fallback: compute from training data
    if feature_means is None:
        if accelerator.is_local_main_process:
            print("Computing normalization stats from training data...")
        df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        train_coords, train_data = separate_and_add_data()
        train_coords = list(dict.fromkeys(flatten_paths(train_coords)))
        train_data = list(dict.fromkeys(flatten_paths(train_data)))
        temp_dataset = NormalizedMultiRasterDatasetMultiYears(train_coords, train_data, df_train)
        feature_means = temp_dataset.get_feature_means()
        feature_stds = temp_dataset.get_feature_stds()

    # Ensure feature stats are on CPU
    if isinstance(feature_means, torch.Tensor):
        feature_means = feature_means.cpu()
    else:
        feature_means = torch.tensor(feature_means).float().cpu()

    if isinstance(feature_stds, torch.Tensor):
        feature_stds = feature_stds.cpu()
    else:
        feature_stds = torch.tensor(feature_stds).float().cpu()

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

    # Generate predictions with uncertainty
    if accelerator.is_local_main_process:
        print("Generating predictions with uncertainty...")

    results = predict_with_uncertainty(
        fold_models=fold_models,
        full_model=full_model,
        full_model_stats=full_model_stats,
        full_model_config=full_model_config,
        dataloader=inference_loader,
        accelerator=accelerator,
        target_transform=args.target_transform
    )

    # Analyze uncertainty quality
    analyze_uncertainty_quality(results, accelerator)

    # Create spatial maps
    spatial_results = create_uncertainty_maps(results, args.output_dir, accelerator)

    # Save metadata
    if accelerator.is_local_main_process:
        metadata = {
            "kfold_experiment_dir": args.kfold_experiment_dir,
            "full_model_path": args.full_model_path,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "batch_size": args.batch_size,
            "target_transform": args.target_transform,
            "num_fold_models": len(fold_models),
            "timestamp": datetime.now().isoformat()
        }

        # Add prediction statistics
        if results['ensemble_predictions'] is not None and len(results['ensemble_predictions']) > 0:
            metadata["ensemble_prediction_stats"] = {
                "min": float(np.min(results['ensemble_predictions'])),
                "max": float(np.max(results['ensemble_predictions'])),
                "mean": float(np.mean(results['ensemble_predictions'])),
                "std": float(np.std(results['ensemble_predictions']))
            }

        if results['prediction_uncertainty'] is not None:
            metadata["uncertainty_stats"] = {
                "min": float(np.min(results['prediction_uncertainty'])),
                "max": float(np.max(results['prediction_uncertainty'])),
                "mean": float(np.mean(results['prediction_uncertainty'])),
                "std": float(np.std(results['prediction_uncertainty']))
            }

        metadata_file = os.path.join(args.output_dir, f"uncertainty_inference_metadata_{args.start_idx}_to_{args.end_idx}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nResults saved to: {args.output_dir}")
        print(f"Metadata saved to: {metadata_file}")

        # Try to create visualizations using existing mapping function
        try:
            if results['ensemble_predictions'] is not None and len(results['ensemble_predictions']) > 0:
                create_prediction_visualizations(
                    INFERENCE_TIME,
                    results['coordinates'],
                    results['ensemble_predictions'],
                    args.output_dir
                )
                print(f"Visualizations saved to {args.output_dir}")
        except Exception as e:
            print(f"Error creating standard visualizations: {e}")

        print("Uncertainty inference completed successfully!")

if __name__ == "__main__":
    main()
