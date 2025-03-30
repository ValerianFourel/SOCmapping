import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import logging
from accelerate import Accelerator
import gc
import os

# Import from training script's modules
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from config import (
    TIME_BEGINNING, TIME_END, MAX_OC, bands_list_order, window_size, time_before,
    file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Yearly, DataYearly,
    SamplesCoordinates_Seasonally, DataSeasonally
)
from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
from modelTransformerVAE import TransformerVAE  # Assuming this is the VAE model used

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TOTAL_YEARS_DATA = TIME_END - TIME_BEGINNING + 1
VAE_NUM_HEADS = 16
VAE_LATENT_DIM_ELEVATION = 24
VAE_LATENT_DIM_OTHERS = 48
VAE_DROPOUT_RATE = 0.3
NORMALIZED_BANDS = ['LST', 'MODIS_NPP', 'TotalEvapotranspiration']

# Helper Functions
def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def compute_band_statistics(dataset, band_name, band_index, batch_size=512, sample_percentage=0.01):
    """Compute mean and std for a specific band using a sample of the dataset."""
    total_samples = len(dataset)
    sample_size = max(1, int(total_samples * sample_percentage))
    sample_size = min(sample_size, total_samples)

    if sample_size == 0:
        logger.warning(f"Dataset size is 0 for band {band_name}, cannot compute statistics.")
        return 0.0, 1.0

    indices = np.random.choice(total_samples, size=sample_size, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()))

    all_band_features = []
    logger.info(f"Computing statistics for band '{band_name}' using {sample_size} samples...")

    is_elevation = (band_name == 'Elevation')

    for batch in tqdm(loader, desc=f"Computing Stats ({band_name})", leave=False):
        try:
            # Batch format: lon, lat, features, encoding_tensor, oc
            _, _, features_batch, encoding_tensor, _ = batch
            band_features = features_batch[:, band_index, :, :, :]  # Shape: (batch_size, W, H, T)

            if is_elevation:
                if band_features.shape[-1] != time_before:
                    logger.warning(f"Elevation band '{band_name}' has unexpected time dim {band_features.shape[-1]}. Using first slice.")
                current_features = band_features[:, :, :, 0].reshape(-1)  # Flatten to (batch_size * W * H)
            else:
                current_features = band_features.reshape(-1)  # Flatten to (batch_size * W * H * T)

            current_features = torch.nan_to_num(current_features.float(), nan=0.0)
            all_band_features.append(current_features.cpu())
            del features_batch, band_features, current_features
        except Exception as e:
            logger.error(f"Error processing batch for {band_name}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    if not all_band_features:
        logger.warning(f"No features collected for statistics computation for band {band_name}.")
        return 0.0, 1.0

    all_band_features_tensor = torch.cat(all_band_features, dim=0)
    if all_band_features_tensor.numel() == 0:
        logger.warning(f"Collected features tensor is empty for band {band_name}.")
        return 0.0, 1.0

    mean = torch.mean(all_band_features_tensor).item()
    std = torch.std(all_band_features_tensor).item()

    del all_band_features, all_band_features_tensor, loader, subset
    gc.collect()

    return mean, std if std > 1e-6 else 1.0

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run VAE Inference for SOC prediction')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained VAE model .pth files')
    parser.add_argument('--inference_year', type=int, required=True, help=f'Target year for inference ({TIME_BEGINNING}-{TIME_END})')
    parser.add_argument('--time_before', type=int, default=time_before, help='Number of years before inference_year to include')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output .npy file containing latent z')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--preload_dataset', action='store_true', help='Preload dataset into memory')
    return parser.parse_args()

# Main Inference Function
if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no')
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # Validate Inference Year
    if not (TIME_BEGINNING <= args.inference_year <= TIME_END):
        raise ValueError(f"Inference year {args.inference_year} must be between {TIME_BEGINNING} and {TIME_END}")

    # Load Dataset Coordinates and Paths
    logger.info("Loading Bavaria 1M dataset coordinates and paths...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()

    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    # Instantiate Dataset
    logger.info(f"Instantiating dataset for inference (preload={args.preload_dataset})...")
    inference_dataset = MultiRasterDatasetMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=train_dataset_df_full_1mil,
        time_before=args.time_before
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=min(4, os.cpu_count()) > 0
    )
    inference_loader = accelerator.prepare(inference_loader)

    # Load VAE Models
    logger.info(f"Loading VAE models from {model_dir}...")
    vae_models = {}
    band_name_to_index = {band_name: i for i, band_name in enumerate(bands_list_order)}

    for band in bands_list_order:
        latent_dim = VAE_LATENT_DIM_ELEVATION if band == 'Elevation' else VAE_LATENT_DIM_OTHERS
        vae = TransformerVAE(
            input_channels=1,
            input_height=window_size,
            input_width=window_size,
            num_heads=VAE_NUM_HEADS,
            latent_dim=latent_dim,
            dropout_rate=VAE_DROPOUT_RATE
        )

        model_path = model_dir / f"{band}.pth"
        if not model_path.exists():
            model_path_alt = model_dir / band / f"{band}.pth"
            if model_path_alt.exists():
                model_path = model_path_alt
            else:
                raise FileNotFoundError(f"Model file not found for band {band} at {model_path} or {model_path_alt}")

        state_dict = torch.load(model_path, map_location='cpu')
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            logger.info(f"Removed 'module.' prefix from state_dict keys for {band}")

        vae.load_state_dict(state_dict)
        vae = accelerator.prepare(vae)
        vae.eval()
        vae_models[band] = vae
        logger.info(f"Loaded and prepared model for band: {band}")

    # Compute Normalization Statistics
    logger.info("Calculating normalization statistics for specified bands...")
    band_statistics = {}
    for band_name in bands_list_order:
        if band_name in NORMALIZED_BANDS:
            band_idx = band_name_to_index[band_name]
            mean, std = compute_band_statistics(inference_dataset, band_name, band_idx, batch_size=args.batch_size)
            band_statistics[band_name] = (mean, std)
            logger.info(f"Computed statistics for {band_name}: mean={mean:.4f}, std={std:.4f}")
        else:
            logger.info(f"Skipping statistics calculation for non-normalized band: {band_name}")

    # Determine Time Range for Inference
    start_year_index = max(0, args.inference_year - args.time_before - TIME_BEGINNING)
    end_year_index = args.inference_year - TIME_BEGINNING
    if end_year_index < 0 or end_year_index >= TOTAL_YEARS_DATA:
        raise ValueError(f"End year index {end_year_index} out of bounds for data range [0, {TOTAL_YEARS_DATA-1}]")
    if start_year_index > end_year_index:
        raise ValueError(f"Start year index {start_year_index} > end year index {end_year_index}")
    inference_time_indices = list(range(start_year_index, end_year_index + 1))
    num_inference_years = len(inference_time_indices)
    logger.info(f"Inference for years {TIME_BEGINNING + start_year_index} to {TIME_BEGINNING + end_year_index}")

    # Prepare Storage for Results
    all_coordinates_list = []
    all_elevation_latents_z_list = []
    other_bands_latents_z_batches = {band: [] for band in bands_list_order if band != 'Elevation'}

    # Inference Loop
    logger.info("Starting inference to extract latent variable z...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Inference Progress")):
            try:
                # Batch format: longitude, latitude, features, encoding_tensor, oc
                lon, lat, features_batch, encoding_tensor, _ = batch
                current_batch_size = lon.size(0)

                coords_batch = torch.stack((lon.cpu(), lat.cpu()), dim=1).numpy()
                all_coordinates_list.append(coords_batch)

                batch_elevation_latent_z = None
                batch_other_latents_z = {
                    band: torch.zeros(current_batch_size, num_inference_years, VAE_LATENT_DIM_OTHERS, device='cpu')
                    for band in other_bands_latents_z_batches.keys()
                }

                for band_idx, band_name in enumerate(bands_list_order):
                    vae = vae_models[band_name]
                    features_band = features_batch[:, band_idx, :, :, :]  # Shape: (batch_size, W, H, T)

                    if band_name == 'Elevation':
                        features_elev_slice = features_band[:, :, :, 0].float()
                        features_elev_slice = torch.nan_to_num(features_elev_slice, nan=0.0)
                        if band_name in NORMALIZED_BANDS:
                            mean, std = band_statistics[band_name]
                            features_elev_slice = (features_elev_slice - mean) / std
                        features_elev_vae_input = features_elev_slice.unsqueeze(1)  # (B, 1, W, H)

                        with torch.autocast(device_type=accelerator.device_type, enabled=accelerator.mixed_precision != 'no'):
                            mu, log_var = vae.encode(features_elev_vae_input)[:2]  # Assuming encode returns (mu, log_var, ...)
                            z = vae.reparameterize(mu, log_var)
                        batch_elevation_latent_z = z.cpu()
                        del features_elev_slice, features_elev_vae_input, mu, log_var, z

                    else:
                        for i, time_idx in enumerate(inference_time_indices):
                            features_t_slice = features_band[:, :, :, time_idx].float()
                            features_t_slice = torch.nan_to_num(features_t_slice, nan=0.0)
                            if band_name in NORMALIZED_BANDS:
                                mean, std = band_statistics[band_name]
                                features_t_slice = (features_t_slice - mean) / std
                            features_t_vae_input = features_t_slice.unsqueeze(1)  # (B, 1, W, H)

                            with torch.autocast(device_type=accelerator.device_type, enabled=accelerator.mixed_precision != 'no'):
                                mu, log_var = vae.encode(features_t_vae_input)[:2]
                                z = vae.reparameterize(mu, log_var)
                            batch_other_latents_z[band_name][:, i, :] = z.cpu()
                            del features_t_slice, features_t_vae_input, mu, log_var, z

                if batch_elevation_latent_z is not None:
                    all_elevation_latents_z_list.append(batch_elevation_latent_z)
                for band in batch_other_latents_z:
                    other_bands_latents_z_batches[band].append(batch_other_latents_z[band])

                del lon, lat, features_batch, encoding_tensor, batch_elevation_latent_z, batch_other_latents_z
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                logger.warning(f"Skipping batch {batch_idx} due to error.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    logger.info("Inference loop finished. Consolidating results...")

    # Consolidate Results
    final_coordinates = np.concatenate(all_coordinates_list, axis=0) if all_coordinates_list else np.array([])
    final_elevation_latent_z = torch.cat(all_elevation_latents_z_list, dim=0).numpy() if all_elevation_latents_z_list else np.array([])
    final_other_latents_z = {}
    for band in other_bands_latents_z_batches:
        if other_bands_latents_z_batches[band]:
            final_other_latents_z[band] = torch.cat(other_bands_latents_z_batches[band], dim=0).numpy()
        else:
            logger.warning(f"No latent vectors collected for band {band}. Saving empty array.")
            final_other_latents_z[band] = np.empty((0, num_inference_years, VAE_LATENT_DIM_OTHERS))

    # Create Output Data Structure
    output_data = {
        'coordinates': final_coordinates,
        'elevation_latent_z': final_elevation_latent_z,
        'other_bands_latent_z': final_other_latents_z,
        'inference_years_range': list(range(TIME_BEGINNING + start_year_index, TIME_BEGINNING + end_year_index + 1)),
        'bands_order': bands_list_order
    }

    # Save Output
    logger.info(f"Saving consolidated results (latent z) to {output_path}...")
    np.save(output_path, output_data, allow_pickle=True)
    logger.info("Inference finished successfully.")