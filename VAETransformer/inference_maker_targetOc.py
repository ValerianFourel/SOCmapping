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
import json

# Import from training script's modules
from dataloader.dataframe_loader import separate_and_add_data_1mil
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataloader.dataloaderMultiYears1Mil import NormalizedMultiRasterDataset1MilMultiYears
from config import (
    TIME_BEGINNING, TIME_END, MAX_OC, bands_list_order, window_size, time_before,
    file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Yearly, DataYearly,
    SamplesCoordinates_Seasonally, DataSeasonally
)
from dataloader.dataframe_loader import separate_and_add_data, filter_dataframe
from modelTransformerVAE import TransformerVAE  # Assuming this is the VAE model used

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIME_END = int(TIME_END)
TIME_BEGINNING = int(TIME_BEGINNING)
# Configuration
TOTAL_YEARS_DATA = TIME_END - TIME_BEGINNING + 1
VAE_NUM_HEADS = 16
VAE_LATENT_DIM_ELEVATION = 24
VAE_LATENT_DIM_OTHERS = 48
VAE_DROPOUT_RATE = 0.3
NORMALIZED_BANDS = ['LST', 'MODIS_NPP', 'TotalEvapotranspiration']
VAE_BATCH_SIZE = 256

# Precomputed Normalization Statistics (from your logs)
BAND_STATISTICS = {
    'LST': (14376.5341796875, 136.1621551513672),
    'MODIS_NPP': (3915.517822265625, 10055.5634765625),
    'TotalEvapotranspiration': (120.5914306640625, 18.143165588378906)
}

# Helper Functions
def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
    """Create a balanced dataset by binning OC values and resampling."""
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    inference_dfs = []
    
    for bin_idx in range(len(bin_counts)):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) > 0:
            if len(bin_data) < min_samples:
                resampled = bin_data.sample(n=min_samples, replace=True)
                inference_dfs.append(resampled)
            else:
                inference_dfs.append(bin_data)
        
    if not inference_dfs:
        raise ValueError("No inference data available after binning")
        
    inference_df = pd.concat(inference_dfs).drop('bin', axis=1)
    return inference_df

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run VAE Inference for SOC prediction')
    parser.add_argument('--model_dir', type=str, required=True, help='Base directory containing VAEsFinal subfolder with band-specific model .pth files')
    parser.add_argument('--inference_year', type=int, required=True, help=f'Target year for inference ({TIME_BEGINNING}-{TIME_END})')
    parser.add_argument('--time_before', type=int, default=time_before, help='Number of years before inference_year to include')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output .parquet file containing latent z and oc')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--preload_dataset', action='store_true', help='Preload dataset into memory')
    return parser.parse_args()

# Main Inference Function
if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no')
    output_path = Path(args.output_file).with_suffix('.parquet')  # Ensure .parquet extension
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_base_dir = Path(args.model_dir) / 'VAEsFinal'  # Point to VAEsFinal subfolder

    # Validate Inference Year
    if not (TIME_BEGINNING <= args.inference_year <= TIME_END):
        raise ValueError(f"Inference year {args.inference_year} must be between {TIME_BEGINNING} and {TIME_END}")

    # Compute Inference Time Indices
    requested_years = list(range(args.inference_year - args.time_before, args.inference_year + 1))
    inference_time_indices = []
    actual_inference_years = []
    for year in requested_years:
        if TIME_BEGINNING <= year <= TIME_END:
            index = year - TIME_BEGINNING
            inference_time_indices.append(index)
            actual_inference_years.append(year)
        else:
            logger.warning(f"Year {year} is outside data range [{TIME_BEGINNING}, {TIME_END}]. Skipping.")
    if not inference_time_indices:
        raise ValueError(f"No valid years found for inference period {requested_years} within [{TIME_BEGINNING}, {TIME_END}]")
    num_inference_years = len(inference_time_indices)
    logger.info(f"Performing inference for {num_inference_years} years: {actual_inference_years}")
    logger.info(f"Data indices (0-based from {TIME_BEGINNING}): {inference_time_indices}")

    # Load Dataset Coordinates and Paths
    logger.info("Loading dataset coordinates and paths...")
    df = filter_dataframe(str(TIME_BEGINNING), str(TIME_END), MAX_OC)
    df_full_inference_oc = create_balanced_dataset(df)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    # Instantiate Dataset
    logger.info(f"Instantiating dataset for inference (preload={args.preload_dataset})...")
    inference_dataset = MultiRasterDatasetMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path,
        data_array_subfolders=data_array_path,
        dataframe=df_full_inference_oc,
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
    logger.info(f"Loading VAE models from {model_base_dir}...")
    vae_models = {}
    band_name_to_index = {band_name: i for i, band_name in enumerate(bands_list_order)}

    for band in bands_list_order:
        latent_dim = VAE_LATENT_DIM_ELEVATION if band == 'Elevation' else VAE_LATENT_DIM_OTHERS
        num_heads = VAE_NUM_HEADS if band == 'Elevation' else VAE_NUM_HEADS * 2
        vae = TransformerVAE(
            input_channels=1,
            input_height=window_size,
            input_width=window_size,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout_rate=VAE_DROPOUT_RATE
        )

        model_path = model_base_dir / band / f"{band}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for band {band} at {model_path}")

        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            logger.info(f"Removed 'module.' prefix from state_dict keys for {band}")

        vae.load_state_dict(state_dict)
        vae = accelerator.prepare(vae)
        vae.eval()
        vae_models[band] = vae
        logger.info(f"Loaded and prepared model for band: {band}")

    # Log Precomputed Statistics
    logger.info("Using precomputed normalization statistics:")
    for band_name, (mean, std) in BAND_STATISTICS.items():
        logger.info(f"{band_name}: mean={mean:.4f}, std={std:.4f}")

    # Prepare Storage for Results
    all_coordinates_list = []
    all_elevation_latents_z_list = []
    all_oc_list = []
    other_bands_latents_z_batches = {band: [] for band in bands_list_order if band != 'Elevation'}

    # Inference Loop
    logger.info("Starting inference to extract latent variable z and oc...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Inference Progress")):
            try:
                # Batch format: longitude, latitude, features, encoding_tensor, oc
                lon, lat, features_batch, encoding_tensor, oc = batch
                current_batch_size = lon.size(0)
                time_dim_size = features_batch.shape[4]  # Actual time dimension size from dataset

                if time_dim_size != num_inference_years:
                    logger.warning(f"Batch {batch_idx}: Expected {num_inference_years} time steps, got {time_dim_size}. Adjusting indices.")
                    adjusted_indices = list(range(min(time_dim_size, num_inference_years)))
                    adjusted_years = actual_inference_years[-time_dim_size:] if time_dim_size < num_inference_years else actual_inference_years
                else:
                    adjusted_indices = inference_time_indices
                    adjusted_years = actual_inference_years

                coords_batch = torch.stack((lon.cpu(), lat.cpu()), dim=1).numpy()
                all_coordinates_list.append(coords_batch)
                all_oc_list.append(oc.cpu().numpy())  # Store oc values

                batch_elevation_latent_z = None
                batch_other_latents_z = {
                    band: torch.zeros(current_batch_size, len(adjusted_indices), VAE_LATENT_DIM_OTHERS, device='cpu')
                    for band in other_bands_latents_z_batches.keys()
                }

                for band_idx, band_name in enumerate(bands_list_order):
                    vae = vae_models[band_name]
                    features_band = features_batch[:, band_idx, :, :, :]  # Shape: (batch_size, W, H, T)

                    if band_name == 'Elevation':
                        # Elevation uses the first available time step (static)
                        features_elev_slice = features_band[:, :, :, 0].float()
                        features_elev_slice = torch.nan_to_num(features_elev_slice, nan=0.0)
                        # Elevation is not in NORMALIZED_BANDS, so no normalization
                        features_elev_vae_input = features_elev_slice.unsqueeze(1)  # (B, 1, W, H)

                        with torch.autocast(device_type=accelerator.device.type, enabled=accelerator.mixed_precision != 'no'):
                            mu, log_var = vae.encode(features_elev_vae_input)[:2]
                            z = vae.reparameterize(mu, log_var)
                        batch_elevation_latent_z = z.cpu()
                        del features_elev_slice, features_elev_vae_input, mu, log_var, z

                    else:
                        for i, time_idx in enumerate(range(time_dim_size)):  # Iterate over actual time steps
                            features_t_slice = features_band[:, :, :, time_idx].float()
                            features_t_slice = torch.nan_to_num(features_t_slice, nan=0.0)
                            if band_name in NORMALIZED_BANDS:
                                mean, std = BAND_STATISTICS[band_name]
                                features_t_slice = (features_t_slice - mean) / std
                            features_t_vae_input = features_t_slice.unsqueeze(1)  # (B, 1, W, H)

                            with torch.autocast(device_type=accelerator.device.type, enabled=accelerator.mixed_precision != 'no'):
                                mu, log_var = vae.encode(features_t_vae_input)[:2]
                                z = vae.reparameterize(mu, log_var)
                            batch_other_latents_z[band_name][:, i, :] = z.cpu()
                            del features_t_slice, features_t_vae_input, mu, log_var, z

                if batch_elevation_latent_z is not None:
                    all_elevation_latents_z_list.append(batch_elevation_latent_z)
                for band in batch_other_latents_z:
                    other_bands_latents_z_batches[band].append(batch_other_latents_z[band])

                del lon, lat, features_batch, encoding_tensor, oc, batch_elevation_latent_z, batch_other_latents_z
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
    final_oc = np.concatenate(all_oc_list, axis=0) if all_oc_list else np.array([])
    final_other_latents_z = {}
    for band in other_bands_latents_z_batches:
        if other_bands_latents_z_batches[band]:
            final_other_latents_z[band] = torch.cat(other_bands_latents_z_batches[band], dim=0).numpy()
        else:
            logger.warning(f"No latent vectors collected for band {band}. Saving empty array.")
            final_other_latents_z[band] = np.empty((0, time_dim_size, VAE_LATENT_DIM_OTHERS))

    # Prepare DataFrame for Parquet
    logger.info("Preparing DataFrame for Parquet output...")
    data_dict = {
        'longitude': final_coordinates[:, 0],
        'latitude': final_coordinates[:, 1],
        'oc': final_oc,
        'elevation_latent_z': [row.tolist() for row in final_elevation_latent_z]  # Convert to list for parquet compatibility
    }

    # Add other bands' latent z with time dimension
    for band in final_other_latents_z:
        for t, year in enumerate(actual_inference_years[-time_dim_size:]):  # Use the last available years
            col_name = f'{band}_latent_z_t{t}_year{year}'
            data_dict[col_name] = [row[t].tolist() for row in final_other_latents_z[band]]

    # Add metadata
    metadata = {
        'inference_years_range': actual_inference_years[-time_dim_size:],  # Adjust to actual years processed
        'bands_order': bands_list_order,
        'normalization_statistics': {band: {'mean': mean, 'std': std} for band, (mean, std) in BAND_STATISTICS.items()}
    }

    # Create DataFrame
    output_df = pd.DataFrame(data_dict)

    # Save to Parquet
    logger.info(f"Saving consolidated results to {output_path}...")
    output_df.to_parquet(output_path, engine='pyarrow', index=False)

    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info("Inference finished successfully.")