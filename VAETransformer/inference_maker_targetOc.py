import numpy as np
import torch
from torch.utils.data import DataLoader
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

# Precomputed Normalization Statistics
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

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run VAE Inference to extract latent space for all OC values')
    parser.add_argument('--model_dir', type=str, required=True, help='Base directory containing VAEsFinal subfolder with band-specific model .pth files')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output .parquet file containing latent z and oc')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--preload_dataset', action='store_true', help='Preload dataset into memory')
    return parser.parse_args()

# Main Inference Function
if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no')
    output_path = Path(args.output_file).with_suffix('.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_base_dir = Path(args.model_dir) / 'VAEsFinal'

    # Load Dataset Coordinates and Paths
    logger.info("Loading dataset coordinates and paths...")
    df = filter_dataframe(str(TIME_BEGINNING), str(TIME_END), MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    # Instantiate Dataset
    logger.info(f"Instantiating dataset for inference (preload={args.preload_dataset})...")
    inference_dataset = MultiRasterDatasetMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path,
        data_array_subfolders=data_array_path,
        dataframe=df,
        time_before=time_before  # Use all available years
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

    # Define all years for metadata
    all_years = list(range(TIME_BEGINNING, TIME_END + 1))
    num_total_years = len(all_years)
    logger.info(f"Processing all {num_total_years} years: {all_years}")

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
    all_year_of_sample_list = []  # New list to store year_of_sample
    other_bands_latents_z_batches = {band: [] for band in bands_list_order if band != 'Elevation'}

    # Inference Loop
    logger.info("Starting inference to extract latent variable z and oc for all data...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Inference Progress")):
            try:
                # Batch format: longitude, latitude, features, encoding_tensor, oc, year_of_sample
                lon, lat, features_batch, encoding_tensor, oc, year_of_sample = batch
                current_batch_size = lon.size(0)
                time_dim_size = features_batch.shape[4]  # Actual time dimension size from dataset

                if time_dim_size != time_before:
                    logger.warning(f"Batch {batch_idx}: Expected {time_before} time steps, got {time_dim_size}.")

                coords_batch = torch.stack((lon.cpu(), lat.cpu()), dim=1).numpy()
                all_coordinates_list.append(coords_batch)
                all_oc_list.append(oc.cpu().numpy())
                all_year_of_sample_list.append(year_of_sample.cpu().numpy())  # Store year_of_sample

                batch_elevation_latent_z = None
                batch_other_latents_z = {
                    band: torch.zeros(current_batch_size, time_dim_size, VAE_LATENT_DIM_OTHERS, device='cpu')
                    for band in other_bands_latents_z_batches.keys()
                }

                for band_idx, band_name in enumerate(bands_list_order):
                    vae = vae_models[band_name]
                    features_band = features_batch[:, band_idx, :, :, :]  # Shape: (batch_size, W, H, T)

                    if band_name == 'Elevation':
                        # Elevation uses the first available time step (static)
                        features_elev_slice = features_band[:, :, :, 0].float()
                        features_elev_slice = torch.nan_to_num(features_elev_slice, nan=0.0)
                        features_elev_vae_input = features_elev_slice.unsqueeze(1)  # (B, 1, W, H)

                        with torch.autocast(device_type=accelerator.device.type, enabled=accelerator.mixed_precision != 'no'):
                            mu, log_var = vae.encode(features_elev_vae_input)[:2]
                            z = vae.reparameterize(mu, log_var)
                        batch_elevation_latent_z = z.cpu()
                        del features_elev_slice, features_elev_vae_input, mu, log_var, z

                    else:
                        for t in range(time_dim_size):
                            features_t_slice = features_band[:, :, :, t].float()
                            features_t_slice = torch.nan_to_num(features_t_slice, nan=0.0)
                            if band_name in NORMALIZED_BANDS:
                                mean, std = BAND_STATISTICS[band_name]
                                features_t_slice = (features_t_slice - mean) / std
                            features_t_vae_input = features_t_slice.unsqueeze(1)  # (B, 1, W, H)

                            with torch.autocast(device_type=accelerator.device.type, enabled=accelerator.mixed_precision != 'no'):
                                mu, log_var = vae.encode(features_t_vae_input)[:2]
                                z = vae.reparameterize(mu, log_var)
                            batch_other_latents_z[band_name][:, t, :] = z.cpu()
                            del features_t_slice, features_t_vae_input, mu, log_var, z

                if batch_elevation_latent_z is not None:
                    all_elevation_latents_z_list.append(batch_elevation_latent_z)
                for band in batch_other_latents_z:
                    other_bands_latents_z_batches[band].append(batch_other_latents_z[band])

                del lon, lat, features_batch, encoding_tensor, oc, year_of_sample, batch_elevation_latent_z, batch_other_latents_z
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
    final_year_of_sample = np.concatenate(all_year_of_sample_list, axis=0) if all_year_of_sample_list else np.array([])  # Consolidate year_of_sample
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
        'year_of_sample': final_year_of_sample,  # Add year_of_sample to DataFrame
        'elevation_latent_z': [row.tolist() for row in final_elevation_latent_z]
    }

    # Add other bands' latent z with time dimension (renamed to 1, 2, ..., time_before)
    for band in final_other_latents_z:
        for t in range(time_dim_size):  # Use 1-based indexing for column names
            col_name = f'{band}_latent_z_{t + 1}'  # e.g., LST_latent_z_1, LST_latent_z_2, etc.
            data_dict[col_name] = [row[t].tolist() for row in final_other_latents_z[band]]

    # Add metadata
    metadata = {
        'years_range': all_years[-time_dim_size:],  # Adjust to actual years processed
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

    # Verification Section
    logger.info("Starting verification of output .parquet and metadata...")

    # Load the saved parquet file
    try:
        loaded_df = pd.read_parquet(output_path)
        logger.info(f"Successfully loaded {output_path} with {len(loaded_df)} rows and {len(loaded_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        raise

    # Check basic structure
    expected_columns = ['longitude', 'latitude', 'oc', 'year_of_sample', 'elevation_latent_z']  # Updated with year_of_sample
    num_other_bands = len(bands_list_order) - 1  # Exclude Elevation
    expected_time_cols_per_band = time_dim_size
    expected_total_cols = len(expected_columns) + num_other_bands * expected_time_cols_per_band

    if len(loaded_df.columns) != expected_total_cols:
        logger.error(f"Column count mismatch: Expected {expected_total_cols}, got {len(loaded_df.columns)}")
    else:
        logger.info(f"Column count matches expected: {expected_total_cols}")

    # Verify column names
    missing_cols = [col for col in expected_columns if col not in loaded_df.columns]
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
    else:
        logger.info("All basic columns (longitude, latitude, oc, year_of_sample, elevation_latent_z) present")

    # Check Elevation latent dimensions
    elev_z_sample = loaded_df['elevation_latent_z'].iloc[0]
    if len(elev_z_sample) != VAE_LATENT_DIM_ELEVATION:
        logger.error(f"Elevation latent z dimension mismatch: Expected {VAE_LATENT_DIM_ELEVATION}, got {len(elev_z_sample)}")
    else:
        logger.info(f"Elevation latent z dimension correct: {VAE_LATENT_DIM_ELEVATION}")

    # Check other bands' latent dimensions and naming
    for band in bands_list_order:
        if band != 'Elevation':
            for t in range(time_dim_size):
                col_name = f'{band}_latent_z_{t + 1}'  # Updated naming convention
                if col_name not in loaded_df.columns:
                    logger.error(f"Missing column: {col_name}")
                else:
                    z_sample = loaded_df[col_name].iloc[0]
                    if len(z_sample) != VAE_LATENT_DIM_OTHERS:
                        logger.error(f"{col_name} latent z dimension mismatch: Expected {VAE_LATENT_DIM_OTHERS}, got {len(z_sample)}")
                    else:
                        logger.info(f"{col_name} present with correct dimension: {VAE_LATENT_DIM_OTHERS}")

    # Verify data integrity
    if loaded_df['longitude'].isnull().any() or loaded_df['latitude'].isnull().any():
        logger.error("Null values found in longitude or latitude")
    else:
        logger.info("No null values in longitude or latitude")

    if loaded_df['oc'].isnull().any():
        logger.error(f"Null values found in oc: {loaded_df['oc'].isnull().sum()} instances")
    else:
        logger.info("No null values in oc")

    if loaded_df['year_of_sample'].isnull().any():
        logger.error(f"Null values found in year_of_sample: {loaded_df['year_of_sample'].isnull().sum()} instances")
    else:
        logger.info("No null values in year_of_sample")

    # Check year_of_sample values
    unique_years = loaded_df['year_of_sample'].unique()
    if not all(year in all_years for year in unique_years):
        logger.warning(f"year_of_sample contains years not in expected range {all_years}: {unique_years}")
    else:
        logger.info(f"year_of_sample values are within expected range: {unique_years}")

    # Check metadata
    try:
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        logger.info("Metadata loaded successfully")
        
        if set(loaded_metadata['bands_order']) != set(bands_list_order):
            logger.error(f"Metadata bands_order mismatch: Expected {bands_list_order}, got {loaded_metadata['bands_order']}")
        else:
            logger.info("Metadata bands_order matches configuration")

        if len(loaded_metadata['years_range']) != time_dim_size:
            logger.error(f"Metadata years_range length mismatch: Expected {time_dim_size}, got {len(loaded_metadata['years_range'])}")
        else:
            logger.info(f"Metadata years_range length correct: {time_dim_size}")

        for band in NORMALIZED_BANDS:
            if band not in loaded_metadata['normalization_statistics']:
                logger.error(f"Missing normalization stats for {band} in metadata")
            else:
                expected_mean, expected_std = BAND_STATISTICS[band]
                loaded_mean = loaded_metadata['normalization_statistics'][band]['mean']
                loaded_std = loaded_metadata['normalization_statistics'][band]['std']
                if not (np.isclose(loaded_mean, expected_mean) and np.isclose(loaded_std, expected_std)):
                    logger.error(f"Normalization stats mismatch for {band}: Expected ({expected_mean}, {expected_std}), got ({loaded_mean}, {loaded_std})")
                else:
                    logger.info(f"Normalization stats for {band} correct")
    except Exception as e:
        logger.error(f"Failed to verify metadata: {e}")
        raise

    # Check file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Parquet file size: {file_size_mb:.2f} MB")

    logger.info("Verification completed.")