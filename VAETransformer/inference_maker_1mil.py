import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import logging
from accelerate import Accelerator
import gc
import os

# --- Import Dataset and Config ---
try:
    from config import (
        TIME_BEGINNING, TIME_END, MAX_OC,
        file_path_coordinates_Bavaria_1mil, bands_list_order, window_size
    )
    from modelTransformerVAE import TransformerVAE
    from dataloader.dataframe_loader import separate_and_add_data_1mil_inference, separate_and_add_data_1mil
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules. Check paths and filenames. Error: {e}")
    exit(1)

from dataloader.dataloaderMultiYears1Mil import NormalizedMultiRasterDataset1MilMultiYears

# --- Configuration ---
TIME_BEGINNING = int(TIME_BEGINNING)  # e.g., 2007
TIME_END = int(TIME_END)              # e.g., 2023
TOTAL_YEARS_DATA = TIME_END - TIME_BEGINNING + 1  # Should be 17

VAE_NUM_HEADS = 16
VAE_LATENT_DIM_ELEVATION = 24
VAE_LATENT_DIM_OTHERS = 48
VAE_DROPOUT_RATE = 0.3
NORMALIZED_BANDS = ['LST', 'MODIS_NPP', 'TotalEvapotranspiration']

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Run VAE Inference for SOC prediction')
    parser.add_argument('--model_dir', type=str, required=True, help='Base directory containing VAEsFinal subfolder with band-specific model .pth files')
    parser.add_argument('--inference_year', type=int, required=True, help='The target year for inference (e.g., 2023)')
    parser.add_argument('--time_before', type=int, default=5, help='Number of years *before* and including inference_year to process (e.g., 5)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output .npy file containing latent z')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--coord_paths_file', type=str, default=None, help='Optional: Path to a file listing coordinate array subfolders')
    parser.add_argument('--data_paths_file', type=str, default=None, help='Optional: Path to a file listing data array subfolders')
    parser.add_argument('--preload_dataset', action='store_true', help='Preload dataset into memory (requires significant RAM)')
    return parser.parse_args()

# --- Modified Dataset Class ---
class NormalizedMultiRasterDataset1MilMultiYears(NormalizedMultiRasterDataset1MilMultiYears):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, inference_time_indices=None, accelerator=None, batch_size=64, num_workers=4, preload=True):
        super().__init__(samples_coordinates_array_subfolders, data_array_subfolders, dataframe, preload)
        self.inference_time_indices = inference_time_indices if inference_time_indices is not None else list(range(TOTAL_YEARS_DATA))
        self.accelerator = accelerator if accelerator else Accelerator()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __getitem__(self, idx):
        longitude, latitude = None, None
        band_tensors = []
        encodings = []

        for band in bands_list_order:
            lon, lat, tensor, encoding = self.band_datasets[band][idx]
            if longitude is None:
                longitude, latitude = lon, lat
            if band == 'Elevation':
                # Static band: repeat single time step to match inference years
                num_inference_years = len(self.inference_time_indices)
                tensor = tensor.repeat(num_inference_years, 1, 1)  # From (1, H, W) to (num_inference_years, H, W)
                encoding = encoding.repeat(num_inference_years)    # From (1,) to (num_inference_years,)
            else:
                # Time-varying band: select only requested time steps
                tensor = tensor[self.inference_time_indices]       # From (17, H, W) to (num_inference_years, H, W)
                encoding = encoding[self.inference_time_indices]   # From (17,) to (num_inference_years,)
            band_tensors.append(tensor)
            encodings.append(encoding)

        final_tensor = torch.stack(band_tensors)  # Shape: (bands, num_inference_years, H, W)
        final_tensor = final_tensor.permute(0, 2, 3, 1)  # Shape: (bands, H, W, num_inference_years)
        encoding_tensor = torch.stack(encodings)  # Shape: (bands, num_inference_years)
        return longitude, latitude, final_tensor, encoding_tensor

# --- Main Inference Function ---
if __name__ == "__main__":
    args = parse_args()
    VAE_BATCH_SIZE = args.batch_size

    # --- Compute Inference Time Indices ---
    requested_years = list(range(args.inference_year - args.time_before + 1, args.inference_year + 1))
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
        raise ValueError(f"No valid years found for inference period {requested_years} within [{TIME_BEGINNING}, {TIME_END}].")
    num_inference_years = len(inference_time_indices)
    logger.info(f"Performing inference for {num_inference_years} years: {actual_inference_years}")
    logger.info(f"Data indices (0-based from {TIME_BEGINNING}): {inference_time_indices}")

    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no')
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_base_dir = Path(args.model_dir) / 'VAEsFinal'

    # --- Precomputed Normalization Statistics ---
    logger.info("Using precomputed normalization statistics...")
    band_statistics = {
        'MODIS_NPP': (3915.517822265625, 10055.5634765625),
        'TotalEvapotranspiration': (120.5914306640625, 18.143165588378906),
        'LST': (14374.1533203125, 139.667724609375)
    }
    for band in NORMALIZED_BANDS:
        if band not in band_statistics:
            logger.warning(f"Precomputed statistics missing for {band}. It will NOT be normalized.")
        else:
            logger.info(f"Stats for {band}: mean={band_statistics[band][0]:.4f}, std={band_statistics[band][1]:.4f}")

    # --- Load Dataset Coordinates and Paths ---
    logger.info("Loading Bavaria 1M dataset coordinates and paths...")
    coordinates_df_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()

    # --- Instantiate Dataset ---
    logger.info(f"Instantiating dataset (preload={args.preload_dataset})...")
    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=coordinates_df_1mil,
        inference_time_indices=inference_time_indices,
        accelerator=accelerator,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count()),
        preload=args.preload_dataset
    )

    if len(inference_dataset) == 0:
        raise ValueError("Inference dataset is empty. Check paths and dataframe.")

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if min(4, os.cpu_count()) > 0 else False,
    )
    logger.info("Preparing DataLoader with Accelerator...")
    inference_loader = accelerator.prepare(inference_loader)
    logger.info("DataLoader prepared.")

    # --- Load VAE Models ---
    logger.info(f"Loading VAE models from {model_base_dir}...")
    vae_models = {}
    band_name_to_index = {band_name: i for i, band_name in enumerate(bands_list_order)}

    for band in bands_list_order:
        is_elevation = (band == 'Elevation')
        latent_dim = VAE_LATENT_DIM_ELEVATION if is_elevation else VAE_LATENT_DIM_OTHERS
        num_heads = VAE_NUM_HEADS if is_elevation else VAE_NUM_HEADS * 2
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
            raise FileNotFoundError(f"Model file not found for {band} at {model_path}")

        logger.info(f"Loading state dict for {band} from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            logger.info(f"Removed 'module.' prefix from state_dict keys for {band}")
        vae.load_state_dict(state_dict)
        vae.eval()
        vae_models[band] = vae
        logger.info(f"Loaded model for band: {band}")

    # --- Prepare Storage for Results ---
    all_coordinates_list = []
    all_latents_z = {band: [] for band in bands_list_order}

    # --- Inference Loop ---
    logger.info("Starting inference...")
    for band in vae_models:
        vae_models[band] = accelerator.prepare(vae_models[band])
    logger.info("Models prepared.")

    with torch.no_grad():
        batch_iterator = tqdm(inference_loader, desc="Inference Progress", total=len(inference_loader))
        device = "cuda" if torch.cuda.is_available() else "cpu"


        for batch_idx, batch in enumerate(batch_iterator):
            try:
                lon, lat, features_batch, _ = batch  # Shape: (batch, bands, H, W, num_inference_years)
                current_batch_size = lon.size(0)

                coords_batch = torch.stack((lon.cpu(), lat.cpu()), dim=1).numpy()
                all_coordinates_list.append(coords_batch)

                batch_latents_z = {}

                for band_idx, band_name in enumerate(bands_list_order):
                    vae = vae_models[band_name]
                    is_elevation = (band_name == 'Elevation')
                    latent_dim = VAE_LATENT_DIM_ELEVATION if is_elevation else VAE_LATENT_DIM_OTHERS
                    band_batch_results = torch.zeros(current_batch_size, num_inference_years, latent_dim, device='cpu')

                    if is_elevation:
                        features_elev_slice = features_batch[:, band_idx, :, :, 0].float()
                        features_elev_slice = torch.nan_to_num(features_elev_slice, nan=0.0)

                        if band_name in NORMALIZED_BANDS and band_name in band_statistics:
                            mean, std = band_statistics[band_name]
                            features_elev_slice = (features_elev_slice - mean) / std

                        features_elev_vae_input = features_elev_slice.unsqueeze(1)
                        with torch.autocast(device_type=device, enabled=accelerator.mixed_precision != 'no'):
                            mu, log_var, _ = vae.encode(features_elev_vae_input)
                            z = vae.reparameterize(mu, log_var)
                        band_batch_results = z.unsqueeze(1).expand(-1, num_inference_years, -1).cpu()
                        del features_elev_slice, features_elev_vae_input, mu, log_var, z

                    else:  # Time-varying bands
                        for i in range(num_inference_years):
                            features_t_slice = features_batch[:, band_idx, :, :, i].float()
                            features_t_slice = torch.nan_to_num(features_t_slice, nan=0.0)

                            if band_name in NORMALIZED_BANDS and band_name in band_statistics:
                                mean, std = band_statistics[band_name]
                                features_t_slice = (features_t_slice - mean) / std

                            features_t_vae_input = features_t_slice.unsqueeze(1)
                            with torch.autocast(device_type="cuda", enabled=accelerator.mixed_precision != 'no'):
                                mu, log_var, _ = vae.encode(features_t_vae_input)
                                z = vae.reparameterize(mu, log_var)
                            band_batch_results[:, i, :] = z.cpu()
                            del features_t_slice, features_t_vae_input, mu, log_var, z

                    batch_latents_z[band_name] = band_batch_results
                    del band_batch_results
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                for band in bands_list_order:
                    all_latents_z[band].append(batch_latents_z[band])

                del lon, lat, features_batch, batch_latents_z, coords_batch
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

    # --- Consolidate Results ---
    final_coordinates = np.concatenate(all_coordinates_list, axis=0) if all_coordinates_list else np.array([])
    final_latents_z = {}
    total_samples_processed = final_coordinates.shape[0]

    if total_samples_processed == 0:
        logger.warning("No samples processed successfully. Output will contain empty arrays.")

    for band in bands_list_order:
        is_elevation = (band == 'Elevation')
        latent_dim = VAE_LATENT_DIM_ELEVATION if is_elevation else VAE_LATENT_DIM_OTHERS
        if all_latents_z[band]:
            final_latents_z[band] = torch.cat(all_latents_z[band], dim=0).numpy()
            logger.info(f"Consolidated '{band}' latents shape: {final_latents_z[band].shape}")
            if final_latents_z[band].shape[0] != total_samples_processed:
                logger.error(f"Mismatch in sample count for {band}: {final_latents_z[band].shape[0]} vs {total_samples_processed}")
        else:
            logger.warning(f"No latent vectors for {band}. Saving empty array.")
            final_latents_z[band] = np.empty((total_samples_processed, num_inference_years, latent_dim))

    # --- Create Output Data Structure ---
    output_data = {
        'coordinates': final_coordinates,
        'latents_z': final_latents_z,
        'inference_years': actual_inference_years,
        'bands_order': bands_list_order
    }

    # --- Save Output ---
    logger.info(f"Saving results to {output_path}...")
    np.save(output_path, output_data, allow_pickle=True)

    logger.info("Inference finished successfully.")