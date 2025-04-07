import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import logging
import gc
import os
from accelerate import Accelerator

# --- Import Dataset and Config ---
try:
    from config import (
        TIME_BEGINNING, TIME_END, MAX_OC, time_before,
        file_path_coordinates_Bavaria_1mil, bands_list_order, window_size, INFERENCE_TIME
    )
    from modelTransformerVAE import TransformerVAE
    from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules. Check paths and filenames. Error: {e}")
    exit(1)

from dataloader.dataloaderMultiYears1Mil import NormalizedMultiRasterDataset1MilMultiYears
INFERENCE_TIME = int(INFERENCE_TIME)

# --- Configuration ---
TIME_BEGINNING = int(TIME_BEGINNING)
TIME_END = int(TIME_END)
TOTAL_YEARS_DATA = TIME_END - TIME_BEGINNING + 1

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
    parser = argparse.ArgumentParser(description='Run VAE Inference for SOC prediction with multi-GPU support')
    parser.add_argument('--model_dir', type=str, required=True, help='Base directory containing VAEsFinal subfolder')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output .parquet file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU for inference')
    parser.add_argument('--coord_paths_file', type=str, default=None, help='Optional: Path to coordinate subfolders file')
    parser.add_argument('--data_paths_file', type=str, default=None, help='Optional: Path to data subfolders file')
    parser.add_argument('--preload_dataset', action='store_true', help='Preload dataset into memory')
    return parser.parse_args()

# --- Modified Dataset Class ---
class NormalizedMultiRasterDataset1MilMultiYears(NormalizedMultiRasterDataset1MilMultiYears):
    def __init__(self, samples_coordinates_array_subfolders, data_array_subfolders, dataframe, 
                 inference_time_indices=None, batch_size=64, num_workers=4, preload=True):
        super().__init__(samples_coordinates_array_subfolders, data_array_subfolders, dataframe, preload)
        self.inference_time_indices = inference_time_indices if inference_time_indices is not None else list(range(TOTAL_YEARS_DATA))
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
                num_inference_years = len(self.inference_time_indices)
                tensor = tensor.repeat(num_inference_years, 1, 1)
                encoding = encoding.repeat(num_inference_years)
            else:
                tensor = tensor[self.inference_time_indices]
                encoding = encoding[self.inference_time_indices]
            band_tensors.append(tensor)
            encodings.append(encoding)

        final_tensor = torch.stack(band_tensors).permute(0, 2, 3, 1)  # (bands, H, W, num_inference_years)
        encoding_tensor = torch.stack(encodings)  # (bands, num_inference_years)
        return longitude, latitude, final_tensor, encoding_tensor

# --- Main Inference Function ---
if __name__ == "__main__":
    args = parse_args()
    VAE_BATCH_SIZE = args.batch_size

    # Initialize Accelerator for multi-GPU support
    accelerator = Accelerator(
        mixed_precision='fp16' if torch.cuda.is_available() else 'no',
        device_placement=True  # Automatically place tensors/models on correct devices
    )

    # --- Compute Inference Time Indices ---
    requested_years = list(range(INFERENCE_TIME - time_before + 1, INFERENCE_TIME + 1))
    inference_time_indices = []
    actual_inference_years = []
    for year in requested_years:
        if TIME_BEGINNING <= year <= TIME_END:
            index = year - TIME_BEGINNING
            inference_time_indices.append(index)
            actual_inference_years.append(year)
        else:
            logger.warning(f"Year {year} outside data range [{TIME_BEGINNING}, {TIME_END}]. Skipping.")
    if not inference_time_indices:
        raise ValueError(f"No valid years for inference period {requested_years} within [{TIME_BEGINNING}, {TIME_END}].")
    num_inference_years = len(inference_time_indices)
    logger.info(f"Inference for {num_inference_years} years: {actual_inference_years}")
    logger.info(f"Data indices: {inference_time_indices}")

    output_path = Path(args.output_file).with_suffix('.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_base_dir = Path(args.model_dir) / 'VAEsFinal'

    # --- Precomputed Normalization Statistics ---
    band_statistics = {
        'MODIS_NPP': (3915.517822265625, 10055.5634765625),
        'TotalEvapotranspiration': (120.5914306640625, 18.143165588378906),
        'LST': (14374.1533203125, 139.667724609375)
    }
    for band in NORMALIZED_BANDS:
        logger.info(f"Stats for {band}: mean={band_statistics.get(band, (0, 1))[0]:.4f}, std={band_statistics.get(band, (0, 1))[1]:.4f}")

    # --- Load Dataset ---
    logger.info("Loading dataset coordinates and paths...")
    coordinates_df_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()

    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=coordinates_df_1mil,
        inference_time_indices=inference_time_indices,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count()),
        preload=args.preload_dataset
    )

    if len(inference_dataset) == 0:
        raise ValueError("Inference dataset is empty.")

    # Adjust batch size based on number of GPUs
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if min(4, os.cpu_count()) > 0 else False)

    # Prepare DataLoader with Accelerator for multi-GPU
    inference_loader = accelerator.prepare_data_loader(inference_loader)
    logger.info(f"DataLoader prepared for {accelerator.num_processes} GPUs.")

    # --- Load VAE Models ---
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

    state_dict = torch.load(model_path, map_location='cpu')
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    vae.load_state_dict(state_dict)
    vae.eval()
    vae_models[band] = accelerator.prepare_model(vae)  # Prepare model for multi-GPU
    logger.info(f"Loaded and prepared model for band: {band}")

# --- Prepare Storage for Results ---
all_coordinates_list = []
all_latents_z = {band: [] for band in bands_list_order}

# --- Inference Loop ---
logger.info("Starting inference across multiple GPUs...")
with torch.no_grad():
    batch_iterator = tqdm(inference_loader, desc=f"Inference Progress (Rank {accelerator.process_index})", 
                          total=len(inference_loader), disable=not accelerator.is_local_main_process)

    for batch_idx, batch in enumerate(batch_iterator):
        try:
            lon, lat, features_batch, _ = batch  # Automatically placed on correct device
            current_batch_size = lon.size(0)

            # Gather coordinates on main process
            coords_batch = torch.stack((lon, lat), dim=1)
            coords_batch_gathered = accelerator.gather(coords_batch).cpu().numpy()
            if accelerator.is_main_process:
                all_coordinates_list.append(coords_batch_gathered)

            batch_latents_z = {}

            for band_idx, band_name in enumerate(bands_list_order):
                vae = vae_models[band_name]
                # Access the underlying model if wrapped in DDP
                model = vae.module if isinstance(vae, torch.nn.parallel.DistributedDataParallel) else vae
                is_elevation = (band_name == 'Elevation')
                latent_dim = VAE_LATENT_DIM_ELEVATION if is_elevation else VAE_LATENT_DIM_OTHERS
                band_batch_results = torch.zeros(current_batch_size, num_inference_years, latent_dim, 
                                               device=accelerator.device)

                if is_elevation:
                    features_elev_slice = features_batch[:, band_idx, :, :, 0].float()
                    features_elev_slice = torch.nan_to_num(features_elev_slice, nan=0.0)
                    if band_name in NORMALIZED_BANDS and band_name in band_statistics:
                        mean, std = band_statistics[band_name]
                        features_elev_slice = (features_elev_slice - mean) / std
                    features_elev_vae_input = features_elev_slice.unsqueeze(1)
                    with torch.autocast(device_type='cuda', enabled=accelerator.mixed_precision == 'fp16'):
                        mu, log_var, _ = model.encode(features_elev_vae_input)
                        z = model.reparameterize(mu, log_var)
                    band_batch_results = z.unsqueeze(1).expand(-1, num_inference_years, -1)
                else:
                    for i in range(num_inference_years):
                        features_t_slice = features_batch[:, band_idx, :, :, i].float()
                        features_t_slice = torch.nan_to_num(features_t_slice, nan=0.0)
                        if band_name in NORMALIZED_BANDS and band_name in band_statistics:
                            mean, std = band_statistics[band_name]
                            features_t_slice = (features_t_slice - mean) / std
                        features_t_vae_input = features_t_slice.unsqueeze(1)
                        with torch.autocast(device_type='cuda', enabled=accelerator.mixed_precision == 'fp16'):
                            mu, log_var, _ = model.encode(features_t_vae_input)
                            z = model.reparameterize(mu, log_var)
                        band_batch_results[:, i, :] = z

                # Gather results across GPUs
                gathered_results = accelerator.gather(band_batch_results).cpu()
                if accelerator.is_main_process:
                    batch_latents_z[band_name] = gathered_results

                accelerator.free_memory()

            if accelerator.is_main_process:
                for band in bands_list_order:
                    all_latents_z[band].append(batch_latents_z[band])

            accelerator.free_memory()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx} on rank {accelerator.process_index}: {e}", exc_info=True)
            accelerator.free_memory()
            continue

    # --- Consolidate Results on Main Process ---
    if accelerator.is_main_process:
        logger.info("Consolidating results...")
        final_coordinates = np.concatenate(all_coordinates_list, axis=0) if all_coordinates_list else np.array([])
        total_samples_processed = final_coordinates.shape[0]

        df_data = {
            'longitude': final_coordinates[:, 0],
            'latitude': final_coordinates[:, 1]
        }

        for band in bands_list_order:
            is_elevation = (band == 'Elevation')
            latent_dim = VAE_LATENT_DIM_ELEVATION if is_elevation else VAE_LATENT_DIM_OTHERS
            if all_latents_z[band]:
                latents_array = torch.cat(all_latents_z[band], dim=0).numpy()
                logger.info(f"Consolidated '{band}' latents shape: {latents_array.shape}")
            else:
                latents_array = np.empty((total_samples_processed, num_inference_years, latent_dim))

            for year_idx in range(num_inference_years):
                year_suffix = year_idx + 1
                column_name = f"variable_latentspace_{band}_{year_suffix}"
                df_data[column_name] = [list(latents_array[i, year_idx, :]) for i in range(total_samples_processed)]

        output_df = pd.DataFrame(df_data)
        logger.info(f"DataFrame shape: {output_df.shape}")

        logger.info(f"Saving to {output_path}...")
        output_path = "/fast/vfourel/MasterThesis/VAETransformerEmbeddings"
        output_df.to_parquet(output_path, engine='pyarrow', index=False)
        logger.info("Inference completed.")
    
    accelerator.wait_for_everyone()