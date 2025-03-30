# inference_script_latent_z.py

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
    # Replace 'your_dataset_module' with the actual filename (without .py)
    from dataloader.dataloaderMultiYears1Mil import NormalizedMultiRasterDataset1MilMultiYears
    from modelTransformerVAE import TransformerVAE
    from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
except ImportError as e:
     logger.error(f"Failed to import necessary modules. Check paths and filenames. Error: {e}")
     exit(1)

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

# --- Statistics Computation (Revised for new Dataset structure) ---
# (Function compute_band_statistics remains the same as the previous version)
def compute_band_statistics(dataset: Dataset, band_name: str, band_index: int, batch_size: int = 512, sample_percentage: float = 0.01):
    """Compute mean and std for a specific band using a sample of the dataset."""
    total_samples = len(dataset)
    sample_size = max(1, int(total_samples * sample_percentage))
    sample_size = min(sample_size, total_samples)

    if sample_size == 0:
         logger.warning(f"Dataset size is 0 for band {band_name}, cannot compute statistics.")
         return 0.0, 1.0

    indices = np.random.choice(total_samples, size=sample_size, replace=False)
    subset = Subset(dataset, indices)
    # Use a reasonable num_workers based on your system
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()))

    all_band_features = []
    logger.info(f"Computing statistics for band '{band_name}' using {sample_size} samples...")

    is_elevation = (band_name == 'Elevation')

    for batch in tqdm(loader, desc=f"Computing Stats ({band_name})", leave=False):
        try:
            # Expected batch format: lon, lat, features, encoding
            # features shape: (batch_size, num_bands, W, H, T) where T=TOTAL_YEARS_DATA
            _, _, features_batch, _ = batch

            # Extract features for the specific band -> shape: (batch_size, W, H, T)
            band_features = features_batch[:, band_index, :, :, :]

            # Handle Elevation (T=1 implicitly or explicitly) vs other bands
            if is_elevation:
                # Shape should be (batch_size, W, H, 1), take slice -> (batch_size, W, H)
                if band_features.shape[-1] != 1:
                    logger.warning(f"Elevation band '{band_name}' has unexpected time dim {band_features.shape[-1]}. Using first slice.")
                current_features = band_features[:, :, :, 0].reshape(-1) # Flatten W, H, batch
            else:
                # Shape (batch_size, W, H, T), flatten all -> (batch_size * W * H * T)
                current_features = band_features.reshape(-1)

            # Convert to float, handle NaNs, and append
            current_features = torch.nan_to_num(current_features.float(), nan=0.0)
            all_band_features.append(current_features.cpu()) # Move to CPU before appending list
            del features_batch, band_features, current_features
        except Exception as e:
            logger.error(f"Error processing batch during stats computation for {band_name}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue # Skip batch on error

    if not all_band_features:
        logger.warning(f"No features collected for statistics computation for band {band_name}.")
        return 0.0, 1.0

    try:
        # Concatenate all features for the band
        all_band_features_tensor = torch.cat(all_band_features, dim=0)

        if all_band_features_tensor.numel() == 0:
            logger.warning(f"Collected features tensor is empty for band {band_name}.")
            return 0.0, 1.0

        # Calculate mean and std
        mean = torch.mean(all_band_features_tensor).item()
        std = torch.std(all_band_features_tensor).item()
    except Exception as e:
        logger.error(f"Error calculating final stats for {band_name}: {e}")
        mean, std = 0.0, 1.0 # Fallback values

    # Clean up memory
    del all_band_features, all_band_features_tensor, loader, subset
    gc.collect()

    # Avoid division by zero or very small std
    return mean, std if std > 1e-6 else 1.0


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Run VAE Inference for SOC prediction')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained VAE model .pth files (e.g., model_dir/NDVI.pth)')
    parser.add_argument('--inference_year', type=int, required=True, help=f'The target year for inference (between {TIME_BEGINNING} and {TIME_END})')
    parser.add_argument('--time_before', type=int, default=5, help='Number of years *before* inference_year to include (e.g., 5 means inference_year-5 to inference_year)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output .npy file containing latent z')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    # Num GPUs is implicitly handled by Accelerator
    # parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use (used by Accelerator)')
    parser.add_argument('--coord_paths_file', type=str, default=None, help='Optional: Path to a file listing coordinate array subfolders')
    parser.add_argument('--data_paths_file', type=str, default=None, help='Optional: Path to a file listing data array subfolders')
    parser.add_argument('--preload_dataset', action='store_true', help='Preload dataset into memory (requires significant RAM)')

    return parser.parse_args()

# --- Main Inference Function ---
if __name__ == "__main__":
    args = parse_args()
    # Accelerator handles device placement, mixed precision, and distributed setup
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no') # Use fp16 only if CUDA is available
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # --- Validate Inference Year ---
    if not (TIME_BEGINNING <= args.inference_year <= TIME_END):
        raise ValueError(f"Inference year {args.inference_year} must be between {TIME_BEGINNING} and {TIME_END}")

    # --- Load Dataset Coordinates and Paths ---
    logger.info("Loading Bavaria 1M dataset coordinates and paths...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)

    if args.coord_paths_file and args.data_paths_file:
         logger.info("Loading paths from provided files...")
         raise NotImplementedError("Loading paths from files not fully implemented yet.")
    else:
         logger.info("Using separate_and_add_data_1mil to get paths...")
         samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()

    # --- Instantiate Full Dataset and DataLoader for Inference ---
    logger.info(f"Instantiating dataset for inference (preload={args.preload_dataset})...")
    inference_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=train_dataset_df_full_1mil,
        # Remove batch_size/num_workers if not used by Dataset constructor
        # batch_size=args.batch_size,
        # num_workers=4,
        preload=args.preload_dataset
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()), # Use reasonable number of workers
        pin_memory=True if torch.cuda.is_available() else False, # Pin memory only if using CUDA
        persistent_workers=True if min(4, os.cpu_count()) > 0 else False # Improve loader speed
    )
    # Prepare loader with accelerator AFTER creating it
    inference_loader = accelerator.prepare(inference_loader)

    # --- Load VAE Models ---
    logger.info(f"Loading VAE models from {model_dir}...")
    vae_models = {}
    band_name_to_index = {band_name: i for i, band_name in enumerate(bands_list_order)}

    for band in bands_list_order:
        latent_dim = VAE_LATENT_DIM_ELEVATION if band == 'Elevation' else VAE_LATENT_DIM_OTHERS
        num_heads = VAE_NUM_HEADS

        # Instantiate model on CPU first
        vae = TransformerVAE(
            input_channels=1, input_height=window_size, input_width=window_size,
            num_heads=num_heads, latent_dim=latent_dim, dropout_rate=VAE_DROPOUT_RATE
        )

        model_path = model_dir / f"{band}.pth"
        if not model_path.exists():
            # Try finding model inside a subfolder named after the band (common pattern)
            model_path_alt = model_dir / band / f"{band}.pth"
            if model_path_alt.exists():
                model_path = model_path_alt
            else:
                raise FileNotFoundError(f"Model file not found for band {band} at {model_path} or {model_path_alt}")
        
        logger.info(f"Loading state dict for {band} from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu') # Load to CPU
        
        # Clean state dict if needed (e.g., remove 'module.' prefix)
        if any(key.startswith("module.") for key in state_dict.keys()):
             clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
             state_dict = clean_state_dict
             logger.info(f"Removed 'module.' prefix from state_dict keys for {band}")

        try:
            vae.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.error(f"Error loading state dict for {band}: {e}")
            logger.error("Model architecture might not match saved weights.")
            raise e # Re-raise error to stop execution

        # Prepare model with accelerator (moves to GPU if available)
        vae = accelerator.prepare(vae)
        vae.eval() # Set model to evaluation mode
        vae_models[band] = vae
        logger.info(f"Loaded and prepared model for band: {band}")

    # --- Compute Normalization Statistics ---
    # (Calculation loop remains the same, uses compute_band_statistics)
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


    # --- Determine Time Range for Inference ---
    # (Time range calculation remains the same)
    start_year_index = max(0, args.inference_year - args.time_before - TIME_BEGINNING)
    end_year_index = args.inference_year - TIME_BEGINNING # Inclusive index relative to TIME_BEGINNING
    if end_year_index < 0 or end_year_index >= TOTAL_YEARS_DATA:
         raise ValueError(f"Calculated end year index {end_year_index} is out of bounds for data range [0, {TOTAL_YEARS_DATA-1}] based on TIME_BEGINNING={TIME_BEGINNING}")
    if start_year_index > end_year_index:
         raise ValueError(f"Calculated start year index {start_year_index} is greater than end year index {end_year_index}")
    inference_time_indices = list(range(start_year_index, end_year_index + 1))
    num_inference_years = len(inference_time_indices)
    logger.info(f"Will perform inference for years {TIME_BEGINNING + start_year_index} to {TIME_BEGINNING + end_year_index} (Indices: {start_year_index} to {end_year_index} relative to TIME_BEGINNING)")

    # --- Prepare Storage for Results (Storing Z) ---
    all_coordinates_list = []
    all_elevation_latents_z_list = [] # Renamed list
    # Renamed dictionary and value lists
    other_bands_latents_z_batches = {band: [] for band in bands_list_order if band != 'Elevation'}

    # --- Inference Loop ---
    logger.info("Starting inference to extract latent variable z...")
    # Ensure model is in eval mode and disable gradients
    for model in vae_models.values():
        model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Inference Progress")):
            try:
                # Batch format: longitude, latitude, features, encoding
                lon, lat, features_batch, _ = batch # features_batch on accelerator device
                current_batch_size = lon.size(0) # features_batch is already on accelerator device

                # Store coordinates for this batch -> shape (batch_size, 2)
                # Move coords to CPU for numpy conversion
                coords_batch = torch.stack((lon.cpu(), lat.cpu()), dim=1).numpy()
                all_coordinates_list.append(coords_batch)

                # Temporary storage for current batch latents z
                batch_elevation_latent_z = None
                batch_other_latents_z = {band: torch.zeros(current_batch_size, num_inference_years, VAE_LATENT_DIM_OTHERS, device='cpu')
                                        for band in other_bands_latents_z_batches.keys()}

                for band_idx, band_name in enumerate(bands_list_order):
                    vae = vae_models[band_name] # Model is already prepared by accelerator

                    # Extract features for the current band -> shape: (batch_size, W, H, T)
                    # Feature batch is already on the correct device from the prepared loader
                    features_band = features_batch[:, band_idx, :, :, :]

                    # --- Handle Elevation (Static) ---
                    if band_name == 'Elevation':
                        if features_band.shape[-1] != 1:
                            logger.warning(f"Elevation band '{band_name}' has unexpected time dim {features_band.shape[-1]}. Using first slice.")
                        features_elev_slice = features_band[:, :, :, 0].float()
                        features_elev_slice = torch.nan_to_num(features_elev_slice, nan=0.0)

                        if band_name in NORMALIZED_BANDS:
                            mean, std = band_statistics[band_name]
                            features_elev_slice = (features_elev_slice - mean) / std

                        features_elev_vae_input = features_elev_slice.unsqueeze(1) # (B, 1, W, H)

                        # Use autocast for potential mixed precision speedup during model calls
                        with torch.autocast(device_type=accelerator.device_type, enabled=accelerator.mixed_precision != 'no'):
                            # Encode and Reparameterize to get z
                            mu, log_var, _ = vae.encode(features_elev_vae_input) # Assuming encode returns mu, log_var, potentially memory
                            z = vae.reparameterize(mu, log_var) # Calculate z

                        batch_elevation_latent_z = z.cpu() # Store z on CPU

                        # Clean up intermediate tensors
                        del features_elev_slice, features_elev_vae_input, mu, log_var, z

                    # --- Handle Other Bands (Time Series) ---
                    else:
                        # Iterate through the required time indices
                        for i, time_idx in enumerate(inference_time_indices):
                            features_t_slice = features_band[:, :, :, time_idx].float()
                            features_t_slice = torch.nan_to_num(features_t_slice, nan=0.0)

                            if band_name in NORMALIZED_BANDS:
                                mean, std = band_statistics[band_name]
                                features_t_slice = (features_t_slice - mean) / std

                            features_t_vae_input = features_t_slice.unsqueeze(1) # (B, 1, W, H)

                            with torch.autocast(device_type=accelerator.device_type, enabled=accelerator.mixed_precision != 'no'):
                                # Encode and Reparameterize to get z
                                mu, log_var, _ = vae.encode(features_t_vae_input)
                                z = vae.reparameterize(mu, log_var)

                            # Store z for this band and time step (on CPU)
                            batch_other_latents_z[band_name][:, i, :] = z.cpu()

                            # Clean up intermediate tensors for this timestep
                            del features_t_slice, features_t_vae_input, mu, log_var, z


                # Append batch results (z) to the main lists
                if batch_elevation_latent_z is not None:
                    all_elevation_latents_z_list.append(batch_elevation_latent_z)
                for band in batch_other_latents_z:
                    other_bands_latents_z_batches[band].append(batch_other_latents_z[band])

                # More aggressive cleanup at the end of the batch processing
                del lon, lat, features_batch, batch_elevation_latent_z, batch_other_latents_z
                if 'features_band' in locals(): del features_band
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                 logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                 # Option: skip batch or re-raise error depending on desired behavior
                 # raise e # Stop execution
                 logger.warning(f"Skipping batch {batch_idx} due to error.")
                 gc.collect()
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                 continue # Continue to next batch


    logger.info("Inference loop finished. Consolidating results...")

    # --- Consolidate Results ---
    final_coordinates = np.concatenate(all_coordinates_list, axis=0) if all_coordinates_list else np.array([])
    final_elevation_latent_z = torch.cat(all_elevation_latents_z_list, dim=0).numpy() if all_elevation_latents_z_list else np.array([])

    final_other_latents_z = {}
    for band in other_bands_latents_z_batches:
        if other_bands_latents_z_batches[band]:
            final_other_latents_z[band] = torch.cat(other_bands_latents_z_batches[band], dim=0).numpy()
        else:
            # Handle cases where a band might have no results (e.g., if errors occurred)
            latent_dim = VAE_LATENT_DIM_OTHERS
            logger.warning(f"No latent vectors collected for band {band}. Saving empty array.")
            final_other_latents_z[band] = np.empty((0, num_inference_years, latent_dim))

    # --- Create Output Data Structure ---
    output_data = {
        'coordinates': final_coordinates,
        'elevation_latent_z': final_elevation_latent_z, # Renamed key
        'other_bands_latent_z': final_other_latents_z, # Renamed key
        'inference_years_range': list(range(TIME_BEGINNING + start_year_index, TIME_BEGINNING + end_year_index + 1)),
        'bands_order': bands_list_order
    }

    # --- Save Output ---
    logger.info(f"Saving consolidated results (latent z) to {output_path}...")
    np.save(output_path, output_data, allow_pickle=True) # Allow pickle for dictionary structure

    logger.info("Inference finished successfully.")