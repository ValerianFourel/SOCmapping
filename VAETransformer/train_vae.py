import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse
import matplotlib.pyplot as plt  # Added for image generation
from config import (
    TIME_BEGINNING, TIME_END, MAX_OC, NUM_EPOCH_VAE_TRAINING,
    file_path_coordinates_Bavaria_1mil, bands_list_order, window_size,
    mean_LST, mean_MODIS_NPP, mean_totalEvapotranspiration,
    std_LST, std_MODIS_NPP, std_totalEvapotranspiration, window_size_LAI
)
import logging
from modelTransformerVAE import TransformerVAE
from losses import (
    ElevationVAELoss, TotalEvapotranspirationVAELoss, SoilEvaporationVAELoss,
    NPPVAELoss, LSTVAELoss, LAIVAELoss, normalize_tensor_01, normalize_tensor_lst, LAIVAELoss_v2
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

TIME_BEGINNING = int(TIME_BEGINNING)
TIME_END = int(TIME_END)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized hyperparameters
BASE_VAE_LR = 0.001
VAE_NUM_HEADS = 2
VAE_LATENT_DIM_ELEVATION = 5
VAE_LATENT_DIM_LAI = 5
VAE_LATENT_DIM_OTHERS = 5
VAE_DROPOUT_RATE = 0.3
VAE_BATCH_SIZE = 256
ACCUM_STEPS = 2

# Bands to normalize
NORMALIZED_BANDS = ['LST', 'MODIS_NPP', 'TotalEvapotranspiration']

# Define base directories
BASE_SAVE_DIR = Path('/home/vfourel/SOCProject/SOCmapping/VAETransformer/weights')
BASE_IMAGE_DIR = Path('/home/vfourel/SOCProject/SOCmapping/VAETransformer/.imageOutput')  # New image output directory
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = BASE_SAVE_DIR

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE for SOC prediction')
    parser.add_argument('--load_vae', type=str, default=None, help='Path to pre-trained VAE model directory (optional)')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--band', type=str, default=None, help='Specific band to train (e.g., NDVI); if None, train all bands')
    parser.add_argument('--use_precomputed_stats', action='store_true', default=True, help='Use precomputed statistics from config.py')
    return parser.parse_args()

def compute_band_statistics(band_dataset):
    total_samples = len(band_dataset)
    sample_size = max(1, int(total_samples * 0.01))
    indices = np.random.choice(total_samples, size=sample_size, replace=False)
    subset = Subset(band_dataset, indices)
    loader = DataLoader(subset, batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=4)
    
    all_features = []
    for _, _, features, _ in loader:
        if len(features.shape) == 5:
            features = features.view(-1, *features.shape[2:4])
        elif len(features.shape) == 4:
            pass
        else:
            raise ValueError(f"Unexpected features shape in compute_band_statistics: {features.shape}")
        features = torch.nan_to_num(features, nan=0.0)
        all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    mean = torch.mean(all_features)
    std = torch.std(all_features)
    return mean.item(), std.item() if std.item() != 0 else 1.0


import torch

import torch
import numpy as np

def get_batch_statistics(features):
    """
    Calculate and print statistical measures and distribution insights for a batch tensor going into a VAE.
    
    Args:
        features (torch.Tensor): Input tensor of shape [x, 1, height, width]
    """
    # Ensure input has correct number of dimensions
    if features.dim() != 4:
        raise ValueError("Input tensor must have 4 dimensions [batch, channels, height, width]")
    if features.size(1) != 1:
        raise ValueError("Input tensor must have 1 channel")
    
    # Flatten all dimensions except batch
    flat_features = features.view(features.size(0), -1)  # Shape: [x, height * width]
    
    # Basic statistics
    print("Per-batch statistics:")
    for i in range(features.size(0)):
        print(f"Batch {i}:")
        print(f"  Mean: {flat_features[i].mean():.4f}")
        print(f"  Std: {flat_features[i].std():.4f}")
        print(f"  Min: {flat_features[i].min():.4f}")
        print(f"  Max: {flat_features[i].max():.4f}")
        print(f"  Median: {flat_features[i].median():.4f}")
    
    # Distribution insights
    all_flat = features.view(-1)  # Flatten everything for overall distribution
    
    # Calculate histogram for overall distribution
    
    # Percentage of zeros
    zero_percentage = (all_flat == 0).float().mean() * 100
    
    # Print distribution statistics
    print("\nOverall distribution statistics:")
    print(f"Overall Mean: {all_flat.mean():.4f}")
    print(f"Overall Std: {all_flat.std():.4f}")
    print(f"Range: {(all_flat.max() - all_flat.min()):.4f}")
    print(f"Zero Percentage: {zero_percentage:.2f}%")


def train_single_vae(vae, band_dataset, num_epochs, accelerator, band, save_dir, time_steps, norm_stats=None):
    optimizer = optim.Adam(vae.parameters(), lr=BASE_VAE_LR)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(accelerator.device)
    
    # Initialize band-specific loss
    data_length = len(band_dataset)
    loss_fns = {
        'Elevation': ElevationVAELoss(data_length, VAE_BATCH_SIZE),
        'TotalEvapotranspiration': TotalEvapotranspirationVAELoss(data_length, VAE_BATCH_SIZE),
        'SoilEvaporation': SoilEvaporationVAELoss(data_length, VAE_BATCH_SIZE),
        'MODIS_NPP': NPPVAELoss(data_length, VAE_BATCH_SIZE),
        'LST': LSTVAELoss(data_length, VAE_BATCH_SIZE),
        'LAI': LAIVAELoss(data_length, VAE_BATCH_SIZE)
    }
    loss_fn = loss_fns.get(band, ElevationVAELoss(data_length, VAE_BATCH_SIZE))
    
    vae = accelerator.prepare(vae)
    optimizer = accelerator.prepare(optimizer)
    loss_fn = accelerator.prepare(loss_fn)

    # Create band-specific image output directory
    image_output_dir = BASE_IMAGE_DIR / band
    image_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training VAE for band {band} with {num_epochs} epochs and random sampling across {time_steps} time steps")
    logger.info(f"Loss weight for {band}: {loss_fn.get_loss_weight():.4f}")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        coord_indices = np.random.choice(len(band_dataset), size=500000, replace=False)
        band_dataset_subset = Subset(band_dataset, coord_indices)
        train_loader = DataLoader(band_dataset_subset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        train_loader = accelerator.prepare(train_loader)
        
        vae.train()
        epoch_loss = 0.0
        total_batches = len(train_loader)
        checkpoints = [int(total_batches * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
        if total_batches > 0 and checkpoints[0] == 0:
            checkpoints[0] = 1
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'VAE Epoch {epoch+1} - Band {band}', leave=False)):
            _, _, features, _ = batch
            batch_size = features.size(0)
            
            if batch_idx == 0 and epoch == 0:
                logger.info(f"Initial features shape for band {band}: {features.shape}")
            
            if band != 'Elevation':
                #if Process time-series bands
                if features.dim() != 4:
                    raise ValueError(f"Expected 4D tensor [batch, time, height, width] for band {band}, got {features.shape}")
                if features.shape[2:4] != (window_size, window_size):
                    raise ValueError(f"Expected spatial dimensions ({window_size}, {window_size}) for band {band}, got {features.shape}")

                original_batch_size = features.shape[0]
                samples_per_batch = 8
                new_batch_size = original_batch_size * samples_per_batch

                batch_indices = torch.repeat_interleave(torch.arange(original_batch_size, device=features.device), samples_per_batch)
                time_indices = torch.randint(0, time_steps, (new_batch_size,), device=features.device)
                features_expanded = features[batch_indices, time_indices, :, :].unsqueeze(1)
                features = features_expanded
            else:
                if features.dim() != 4 or features.shape[1] != 1 or features.shape[2:4] != (window_size, window_size):
                    raise ValueError(f"Expected [batch, 1, {window_size}, {window_size}] for Elevation, got {features.shape}")
            
            features = torch.nan_to_num(features.to(accelerator.device), nan=0.0)
            if band == 'LAI':
                features = torch.nan_to_num(features.to(accelerator.device), nan=0.0, posinf=10.0, neginf=0.0)

            if batch_idx == 0 and epoch == 0:
                logger.info(f"Features shape after processing for band {band}: {features.shape}")
                
                # Analyze distribution before VAE
                get_batch_statistics(features)


            if norm_stats and band in NORMALIZED_BANDS:
                mean, std = norm_stats
                features = (features - mean) / std
            
            optimizer.zero_grad()
            with accelerator.accumulate(vae):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    recon, mu, log_var = vae(features)
                    
                    if band == 'Elevation':
                        total_loss = loss_fn(recon, recon, features, mu, log_var, lpips)
                    elif band == 'LST':
                        total_loss = loss_fn(recon, features, features, mu, log_var, lpips)
                    elif band == 'LAI':
                        total_loss = loss_fn(recon, features, mu, log_var, lpips)
                    elif band in ['TotalEvapotranspiration', 'SoilEvaporation', 'MODIS_NPP']:
                        total_loss = loss_fn(recon, features, mu, log_var, lpips)
                    else:
                        raise ValueError(f"No loss function defined for band {band}")
                
                accelerator.backward(total_loss)
                if band == 'LAI':
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer.step()
            
            batch_total_loss = total_loss.item()
            epoch_loss += batch_total_loss
            
            if accelerator.is_main_process:
                global_step += 1
                wandb.log({
                    f'vae_total_loss_{band}': batch_total_loss,
                    'epoch': epoch + 1,
                    'global_step': global_step
                })

                if batch_idx + 1 in checkpoints:
                    vae.eval()
                    with torch.no_grad():
                        sample_features = features[:1].cpu().numpy()
                        sample_recon = recon[:1].cpu().numpy()
                        
                        if sample_features.shape[1] == 1:
                            sample_features = sample_features.squeeze(1)
                            sample_recon = sample_recon.squeeze(1)
                        
                        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                        axs[0].imshow(sample_features[0], cmap='viridis')
                        axs[0].set_title('Original')
                        axs[0].axis('off')
                        axs[1].imshow(sample_recon[0], cmap='viridis')
                        axs[1].set_title('Reconstructed')
                        axs[1].axis('off')
                        progress_percent = int(((batch_idx + 1) / total_batches) * 100)
                        plt.suptitle(f'Band: {band} - Epoch: {epoch + 1} - {progress_percent}%')
                        
                        image_path = image_output_dir / f'epoch_{epoch + 1}_progress_{progress_percent}.png'
                        plt.savefig(image_path, bbox_inches='tight')
                        plt.close(fig)
                        
                        wandb.log({
                            f'vae_reconstruction_{band}_epoch_{epoch + 1}_progress_{progress_percent}': wandb.Image(str(image_path))
                        })
                    vae.train()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if accelerator.is_main_process:
            logger.info(f'VAE Epoch {epoch+1} - Band {band} - Average Training Loss: {avg_epoch_loss:.4f}')
            wandb.log({f'vae_train_loss_avg_{band}': avg_epoch_loss, 'epoch': epoch + 1})

    vae_path = save_dir / f"{band}.pth"
    if accelerator.is_main_process:
        accelerator.save(vae.state_dict(), vae_path)
        wandb.save(str(vae_path))
    
    return vae



if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16')
    window_size_original = window_size
    bands_to_train = [args.band] if args.band else bands_list_order
    if args.band and args.band not in bands_list_order:
        raise ValueError(f"Band {args.band} not in {bands_list_order}")

    wandb.init(project="socmapping-VAETransformer", config={
        "max_oc": MAX_OC,
        "time_beginning": TIME_BEGINNING,
        "time_end": TIME_END,
        "window_size": window_size,
        "bands": len(bands_to_train),
        "vae_epochs": NUM_EPOCH_VAE_TRAINING,
        "vae_lr": BASE_VAE_LR,
        "vae_num_heads": VAE_NUM_HEADS,
        "vae_latent_dim_elevation": VAE_LATENT_DIM_ELEVATION,
        "vae_latent_dim_others": VAE_LATENT_DIM_OTHERS,
        "vae_dropout_rate": VAE_DROPOUT_RATE,
        "vae_batch_size": VAE_BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "trained_band": args.band if args.band else "all",
        "normalized_bands": NORMALIZED_BANDS,
        "use_precomputed_stats": args.use_precomputed_stats
    })

    from dataloader.dataloaderMultiYears1Mil import NormalizedMultiRasterDataset1MilMultiYears
    from dataloader.dataframe_loader import separate_and_add_data_1mil

    logger.info("Loading Bavaria 1M dataset...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)
    
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil()
    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    train_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path_1mil,
        data_array_path_1mil,
        train_dataset_df_full_1mil,
        accelerator=accelerator,
        batch_size=VAE_BATCH_SIZE,
        num_workers=4,
        preload=True
    )

    device_map = {band: f'cuda:{i % args.num_gpus}' for i, band in enumerate(bands_list_order)}

    training_counter = 0
    existing_dirs = [int(d.name.split('_')[0]) for d in SAVE_DIR.iterdir() if d.is_dir() and d.name.split('_')[0].isdigit()]
    training_counter_start = max(existing_dirs) + 1 if existing_dirs else 1

    vae_models = {}
    band_statistics = {}

    time_steps = TIME_END - TIME_BEGINNING + 1 if 'Elevation' not in bands_to_train else 1

    precomputed_stats = {
        'LST': (mean_LST, std_LST),
        'MODIS_NPP': (mean_MODIS_NPP, std_MODIS_NPP),
        'TotalEvapotranspiration': (mean_totalEvapotranspiration, std_totalEvapotranspiration)
    }

    for band in NORMALIZED_BANDS:
        if band in bands_to_train:
            if args.use_precomputed_stats and band in precomputed_stats:
                band_statistics[band] = precomputed_stats[band]
                logger.info(f"Using precomputed statistics for {band}: mean={precomputed_stats[band][0]}, std={precomputed_stats[band][1]}")
            else:
                band_dataset = train_dataset.get_band_dataset(band)
                mean, std = compute_band_statistics(band_dataset)
                band_statistics[band] = (mean, std)
                logger.info(f"Computed statistics for {band}: mean={mean}, std={std}")

    for band in bands_to_train:
        logger.info(f"Starting training for band: {band}")
        
        training_counter += 1
        subfolder_name = f"{training_counter_start + training_counter - 1}_{band}"
        save_dir = SAVE_DIR / subfolder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        latent_dim = VAE_LATENT_DIM_ELEVATION if band == 'Elevation' else VAE_LATENT_DIM_OTHERS
        latent_dim = VAE_LATENT_DIM_LAI if band == 'LAI' else latent_dim
        num_heads = VAE_NUM_HEADS if band == 'Elevation' else VAE_NUM_HEADS * 2
        vae = TransformerVAE(
            input_channels=1,
            input_height=window_size,
            input_width=window_size,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout_rate=VAE_DROPOUT_RATE
        )
        
        vae_path = save_dir / f"{band}.pth"
        if args.load_vae and Path(args.load_vae).exists() and vae_path.exists():
            logger.info(f"Loading pre-trained VAE for band {band} from {vae_path}")
            vae.load_state_dict(torch.load(vae_path, map_location=device_map[band]))
        else:
            band_dataset = train_dataset.get_band_dataset(band)
            norm_stats = band_statistics.get(band, None)
            vae = train_single_vae(
                vae,
                band_dataset,
                NUM_EPOCH_VAE_TRAINING,
                accelerator,
                band,
                save_dir,
                time_steps if band != 'Elevation' else 1,
                norm_stats=norm_stats
            )
        
        vae_models[band] = vae
        logger.info(f"Completed training for band: {band}")

    if accelerator.is_main_process:
        wandb.finish()
    logger.info("All VAE training completed.")