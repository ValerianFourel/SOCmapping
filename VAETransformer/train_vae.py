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
from config import (
    TIME_BEGINNING, TIME_END, MAX_OC, NUM_EPOCH_VAE_TRAINING,
    file_path_coordinates_Bavaria_1mil, bands_list_order, window_size
)
import logging
from modelTransformerVAE import TransformerVAE

TIME_BEGINNING = int(TIME_BEGINNING)
TIME_END = int(TIME_END)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized hyperparameters
BASE_VAE_LR = 0.001
VAE_KLD_WEIGHT = 0.1
VAE_NUM_HEADS = 16
VAE_LATENT_DIM_ELEVATION = 24
VAE_LATENT_DIM_OTHERS = 48
VAE_DROPOUT_RATE = 0.3
VAE_BATCH_SIZE = 256
VAE_ALPHA = 0.5
ACCUM_STEPS = 2

# Bands to normalize
NORMALIZED_BANDS = ['LST', 'MODIS_NPP', 'TotalEvapotranspiration']

# Band-specific KLD weights
band_kld_weights = {
    'LST': 0.003,
    'MODIS_NPP': 1.0,
    'SoilEvaporation': 1.0,
    'TotalEvapotranspiration': 0.5
}

# Define base directory
BASE_SAVE_DIR = Path('/home/vfourel/SOCProject/SOCmapping/VAETransformer/weights')
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = BASE_SAVE_DIR

# Composite MSE-L1 loss with NaN handling
def composite_mse_l1_loss(outputs, targets, alpha=VAE_ALPHA):
    outputs = torch.nan_to_num(outputs, nan=0.0)
    targets = torch.nan_to_num(targets, nan=0.0)
    mse_loss = torch.mean((targets - outputs) ** 2)
    l1_loss = torch.mean(torch.abs(targets - outputs))
    return (alpha * mse_loss + (1 - alpha) * l1_loss) * 1e1 / (targets.numel() / targets.size(0))

# VAE loss function
def vae_loss(recon_x, x, mu, log_var, kld_weight=VAE_KLD_WEIGHT):
    recon_loss = composite_mse_l1_loss(recon_x, x)
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp() + 1e-6)
    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE for SOC prediction')
    parser.add_argument('--load_vae', type=str, default=None, help='Path to pre-trained VAE model directory (optional)')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--band', type=str, default=None, help='Specific band to train (e.g., NDVI); if None, train all bands')
    return parser.parse_args()

def compute_band_statistics(band_dataset):
    """Compute mean and std for a band's dataset using 1% of features"""
    total_samples = len(band_dataset)
    sample_size = max(1, int(total_samples * 0.01))  # 1% of dataset, at least 1 sample
    indices = np.random.choice(total_samples, size=sample_size, replace=False)
    subset = Subset(band_dataset, indices)
    loader = DataLoader(subset, batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=4)
    
    all_features = []
    for _, _, features, _ in loader:
        features = torch.nan_to_num(features, nan=0.0)
        all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    mean = torch.mean(all_features)
    std = torch.std(all_features)
    return mean.item(), std.item() if std.item() != 0 else 1.0  # Avoid division by zero

def train_single_vae(vae, band_dataset, num_epochs, accelerator, band, save_dir, time_steps, kld_weight=VAE_KLD_WEIGHT, norm_stats=None):
    """
    Train a single VAE with optional normalization for specific bands.
    norm_stats: tuple (mean, std) for normalization, None if no normalization.
    """
    optimizer = optim.Adam(vae.parameters(), lr=BASE_VAE_LR)
    vae = accelerator.prepare(vae)
    optimizer = accelerator.prepare(optimizer)

    logger.info(f"Training VAE for band {band} with {num_epochs} epochs, {time_steps} time steps, and dynamic 100k sampling.")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        indices = np.random.choice(len(band_dataset), size=100000, replace=False)
        band_dataset_subset = Subset(band_dataset, indices)
        train_loader = DataLoader(band_dataset_subset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        train_loader = accelerator.prepare(train_loader)

        vae.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'VAE Epoch {epoch+1} - Band {band}', leave=False)):
            _, _, features, _ = batch
            features = torch.nan_to_num(features.to(accelerator.device), nan=0.0)
            
            # Normalize if band is in NORMALIZED_BANDS and stats are provided
            if norm_stats and band in NORMALIZED_BANDS:
                mean, std = norm_stats
                features = (features - mean) / std
            
            optimizer.zero_grad()
            with accelerator.accumulate(vae):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    features = features.permute(0, 2, 3, 1).unsqueeze(1)
                    recon, mu, log_var = vae(features)
                    log_var = torch.clamp(log_var, min=-10, max=10)
                    total_loss, recon_loss, kld_loss = vae_loss(recon, features, mu, log_var, kld_weight=kld_weight)
                
                accelerator.backward(total_loss)
                optimizer.step()
            
            batch_total_loss = total_loss.item()
            batch_recon_loss = recon_loss.item()
            batch_kld_loss = kld_loss.item()
            epoch_loss += batch_total_loss
            
            if accelerator.is_main_process:
                global_step += 1
                wandb.log({
                    f'vae_total_loss_{band}': batch_total_loss,
                    f'vae_recon_loss_{band}': batch_recon_loss,
                    f'vae_kld_loss_{band}': batch_kld_loss,
                    'epoch': epoch + 1,
                    'global_step': global_step
                })
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if accelerator.is_main_process:
            logger.info(f'VAE Epoch {epoch+1} - Band {band} - Average Training Loss: {avg_epoch_loss:.4f}')
            wandb.log({f'vae_train_loss_avg_{band}': avg_epoch_loss, 'epoch': epoch + 1})

    vae_path = save_dir / f"{band}.pth"
    if accelerator.is_main_process:
        accelerator.save(vae.state_dict(), vae_path)
        wandb.save(str(vae_path))
    
    return vae

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16')

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
        "vae_kld_weight": VAE_KLD_WEIGHT,
        "vae_num_heads": VAE_NUM_HEADS,
        "vae_latent_dim_elevation": VAE_LATENT_DIM_ELEVATION,
        "vae_latent_dim_others": VAE_LATENT_DIM_OTHERS,
        "vae_dropout_rate": VAE_DROPOUT_RATE,
        "vae_batch_size": VAE_BATCH_SIZE,
        "vae_alpha": VAE_ALPHA,
        "accum_steps": ACCUM_STEPS,
        "trained_band": args.band if args.band else "all",
        "normalized_bands": NORMALIZED_BANDS
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
    band_statistics = {}  # Store normalization stats for normalized bands

    # Compute statistics for normalized bands
    for band in NORMALIZED_BANDS:
        if band in bands_to_train:
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

        time_steps = 1 if band == 'Elevation' else (TIME_END - TIME_BEGINNING + 1)
        latent_dim = VAE_LATENT_DIM_ELEVATION if band == 'Elevation' else VAE_LATENT_DIM_OTHERS
        num_heads = VAE_NUM_HEADS if band == 'Elevation' else VAE_NUM_HEADS * 2

        vae = TransformerVAE(
            input_channels=1,
            input_height=window_size,
            input_width=window_size,
            input_time=time_steps,
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
            kld_weight = band_kld_weights.get(band, VAE_KLD_WEIGHT)
            norm_stats = band_statistics.get(band, None)  # Get stats if band is in NORMALIZED_BANDS, else None
            vae = train_single_vae(vae, band_dataset, NUM_EPOCH_VAE_TRAINING, accelerator, band, save_dir, time_steps, kld_weight=kld_weight, norm_stats=norm_stats)
        
        vae_models[band] = vae
        logger.info(f"Completed training for band: {band}")

    if accelerator.is_main_process:
        wandb.finish()
    logger.info("All VAE training completed.")