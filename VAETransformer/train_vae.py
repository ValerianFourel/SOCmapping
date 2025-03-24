import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse
from config import (
    TIME_BEGINNING, TIME_END, MAX_OC, NUM_EPOCH_VAE_TRAINING,
    file_path_coordinates_Bavaria_1mil, bands_list_order, window_size, time_before
)
import logging
from modelTransformerVAE import TransformerVAE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimized hyperparameters
BASE_VAE_LR = 0.001
VAE_KLD_WEIGHT = 0.1
VAE_NUM_HEADS = 16
VAE_LATENT_DIM = 24
VAE_DROPOUT_RATE = 0.3
VAE_BATCH_SIZE = 256
VAE_ALPHA = 0.5
ACCUM_STEPS = 2

# Define directories
BASE_SAVE_DIR = Path('/home/vfourel/SOCProject/SOCmapping/VAETransformer/weights')
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
SUBFOLDER_NAME = f"vae_transformer_250k_MAX_OC_{MAX_OC}_TIME_{TIME_BEGINNING}_to_{TIME_END}"
SAVE_DIR = BASE_SAVE_DIR / SUBFOLDER_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld_loss = torch.nan_to_num(kld_loss, nan=0.0) / VAE_LATENT_DIM * 1e2
    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE for SOC prediction')
    parser.add_argument('--load_vae', type=str, default=None, help='Path to pre-trained VAE model directory (optional)')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    return parser.parse_args()

class MultiChannelVAE(nn.Module):
    def __init__(self, num_channels, device_map):
        super().__init__()
        self.num_channels = num_channels
        self.vaes = nn.ModuleDict()
        self.device_map = device_map
        
        # Initialize one VAE per channel
        for i, band in enumerate(bands_list_order):
            vae = TransformerVAE(
                input_channels=1,
                input_height=window_size,
                input_width=window_size,
                input_time=time_before,
                num_heads=VAE_NUM_HEADS,
                latent_dim=VAE_LATENT_DIM,
                dropout_rate=VAE_DROPOUT_RATE
            )
            device = self.device_map.get(band, f'cuda:{i % torch.cuda.device_count()}')
            self.vaes[band] = vae.to(device)

    def forward(self, x):
        recon_batch = []
        mu_batch = []
        log_var_batch = []
        
        for i, band in enumerate(bands_list_order):
            channel_data = x[:, i:i+1, :, :, :].to(self.vaes[band].device)
            recon, mu, log_var = self.vaes[band](channel_data)
            recon_batch.append(recon.to(x.device))
            mu_batch.append(mu.to(x.device))
            log_var_batch.append(log_var.to(x.device))
        
        return (torch.cat(recon_batch, dim=1),
                torch.cat(mu_batch, dim=1),
                torch.cat(log_var_batch, dim=1))

def train_multi_vae(model, train_loader, num_epochs, accelerator):
    optimizers = {
        band: optim.Adam(model.vaes[band].parameters(), lr=BASE_VAE_LR)
        for band in bands_list_order
    }
    train_loader = accelerator.prepare(train_loader)
    optimizers = {band: accelerator.prepare(opt) for band, opt in optimizers.items()}
    model = accelerator.prepare(model)

    logger.info(f"Training {len(bands_list_order)} VAEs on {len(train_loader)} batches with {ACCUM_STEPS} accumulation steps.")
    
    for epoch in range(num_epochs):
        model.train()
        running_losses = {band: 0.0 for band in bands_list_order}
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'VAE Epoch {epoch+1}', leave=False)):
            features = torch.nan_to_num(batch[2].to(accelerator.device), nan=0.0)  # Get features from batch
            
            for band in bands_list_order:
                optimizers[band].zero_grad()
            
            with accelerator.accumulate(model):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    recon, mu, log_var = model(features)
                    total_loss = 0
                    for i, band in enumerate(bands_list_order):
                        channel_data = features[:, i:i+1, :, :, :]
                        recon_channel = recon[:, i:i+1, :, :, :]
                        mu_channel = mu[:, i * VAE_LATENT_DIM:(i + 1) * VAE_LATENT_DIM]
                        log_var_channel = log_var[:, i * VAE_LATENT_DIM:(i + 1) * VAE_LATENT_DIM]
                        loss, _, _ = vae_loss(recon_channel, channel_data, mu_channel, log_var_channel)
                        running_losses[band] += loss.item()
                        total_loss += loss
                
                accelerator.backward(total_loss)
                for band in bands_list_order:
                    optimizers[band].step()
        
        if accelerator.is_main_process:
            for band in bands_list_order:
                avg_loss = running_losses[band] / len(train_loader)
                logger.info(f'VAE Epoch {epoch+1} - Band {band} - Training Loss: {avg_loss:.4f}')
                wandb.log({f'vae_train_loss_avg_{band}': avg_loss})

    return model

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

    wandb.init(project="socmapping-VAETransformer", config={
        "max_oc": MAX_OC,
        "time_beginning": TIME_BEGINNING,
        "time_end": TIME_END,
        "window_size": window_size,
        "time_before": time_before,
        "bands": len(bands_list_order),
        "vae_epochs": NUM_EPOCH_VAE_TRAINING,
        "vae_lr": BASE_VAE_LR,
        "vae_kld_weight": VAE_KLD_WEIGHT,
        "vae_num_heads": VAE_NUM_HEADS,
        "vae_latent_dim": VAE_LATENT_DIM,
        "vae_dropout_rate": VAE_DROPOUT_RATE,
        "vae_batch_size": VAE_BATCH_SIZE,
        "vae_alpha": VAE_ALPHA,
        "accum_steps": ACCUM_STEPS
    })

    # Load and prepare dataset
    from dataloader.dataloaderMultiYears1Mil import NormalizedMultiRasterDataset1MilMultiYears
    from dataloader.dataframe_loader import separate_and_add_data_1mil

    logger.info("Loading Bavaria 1M dataset...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)
    train_dataset_df_250k = train_dataset_df_full_1mil.sample(n=350000, random_state=42)
    
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil()
    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    train_dataset = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path_1mil,
        data_array_path_1mil,
        train_dataset_df_250k,
        accelerator=accelerator,
        batch_size=VAE_BATCH_SIZE,
        num_workers=4,
        preload=True
    )
    train_dataset_subset = Subset(train_dataset, range(min(350000, len(train_dataset))))
    train_loader = DataLoader(train_dataset_subset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Create device map for distributing VAEs across GPUs
    device_map = {band: f'cuda:{i % args.num_gpus}' for i, band in enumerate(bands_list_order)}
    
    # Initialize and train multi-channel VAE
    multi_vae = MultiChannelVAE(num_channels=len(bands_list_order), device_map=device_map)
    
    if args.load_vae and Path(args.load_vae).exists():
        logger.info(f"Loading pre-trained VAEs from {args.load_vae}")
        for band in bands_list_order:
            vae_path = Path(args.load_vae) / f"{band}.pth"
            if vae_path.exists():
                multi_vae.vaes[band].load_state_dict(torch.load(vae_path, map_location=device_map[band]))
    else:
        multi_vae = train_multi_vae(multi_vae, train_loader, NUM_EPOCH_VAE_TRAINING, accelerator)
        
        if accelerator.is_main_process:
            for band in bands_list_order:
                vae_path = SAVE_DIR / f"{band}.pth"
                accelerator.save(multi_vae.vaes[band].state_dict(), vae_path)
                wandb.save(str(vae_path))

    if accelerator.is_main_process:
        wandb.finish()
    logger.info("All VAE training completed.")