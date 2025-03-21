import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloaderMultiYears1Mil import MultiRasterDataset1MilMultiYears
from dataloader.dataframe_loader import separate_and_add_data_1mil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse
from config import (TIME_BEGINNING, TIME_END, MAX_OC, NUM_EPOCH_VAE_TRAINING,
                   file_path_coordinates_Bavaria_1mil, bands_list_order, window_size, time_before)
from torch.utils.data import DataLoader, Subset
import logging
from modelTransformerVAE import TransformerVAE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed hyperparameters
BASE_VAE_LR = 0.001  # Base learning rate
VAE_KLD_WEIGHT = 0.1
VAE_NUM_HEADS = 16
VAE_LATENT_DIM = 24
VAE_DROPOUT_RATE = 0.3
VAE_BATCH_SIZE = 128
VAE_ALPHA = 0.5
ACCUM_STEPS = 4

# Composite MSE-L1 loss with NaN handling
def composite_mse_l1_loss(outputs, targets, alpha=VAE_ALPHA):
    outputs = torch.nan_to_num(outputs, nan=0.0)
    targets = torch.nan_to_num(targets, nan=0.0)
    mse_loss = torch.mean((targets - outputs) ** 2)
    l1_loss = torch.mean(torch.abs(targets - outputs))
    composite_loss = alpha * mse_loss + (1 - alpha) * l1_loss
    num_elements = torch.numel(targets) / targets.size(0)
    return composite_loss * 1e1 / num_elements

# VAE loss function
def vae_loss(recon_x, x, mu, log_var, kld_weight=VAE_KLD_WEIGHT):
    recon_loss = composite_mse_l1_loss(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld_loss = torch.nan_to_num(kld_loss, nan=0.0) / VAE_LATENT_DIM * 1e2
    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE for SOC prediction')
    parser.add_argument('--load_vae', type=str, default=None, help='Path to pre-trained VAE model (optional)')
    return parser.parse_args()

def compute_adaptive_learning_rates(model, dataset, device):
    """Compute adaptive learning rates based on initial loss magnitudes for each band."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=4)
    initial_losses = {band: 0.0 for band in bands_list_order}
    num_batches = min(10, len(dataloader))  # Use first 10 batches or less

    with torch.no_grad():
        for batch_idx, (longitudes, latitudes, features, encoding) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            features = torch.nan_to_num(features.to(device), nan=0.0)
            
            for band_idx in range(len(bands_list_order)):
                band_features = features[:, band_idx:band_idx+1, :, :, :]
                recon, mu, log_var = model(band_features, band_idx)
                _, recon_loss, _ = vae_loss(recon, band_features, mu, log_var)
                initial_losses[bands_list_order[band_idx]] += recon_loss.item()
    
    # Average initial losses
    for band in bands_list_order:
        initial_losses[band] /= num_batches
    
    # Normalize losses and compute adaptive learning rates
    max_loss = max(initial_losses.values())
    adaptive_lrs = {band: BASE_VAE_LR * (max_loss / (initial_losses[band] + 1e-8)) for band in bands_list_order}
    logger.info(f"Adaptive learning rates: {adaptive_lrs}")
    return adaptive_lrs

def train_vae(model, train_loader, num_epochs, accelerator):
    optimizer = optim.Adam(model.parameters(), lr=BASE_VAE_LR)  # Use base LR initially
    
    logger.info("Preparing VAE training with Accelerator...")
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    logger.info(f"Training on {len(train_loader)} batches per epoch with {ACCUM_STEPS} accumulation steps.")

    # Compute adaptive learning rates
    adaptive_lrs = compute_adaptive_learning_rates(model, train_loader.dataset, accelerator.device)
    for param_group in optimizer.param_groups:
        param_group['lr'] = sum(adaptive_lrs.values()) / len(adaptive_lrs)  # Initial average LR

    for epoch in range(num_epochs):
        model.train()
        running_loss = {band: 0.0 for band in bands_list_order}
        running_recon_loss = {band: 0.0 for band in bands_list_order}
        running_kld_loss = {band: 0.0 for band in bands_list_order}
        
        logger.info(f"Starting VAE Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (longitudes, latitudes, features, encoding) in enumerate(tqdm(train_loader, desc=f'VAE Epoch {epoch+1}')):
            logger.debug(f"Processing batch {batch_idx+1}/{len(train_loader)}")
            features = torch.nan_to_num(features.to(accelerator.device), nan=0.0)  # [128, 6, 33, 33, 5]
            encoding = encoding.to(accelerator.device)  # [128, 6, 5]
            
            optimizer.zero_grad()
            
            # Accumulate loss across all bands
            total_loss = 0.0
            band_recon_losses = {}
            band_kld_losses = {}
            
            with accelerator.accumulate(model):
                for band_idx in range(len(bands_list_order)):
                    # Select only the current band's data
                    band_features = features[:, band_idx:band_idx+1, :, :, :]  # [128, 1, 33, 33, 5]
                    
                    recon, mu, log_var = model(band_features, band_idx)
                    loss, recon_loss, kld_loss = vae_loss(recon, band_features, mu, log_var)
                    
                    mu = torch.clamp(mu, -100, 100)
                    log_var = torch.clamp(log_var, -20, 20)
                    
                    total_loss = total_loss + loss  # Accumulate loss
                    
                    # Store individual losses for logging
                    band_recon_losses[bands_list_order[band_idx]] = recon_loss.item()
                    band_kld_losses[bands_list_order[band_idx]] = kld_loss.item()
                    running_loss[bands_list_order[band_idx]] += loss.item()
                    running_recon_loss[bands_list_order[band_idx]] += recon_loss.item()
                    running_kld_loss[bands_list_order[band_idx]] += kld_loss.item()
                
                # Single backward pass for all bands
                accelerator.backward(total_loss)
                optimizer.step()
            
            # Logging every 10 batches
            if accelerator.is_main_process and batch_idx % 10 == 0:
                for band in bands_list_order:
                    wandb.log({
                        f'vae_train_loss_{band}': running_loss[band] - running_loss[band] + total_loss.item()/len(bands_list_order),  # Approximate per-band loss
                        f'vae_train_recon_loss_{band}': band_recon_losses[band],
                        f'vae_train_kld_loss_{band}': band_kld_losses[band],
                        'vae_batch': batch_idx + 1 + epoch * len(train_loader),
                        'vae_epoch': epoch + 1
                    })
                    logger.debug(f"Batch {batch_idx+1} - Band {band} - Loss: {total_loss.item()/len(bands_list_order):.4f}")
        
        # Average losses per band
        for band in bands_list_order:
            train_loss = running_loss[band] / len(train_loader)
            if accelerator.is_main_process:
                wandb.log({
                    'vae_epoch': epoch + 1,
                    f'vae_train_loss_avg_{band}': train_loss,
                    f'vae_train_recon_loss_avg_{band}': running_recon_loss[band] / len(train_loader),
                    f'vae_train_kld_loss_avg_{band}': running_kld_loss[band] / len(train_loader)
                })
                logger.info(f'VAE Epoch {epoch+1} - Band {band} - Training Loss: {train_loss:.4f}')
                print(f'VAE Epoch {epoch+1} - Band {band} - Training Loss: {train_loss:.4f}')

    logger.info("VAE training completed.")
    return model

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()

    def flatten_paths(path_list):
        flattened = []
        for item in path_list:
            if isinstance(item, list):
                flattened.extend(flatten_paths(item))
            else:
                flattened.append(item)
        return flattened

    # VAE Training
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

    logger.info("Loading Bavaria 1M dataset...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil).fillna(0.0)
    
    # Sample 250k elements
    train_dataset_df_250k = train_dataset_df_full_1mil.sample(n=250000, random_state=42)
    logger.info(f"Reduced dataset to {len(train_dataset_df_250k)} elements.")

    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil()
    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    logger.info("Initializing VAE dataset...")
    train_dataset = MultiRasterDataset1MilMultiYears(samples_coordinates_array_path_1mil, data_array_path_1mil, train_dataset_df_250k)
    
    train_dataset_subset = Subset(train_dataset, range(min(250000, len(train_dataset))))
    train_loader = DataLoader(train_dataset_subset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4)
    logger.info(f"VAE DataLoader created with {len(train_loader)} batches.")

    logger.info("Testing data loading with one batch...")
    for batch in train_loader:
        longitudes, latitudes, features, encoding = batch
        features = torch.nan_to_num(features, nan=0.0)
        logger.info(f"Batch loaded - Features shape: {features.shape}, Encoding shape: {encoding.shape}")
        break

    vae_model = TransformerVAE(
        input_channels=1,  # Single channel for individual band
        input_height=window_size,
        input_width=window_size,
        input_time=time_before,
        num_heads=VAE_NUM_HEADS,
        latent_dim=VAE_LATENT_DIM,
        dropout_rate=VAE_DROPOUT_RATE,
        bands_list_order=bands_list_order
    ).to(accelerator.device)

    if args.load_vae:
        if accelerator.is_main_process:
            logger.info(f"Loading pre-trained VAE from {args.load_vae}")
            print(f"Loading pre-trained VAE from {args.load_vae}")
        vae_model.load_state_dict(torch.load(args.load_vae, map_location=accelerator.device))
    else:
        logger.info("Starting VAE training...")
        vae_model = train_vae(vae_model, train_loader, num_epochs=NUM_EPOCH_VAE_TRAINING, accelerator=accelerator)
        if accelerator.is_main_process:
            vae_path = f'vae_transformer_250k_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
            accelerator.save(vae_model.state_dict(), vae_path)
            wandb.save(vae_path)
            wandb.run.summary["vae_parameters"] = vae_model.count_parameters()
            logger.info(f"VAE parameters: {vae_model.count_parameters()}")
            print(f"VAE parameters: {vae_model.count_parameters()}")

    wandb.finish()