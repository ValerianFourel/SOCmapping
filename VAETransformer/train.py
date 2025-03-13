import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataloader.dataloaderMultiYears1Mil import MultiRasterDataset1MilMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, separate_and_add_data_1mil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse
from config import (TIME_BEGINNING, TIME_END, MAX_OC, NUM_EPOCH_VAE_TRAINING, NUM_EPOCH_MLP_TRAINING,
                   file_path_coordinates_Bavaria_1mil, bands_list_order, window_size, time_before)
from torch.utils.data import DataLoader
from modelTransformerVAE import TransformerVAE
from modelMLPRegressor import MLPRegressor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed hyperparameters
VAE_LR = 0.001
VAE_KLD_WEIGHT = 0.1  # Boosted from 0.01 to 0.1
VAE_NUM_HEADS = 2
VAE_LATENT_DIM = 128
VAE_DROPOUT_RATE = 0.3
VAE_BATCH_SIZE = 8
VAE_ALPHA = 0.5  # For composite MSE-L1 loss
MLP_LR = 0.002
MLP_HIDDEN_DIM = 32
MLP_BATCH_SIZE = 256
ACCUM_STEPS = 64

# New composite MSE-L1 loss
def composite_mse_l1_loss(outputs, targets, alpha=VAE_ALPHA):
    """Composite loss combining MSE and L1 loss."""
    # Compute MSE loss
    mse_loss = torch.mean((targets - outputs) ** 2)
    
    # Compute L1 loss
    l1_loss = torch.mean(torch.abs(targets - outputs))
    
    # Combine losses with alpha weighting
    return alpha * mse_loss + (1 - alpha) * l1_loss

# Define the NormalizedMultiRasterDatasetMultiYears class (unchanged)
class NormalizedMultiRasterDatasetMultiYears(MultiRasterDatasetMultiYears):
    """Wrapper around MultiRasterDatasetMultiYears that adds feature normalization"""
    def __init__(self, samples_coordinates_array_path, data_array_path, df):
        super().__init__(samples_coordinates_array_path, data_array_path, df)
        self.compute_statistics()
        
    def compute_statistics(self):
        """Compute mean and std across all features for normalization"""
        features_list = []
        for i in range(len(self)):
            _, _, features, _ = super().__getitem__(i)
            features_list.append(features.numpy())
        features_array = np.stack(features_list)
        self.feature_means = torch.tensor(np.mean(features_array, axis=(0, 2, 3)), dtype=torch.float32)
        self.feature_stds = torch.tensor(np.std(features_array, axis=(0, 2, 3)), dtype=torch.float32)
        self.feature_stds = torch.clamp(self.feature_stds, min=1e-8)
        
    def __getitem__(self, idx):
        longitude, latitude, features, target = super().__getitem__(idx)
        features = (features - self.feature_means[:, None, None]) / self.feature_stds[:, None, None]
        return longitude, latitude, features, target

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE and MLP for SOC prediction')
    parser.add_argument('--load_vae', type=str, default=None, help='Path to pre-trained VAE model (optional)')
    return parser.parse_args()

def create_balanced_dataset(df, n_bins=128, min_ratio=0.75):
    logger.info("Creating balanced dataset...")
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    validation_indices = []
    training_dfs = []
    
    for bin_idx in range(len(bin_counts)):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) >= 4:
            val_samples = bin_data.sample(n=min(8, len(bin_data)))
            validation_indices.extend(val_samples.index)
            train_samples = bin_data.drop(val_samples.index)
            if len(train_samples) > 0:
                if len(train_samples) < min_samples:
                    resampled = train_samples.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(train_samples)
    
    if not training_dfs or not validation_indices:
        raise ValueError("No training or validation data available after binning")
    
    training_df = pd.concat(training_dfs).drop('bin', axis=1)
    validation_df = df.loc[validation_indices].drop('bin', axis=1)
    logger.info("Balanced dataset created.")
    return training_df, validation_df

def train_vae(model, train_loader, num_epochs, accelerator):
    optimizer = optim.Adam(model.parameters(), lr=VAE_LR)
    
    logger.info("Preparing VAE training with Accelerator...")
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    logger.info(f"Training on {len(train_loader)} batches per epoch with {ACCUM_STEPS} accumulation steps.")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kld_loss = 0.0
        
        logger.info(f"Starting VAE Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (longitudes, latitudes, features) in enumerate(tqdm(train_loader, desc=f'VAE Epoch {epoch+1}')):
            logger.debug(f"Processing batch {batch_idx+1}/{len(train_loader)}")
            features = features.to(accelerator.device)
            
            with accelerator.accumulate(model):
                reconstruction, mu, log_var = model(features)
                recon_loss = composite_mse_l1_loss(reconstruction, features, alpha=VAE_ALPHA)
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + VAE_KLD_WEIGHT * kld_loss
                
                accelerator.backward(loss)
                
                if ((batch_idx + 1) % ACCUM_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                running_kld_loss += kld_loss.item()
                
                if accelerator.is_main_process and batch_idx % 10 == 0:
                    wandb.log({
                        'vae_train_loss': loss.item(),
                        'vae_train_recon_loss': recon_loss.item(),
                        'vae_train_kld_loss': kld_loss.item(),
                        'vae_batch': batch_idx + 1 + epoch * len(train_loader),
                        'vae_epoch': epoch + 1
                    })
                    logger.debug(f"Batch {batch_idx+1} - Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        if accelerator.is_main_process:
            wandb.log({
                'vae_epoch': epoch + 1,
                'vae_train_loss_avg': train_loss,
                'vae_train_recon_loss_avg': running_recon_loss / len(train_loader),
                'vae_train_kld_loss_avg': running_kld_loss / len(train_loader)
            })
            logger.info(f'VAE Epoch {epoch+1} - Training Loss: {train_loss:.4f}')
            print(f'VAE Epoch {epoch+1} - Training Loss: {train_loss:.4f}')

    logger.info("VAE training completed.")
    return model

def train_mlp(vae_model, mlp_model, train_loader, val_loader, num_epochs, accelerator):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=MLP_LR)
    
    train_loader, val_loader, mlp_model, optimizer = accelerator.prepare(train_loader, val_loader, mlp_model, optimizer)
    vae_model = accelerator.prepare(vae_model)
    vae_model.eval()

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        mlp_model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader, desc=f'MLP Epoch {epoch+1}')):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()
            
            with accelerator.accumulate(mlp_model):
                with torch.no_grad():
                    _, mu, _ = vae_model(features)
                outputs = mlp_model(mu)
                loss = criterion(outputs.squeeze(), targets)
                
                accelerator.backward(loss)
                
                if ((batch_idx + 1) % ACCUM_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += loss.item()
                
                if accelerator.is_main_process:
                    wandb.log({
                        'mlp_train_loss': loss.item(),
                        'mlp_batch': batch_idx + 1 + epoch * len(train_loader),
                        'mlp_epoch': epoch + 1
                    })

        train_loss = running_loss / len(train_loader)
        
        mlp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for longitudes, latitudes, features, targets in val_loader:
                features = features.to(accelerator.device)
                targets = targets.to(accelerator.device).float()
                _, mu, _ = vae_model(features)
                outputs = mlp_model(mu)
                val_loss += criterion(outputs.squeeze(), targets).item()
        
        val_loss = val_loss / len(val_loader)
        
        if accelerator.is_main_process:
            wandb.log({
                'mlp_epoch': epoch + 1,
                'mlp_train_loss_avg': train_loss,
                'mlp_val_loss': val_loss
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.run.summary['mlp_best_val_loss'] = best_val_loss
                accelerator.save(mlp_model.state_dict(), f'mlp_checkpoint_epoch_{epoch+1}.pth')
                wandb.save(f'mlp_checkpoint_epoch_{epoch+1}.pth')
            logger.info(f'MLP Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'MLP Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return mlp_model

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
        "vae_lr": VAE_LR,
        "vae_kld_weight": VAE_KLD_WEIGHT,
        "vae_num_heads": VAE_NUM_HEADS,
        "vae_latent_dim": VAE_LATENT_DIM,
        "vae_dropout_rate": VAE_DROPOUT_RATE,
        "vae_batch_size": VAE_BATCH_SIZE,
        "vae_alpha": VAE_ALPHA,
        "accum_steps": ACCUM_STEPS
    })

    logger.info("Loading Bavaria 1M dataset...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil()
    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    logger.info("Initializing VAE dataset...")
    train_dataset = MultiRasterDataset1MilMultiYears(samples_coordinates_array_path_1mil, data_array_path_1mil, train_dataset_df_full_1mil)
    train_loader = DataLoader(train_dataset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4)
    logger.info(f"VAE DataLoader created with {len(train_loader)} batches.")

    logger.info("Testing data loading with one batch...")
    for batch in train_loader:
        longitudes, latitudes, features = batch
        logger.info(f"Batch loaded - Features shape: {features.shape}")
        break

    vae_model = TransformerVAE(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before,
        num_heads=VAE_NUM_HEADS,
        latent_dim=VAE_LATENT_DIM,
        dropout_rate=VAE_DROPOUT_RATE
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
            vae_path = f'vae_transformer_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
            accelerator.save(vae_model.state_dict(), vae_path)
            wandb.save(vae_path)
            wandb.run.summary["vae_parameters"] = vae_model.count_parameters()
            logger.info(f"VAE parameters: {vae_model.count_parameters()}")
            print(f"VAE parameters: {vae_model.count_parameters()}")

    wandb.finish()

    # MLP Training
    wandb.init(project="socmapping-VAETransformer-MLPRegressor", config={
        "max_oc": MAX_OC,
        "time_beginning": TIME_BEGINNING,
        "time_end": TIME_END,
        "window_size": window_size,
        "time_before": time_before,
        "bands": len(bands_list_order),
        "mlp_epochs": NUM_EPOCH_MLP_TRAINING,
        "mlp_lr": MLP_LR,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "vae_latent_dim": 8,
        "mlp_batch_size": MLP_BATCH_SIZE,
        "accum_steps": ACCUM_STEPS
    })

    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    train_df, val_df = create_balanced_dataset(df)
    train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
    train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False, num_workers=4)

    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df)
        logger.info(f"Training set size: {len(train_df)}")
        print(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        print(f"Validation set size: {len(val_df)}")

    mlp_model = MLPRegressor(input_dim=8).to(accelerator.device)
    mlp_model = train_mlp(vae_model, mlp_model, train_loader, val_loader, num_epochs=NUM_EPOCH_MLP_TRAINING, accelerator=accelerator)

    if accelerator.is_main_process:
        mlp_path = f'mlp_regressor_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
        accelerator.save(mlp_model.state_dict(), mlp_path)
        wandb.save(mlp_path)
        wandb.run.summary["mlp_parameters"] = mlp_model.count_parameters()
        logger.info(f"MLP parameters: {mlp_model.count_parameters()}")
        print(f"MLP parameters: {mlp_model.count_parameters()}")

    wandb.finish()