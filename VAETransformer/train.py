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

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE and MLP for SOC prediction')
    parser.add_argument('--load_vae', type=str, default=None, help='Path to pre-trained VAE model (optional)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accum_steps', type=int, default=64, help='Number of gradient accumulation steps')
    return parser.parse_args()

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
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

def train_vae(model, train_loader, num_epochs, accelerator, accum_steps):
    reconstruction_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    kld_weight = 0.01
    
    logger.info("Preparing VAE training with Accelerator...")
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    logger.info(f"Training on {len(train_loader)} batches per epoch with {accum_steps} accumulation steps.")

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
            
            with accelerator.accumulate(model):  # Gradient accumulation
                reconstruction, mu, log_var = model(features)
                recon_loss = reconstruction_criterion(reconstruction, features)
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kld_weight * kld_loss
                
                accelerator.backward(loss)
                
                if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                running_kld_loss += kld_loss.item()
                
                if accelerator.is_main_process and batch_idx % 10 == 0:  # Log every 10 batches
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

def train_mlp(vae_model, mlp_model, train_loader, val_loader, num_epochs, accelerator, accum_steps):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    
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
            
            with accelerator.accumulate(mlp_model):  # Gradient accumulation
                with torch.no_grad():
                    _, mu, _ = vae_model(features)
                outputs = mlp_model(mu)
                loss = criterion(outputs.squeeze(), targets)
                
                accelerator.backward(loss)
                
                if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader)):
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
            print(f'MLP Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return mlp_model

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()

    # Flatten paths helper
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
        "vae_lr": 0.001,
        "kld_weight": 0.01,
        "vae_num_heads": 2,
        "vae_latent_dim": 8,
        "vae_dropout_rate": 0.3,
        "accum_steps": args.accum_steps
    })

    # Load Bavaria 1M points for VAE training
    logger.info("Loading Bavaria 1M dataset...")
    train_dataset_df_full_1mil = pd.read_csv(file_path_coordinates_Bavaria_1mil)
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil()
    samples_coordinates_array_path_1mil = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path_1mil)))
    data_array_path_1mil = list(dict.fromkeys(flatten_paths(data_array_path_1mil)))

    logger.info("Initializing VAE dataset...")
    train_dataset = MultiRasterDataset1MilMultiYears(samples_coordinates_array_path_1mil, data_array_path_1mil, train_dataset_df_full_1mil)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    logger.info(f"VAE DataLoader created with {len(train_loader)} batches.")

    # Test a single batch to check data loading
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
        num_heads=2,
        latent_dim=8,
        dropout_rate=0.3
    ).to(accelerator.device)

    if args.load_vae:
        if accelerator.is_main_process:
            logger.info(f"Loading pre-trained VAE from {args.load_vae}")
            print(f"Loading pre-trained VAE from {args.load_vae}")
        vae_model.load_state_dict(torch.load(args.load_vae, map_location=accelerator.device))
    else:
        logger.info("Starting VAE training...")
        vae_model = train_vae(vae_model, train_loader, num_epochs=NUM_EPOCH_VAE_TRAINING, 
                            accelerator=accelerator, accum_steps=args.accum_steps)
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
        "mlp_lr": 0.001,
        "mlp_hidden_dim": 32,
        "vae_latent_dim": 8,
        "accum_steps": args.accum_steps
    })

    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    train_df, val_df = create_balanced_dataset(df)
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df)
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")

    mlp_model = MLPRegressor(input_dim=8).to(accelerator.device)
    mlp_model = train_mlp(vae_model, mlp_model, train_loader, val_loader, 
                         num_epochs=NUM_EPOCH_MLP_TRAINING, accelerator=accelerator, 
                         accum_steps=args.accum_steps)

    if accelerator.is_main_process:
        mlp_path = f'mlp_regressor_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
        accelerator.save(mlp_model.state_dict(), mlp_path)
        wandb.save(mlp_path)
        wandb.run.summary["mlp_parameters"] = mlp_model.count_parameters()
        print(f"MLP parameters: {mlp_model.count_parameters()}")

    wandb.finish()