import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse
import os
import statsmodels.api as sm
from config import (
    TIME_BEGINNING, TIME_END, file_path_coordinates_Bavaria_1mil, MAX_OC,
    bands_list_order, window_size, time_before
)
import logging
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data

from modelTransformerVAE import TransformerVAE
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataloader.dataframe_loader import separate_and_add_data_1mil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define MLP Regressor
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output is a single value (SOC)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
    """
    Create a balanced dataset by binning OC values and resampling.
    If use_validation is True, splits into training and validation sets.
    If use_validation is False, returns only a balanced training set.
    """
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    training_dfs = []
    
    if use_validation:
        validation_indices = []
        
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
        return training_df, validation_df
    
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        
        if not training_dfs:
            raise ValueError("No training data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        return training_df

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with VAEs and train MLP/OLS for SOC prediction')
    parser.add_argument('--weights_dir', type=str, default='/clusterfs/SOCProject/SOCmapping/VAETransformer/weights', 
                        help='Directory containing VAE weights')
    parser.add_argument('--num_epochs_mlp', type=int, default=50, help='Number of epochs to train MLP')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference and training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer size for MLP')
    parser.add_argument('--use_validation', action='store_true', help='Use validation set during training')
    return parser.parse_args()

def load_vae_models(weights_dir, bands_list_order, device_map):
    """Load pre-trained VAEs from weights directory."""
    weights_path = Path(weights_dir)
    vae_models = {}
    
    for band in bands_list_order:
        vae_path = weights_path / band / f"{band}.pth"
        if vae_path.exists():
            time_steps = 1 if band == 'Elevation' else (int(TIME_END) - int(TIME_BEGINNING) + 1)
            latent_dim = 24 if band == 'Elevation' else 48
            num_heads = 16 if band == 'Elevation' else 32
            
            vae = TransformerVAE(
                input_channels=1,
                input_height=window_size,
                input_width=window_size,
                input_time=time_steps,
                num_heads=num_heads,
                latent_dim=latent_dim,
                dropout_rate=0.3
            )
            vae.load_state_dict(torch.load(vae_path, map_location=device_map[band]))
            vae.eval()
            vae_models[band] = vae
            logger.info(f"Loaded VAE for band {band} from {vae_path}")
        else:
            logger.warning(f"VAE weights not found for band {band} at {vae_path}")
    
    return vae_models

def get_latent_representations(vae_models, dataset, accelerator, batch_size):
    """Perform inference with VAEs to get latent space z."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader = accelerator.prepare(dataloader)
    
    all_latents = []
    all_targets = []
    all_coordinates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent representations"):
            longitude, latitude, features, oc, _ = batch
            batch_latents = []
            
            for band_idx, band in enumerate(bands_list_order):
                if band in vae_models:
                    vae = vae_models[band].to(accelerator.device)
                    band_features = features[:, band_idx:band_idx+1, :, :, :]
                    recon, mu, log_var = vae(band_features)
                    # Sample z from the latent distribution
                    z = vae.reparameterize(mu, log_var)
                    batch_latents.append(z)
            
            if batch_latents:
                latents = torch.cat(batch_latents, dim=1)
                latents = accelerator.gather(latents).cpu().numpy()
                oc = accelerator.gather(oc).cpu().numpy()
                coords = torch.stack([longitude, latitude], dim=1)
                coords = accelerator.gather(coords).cpu().numpy()
                
                all_latents.append(latents)
                all_targets.append(oc)
                all_coordinates.append(coords)
    
    if accelerator.is_main_process:
        latents = np.concatenate(all_latents, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        coordinates = np.concatenate(all_coordinates, axis=0)
        logger.info(f"Latent representations (z) shape: {latents.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        logger.info(f"Coordinates shape: {coordinates.shape}")
        return latents, targets, coordinates
    return None, None, None

def train_mlp(mlp, train_latents, train_targets, val_latents, val_targets, num_epochs, batch_size, accelerator):
    """Train the MLP regressor with optional validation."""
    train_dataset = TensorDataset(
        torch.tensor(train_latents, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = accelerator.prepare(train_loader)
    
    if val_latents is not None and val_targets is not None:
        val_dataset = TensorDataset(
            torch.tensor(val_latents, dtype=torch.float32),
            torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        val_loader = accelerator.prepare(val_loader)
    else:
        val_loader = None
    
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    mlp, optimizer = accelerator.prepare(mlp, optimizer)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        mlp.train()
        train_loss = 0.0
        for batch_latents, batch_targets in tqdm(train_loader, desc=f"MLP Train Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            outputs = mlp(batch_latents)
            loss = criterion(outputs, batch_targets)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        if val_loader:
            mlp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_latents, batch_targets in tqdm(val_loader, desc=f"MLP Val Epoch {epoch+1}", leave=False):
                    outputs = mlp(batch_latents)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = accelerator.unwrap_model(mlp).state_dict()
            
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                wandb.log({"mlp_train_loss": avg_train_loss, "mlp_val_loss": avg_val_loss, "epoch": epoch+1})
        else:
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
                wandb.log({"mlp_train_loss": avg_train_loss, "epoch": epoch+1})
    
    return mlp, best_model_state, best_val_loss if val_loader else None

def train_ols(train_latents, train_targets, val_latents, val_targets):
    """Train an OLS regression model."""
    X_train = sm.add_constant(train_latents)  # Add intercept term
    ols_model = sm.OLS(train_targets, X_train).fit()
    
    if val_latents is not None and val_targets is not None:
        X_val = sm.add_constant(val_latents)
        val_predictions = ols_model.predict(X_val)
        val_mse = np.mean((val_targets - val_predictions) ** 2)
        logger.info(f"OLS Validation MSE: {val_mse:.4f}")
        wandb.log({"ols_val_mse": val_mse})
    else:
        val_mse = None
    
    train_predictions = ols_model.predict(X_train)
    train_mse = np.mean((train_targets - train_predictions) ** 2)
    logger.info(f"OLS Training MSE: {train_mse:.4f}")
    wandb.log({"ols_train_mse": train_mse})
    
    return ols_model, train_mse, val_mse

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16')
    
    # Initialize W&B
    wandb.init(project="socmapping-VAETransformer-MLPRegressor", config={
        "time_beginning": TIME_BEGINNING,
        "time_end": TIME_END,
        "window_size": window_size,
        "bands": bands_list_order,
        "mlp_epochs": args.num_epochs_mlp,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "weights_dir": args.weights_dir,
        "use_validation": args.use_validation
    })
    
    # Load and balance dataset
    logger.info("Loading and balancing dataset...")
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coords, data_paths = separate_and_add_data_1mil()
    samples_coords = list(dict.fromkeys([str(p) for p in samples_coords]))
    data_paths = list(dict.fromkeys([str(p) for p in data_paths]))
    
    if args.use_validation:
        train_df, val_df = create_balanced_dataset(df, use_validation=True)
        train_dataset = MultiRasterDatasetMultiYears(samples_coords, data_paths, train_df, time_before=time_before)
        val_dataset = MultiRasterDatasetMultiYears(samples_coords, data_paths, val_df, time_before=time_before)
        if accelerator.is_main_process:
            logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    else:
        train_df = create_balanced_dataset(df, use_validation=False)
        train_dataset = MultiRasterDatasetMultiYears(samples_coords, data_paths, train_df, time_before=time_before)
        val_dataset = None
        if accelerator.is_main_process:
            logger.info(f"Train dataset size: {len(train_dataset)}")
    
    # Device mapping for VAEs
    device_map = {band: f'cuda:{i % torch.cuda.device_count()}' for i, band in enumerate(bands_list_order)}
    
    # Load VAEs
    vae_models = load_vae_models(args.weights_dir, bands_list_order, device_map)
    
    # Get latent representations (z)
    train_latents, train_targets, train_coordinates = get_latent_representations(vae_models, train_dataset, accelerator, args.batch_size)
    if args.use_validation:
        val_latents, val_targets, val_coordinates = get_latent_representations(vae_models, val_dataset, accelerator, args.batch_size)
    else:
        val_latents, val_targets, val_coordinates = None, None, None
    
    if train_latents is not None and train_targets is not None:
        # Calculate total latent dimension
        total_latent_dim = sum([vae_models[band].latent_dim for band in vae_models])
        
        # Train MLP
        mlp = MLPRegressor(input_dim=total_latent_dim, hidden_dim=args.hidden_dim)
        mlp, best_model_state, best_val_loss = train_mlp(
            mlp, train_latents, train_targets, val_latents, val_targets, 
            args.num_epochs_mlp, args.batch_size, accelerator
        )
        
        # Train OLS
        ols_model, ols_train_mse, ols_val_mse = train_ols(train_latents, train_targets, val_latents, val_targets)
        
        # Save results
        if accelerator.is_main_process:
            np.save("train_latent_representations.npy", train_latents)
            np.save("train_targets.npy", train_targets)
            np.save("train_coordinates.npy", train_coordinates)
            if args.use_validation:
                np.save("val_latent_representations.npy", val_latents)
                np.save("val_targets.npy", val_targets)
                np.save("val_coordinates.npy", val_coordinates)
            
            if best_model_state is not None:
                final_model_path = f"mlp_regressor_best_val_loss_{best_val_loss:.4f}.pth" if args.use_validation else "mlp_regressor.pth"
                accelerator.save(best_model_state, final_model_path)
                wandb.save(final_model_path)
                logger.info(f"Saved MLP model to {final_model_path}")
            else:
                torch.save(mlp.state_dict(), "mlp_regressor.pth")
                wandb.save("mlp_regressor.pth")
                logger.info("Saved MLP model to mlp_regressor.pth")
            
            # Save OLS summary
            with open("ols_summary.txt", "w") as f:
                f.write(str(ols_model.summary()))
            wandb.save("ols_summary.txt")
            logger.info("Saved OLS summary to ols_summary.txt")
    
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()