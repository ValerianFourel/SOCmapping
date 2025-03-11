import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataloaderMappingMultiYear import MultiRasterDatasetMappingMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
import argparse
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, NUM_EPOCH_VAE_TRAINING, NUM_EPOCH_MLP_TRAINING,
                   file_path_coordinates_Bavaria_1mil,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from torch.utils.data import Dataset, DataLoader
from modelTransformerVAE import TransformerVAE
from mapping import create_prediction_visualizations

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE and MLP for SOC prediction with regional embeddings')
    parser.add_argument('--load_vae', type=str, default=None,
                       help='Path to pre-trained VAE model to load (optional)')
    return parser.parse_args()

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
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
    
    return training_df, validation_df

def modify_matrix_coordinates(MatrixCoordinates_1mil_Yearly=MatrixCoordinates_1mil_Yearly,
                             MatrixCoordinates_1mil_Seasonally=MatrixCoordinates_1mil_Seasonally, 
                             INFERENCE_TIME=INFERENCE_TIME):
    for i, path in enumerate(MatrixCoordinates_1mil_Seasonally):
        folders = path.split('/')
        last_folder = folders[-1]
        if last_folder == 'Elevation':
            continue
        elif last_folder == 'MODIS_NPP':
            new_path = f"{path}/{INFERENCE_TIME[:4]}"
        else:
            new_path = f"{path}/{INFERENCE_TIME}"
        MatrixCoordinates_1mil_Seasonally[i] = new_path

    for i, path in enumerate(MatrixCoordinates_1mil_Yearly):
        if 'Elevation' in path:
            continue
        new_path = f"{path}/{INFERENCE_TIME[:4]}"
        MatrixCoordinates_1mil_Yearly[i] = new_path

    return MatrixCoordinates_1mil_Yearly, MatrixCoordinates_1mil_Seasonally

def parallel_predict(df_full, vae_model, bands_yearly, batch_size=256, accelerator=None):
    dataset = MultiRasterDatasetMapping(bands_yearly, df_full)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    vae_model, dataloader = accelerator.prepare(vae_model, dataloader)
    vae_model.eval()

    coordinates, latent_embeddings = [], []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Generating VAE embeddings', leave=True)
        for longitudes, latitudes, features in progress_bar:
            coordinates.append(np.column_stack((longitudes.cpu().numpy(), latitudes.cpu().numpy())))
            features_stacked = torch.stack(list(features.values()), dim=1)
            features_stacked = features_stacked.to(accelerator.device)
            _, mu, _ = vae_model(features_stacked)
            latent_embeddings.extend(mu.cpu().numpy())

    return np.vstack(coordinates), np.array(latent_embeddings)

def train_vae(model, train_loader, num_epochs=10, accelerator=None):
    reconstruction_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    kld_weight = 0.01
    
    train_loader, model, optimizer = accelerator.prepare(
        train_loader, model, optimizer
    )

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kld_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

            optimizer.zero_grad()
            reconstruction, mu, log_var = model(features)
            
            recon_loss = reconstruction_criterion(reconstruction, features)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kld_weight * kld_loss
            
            accelerator.backward(loss)
            optimizer.step()
            
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kld_loss += kld_loss.item()
            
            if accelerator.is_main_process:
                wandb.log({
                    'vae_train_loss': loss.item(),
                    'vae_train_recon_loss': recon_loss.item(),
                    'vae_train_kld_loss': kld_loss.item(),
                    'vae_batch': batch_idx + 1 + epoch * len(train_loader),
                    'vae_epoch': epoch + 1
                })

        train_loss = running_loss / len(train_loader)
        train_recon_loss = running_recon_loss / len(train_loader)
        train_kld_loss = running_kld_loss / len(train_loader)
        
        
        
        if accelerator.is_main_process:
            wandb.log({
                'vae_epoch': epoch + 1,
                'vae_train_loss_avg': train_loss,
                'vae_train_recon_loss_avg': train_recon_loss,
                'vae_train_kld_loss_avg': train_kld_loss,

            })
        
        accelerator.print(f'VAE Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KLD: {train_kld_loss:.4f})')

    return model

def train_mlp(vae_model, mlp_model, train_loader, val_loader, num_epochs=10, accelerator=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    
    train_loader, val_loader, mlp_model, optimizer = accelerator.prepare(
        train_loader, val_loader, mlp_model, optimizer
    )
    vae_model = accelerator.prepare(vae_model)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        mlp_model.train()
        vae_model.eval()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

            optimizer.zero_grad()
            with torch.no_grad():
                _, mu, _ = vae_model(features)
            
            outputs = mlp_model(mu)
            loss = criterion(outputs.squeeze(), targets)
            
            accelerator.backward(loss)
            optimizer.step()
            
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
        val_outputs = []
        val_targets_list = []
        
        with torch.no_grad():
            for longitudes, latitudes, features, targets in val_loader:
                features = features.to(accelerator.device)
                targets = targets.to(accelerator.device).float()
                _, mu, _ = vae_model(features)
                outputs = mlp_model(mu)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                val_outputs.extend(outputs.squeeze().cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        
        val_outputs = np.array(val_outputs)
        val_targets_list = np.array(val_targets_list)
        correlation = np.corrcoef(val_outputs, val_targets_list)[0, 1]
        r_squared = correlation ** 2
        mse = np.mean((val_outputs - val_targets_list) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(val_outputs - val_targets_list))

        if accelerator.is_main_process:
            wandb.log({
                'mlp_epoch': epoch + 1,
                'mlp_train_loss_avg': train_loss,
                'mlp_val_loss': val_loss,
                'mlp_correlation': correlation,
                'mlp_r_squared': r_squared,
                'mlp_mse': mse,
                'mlp_rmse': rmse,
                'mlp_mae': mae
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.run.summary['mlp_best_val_loss'] = best_val_loss
                accelerator.save_state(f'mlp_checkpoint_epoch_{epoch+1}.pth')
                wandb.save(f'mlp_checkpoint_epoch_{epoch+1}.pth')
        
        accelerator.print(f'MLP Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        accelerator.print(f'Validation Loss: {val_loss:.4f}\n')

    return mlp_model, val_outputs, val_targets_list

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    
    wandb.init(
        project="socmapping-VAETransformer-MLPRegressor",
        config={
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands": len(bands_list_order),
            "vae_epochs": 10,
            "mlp_epochs": 10,
            "batch_size": 256,
            "vae_lr": 0.001,
            "mlp_lr": 0.001,
            "kld_weight": 0.01,
            "vae_num_heads": 2,
            "vae_latent_dim": 8,
            "vae_dropout_rate": 0.3,
            "mlp_hidden_dim": 32
        }
    )

    # Training data preparation (for VAE and MLP)
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    def flatten_paths(path_list):
        flattened = []
        for item in path_list:
            if isinstance(item, list):
                flattened.extend(flatten_paths(item))
            else:
                flattened.append(item)
        return flattened

    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    train_df, val_df = create_balanced_dataset(df)
    
    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df)
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
    train_dataset_df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)

    # train_dataset = MultiRasterDatasetMapping(samples_coordinates_array_path, data_array_path, train_dataset_df_full)
    train_dataset = MultiRasterDatasetMappingMultiYears(samples_coordinates_array_path, train_dataset_df_full)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    for batch in train_loader:
        _, _, first_batch, _ = batch
        break
    first_batch_size = first_batch.shape
    if accelerator.is_main_process:
        print("Size of the first batch:", first_batch_size)

    # Initialize VAE
    vae_model = TransformerVAE(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before,
        num_heads=2,
        latent_dim=8,
        dropout_rate=0.3
    )

    if args.load_vae:
        if accelerator.is_main_process:
            print(f"Loading pre-trained VAE from {args.load_vae}")
        vae_model.load_state_dict(torch.load(args.load_vae, map_location=accelerator.device))
    else:
        # Train VAE
        vae_model = train_vae(vae_model, train_loader, num_epochs=10, accelerator=accelerator)
        if accelerator.is_main_process:
            vae_path = f'vaetransformer_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
            accelerator.save(vae_model.state_dict(), vae_path)
            wandb.save(vae_path)

    # Initialize MLP
    mlp_model = MLPRegressor(input_dim=8)
    
    if accelerator.is_main_process:
        wandb.run.summary["vae_parameters"] = vae_model.count_parameters()
        wandb.run.summary["mlp_parameters"] = mlp_model.count_parameters()
        print(f"VAE parameters: {vae_model.count_parameters()}")
        print(f"MLP parameters: {mlp_model.count_parameters()}")

    # Train MLP
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    mlp_model, val_outputs, val_targets = train_mlp(vae_model, mlp_model, train_loader, val_loader, 
                                                    num_epochs=10, accelerator=accelerator)

    # Save final MLP model
    if accelerator.is_main_process:
        mlp_path = f'mlp_regressor_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
        accelerator.save(mlp_model.state_dict(), mlp_path)
        wandb.save(mlp_path)

    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_main_process:
            print(df_full.head())
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error loading 1M points file: {e}")
        wandb.finish()
        exit()

    BandsYearly_1milPoints, _ = modify_matrix_coordinates()
    coordinates, latent_embeddings = parallel_predict(
        df_full=df_full,
        vae_model=vae_model,
        bands_yearly=BandsYearly_1milPoints,
        batch_size=256,
        accelerator=accelerator
    )

    # Log embeddings statistics to wandb
    if accelerator.is_main_process:
        wandb.run.summary["num_embeddings"] = len(latent_embeddings)
        wandb.run.summary["embedding_dim"] = latent_embeddings.shape[1]
        wandb.run.summary["embedding_mean"] = np.mean(latent_embeddings)
        wandb.run.summary["embedding_std"] = np.std(latent_embeddings)

        # Optionally save embeddings and coordinates
        np.save('latent_embeddings.npy', latent_embeddings)
        np.save('coordinates.npy', coordinates)
        wandb.save('latent_embeddings.npy')
        wandb.save('coordinates.npy')

        # Create visualization of embeddings (e.g., mean across latent dimensions)
        mean_embeddings = np.mean(latent_embeddings, axis=1)
        save_path = '/home/vfourel/SOCProject/SOCmapping/predictions_plots/vae_embeddings'
        create_prediction_visualizations(INFERENCE_TIME, coordinates, mean_embeddings, save_path)
        print("VAE embeddings generated and saved successfully!")

    wandb.finish()
