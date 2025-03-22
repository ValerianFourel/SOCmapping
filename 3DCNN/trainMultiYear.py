import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, num_epochs,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from torch.utils.data import Dataset, DataLoader
from modelCNNMultiYear import Small3DCNN

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
    """
    Create a balanced dataset by binning OC values and resampling to ensure more homogeneous distribution
    """
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

    if not training_dfs:
        raise ValueError("No training data available after binning and sampling")
    if not validation_indices:
        raise ValueError("No validation data available after binning and sampling")

    training_df = pd.concat(training_dfs)
    validation_df = df.loc[validation_indices]
    training_df = training_df.drop('bin', axis=1)
    validation_df = validation_df.drop('bin', axis=1)

    print(f"Number of bins with data: {len(bin_counts)}")
    print(f"Min Number in a bins with data: {min_samples}")
    print(f"Original data size: {len(df)}")
    print(f"Training set size: {len(training_df)}")
    print(f"Validation set size: {len(validation_df)}")

    return training_df, validation_df

def train_model(model, train_loader, val_loader, num_epochs=100, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for longitudes, latitudes, features, targets in tqdm(train_loader):
            features = features.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets = []

        with torch.no_grad():
            for longitudes, latitudes, features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device).float()
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_outputs.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss = val_loss / len(val_loader)

        # Calculate metrics
        val_outputs = np.array(val_outputs)
        val_targets = np.array(val_targets)
        correlation = np.corrcoef(val_outputs, val_targets)[0, 1]
        r_squared = correlation ** 2
        mse = np.mean((val_outputs - val_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(val_outputs - val_targets))

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'correlation': correlation,
            'r_squared': r_squared,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            wandb.run.summary["best_val_loss"] = best_val_loss

        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')

    model.load_state_dict(best_model)
    return model, val_outputs, val_targets

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="socmapping-3dcnn",
        config={
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands": len(bands_list_order),
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 0.001
        }
    )

    # Data preparation
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

    # Create balanced datasets
    train_df, val_df = create_balanced_dataset(df)
    
    # Log dataset sizes
    wandb.run.summary["train_size"] = len(train_df)
    wandb.run.summary["val_size"] = len(val_df)

    # Create datasets and dataloaders
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    for batch in train_loader:
        _ , _ , first_batch , _ = batch 
        break

    first_batch_size = first_batch.shape
    print("Size of the first batch:", first_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize and train model
    model = Small3DCNN(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before
    )
    
    # Log model parameters
    wandb.run.summary["model_parameters"] = model.count_parameters()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, val_outputs, val_targets = train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)

    # Save model to wandb
    model_path = f'cnn_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    # Finish wandb run
    wandb.finish()
    print("Model trained and saved successfully!")
