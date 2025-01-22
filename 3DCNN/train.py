import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset , normalize_batch
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, filter_and_rebalance_dataframe , filter_and_uniform_sample
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import (window_size, TIME_BEGINNING, TIME_END, YEARS_BACK, seasons, 
                   years_padded, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, 
                   DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC,MAX_OC)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import SOCPredictor3DCNN , get_trainable_params
from visualisation_utils import analyze_oc_distribution , visualize_batch_distributions, plot_output_target_difference
from losses import ChiSquareLoss, HuberLoss, calculate_losses, InverseHuberLoss
import wandb
from accelerate import Accelerator
import argparse


import torch
import torch.nn as nn

class SimpleSOCPredictor(nn.Module):
    def __init__(self, window_size=3):
        super(SimpleSOCPredictor, self).__init__()

        # Calculate input size
        # window_size * window_size * 6 features * 3 * 3 (from window_size*3)
        input_size = 1152 

        # Simple feedforward network
        self.network = nn.Sequential(
            # First flatten the input
            nn.Flatten(),

            # Layer 1
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: (batch_size, 1, window_size*3, window_size*3, 6)
        return self.network(x)

# Example usage:
# model = SimpleSOCPredictor(window_size=3)
# x = torch.randn(256, 1, 9, 9, 6)  # Example batch
# output = model(x)


def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def check_batch_variation(batch_features, batch_targets):
    # Check if features have variation
    feature_std = torch.std(batch_features, dim=0)
    target_std = torch.std(batch_targets, dim=0)

    # Define thresholds for minimum acceptable standard deviation
    min_std_threshold = 1e-6

    # Check if features are too uniform
    features_uniform = torch.any(feature_std < min_std_threshold).item()
    targets_uniform = torch.any(target_std < min_std_threshold).item()

    if features_uniform:
        print(f"Warning: Features have low variation. Std: {feature_std}")
    if targets_uniform:
        print(f"Warning: Targets have low variation. Std: {target_std}")

    return not (features_uniform or targets_uniform)


def get_loss_function(loss_type='l1'):
    """
    Returns the specified loss function.

    Args:
        loss_type (str): Type of loss function ('l1', 'l2', 'huber')

    Returns:
        torch.nn loss function
    """
    
    loss_functions = {
        'l1': nn.L1Loss(),
        'l2': nn.MSELoss(),
        'huber': nn.HuberLoss(),
        'inverseHuber': InverseHuberLoss()
    }
    return loss_functions.get(str(loss_type).lower(), nn.L1Loss())

def train_model(args):
    # Initialize accelerator
    accelerator = Accelerator()
    loss_type = args.loss_type
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
            # Define the base bands list
    bands_final = ['LAI', 'LST', 'SoilEvaporation', 'MODIS_NPP', 'Elevation', 'TotalEvapotranspiration']

    # Filter bands based on include/exclude arguments
    if args.include_bands:
        bands_final = [band for band in bands_final if band in args.include_bands]
    if args.exclude_bands:
        bands_final = [band for band in bands_final if band not in args.exclude_bands]

    # Initialize wandb
    #wandb.init(
    #    project="soil-prediction",
    #    config={
    #        "learning_rate": learning_rate,
    #        "architecture": "SOCPredictor3DCNN",
    #        "epochs": 100,
    #        "batch_size": batch_size,
    #        "loss_function": loss_type
    #    }
    #)

    # Data preparation
    df = filter_and_uniform_sample(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    analyze_oc_distribution(df)
    df = df.head(256)
    # Flatten and deduplicate paths
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    # Create dataset
    dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, df, 
                               YEARS_BACK, seasons, years_padded,bands_final)
    window_size =3 
    # Model parameters
    input_channels = YEARS_BACK
    input_depth = len(bands_final)
    input_height = window_size*4
    input_width = window_size*4 

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and criterion
    #model = SOCPredictor3DCNN(input_channels, input_depth, input_height, input_width)
    model = SimpleSOCPredictor()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = get_loss_function(loss_type)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_epochs = epochs
    best_loss = float('inf')

    print('get_trainable_params:  ', get_trainable_params(model))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_outputs = []
        epoch_targets = []

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

        for batch_idx, (longitudes, latitudes, batch_features, batch_targets) in enumerate(pbar):
            batch_features = normalize_batch(batch_features.float()) # we need to normalized the features
            batch_features = batch_features.float()
            batch_targets = batch_targets.float()

            if not check_batch_variation(batch_features, batch_targets):
                continue

            optimizer.zero_grad()
            outputs = model(batch_features)

            #if epoch >= 4:
                #print(outputs)

            epoch_outputs.append(outputs.detach().cpu()) # we want to visuazlie the outputs and targets (Y) value
            epoch_targets.append(batch_targets.detach().cpu())

            loss = criterion(outputs, batch_targets) # we now compute the loss 
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            #wandb.log({
            #    "batch_loss": loss.item(),
            #    "batch": batch_idx + epoch * len(dataloader)
            #})
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # End of epoch processing
        all_outputs = torch.cat(epoch_outputs, dim=0).numpy()
        all_targets = torch.cat(epoch_targets, dim=0).numpy()
        l1_loss, l2_loss = calculate_losses(all_outputs, all_targets)
        print('L1 Loss:   ', l1_loss)
        print('L2 Loss:   ', l2_loss)

        plot_output_target_difference(all_outputs, all_targets, epoch, 
                                    f'lr_0_005_window8_rebalanced_yearly_{loss_type}_{num_epochs}')

        correlation = np.corrcoef(all_outputs.flatten(), all_targets.flatten())[0,1]
        print(f'Correlation: {correlation}')

        epoch_loss = running_loss / len(dataloader)
        #wandb.log({
        #    "epoch": epoch,
        #    "epoch_correlation": correlation,
        #    "epoch_loss": epoch_loss,
        #    "epoch_l1_loss": l1_loss,
        #    "epoch_l2_loss": l2_loss
        #})

        print(f'\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            unwrapped_model = accelerator.unwrap_model(model)
            model_path = f'best_model_5epoch_2015_10yb_{loss_type}_{str(epoch)}.pth'
            #torch.save(unwrapped_model.state_dict(), model_path)
            #wandb.save(model_path)

    print('Training finished!')
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description='Train SOC Predictor model with different loss functions')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'huber', 'inverseHuber'],
                        help='Type of loss function to use (default: l1)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--include_bands', nargs='+', default=[],
                        help='List of bands to include (if empty, includes all except excluded)')
    parser.add_argument('--exclude_bands', nargs='+', default=[],
                        help='List of bands to exclude')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    train_model(args)