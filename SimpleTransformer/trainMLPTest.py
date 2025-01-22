import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset , normalize_batch
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, filter_and_uniform_sample,filter_and_rebalance_dataframe
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import (window_size, TIME_BEGINNING, TIME_END, YEARS_BACK, seasons, 
                   years_padded, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, 
                   DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC,MAX_OC,
                   feature_dim, num_heads, num_layers, hidden_dim, seq_len, spatial_dim, output_dim,learning_rate)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import TransformerForRegression , get_trainable_params
from visualisation_utils import analyze_oc_distribution , visualize_batch_distributions, plot_output_target_difference
from losses import ChiSquareLoss, HuberLoss, calculate_losses, InverseHuberLoss
import wandb
from accelerate import Accelerator
import argparse

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
        'huber': nn.HuberLoss(), 'inverseHuber': InverseHuberLoss()
    }
    return loss_functions.get(str(loss_type).lower(), nn.L1Loss())

def train_model(args):
    # Initialize accelerator
    accelerator = Accelerator()
    loss_type = args.loss_type
    epochs = args.epochs
    # Initialize wandb
    wandb.init(
        project="soil-prediction-MLP",  # Changed project name
        config={
            "learning_rate": learning_rate,
            "architecture": "MLPRegression",  # Changed architecture name
            "epochs": 100,
            "batch_size": 256,
            "loss_function": loss_type
        }
    )

    # Data preparation
    df = filter_and_uniform_sample(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    analyze_oc_distribution(df)

    # Flatten and deduplicate paths
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    # Create dataset
    dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, df, 
                               YEARS_BACK, seasons, years_padded)

    # Model parameters
    input_channels = 10
    input_depth = 6
    input_height = window_size*4 + 1
    input_width = window_size*4 + 1
    input_size = input_channels * input_depth * input_height * input_width  # Flatten input dimensions
    hidden_size1 = 512
    hidden_size2 = 256
    output_size = 1  # Assuming single value regression
    batch_size = 256

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the MLP model
    class MLPRegression(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(MLPRegression, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size1)
            self.layer2 = nn.Linear(hidden_size1, hidden_size2)
            self.layer3 = nn.Linear(hidden_size2, output_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            # Flatten the input
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.layer1(x)))
            x = self.dropout(self.relu(self.layer2(x)))
            x = self.layer3(x)
            return x

    # Initialize model, optimizer, and criterion
    model = MLPRegression(input_size, hidden_size1, hidden_size2, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
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
             # batch_features = normalize_batch(batch_features.float())
            batch_features = batch_features.float()
            batch_targets = batch_targets.float()

            if not check_batch_variation(batch_features, batch_targets):
                continue

            optimizer.zero_grad()
            outputs = model(batch_features)

            if epoch >= 80:
                print(outputs)

            epoch_outputs.append(outputs.detach().cpu())
            epoch_targets.append(batch_targets.detach().cpu())

            loss = criterion(outputs, batch_targets)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(dataloader)
            })
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
        wandb.log({
            "epoch": epoch,
            "epoch_correlation": correlation,
            "epoch_loss": epoch_loss,
            "epoch_l1_loss": l1_loss,
            "epoch_l2_loss": l2_loss
        })

        print(f'\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            unwrapped_model = accelerator.unwrap_model(model)
            model_path = f'best_model_mlp_{loss_type}_{str(epoch)}.pth'
            torch.save(unwrapped_model.state_dict(), model_path)
            wandb.save(model_path)

    print('Training finished!')
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description='Train SOC Predictor model with different loss functions')
    parser.add_argument('--loss_type', type=str, default='inverseHuber', choices=['l1', 'l2', 'huber','inverseHubers'],
                        help='Type of loss function to use (default: l1)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)