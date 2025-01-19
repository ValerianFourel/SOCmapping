import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data, filter_and_rebalance_dataframe
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
from losses import ChiSquareLoss, HuberLoss, calculate_losses
import wandb
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Initialize wandb
wandb.init(
    project="soil-prediction",
    config={
        "learning_rate": 0.01,
        "architecture": "SOCPredictor3DCNN",
        "epochs": 100,
        "batch_size": 256
    }
)

# Your existing data preparation code remains the same until dataloader creation
# df = filter_dataframe(TIME_BEGINNING, TIME_END,MAX_OC)
df = filter_and_rebalance_dataframe(TIME_BEGINNING, TIME_END,MAX_OC)
samples_coordinates_array_path, data_array_path = separate_and_add_data()

analyze_oc_distribution(df)



def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

samples_coordinates_array_path = flatten_paths(samples_coordinates_array_path)
data_array_path = flatten_paths(data_array_path)
samples_coordinates_array_path = list(dict.fromkeys(samples_coordinates_array_path))
data_array_path = list(dict.fromkeys(data_array_path))

dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, df, 
                           YEARS_BACK, seasons, years_padded)

# Model parameters
input_channels = 10
input_depth = 6
input_height = window_size*4 + 1
input_width = window_size*4 + 1
batch_size = 256

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and criterion
model = SOCPredictor3DCNN(input_channels, input_depth, input_height, input_width)
optimizer = optim.Adam(model.parameters(), lr=0.005)
# criterion = nn.L1Loss()

# Prepare for distributed training
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

num_epochs = 100
best_loss = float('inf')



# Initialize loss function
criterion = HuberLoss()


#criterion= LogCoshLoss()
print('get_trainable_params:  ',get_trainable_params(model))

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

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    # Lists to store all outputs and targets for correlation calculation
    epoch_outputs = []
    epoch_targets = []
    for batch_idx, (longitudes, latitudes, batch_features, batch_targets) in enumerate(pbar):
        # Move data to appropriate device (handled by accelerator)
        batch_features = batch_features.float()
        batch_targets = batch_targets.float()
        # Add quality checks
        if not check_batch_variation(batch_features, batch_targets):
            print(f"Batch {batch_idx} has insufficient variation!")
            # Optional: You can also log some statistics
            print(f"Features mean: {torch.mean(batch_features, dim=0)}")
            print(f"Features std: {torch.std(batch_features, dim=0)}")
            print(f"Targets mean: {torch.mean(batch_targets, dim=0)}")
            print(f"Targets std: {torch.std(batch_targets, dim=0)}")
            continue  # Skip this batch if it has insufficient variation


        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_features)
        
        # Calculate loss
        #visualize_batch_distributions(outputs,batch_targets)
        # Store outputs and targets for correlation
        # Detach outputs from computational graph and move to CPU
        epoch_outputs.append(outputs.detach().cpu())
        epoch_targets.append(batch_targets.detach().cpu())

        loss = criterion(outputs, batch_targets)

        # Backward pass and optimize (handled by accelerator)
        accelerator.backward(loss)
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Log batch loss to wandb
        wandb.log({
            "batch_loss": loss.item(),
            "batch": batch_idx + epoch * len(dataloader)
        })

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
          # Calculate correlation at end of epoch
    all_outputs = torch.cat(epoch_outputs, dim=0).numpy()
    all_targets = torch.cat(epoch_targets, dim=0).numpy()
    l1_loss, l2_loss = calculate_losses(all_outputs, all_targets)
    print('L1 Loss:   ',l1_loss)
    print('L2 Loss:   ',l2_loss)
    plot_output_target_difference(all_outputs, all_targets, epoch, 'lr_0_005_window8_rebalanced_yearly')

    # Calculate correlation coefficient
    correlation = np.corrcoef(all_outputs.flatten(), all_targets.flatten())[0,1]
    print(correlation)

    # Calculate and log epoch loss
    epoch_loss = running_loss / len(dataloader)
    wandb.log({
        "epoch": epoch,
        "epoch_correlation": correlation,
        "epoch_loss": epoch_loss,
        "epoch_l1_loss": l1_loss,
        "epoch_l2_loss": l2_loss
    })

    # Print epoch statistics
    print(f'\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        # Unwrap model before saving
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), 'best_model_5epoch_2015_10yb.pth')
        wandb.save('best_model_4epoch_2015_10yb_100ocmax.pth')

print('Training finished!')
wandb.finish()
