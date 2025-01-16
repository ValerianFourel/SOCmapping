import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
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
from model import SOCPredictor3DCNN
from visualisation_utils import analyze_oc_distribution , visualize_batch_distributions
from losses import ChiSquareLoss
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
df = filter_dataframe(TIME_BEGINNING, TIME_END,MAX_OC)
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
optimizer = optim.Adam(model.parameters(), lr=0.05)
# criterion = nn.L1Loss()

# Prepare for distributed training
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

num_epochs = 100
best_loss = float('inf')



# Initialize loss function
criterion = ChiSquareLoss()


#criterion= LogCoshLoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

    for batch_idx, (longitudes, latitudes, batch_features, batch_targets) in enumerate(pbar):
        # Move data to appropriate device (handled by accelerator)
        batch_features = batch_features.float()
        batch_targets = batch_targets.float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_features)
        
        # Calculate loss
        visualize_batch_distributions(outputs,batch_targets)
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

    # Calculate and log epoch loss
    epoch_loss = running_loss / len(dataloader)
    wandb.log({
        "epoch": epoch,
        "epoch_loss": epoch_loss
    })

    # Print epoch statistics
    print(f'\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        # Unwrap model before saving
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), 'best_model_5epoch_2015_10yb.pth')
        wandb.save('best_model_100epoch_2015_10yb_200ocmax.pth')

print('Training finished!')
wandb.finish()
