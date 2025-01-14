import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe , separate_and_add_data
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import   window_size, TIME_BEGINNING ,TIME_END , YEARS_BACK, seasons, years_padded  , SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally ,file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC 
from XGBoost_map.mapping import  create_prediction_visualizations , parallel_predict

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model import SOCPredictor3DCNN
import torch
import torch.nn as nn
import torch.optim as optim


print(years_padded)


##################################################################

# Drawing the mapping

file_path = file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC

def get_top_sampling_years(file_path, top_n=3):
    """
    Read the Excel file and return the top n years with the most samples

    Parameters:
    file_path: str, path to the Excel file
    top_n: int, number of top years to return (default=3)
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Count samples per year and sort in descending order
        year_counts = df['year'].value_counts()
        top_years = year_counts.head(top_n)

        print(f"\nTop {top_n} years with the most samples:")
        for year, count in top_years.items():
            print(f"Year {year}: {count} samples")

        return df, top_years

    except Exception as e:
        print(f"Error reading file: {str(e)}")


df = filter_dataframe(TIME_BEGINNING,TIME_END)



# Loop to update variables dynamically
samples_coordinates_array_path ,  data_array_path = separate_and_add_data()
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

# Remove duplicates
samples_coordinates_array_path = list(dict.fromkeys(samples_coordinates_array_path))
data_array_path = list(dict.fromkeys(data_array_path))


# Create dataset and dataloader
dataset = MultiRasterDataset(samples_coordinates_array_path ,  data_array_path , df,YEARS_BACK, seasons, years_padded )

print("Dataset length:", len(df))
# If using a custom dataset, verify the data is loaded correctly

# Example parameters for soil data
input_channels = 16    # Number of soil properties or spectral bands
input_depth = 1     # Soil depth layers
input_height = window_size*4    # Spatial dimension height
input_width = window_size*4     # Spatial dimension width
batch_size = 256       # Number of samples to process at once

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#############################


# Prepare data for XGBoost
X_train, y_train = [], []
coordinates = []  # To store longitude and latitude


# Initialize model
model = SOCPredictor3DCNN(input_channels, input_depth, input_height, input_width)
criterion = nn.MSELoss()  # L2 loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 5
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Add tqdm progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

    for longitudes, latitudes, batch_features, batch_targets in pbar:
        # Reshape batch_features to include the extra dimension

        # Move data to device
        batch_features = batch_features.to(device).float().unsqueeze(2)  # Now shape is [4, 6, 1, 17, 17]
        batch_targets = batch_targets.to(device).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_features)

        # Calculate loss
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloader)

    # Print epoch statistics
    print(f'\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_model_5epoch.pth')

print('Training finished!')
