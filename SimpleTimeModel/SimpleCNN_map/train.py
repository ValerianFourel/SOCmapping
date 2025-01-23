import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from config import (TIME_BEGINNING, TIME_END, seasons, years_padded, 
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, 
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, MAX_OC)
from torch.utils.data import Dataset, DataLoader
from modelCNN import SmallCNN


def train_model(model, dataloader, num_epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store all predictions and targets
    all_outputs = []
    all_targets = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_outputs = []
        epoch_targets = []

        for longitudes, latitudes, features, targets in tqdm(dataloader):
            # Move data to device
            features = features.to(device)
            targets = targets.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Store predictions and targets
            epoch_outputs.extend(outputs.cpu().detach().numpy())
            epoch_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)

        # Convert lists to numpy arrays for correlation calculation
        epoch_outputs = np.array(epoch_outputs)
        epoch_targets = np.array(epoch_targets)

        # Calculate correlations
        correlation = np.corrcoef(epoch_outputs, epoch_targets)[0, 1]
        r_squared = correlation ** 2

        print(f'Epoch {epoch+1}:')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Correlation: {correlation:.4f}')
        print(f'R²: {r_squared:.4f}\n')

        # Store the predictions and targets for the last epoch
        if epoch == num_epochs - 1:
            all_outputs = epoch_outputs
            all_targets = epoch_targets

    # Create scatter plot of predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_outputs, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], 
             [min(all_targets), max(all_targets)], 
             'r--', label='Perfect prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()

    # Save the plot
    plt.savefig('predictions_vs_true_values.png')
    plt.close()

    # Save predictions and targets to CSV
    results_df = pd.DataFrame({
        'True_Values': all_targets,
        'Predictions': all_outputs
    })
    results_df.to_csv('model_predictions.csv', index=False)

    # Calculate final statistics
    final_correlation = np.corrcoef(all_outputs, all_targets)[0, 1]
    final_r_squared = final_correlation ** 2
    mse = np.mean((all_outputs - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_outputs - all_targets))

    # Save statistics to text file
    with open('model_statistics.txt', 'w') as f:
        f.write(f'Final Statistics:\n')
        f.write(f'Correlation: {final_correlation:.4f}\n')
        f.write(f'R²: {final_r_squared:.4f}\n')
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'MAE: {mae:.4f}\n')

    return model, all_outputs, all_targets

# Main execution
if __name__ == "__main__":
    # Data preparation
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    # Flatten and remove duplicates
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

    # Create dataset and dataloader
    dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, df)
    print("Dataset length:", len(df))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Initialize and train the model
    model = SmallCNN(input_channels=6)  # Adjust input_channels based on your data
    print(model.count_parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, all_outputs, all_targets = train_model(model, dataloader, num_epochs=100, device=device)

    # Save the model
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("Model trained and saved successfully!")
