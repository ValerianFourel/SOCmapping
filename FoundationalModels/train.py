
from functools import partial
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from IEEE_TPAMI_SpectralGPT.util import video_vit
from IEEE_TPAMI_SpectralGPT.util.logging import master_print as print
from IEEE_TPAMI_SpectralGPT.util.pos_embed import interpolate_pos_embed

from IEEE_TPAMI_SpectralGPT.models_mae_spectral import MaskedAutoencoderViT
from config import (LOADING_TIME_BEGINNING , TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, imageSize, bands_dict , 
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from IEEE_TPAMI_SpectralGPT import models_vit_tensor

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Model configuration parser')

    # Model architecture argument
    parser.add_argument('--model', default='vit_base_patch8', type=str, metavar='MODEL',
                        # vit_tensor_base_patch16 vit_base_patch16
                        help='Name of model to train')

    # Drop path rate argument
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Number of classes argument
    parser.add_argument('--nb_classes', default=62, type=int,
                        help='number of the classification types')
    
    parser.add_argument('--finetune', default='/home/vfourel/SOCProject/SOCmapping/FoundationalModels/models/SpectralGPT/SpectralGPT+.pth',
                        help='finetune from checkpoint')

    args = parser.parse_args()

    return args


def load_model(model, args, device='cuda'):
    if args.finetune:
        # Load checkpoint
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print(f"Loading pre-trained checkpoint from: {args.finetune}")

        checkpoint_model = checkpoint['model']

        # Remove incompatible keys
        skip_keys = [
            'patch_embed.0.proj.weight', 
            'patch_embed.1.proj.weight', 
            'patch_embed.2.proj.weight',
            'patch_embed.2.proj.bias', 
            'head.weight', 
            'head.bias'
        ]

        for k in skip_keys:
            if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Handle position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # Load state dict
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys:", msg.missing_keys)

        # Initialize head layer
        trunc_normal_(model.head.weight, std=2e-5)

    # Move model to device
    model = model.to(device)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f'Number of parameters: {n_parameters/1e6:.2f}M')

    return model


def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
    """
    Create a balanced dataset by binning OC values and resampling to ensure more homogeneous distribution

    Args:
        df: Input DataFrame
        n_bins: Number of bins for OC values
        min_ratio: Minimum ratio of samples in each bin compared to the maximum bin
    """
    # Create bins
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins

    # Count samples in each bin
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)  # Ensure at least 5 samples

    # Create validation set by taking 4 samples from each bin
    validation_indices = []
    training_dfs = []

    for bin_idx in range(len(bin_counts)):
        bin_data = df[df['bin'] == bin_idx]

        if len(bin_data) >= 4:
            # Randomly select 4 samples for validation
            val_samples = bin_data.sample(n=min(8, len(bin_data)))
            validation_indices.extend(val_samples.index)

            # Remaining samples for training
            train_samples = bin_data.drop(val_samples.index)

            if len(train_samples) > 0:  # Only process if there are remaining samples
                # Resample if needed
                if len(train_samples) < min_samples:
                    # Resample with replacement to reach min_samples
                    resampled = train_samples.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(train_samples)

    if not training_dfs:  # Check if we have any training data
        raise ValueError("No training data available after binning and sampling")

    if not validation_indices:  # Check if we have any validation data
        raise ValueError("No validation data available after binning and sampling")

    # Combine all training samples
    training_df = pd.concat(training_dfs)
    validation_df = df.loc[validation_indices]

    # Remove the temporary bin column
    training_df = training_df.drop('bin', axis=1)
    validation_df = validation_df.drop('bin', axis=1)

    print(f"Number of bins with data: {len(bin_counts)}")
    print(f"Min Number in a bins with data: {min_samples}")
    print(f"Original data size: {len(df)}")
    print(f"Training set size: {len(training_df)}")
    print(f"Validation set size: {len(validation_df)}")

    return training_df, validation_df



class LatentMLP(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, dropout=0.1):
        """
        Args:
            input_dim (int): Dimension of input features (1280 from latent)
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout rate
        """
        super().__init__()

        # First we'll process each token, then aggregate
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # After aggregating tokens, final MLP layers
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)  # Final output dimension of 1
        )

    def forward(self, latent):
        """
        Args:
            latent: Tensor of shape [batch_size, num_tokens, embedding_dim]
                   In this case [2, 25, 1280]
        Returns:
            output: Tensor of shape [batch_size, 1]
                   In this case [2, 1]
        """
        # Process each token: [2, 25, 1280] -> [2, 25, hidden_dim]
        token_features = self.token_mlp(latent)

        # Average across tokens: [2, 25, hidden_dim] -> [2, hidden_dim]
        pooled_features = torch.mean(token_features, dim=1)

        # Final MLP to get output: [2, hidden_dim] -> [2, 1]
        output = self.final_mlp(pooled_features)

        return output

# Example usage:
def process_latent(latent, target=None):
    """
    Args:
        latent: Tensor from encoder [2, 25, 1280]
        target: Optional target values for training [2, 1]
    """
    # Initialize model
    model = LatentMLP(input_dim=1280)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    latent = latent.to(device)

    # Forward pass
    output = model(latent)

    if target is not None:
        # Define loss function (MSE for regression)
        criterion = nn.MSELoss()
        target = target.to(device)
        loss = criterion(output, target)

        # Example of backward pass (if training)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return output, loss

    return output

# Example of how to use it:
"""
# Your latent tensor
latent = torch.randn(2, 25, 1280)

# For training:
target = torch.randn(2, 1)  # Example target values
output, loss = process_latent(latent, target)
print(f"Output shape: {output.shape}")  # [2, 1]
print(f"Loss: {loss.item()}")

# For inference:
output = process_latent(latent)
print(f"Output shape: {output.shape}")  # [2, 1]
"""

def print_tensor_shapes(item, prefix=''):
    if isinstance(item, torch.Tensor):
        print(f"{prefix}Tensor shape:", item.shape)
    elif isinstance(item, (list, tuple)):
        print(f"{prefix}List/Tuple of length:", len(item))
        for i, subitem in enumerate(item):
            print_tensor_shapes(subitem, prefix=f"{prefix}[{i}] ")
    elif isinstance(item, dict):
        for key, value in item.items():
            print_tensor_shapes(value, prefix=f"{prefix}['{key}'] ")
    else:
        print(f"{prefix}Not a tensor, type:", type(item))


import pandas as pd
import torch
from pathlib import Path
import numpy as np

def encode_and_save_tensors(data_dict, model, output_path, longitude, latitude, oc):
    """
    Encode tensors and save results with metadata to parquet file.

    Parameters:
    data_dict: dictionary containing tensors and metadata
    model: the encoder model
    output_path: where to save the parquet file
    longitude, latitude, oc: additional metadata
    """
    # Initialize lists to store all data
    all_data = {
        'longitude': [],
        'latitude': [],
        'oc': [],
        'band': [],
        'year': [],
        'type': [],
        'encoding': []  # Will store flattened encodings
    }

    # Process each band
    with torch.no_grad():  # Disable gradient computation for efficiency
        for band_name, band_data in data_dict.items():
            if len(band_data['tensors']) > 0:
                # Get encodings for all tensors in this band
                tensors = band_data['tensors']
                years = band_data['years'].tolist()
                types = band_data['types']

                # Process each tensor individually
                for idx, tensor in enumerate(tensors):
                    # Add batch dimension if needed
                    if len(tensor.shape) == 3:  # Assuming (C, H, W) format
                        tensor = tensor.unsqueeze(0)  # Make it (1, C, H, W)

                    # Encode tensor
                    encoding = model.encode(tensor)

                    # Convert encoding to numpy and flatten
                    encoding_np = encoding.cpu().numpy().flatten()

                    # Append all data
                    all_data['longitude'].append(longitude)
                    all_data['latitude'].append(latitude)
                    all_data['oc'].append(oc)
                    all_data['band'].append(band_name)
                    all_data['year'].append(years[idx])
                    all_data['type'].append(types[idx])
                    all_data['encoding'].append(encoding_np)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save to parquet
    output_file = Path(output_path) / f"encodings_{longitude}_{latitude}.parquet"
    df.to_parquet(output_file, index=False)

    return output_file

# Example usage:
def process_batch(batch, model, output_dir):
    """
    Process a single batch of data
    """
    longitude, latitude, data_dict, oc = batch

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Encode and save
    saved_file = encode_and_save_tensors(
        data_dict, 
        model, 
        output_path, 
        longitude, 
        latitude, 
        oc
    )

    print(f"Saved encodings to {saved_file}")

# Process all batches
def process_all_data(dataloader, model, output_dir):
    """
    Process all batches in the dataloader
    """
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx}")
        process_batch(batch, model, output_dir)


def transform_data(stacked_tensor, metadata):
    """
    Parameters:
    stacked_tensor: tensor of shape [B, total_steps, H, W]
    metadata: tensor of shape [B, total_steps, 2]
    index_to_band: dictionary mapping indices to band names
    time_before: integer specifying the time dimension T
    """
    B, total_steps, H, W = stacked_tensor.shape
    index_to_band = {k: v for k, v in bands_dict.items()}
    num_channels = (total_steps - 1) // time_before  # -1 for elevation instrument

    # 1. Extract elevation instrument data (metadata [0,0])
    elevation_mask = (metadata[:, :, 0] == 0) & (metadata[:, :, 1] == 0)
    elevation_indices = elevation_mask.nonzero(as_tuple=True)
    elevation_instrument = stacked_tensor[
        elevation_indices[0], 
        elevation_indices[1], 
        :, 
    ].reshape(B, 1, H, W)

    # 2. Process remaining data
    remaining_indices = ~elevation_mask
    metadata_strings = []
    channel_time_data = {}  # Dictionary to store data by channel and time

    # Initialize dictionary for all channels and times
    for c in range(num_channels):
        channel_time_data[c] = {}
        for t in range(time_before):
            channel_time_data[c][t] = []
    # Organize data by channel and time
    for b in range(B):
        channel_year_data = {}  # Temporary storage to sort by year

        # First collect all data for this batch
        for i in range(total_steps):
            if not elevation_mask[b, i]:
                band_idx = int(metadata[b, i, 0])
                year = int(metadata[b, i, 1])
                band_name = index_to_band[band_idx]

                # Initialize channel if not exists
                channel = band_idx - 1  # Assuming band_idx starts from 1
                if channel not in channel_year_data:
                    channel_year_data[channel] = []

                # Store data with year for sorting
                channel_year_data[channel].append((year, stacked_tensor[b, i]))

        # Sort and store data for each channel
        for channel in channel_year_data:
            # Sort by year in descending order
            sorted_data = sorted(channel_year_data[channel], 
                            key=lambda x: x[0], 
                            reverse=True) 

            # Store in the final structure
            for time_idx, (year, tensor_data) in enumerate(sorted_data):
                if time_idx < time_before:  # Ensure we don't exceed time dimension
                    metadata_strings.append(f"{index_to_band[channel+1]}_{year}")

                    # Initialize if needed
                    if len(channel_time_data[channel][time_idx]) <= b:
                        channel_time_data[channel][time_idx].append(tensor_data)

    # 3. Reshape into final tensor
    remaining_tensor = torch.zeros((B, num_channels, time_before, H, W))

    # Fill the tensor
    for c in range(num_channels):
        for t in range(time_before):
            if channel_time_data[c][t]:
                data = torch.stack(channel_time_data[c][t])
                remaining_tensor[:len(data), c, t] = data

    return elevation_instrument, remaining_tensor, metadata_strings

# Example usage:
"""
B, total_steps, H, W = 256, 61, 96, 96
time_before = 12
num_channels = (total_steps - 1) // time_before  # Should be 5

stacked_tensor = torch.randn(B, total_steps, H, W)
metadata = torch.zeros(B, total_steps, 2)
index_to_band = {1: "band_A", 2: "band_B", 3: "band_C", 4: "band_D", 5: "band_E"}

elevation_instrument, ordered_tensor, metadata_strings = transform_data(
    stacked_tensor, metadata, index_to_band, time_before=12
)

# ordered_tensor shape: [256, 5, 12, 96, 96]
# elevation_instrument shape: [256, 1, 96, 96]
"""

# Main execution
if __name__ == "__main__":
    args = parse_args()
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

    # Create balanced datasets
    train_df, val_df = create_balanced_dataset(df)
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    # Create datasets and dataloaders
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        # Iterate through the DataLoader to get the first batch
    for batch in train_loader:
        ongitude, latitude, stacked_tensor, metadata, oc = batch 
        break
    print(stacked_tensor.shape)
    print(metadata.shape)
    print(metadata[2])
    #first_batch = decode_data_dict(stacked_tensor, metadata)
    elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
    print(remaining_tensor.shape)
#######################################################################

    model = models_vit_tensor.__dict__[args.model](drop_path_rate=args.drop_path,
         num_classes=args.nb_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stacked_values= stacked_tensor.unsqueeze(dim=1)
    print('remaining_tensor[0,0] ', remaining_tensor[0,0].shape)
    x = remaining_tensor[0,0].unsqueeze(dim=0).unsqueeze(dim=0)
    print('x ', x.shape)
    x = x.to(device)
    model = load_model(model, args, device)
    model.eval()
    embeddings = model.patch_embed(x)
    #     # Setup device
    

    # # Set to eval mode for inference
    # 
    print(embeddings.shape)

    
