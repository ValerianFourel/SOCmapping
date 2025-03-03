from functools import partial
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import wandb
from datetime import datetime
import os
import time

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
    metadata_strings = [[[[] for _ in range(time_before)] 
                        for _ in range(num_channels)] 
                        for _ in range(B)]
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
                    metadata_str = f"{index_to_band[channel+1]}_{year}"
                    metadata_strings[b][channel][time_idx] = metadata_str

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





import pandas as pd
import torch
import numpy as np
from itertools import product

def process_batch_to_dataframe(longitude, latitude, elevation_instrument, remaining_tensor, 
                             metadata_strings, oc, model, device):
    # Lists to store all data
    records = []

    # Process each sample in the batch
    for idx in range(longitude.shape[0]):
        # Base record with location and oc
        base_record = {
            'longitude': longitude[idx].item(),
            'latitude': latitude[idx].item(),
            'organic_carbon': oc[idx].item()
        }

        # Process elevation embedding
        elevation_input = elevation_instrument[idx:idx+1].unsqueeze(1)  # [1, 1, 96, 96]
        elevation_input = elevation_input.to(device).expand(1, 1, 3, 96, 96)
        elevation_embedding = model.patch_embed(elevation_input).squeeze().cpu().detach().numpy()

        # Add elevation embedding to base record
        base_record['elevation_embedding'] = elevation_embedding.tolist()

        # Process channel/time embeddings
        channel_time_embeddings = {}

        # Iterate through all channel and time combinations
        for channel, time in product(range(5), range(5)):
            # Get the specific tensor slice
            x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 96]
            x = x.expand(1, 1, 3, 96, 96)  # Expand to match model input requirements
            x = x.to(device)

            # Get embedding
            embedding = model.patch_embed(x).squeeze().cpu().detach().numpy()

            # Create key using metadata string
            tag = metadata_strings[idx][channel][time]

            # Store embedding with its tag
            channel_time_embeddings[tag] = embedding.tolist()

        # Add channel/time embeddings to base record
        base_record['channel_time_embeddings'] = channel_time_embeddings

        records.append(base_record)

    return pd.DataFrame(records)


def save_run_documentation(folder_path, args, run_number, timing_info):
    """Create documentation file for the run"""
    doc_path = os.path.join(folder_path, f'run_{run_number}_documentation.txt')

    with open(doc_path, 'w') as f:
        f.write(f"Run {run_number} Documentation\n")
        f.write("=" * 50 + "\n\n")

        # Timing information
        f.write("Timing Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"LOADING_TIME_BEGINNING: {timing_info['loading_time']}\n")
        f.write(f"TIME_BEGINNING: {timing_info['start_time']}\n")
        f.write(f"TIME_END: {timing_info['end_time']}\n")
        f.write(f"INFERENCE_TIME: {timing_info['inference_time']}\n")
        f.write(f"MAX_OC: {timing_info['max_oc']}\n")
        f.write(f"time_before: {timing_info['time_before']}\n")
        f.write(f"imageSize: {timing_info['image_size']}\n\n")

        # Arguments information
        f.write("Arguments Configuration:\n")
        f.write("-" * 20 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

def save_to_parquet(train_loader, model, device, args, 
                   base_folder='/fast/vfourel/SOCProject'):
    # Create run folder with timestamp
    run_number = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_folder, f'run_{run_number}')
    os.makedirs(folder_path, exist_ok=True)

    # Initialize timing information
    timing_info = {
        'loading_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'start_time': time.time(),
        'time_before': time.time(),
        'max_oc': float('-inf'),
        'image_size': (96, 96),  # Update if different
        'inference_time': 0
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()

            longitude, latitude, stacked_tensor, metadata, oc = batch
            batch_size = len(longitude)

            # Update max_oc
            timing_info['max_oc'] = max(timing_info['max_oc'], torch.max(oc).item())

            # Process the batch data
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(
                stacked_tensor, metadata)

            # Convert batch to DataFrame
            batch_df = process_batch_to_dataframe(
                longitude, latitude, elevation_instrument, remaining_tensor,
                metadata_strings, oc, model, device
            )

            # Save individual batch to parquet
            batch_filename = f'batch_{batch_idx:04d}_size_{batch_size}.parquet'
            batch_path = os.path.join(folder_path, batch_filename)
            batch_df.to_parquet(batch_path)

            # Update timing information
            batch_end_time = time.time()
            timing_info['inference_time'] += (batch_end_time - batch_start_time)

    # Final timing updates
    timing_info['end_time'] = time.time()

    # Save documentation
    save_run_documentation(folder_path, args, run_number, timing_info)

    return folder_path

if __name__ == "__main__":
    args = parse_args()
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Initialize models
    vit_model = models_vit_tensor.__dict__[args.model](
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes
    ).to(device)
    vit_model = load_model(vit_model, args, device)
    df = save_to_parquet(train_loader, vit_model, device,args)
    
