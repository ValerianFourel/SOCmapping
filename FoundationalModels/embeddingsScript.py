from functools import partial
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from timm.models.layers import trunc_normal_
from IEEE_TPAMI_SpectralGPT.util import video_vit
from IEEE_TPAMI_SpectralGPT.util.logging import master_print as print
from IEEE_TPAMI_SpectralGPT.util.pos_embed import interpolate_pos_embed
from IEEE_TPAMI_SpectralGPT.models_mae_spectral import MaskedAutoencoderViT
from IEEE_TPAMI_SpectralGPT import models_vit_tensor
from config import (
    LOADING_TIME_BEGINNING, TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
    seasons, years_padded, time_before, imageSize, bands_dict, SamplesCoordinates_Yearly,
    MatrixCoordinates_1mil_Yearly, DataYearly, SamplesCoordinates_Seasonally,
    bands_list_order, MatrixCoordinates_1mil_Seasonally, DataSeasonally,
    window_size, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before
)
import argparse
import os
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from itertools import product

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Generation with Size Validation')
    parser.add_argument('--model', default='vit_base_patch8', type=str, metavar='MODEL',
                        help='Name of model to train (e.g., vit_base_patch8)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--nb_classes', default=62, type=int,
                        help='Number of classification types')
    parser.add_argument('--finetune', default='/home/vfourel/SOCProject/SOCmapping/FoundationalModels/models/SpectralGPT/SpectralGPT+.pth',
                        help='Path to finetune checkpoint')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for dataloader')
    parser.add_argument('--output_dir', default='/fast/vfourel/SOCProject', type=str,
                        help='Base folder for output Parquet files')
    args = parser.parse_args()
    return args

# Load and initialize the model
def load_model(model, args, device='cuda'):
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print(f"Loading pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint['model']

        skip_keys = [
            'patch_embed.0.proj.weight', 'patch_embed.1.proj.weight',
            'patch_embed.2.proj.weight', 'patch_embed.2.proj.bias',
            'head.weight', 'head.bias'
        ]
        for k in skip_keys:
            if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys:", msg.missing_keys)
        trunc_normal_(model.head.weight, std=2e-5)

    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f'Number of parameters: {n_parameters/1e6:.2f}M')
    return model

# Create balanced dataset
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

    print(f"Number of bins with data: {len(bin_counts)}")
    print(f"Min samples per bin: {min_samples}")
    print(f"Original data size: {len(df)}")
    print(f"Training set size: {len(training_df)}")
    print(f"Validation set size: {len(validation_df)}")
    return training_df, validation_df

# Transform input tensors
def transform_data(stacked_tensor, metadata):
    B, total_steps, H, W = stacked_tensor.shape
    print(f"Input stacked_tensor shape: [B={B}, total_steps={total_steps}, H={H}, W={W}]")
    
    index_to_band = {k: v for k, v in bands_dict.items()}
    num_channels = (total_steps - 1) // time_before
    print(f"Calculated num_channels: {num_channels} (total_steps-1={total_steps-1} / time_before={time_before})")

    # Extract elevation
    elevation_mask = (metadata[:, :, 0] == 0) & (metadata[:, :, 1] == 0)
    elevation_indices = elevation_mask.nonzero(as_tuple=True)
    elevation_instrument = stacked_tensor[elevation_indices[0], elevation_indices[1], :, :].reshape(B, 1, H, W)
    print(f"Elevation instrument shape: {elevation_instrument.shape} (expected [B={B}, 1, H={H}, W={W}])")

    # Process remaining data
    remaining_indices = ~elevation_mask
    metadata_strings = [[[[] for _ in range(time_before)] for _ in range(num_channels)] for _ in range(B)]
    channel_time_data = {c: {t: [] for t in range(time_before)} for c in range(num_channels)}

    for b in range(B):
        channel_year_data = {}
        for i in range(total_steps):
            if not elevation_mask[b, i]:
                band_idx = int(metadata[b, i, 0])
                year = int(metadata[b, i, 1])
                channel = band_idx - 1
                if channel not in channel_year_data:
                    channel_year_data[channel] = []
                channel_year_data[channel].append((year, stacked_tensor[b, i]))

        for channel in channel_year_data:
            sorted_data = sorted(channel_year_data[channel], key=lambda x: x[0], reverse=True)
            for time_idx, (year, tensor_data) in enumerate(sorted_data):
                if time_idx < time_before:
                    metadata_strings[b][channel][time_idx] = f"{index_to_band[channel+1]}_{year}"
                    channel_time_data[channel][time_idx].append(tensor_data)
                    print(f"Channel {channel}, Time {time_idx}, Tensor shape: {tensor_data.shape} (expected [H={H}, W={W}])")

    remaining_tensor = torch.zeros((B, num_channels, time_before, H, W))
    for c in range(num_channels):
        for t in range(time_before):
            if channel_time_data[c][t]:
                data = torch.stack(channel_time_data[c][t])
                remaining_tensor[:len(data), c, t] = data
                print(f"Remaining tensor slice [:{len(data)}, {c}, {t}] shape: {data.shape} (expected [B_slice, H={H}, W={W}])")

    print(f"Final remaining_tensor shape: {remaining_tensor.shape} (expected [B={B}, num_channels={num_channels}, time_before={time_before}, H={H}, W={W}])")
    return elevation_instrument, remaining_tensor, metadata_strings

# Process batch to DataFrame with embedding size validation
def process_batch_to_dataframe(longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, model, device):
    records = []
    console = Console()

    for idx in range(longitude.shape[0]):
        base_record = {
            'longitude': longitude[idx].item(),
            'latitude': latitude[idx].item(),
            'organic_carbon': oc[idx].item()
        }

        # Elevation embedding
        elevation_input = elevation_instrument[idx:idx+1].unsqueeze(1)  # [1, 1, H, W]
        print(f"Elevation input shape before expansion (idx={idx}): {elevation_input.shape}")
        elevation_input = elevation_input.to(device).expand(1, 1, 3, 96, 96)  # [1, 1, 3, 96, 96]
        print(f"Elevation input shape after expansion: {elevation_input.shape}")
        
        elevation_embedding = model.patch_embed(elevation_input).squeeze().cpu().detach().numpy()
        elevation_emb_size = elevation_embedding.flatten().size
        print(f"Elevation embedding size (idx={idx}): {elevation_emb_size} (shape: {elevation_embedding.shape})")
        base_record['elevation_embedding'] = elevation_embedding.tolist()

        # Channel-time embeddings
        channel_time_embeddings = {}
        for channel, time in product(range(len(bands_list_order)-1), range(time_before)):
            x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            print(f"Channel {channel}, Time {time} input shape before expansion: {x.shape}")
            x = x.expand(1, 1, 3, 96, 96).to(device)  # [1, 1, 3, 96, 96]
            print(f"Channel {channel}, Time {time} input shape after expansion: {x.shape}")
            
            embedding = model.patch_embed(x).squeeze().cpu().detach().numpy()
            embedding_size = embedding.flatten().size
            print(f"Channel {channel}, Time {time} embedding size: {embedding_size} (shape: {embedding.shape})")
            tag = metadata_strings[idx][channel][time]
            channel_time_embeddings[tag] = embedding.tolist()

        base_record['channel_time_embeddings'] = channel_time_embeddings
        records.append(base_record)

    df = pd.DataFrame(records)
    print(f"Batch DataFrame created with {len(df)} rows")
    return df

# Save to Parquet with documentation
def save_to_parquet(train_loader, model, device, args, base_folder='/fast/vfourel/SOCProject'):
    run_number = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_folder, f'run_{run_number}')
    os.makedirs(folder_path, exist_ok=True)

    timing_info = {
        'loading_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'start_time': time.time(),
        'time_before': time.time(),
        'max_oc': float('-inf'),
        'image_size': (96, 96),
        'inference_time': 0
    }

    console = Console()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            longitude, latitude, stacked_tensor, metadata, oc = batch
            batch_size = len(longitude)
            print(f"Batch {batch_idx}: {batch_size} samples")

            elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
            batch_df = process_batch_to_dataframe(
                longitude, latitude, elevation_instrument, remaining_tensor,
                metadata_strings, oc, model, device
            )

            batch_filename = f'batch_{batch_idx:04d}_size_{batch_size}.parquet'
            batch_path = os.path.join(folder_path, batch_filename)
            batch_df.to_parquet(batch_path)
            print(f"Saved {batch_filename} with {len(batch_df)} rows")

            timing_info['max_oc'] = max(timing_info['max_oc'], torch.max(oc).item())
            timing_info['inference_time'] += time.time() - batch_start_time

    timing_info['end_time'] = time.time()
    doc_path = os.path.join(folder_path, f'run_{run_number}_documentation.txt')
    with open(doc_path, 'w') as f:
        f.write(f"Run {run_number} Documentation\n")
        f.write("=" * 50 + "\n\n")
        f.write("Timing Information:\n")
        f.write(f"LOADING_TIME_BEGINNING: {timing_info['loading_time']}\n")
        f.write(f"TIME_BEGINNING: {timing_info['start_time']}\n")
        f.write(f"TIME_END: {timing_info['end_time']}\n")
        f.write(f"INFERENCE_TIME: {timing_info['inference_time']}\n")
        f.write(f"MAX_OC: {timing_info['max_oc']}\n")
        f.write(f"time_before: {timing_info['time_before']}\n")
        f.write(f"imageSize: {timing_info['image_size']}\n\n")
        f.write("Arguments Configuration:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Run completed. Output saved to {folder_path}")
    return folder_path

# Main execution
if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console = Console()
    console.print(f"Using device: {device}")

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
    print(f"Flattened coordinates paths: {len(samples_coordinates_array_path)}")
    print(f"Flattened data paths: {len(data_array_path)}")

    train_df, val_df = create_balanced_dataset(df)
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize and load model
    vit_model = models_vit_tensor.__dict__[args.model](
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes
    ).to(device)
    vit_model = load_model(vit_model, args, device)

    # Process and save embeddings
    output_folder = save_to_parquet(train_loader, vit_model, device, args, base_folder=args.output_dir)
