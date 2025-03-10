from functools import partial
import numpy as np
import torch
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
    window_size, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
import argparse
import os
from datetime import datetime
from rich.console import Console
from itertools import product
from accelerate import Accelerator
from model_mlp import MLPRegressor

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='MLP Regression on Embeddings with Multi-GPU Support')
    parser.add_argument('--model', default='vit_base_patch8', type=str, metavar='MODEL',
                        help='Name of model to extract embeddings (e.g., vit_base_patch8)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--nb_classes', default=62, type=int,
                        help='Number of classification types for ViT')
    parser.add_argument('--finetune', default='/home/vfourel/SOCProject/SOCmapping/FoundationalModels/models/SpectralGPT/SpectralGPT+.pth',
                        help='Path to finetune checkpoint')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for dataloader')
    parser.add_argument('--output_dir', default='/fast/vfourel/SOCProject', type=str,
                        help='Base folder for saving model checkpoints')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate for MLP')
    parser.add_argument('--accum_steps', default=8, type=int, help='Gradient accumulation steps')
    args = parser.parse_args()
    return args

# Load and initialize the ViT model
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
    # print(f"Input stacked_tensor shape: [B={B}, total_steps={total_steps}, H={H}, W={W}]")
    
    index_to_band = {k: v for k, v in bands_dict.items()}
    num_channels = (total_steps - 1) // time_before
    # print(f"Calculated num_channels: {num_channels} (total_steps-1={total_steps-1} / time_before={time_before})")

    elevation_mask = (metadata[:, :, 0] == 0) & (metadata[:, :, 1] == 0)
    elevation_indices = elevation_mask.nonzero(as_tuple=True)
    elevation_instrument = stacked_tensor[elevation_indices[0], elevation_indices[1], :, :].reshape(B, 1, H, W)
    # print(f"Elevation instrument shape: {elevation_instrument.shape} (expected [B={B}, 1, H={H}, W={W}])")

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

    remaining_tensor = torch.zeros((B, num_channels, time_before, H, W))
    for c in range(num_channels):
        for t in range(time_before):
            if channel_time_data[c][t]:
                data = torch.stack(channel_time_data[c][t])
                remaining_tensor[:len(data), c, t] = data

    # print(f"Final remaining_tensor shape: {remaining_tensor.shape} (expected [B={B}, num_channels={num_channels}, time_before={time_before}, H={H}, W={W}])")
    return elevation_instrument, remaining_tensor, metadata_strings

def process_batch_to_embeddings(longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, model, device):
    B = longitude.shape[0]
    num_channels = remaining_tensor.shape[1]
    num_times = remaining_tensor.shape[2]
    
    all_embeddings = []
    targets = oc.to(device)

    with torch.no_grad():
        for idx in range(B):
            sample_embeddings = []

            # Elevation embedding
            elevation_input = elevation_instrument[idx:idx+1].unsqueeze(1)  # [1, 1, H, W]
            elevation_input = elevation_input.to(device).expand(1, 1, 3, 96, 96)  # [1, 1, 3, 96, 96]
            elevation_emb = model.patch_embed(elevation_input).squeeze()  # [num_patches, embedding_dim]
            elevation_emb = elevation_emb.flatten()  # Flatten to [num_patches * embedding_dim]
            sample_embeddings.append(elevation_emb)

            # Channel-time embeddings
            for channel in range(num_channels):
                for time in range(num_times):
                    x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    x = x.to(device).expand(1, 1, 3, 96, 96)  # [1, 1, 3, 96, 96]
                    emb = model.patch_embed(x).squeeze()  # [num_patches, embedding_dim]
                    emb = emb.flatten()  # Flatten to [num_patches * embedding_dim]
                    sample_embeddings.append(emb)

            # Concatenate all embeddings for this sample
            sample_embeddings = torch.cat(sample_embeddings, dim=0)  # [total_embedding_size]
            all_embeddings.append(sample_embeddings)

    # Stack all samples into a single tensor
    all_embeddings = torch.stack(all_embeddings, dim=0)  # [B, total_embedding_size]
    # print(f"All embeddings shape: {all_embeddings.shape}, Targets shape: {targets.shape}")
    return all_embeddings, targets

def train_mlp(train_loader, vit_model, args, accelerator):
    # Calculate embedding dimension
    num_patches = (96 // 8) * (96 // 8)  # 144 patches for patch_size=8
    embedding_dim_per_patch = 768  # Correct ViT base embedding dim per patch (not 110592)
    total_embedding_dim = embedding_dim_per_patch * num_patches * (1 + (len(bands_list_order) - 1) * time_before)  # Elevation + channel-time
    mlp_model = MLPRegressor(input_dim=total_embedding_dim)

    optimizer = optim.Adam(mlp_model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # Prepare with Accelerate for multi-GPU
    train_loader, mlp_model, optimizer = accelerator.prepare(train_loader, mlp_model, optimizer)

    for epoch in range(args.epochs):
        mlp_model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            longitude, latitude, stacked_tensor, metadata, oc = batch

            # Extract embeddings using ViT
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
            embeddings, targets = process_batch_to_embeddings(
                longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
            )

            # Ensure tensors are in float32
            embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
            embeddings = embeddings.to(dtype=torch.float32)
            targets = targets.to(dtype=torch.float32)

            # Forward pass
            outputs = mlp_model(embeddings)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass with gradient accumulation
            accelerator.backward(loss / args.accum_steps)
            total_loss += loss.item() * args.accum_steps

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            avg_loss = total_loss / len(train_loader.dataset)
            # print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if accelerator.is_main_process:
            checkpoint_path = os.path.join(args.output_dir, f'mlp_epoch_{epoch+1}.pth')
            accelerator.save_state(checkpoint_path)

    return mlp_model

# Main execution
if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    console = Console()
    console.print(f"Using device: {accelerator.device}")

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

    # Initialize and load ViT model
    vit_model = models_vit_tensor.__dict__[args.model](
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes
    ).to(accelerator.device)
    vit_model = load_model(vit_model, args, accelerator.device)

    # Train MLP on embeddings
    mlp_model = train_mlp(train_loader, vit_model, args, accelerator)