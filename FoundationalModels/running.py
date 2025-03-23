from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC,
    MAX_OC, bands_dict
)
from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
from dataloader.dataloaderMapping import MultiRasterDataset1MilMultiYears
from mapping import create_prediction_visualizations
from model_transformer import TransformerRegressor
from tqdm import tqdm
from accelerate import Accelerator
from IEEE_TPAMI_SpectralGPT.models_mae_spectral import MaskedAutoencoderViT
from IEEE_TPAMI_SpectralGPT import models_vit_tensor

def transform_data(stacked_tensor, metadata):
    B, total_steps, H, W = stacked_tensor.shape
    index_to_band = {k: v for k, v in bands_dict.items()}
    num_channels = len(bands_list_order) - 1  # Number of non-elevation bands
    expected_steps = 1 + num_channels * time_before

    # Handle mismatched steps
    if total_steps != expected_steps:
        if total_steps < expected_steps:
            print(f"Warning: total_steps ({total_steps}) < expected_steps ({expected_steps}). Padding with zeros.")
            padding = torch.zeros((B, expected_steps - total_steps, H, W), device=stacked_tensor.device)
            stacked_tensor = torch.cat([stacked_tensor, padding], dim=1)
            metadata_padding = torch.zeros((B, expected_steps - total_steps, 2), device=metadata.device)
            metadata = torch.cat([metadata, metadata_padding], dim=1)
        elif total_steps > expected_steps:
            print(f"Warning: total_steps ({total_steps}) > expected_steps ({expected_steps}). Truncating.")
            stacked_tensor = stacked_tensor[:, :expected_steps]
            metadata = metadata[:, :expected_steps]
        total_steps = expected_steps

    elevation_mask = (metadata[:, :, 0] == 0) & (metadata[:, :, 1] == 0)
    elevation_indices = elevation_mask.nonzero(as_tuple=True)
    elevation_instrument = stacked_tensor[elevation_indices[0], elevation_indices[1], :, :].reshape(B, 1, H, W)

    metadata_strings = [[[[] for _ in range(time_before)] for _ in range(num_channels)] for _ in range(B)]
    channel_time_data = {c: {t: [] for t in range(time_before)} for c in range(num_channels)}

    for b in range(B):
        channel_year_data = {}
        for i in range(total_steps):
            if not elevation_mask[b, i]:
                band_idx = int(metadata[b, i, 0])
                year = int(metadata[b, i, 1])
                channel = band_idx - 1  # Adjust for 1-based indexing
                if channel not in channel_year_data:
                    channel_year_data[channel] = []
                channel_year_data[channel].append((year, stacked_tensor[b, i]))

        for channel in channel_year_data:
            sorted_data = sorted(channel_year_data[channel], key=lambda x: x[0], reverse=True)
            for time_idx, (year, tensor_data) in enumerate(sorted_data[:time_before]):
                metadata_strings[b][channel][time_idx] = f"{index_to_band[channel+1]}_{year}"
                channel_time_data[channel][time_idx].append(tensor_data)

    remaining_tensor = torch.zeros((B, num_channels, time_before, H, W))
    for c in range(num_channels):
        for t in range(time_before):
            if channel_time_data[c][t]:
                data = torch.stack(channel_time_data[c][t])
                remaining_tensor[:len(data), c, t] = data

    return elevation_instrument, remaining_tensor, metadata_strings

def process_batch_to_embeddings(longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, vit_model, device):
    B = longitude.shape[0]
    num_channels = remaining_tensor.shape[1]
    num_times = remaining_tensor.shape[2]
    all_embeddings = []
    with torch.no_grad():
        for idx in range(B):
            sample_embeddings = []
            elevation_input = elevation_instrument[idx:idx+1].unsqueeze(1).to(device).expand(1, 1, 3, 96, 96)
            # Fix: Access patch_embed via .module for DDP compatibility
            elevation_emb = vit_model.module.patch_embed(elevation_input).squeeze().flatten()
            sample_embeddings.append(elevation_emb)
            for channel in range(num_channels):
                for time in range(num_times):
                    x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0).to(device).expand(1, 1, 3, 96, 96)
                    # Fix: Access patch_embed via .module for DDP compatibility
                    emb = vit_model.module.patch_embed(x).squeeze().flatten()
                    sample_embeddings.append(emb)
            sample_embeddings = torch.cat(sample_embeddings, dim=0)
            all_embeddings.append(sample_embeddings)
    all_embeddings = torch.stack(all_embeddings, dim=0)
    return all_embeddings

def load_models(
    vit_checkpoint_path="/home/vfourel/SOCProject/SOCmapping/FoundationalModels/models/SpectralGPT/SpectralGPT+.pth",
    transformer_path="/home/vfourel/SOCProject/SOCmapping/FoundationalModels/spectralGPT_TransformerRegressor_MAX_OC_180_TIME_BEGINNING_2007_TIME_END_2023.pth"
):
    accelerator = Accelerator()
    device = accelerator.device

    # Load ViT model
    vit_model = models_vit_tensor.vit_base_patch8(
        drop_path_rate=0.1,
        num_classes=62
    )
    if vit_checkpoint_path:
        checkpoint = torch.load(vit_checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        skip_keys = [
            'patch_embed.0.proj.weight', 'patch_embed.1.proj.weight',
            'patch_embed.2.proj.weight', 'patch_embed.2.proj.bias',
            'head.weight', 'head.bias'
        ]
        for k in skip_keys:
            if k in checkpoint_model and checkpoint_model[k].shape != vit_model.state_dict()[k].shape:
                del checkpoint_model[k]
        msg = vit_model.load_state_dict(checkpoint_model, strict=False)
        if accelerator.is_local_main_process:
            print(f"Loaded ViT checkpoint: {msg}")
    vit_model.eval()

    # Load Transformer model
    num_patches = (96 // 8) * (96 // 8)
    embedding_dim_per_patch = 768
    total_embedding_dim = embedding_dim_per_patch * num_patches * (1 + (len(bands_list_order) - 1) * time_before)
    num_tokens = 1 + (len(bands_list_order) - 1) * time_before

    transformer_model = TransformerRegressor(
        input_dim=total_embedding_dim,
        num_tokens=num_tokens,
        d_model=512,
        nhead=8,
        num_layers=2,
        dim_feedforward=1024,
        output_dim=1
    )

    # Prepare models for distributed inference *before* loading state dict
    vit_model, transformer_model = accelerator.prepare(vit_model, transformer_model)

    # Load the state dictionary *after* preparing
    transformer_model.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    transformer_model.eval()

    return vit_model, transformer_model, device, accelerator

def run_inference(vit_model, transformer_model, dataloader, accelerator, target_mean, target_std):
    vit_model.eval()
    transformer_model.eval()
    all_coordinates = []
    all_predictions = []

    process_embeddings = partial(process_batch_to_embeddings, vit_model=vit_model, device=accelerator.device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitude, latitude, stacked_tensor, metadata = batch
            stacked_tensor = stacked_tensor.to(accelerator.device)
            metadata = metadata.to(accelerator.device)

            # Transform data into elevation and remaining tensors
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)

            # Process into embeddings using ViT
            embeddings = process_embeddings(longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings)

            # Normalize embeddings (as done in training)
            embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
            embeddings = embeddings.to(torch.float32)

            # Run Transformer inference
            outputs = transformer_model(embeddings)
            # Denormalize predictions (assuming training normalized targets)
            predictions = (outputs.squeeze() * target_std + target_mean).cpu().numpy()

            # Store coordinates and predictions
            coords = np.stack([longitude.cpu().numpy(), latitude.cpu().numpy()], axis=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    # Concatenate local results
    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

    # Gather results from all GPUs
    all_coordinates = accelerator.gather(torch.tensor(all_coordinates, device=accelerator.device)).cpu().numpy()
    all_predictions = accelerator.gather(torch.tensor(all_predictions, device=accelerator.device)).cpu().numpy()

    return all_coordinates, all_predictions

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def main():
    accelerator = Accelerator()

    # Load inference dataframe
    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print("Loaded inference dataframe:")
            print(df_full.head())
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    # Prepare data paths
    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()
    samples_coordinates_array_path_1mil = flatten_paths(samples_coordinates_array_path_1mil)
    data_array_path_1mil = flatten_paths(data_array_path_1mil)
    samples_coordinates_array_path_1mil = list(dict.fromkeys(samples_coordinates_array_path_1mil))
    data_array_path_1mil = list(dict.fromkeys(data_array_path_1mil))

    # Initialize dataset
    inference_dataset = MultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=df_full,
        time_before=time_before
    )

    # Create DataLoader
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    inference_loader = accelerator.prepare(inference_loader)

    # Load models
    vit_model, transformer_model, device, accelerator = load_models()

    # Hardcoded target mean and std (should ideally be saved from training)
    target_mean = 0.0  # Placeholder; replace with actual value from training
    target_std = 1.0   # Placeholder; replace with actual value from training
    if accelerator.is_local_main_process:
        print(f"Using target_mean={target_mean}, target_std={target_std} for denormalization")

    # Run inference
    coordinates, predictions = run_inference(vit_model, transformer_model, inference_loader, accelerator, target_mean, target_std)

    # Only the main process handles visualization
    if accelerator.is_local_main_process:
        print(f"Inference completed. Coordinates shape: {coordinates.shape}, Predictions shape: {predictions.shape}")
        create_prediction_visualizations(
            INFERENCE_TIME,
            coordinates,
            predictions,
            save_path_predictions_plots
        )
        print(f"Visualizations saved to {save_path_predictions_plots}")

if __name__ == "__main__":
    main()