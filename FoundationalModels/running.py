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
from dataloader.dataframe_loader import separate_and_add_data_1mil_inference, separate_and_add_data,filter_dataframe
from dataloader.dataloaderMapping import MultiRasterDataset1MilMultiYears
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from mapping import create_prediction_visualizations
from model_transformer import TransformerRegressor
from tqdm import tqdm
from accelerate import Accelerator
from IEEE_TPAMI_SpectralGPT.models_mae_spectral import MaskedAutoencoderViT
from IEEE_TPAMI_SpectralGPT import models_vit_tensor
from balance_dataset import create_balanced_dataset



def transform_data(stacked_tensor, metadata):
    B, total_steps, H, W = stacked_tensor.shape
    index_to_band = {k: v for k, v in bands_dict.items()}
    num_channels = len(bands_list_order) - 1  # Number of non-elevation bands
    expected_steps = 1 + num_channels * time_before

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
                channel = band_idx - 1
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
            elevation_emb = vit_model.module.patch_embed(elevation_input).squeeze().flatten()
            sample_embeddings.append(elevation_emb)
            for channel in range(num_channels):
                for time in range(num_times):
                    x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0).to(device).expand(1, 1, 3, 96, 96)
                    emb = vit_model.module.patch_embed(x).squeeze().flatten()
                    sample_embeddings.append(emb)
            sample_embeddings = torch.cat(sample_embeddings, dim=0)
            all_embeddings.append(sample_embeddings)
    all_embeddings = torch.stack(all_embeddings, dim=0)
    return all_embeddings

def load_models(
    vit_checkpoint_path="/home/vfourel/SOCProject/SOCmapping/FoundationalModels/models/SpectralGPT/SpectralGPT+.pth",
    transformer_path="/home/vfourel/SOCProject/SOCmapping/FoundationalModels/spectralGPT_TransformerRegressor_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_0_6025.pth
):
    accelerator = Accelerator()
    device = accelerator.device

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

    vit_model, transformer_model = accelerator.prepare(vit_model, transformer_model)
    transformer_model.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    transformer_model.eval()

    return vit_model, transformer_model, device, accelerator

def run_inference(vit_model, transformer_model, dataloader, accelerator, oc_mean, oc_std):
    vit_model.eval()
    transformer_model.eval()
    all_coordinates = []
    all_predictions = []

    process_embeddings = partial(process_batch_to_embeddings, vit_model=vit_model, device=accelerator.device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            longitude, latitude, stacked_tensor, metadata = batch  # Ignore normalized OC since it's not needed
            stacked_tensor = stacked_tensor.to(accelerator.device)
            metadata = metadata.to(accelerator.device)

            elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
            embeddings = process_embeddings(longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings)

            embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
            embeddings = embeddings.to(torch.float32)

            outputs = transformer_model(embeddings)
            # Denormalize predictions using the provided mean and std
            predictions_normalized = outputs.squeeze()  # Model outputs normalized predictions
            predictions = (predictions_normalized * oc_std) + oc_mean  # Reverse normalization: y = (x * std) + mean
            predictions = predictions.cpu().numpy()

            coords = np.stack([longitude.cpu().numpy(), latitude.cpu().numpy()], axis=1)
            all_coordinates.append(coords)
            all_predictions.append(predictions)

    all_coordinates = np.concatenate(all_coordinates, axis=0) if all_coordinates else np.array([])
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

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

def compute_oc_statistics():
    """Compute OC statistics from the training dataset"""
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    train_df, _ = create_balanced_dataset(df,False) # we use no validation set 
    train_coords, train_data = separate_and_add_data()
    train_coords = list(dict.fromkeys(flatten_paths(train_coords)))
    train_data = list(dict.fromkeys(flatten_paths(train_data)))

    train_dataset = MultiRasterDatasetMultiYears(
        samples_coordinates_array_subfolders=train_coords,
        data_array_subfolders=train_data,
        dataframe=train_df,
        time_before=time_before
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    all_targets = []
    print('get the OC: \n ')    
    for _, _, _, _, oc in train_loader:
        all_targets.append(oc)
    all_targets = torch.cat(all_targets)
    target_mean, target_std = all_targets.mean().item(), all_targets.std().item()
    return target_mean, target_std
def main():
    print('4th quarter')
    # oc_mean, oc_std = compute_oc_statistics()
    oc_mean = 22.204912500000002
    oc_std= 19.407142758253574
    print(f"OC statistics from training data: mean={oc_mean}, std={oc_std}")
    # OC statistics from training data: mean=23.3523966756125, range=22.460167800508092

    accelerator = Accelerator(device_placement=False)  # Disable auto device placement
    torch.cuda.set_device(0)  # Force GPU 0

    try:
        df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
        if accelerator.is_local_main_process:
            print("Loaded inference dataframe:")
            print(df_full.head())
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Error loading inference dataframe: {e}")
        return

    samples_coordinates_array_path_1mil, data_array_path_1mil = separate_and_add_data_1mil_inference()
    samples_coordinates_array_path_1mil = flatten_paths(samples_coordinates_array_path_1mil)
    data_array_path_1mil = flatten_paths(data_array_path_1mil)
    samples_coordinates_array_path_1mil = list(dict.fromkeys(samples_coordinates_array_path_1mil))
    data_array_path_1mil = list(dict.fromkeys(data_array_path_1mil))

    # Calculate the size of each quarter and select the 1st quarter
    quarter_size = len(df_full) // 4
    #print(len(samples_coordinates_array_path_1mil))
    #print(len(data_array_path_1mil))
    print(df_full)
    inference_dataset = MultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=samples_coordinates_array_path_1mil,
        data_array_subfolders=data_array_path_1mil,
        dataframe=df_full[3*quarter_size:],
        time_before=time_before
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    vit_model, transformer_model, device, accelerator = load_models()
    coordinates, predictions = run_inference(vit_model, transformer_model, inference_loader, accelerator, oc_mean, oc_std)

    save_path_coords = "coordinates_1mil_4thQuarter.npy"
    save_path_preds = "predictions_1mil_4thQuarter.npy"
    np.save(save_path_coords, coordinates)
    np.save(save_path_preds, predictions)

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