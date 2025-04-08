from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from timm.models.layers import trunc_normal_
from IEEE_TPAMI_SpectralGPT.util import video_vit
from IEEE_TPAMI_SpectralGPT.util.logging import master_print as print
from IEEE_TPAMI_SpectralGPT.util.pos_embed import interpolate_pos_embed
from IEEE_TPAMI_SpectralGPT.models_mae_spectral import MaskedAutoencoderViT
from IEEE_TPAMI_SpectralGPT import models_vit_tensor
from config import (
    LOADING_TIME_BEGINNING, TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, num_epochs,
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
from modelTFT import TemporalFusionTransformerRegressor
from balance_dataset import create_balanced_dataset

####################
embedding_dim = 110592
num_channels = 5
num_temporal_steps = 5
d_model = 128
output_dim = 1




def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Regression on Embeddings with Multi-GPU Support')
    parser.add_argument('--model', default='vit_base_patch8', type=str, metavar='MODEL',
                        help='Name of model to extract embeddings')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--nb_classes', default=62, type=int,
                        help='Number of classification types for ViT')
    parser.add_argument('--finetune', default='/home/vfourel/SOCProject/SOCmapping/FoundationalModels/models/SpectralGPT/SpectralGPT+.pth',
                        help='Path to finetune checkpoint')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for dataloader')
    parser.add_argument('--output_dir', default='/fast/vfourel/SOCProject', type=str,
                        help='Base folder for saving model checkpoints')
    parser.add_argument('--epochs', default=num_epochs, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for Transformer')
    parser.add_argument('--accum_steps', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--use_validation', default=True, type=bool, help='Full Training Or Not')
    return parser.parse_args()

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


def transform_data(stacked_tensor, metadata):
    B, total_steps, H, W = stacked_tensor.shape
    index_to_band = {k: v for k, v in bands_dict.items()}
    num_channels = (total_steps - 1) // time_before
    elevation_mask = (metadata[:, :, 0] == 0) & (metadata[:, :, 1] == 0)
    elevation_indices = elevation_mask.nonzero(as_tuple=True)
    elevation_instrument = stacked_tensor[elevation_indices[0], elevation_indices[1], :, :].reshape(B, 1, H, W)
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
    return elevation_instrument, remaining_tensor, metadata_strings

def process_batch_to_embeddings(longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, model, device):
    B = longitude.shape[0]  # Batch size, e.g., 64
    num_channels = remaining_tensor.shape[1]  # e.g., 5
    num_times = remaining_tensor.shape[2]  # e.g., 5
    targets = oc.to(device)

    # Lists to store embeddings
    elevation_embeddings = []
    remaining_embeddings = torch.zeros(B, num_channels, num_times, 110592).to(device)  # Pre-allocate for efficiency

    with torch.no_grad():
        for idx in range(B):
            # Process elevation_instrument
            elevation_input = elevation_instrument[idx:idx+1].unsqueeze(1).to(device).expand(1, 1, 3, 96, 96)
            elevation_emb = model.patch_embed(elevation_input).squeeze().flatten()  # Shape: [110592]
            elevation_embeddings.append(elevation_emb)

            # Process remaining_tensor
            for channel in range(num_channels):
                for time in range(num_times):
                    x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0).to(device).expand(1, 1, 3, 96, 96)
                    emb = model.patch_embed(x).squeeze().flatten()  # Shape: [110592]
                    remaining_embeddings[idx, channel, time] = emb

    # Stack elevation embeddings into [B, 1, 110592]
    embeddings_elevation_instrument = torch.stack(elevation_embeddings, dim=0).unsqueeze(1)  # [64, 1, 110592]

    # Remaining embeddings already in [B, num_channels, num_times, 110592], e.g., [64, 5, 5, 110592]
    embeddings_remaining_tensor = remaining_embeddings

    return embeddings_elevation_instrument, embeddings_remaining_tensor, targets

def train_transformer(train_loader, val_loader, vit_model, args, accelerator):
    num_patches = (96 // 8) * (96 // 8)
    embedding_dim_per_patch = 768
    total_embedding_dim = embedding_dim_per_patch * num_patches * (1 + (len(bands_list_order) - 1) * time_before)
    num_tokens = 1 + (len(bands_list_order) - 1) * time_before

    transformer_model = TemporalFusionTransformerRegressor(
        embedding_dim=embedding_dim,
        num_channels=num_channels,
        num_temporal_steps=num_temporal_steps,
        d_model=d_model,
        output_dim=output_dim
    )
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_loader, transformer_model, optimizer = accelerator.prepare(train_loader, transformer_model, optimizer)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    all_targets = []
    for _, _, _, _, oc in train_loader:
        all_targets.append(oc)
    all_targets = torch.cat(all_targets)
    target_mean, target_std = all_targets.mean().item(), all_targets.std().item()
    print("  target_mean:      ",target_mean,"  target_std:   ",target_std)

    best_r_squared = 0.0
    # 
    for epoch in range(args.epochs):
        transformer_model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            longitude, latitude, stacked_tensor, metadata, oc = batch
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
            #print(' elevation_instrument    ', elevation_instrument)
            embeddings_elevation_instrument, embeddings_remaining_tensor, targets = process_batch_to_embeddings(
                longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
            )
            #print('oc.shape   ',oc.shape )
            #print('stacked_tensor.shape   ',stacked_tensor.shape)
            #print('elevation_instrument.shape   ',elevation_instrument.shape)
            #print('remaining_tensor.shape   ',remaining_tensor.shape)
            #print('embeddings_elevation_instrument.shape   ',embeddings_elevation_instrument.shape )
            #print('embeddings_remaining_tensor.shape   ',embeddings_remaining_tensor.shape )

            targets = (targets - target_mean) / (target_std + 1e-8)
            embeddings_elevation_instrument, embeddings_remaining_tensor,  targets = embeddings_elevation_instrument.to(torch.float32), embeddings_remaining_tensor.to(torch.float32), targets.to(torch.float32)

            outputs = transformer_model(embeddings_elevation_instrument, embeddings_remaining_tensor)
            loss = criterion(outputs.squeeze(), targets)
            accelerator.backward(loss / args.accum_steps)
            total_train_loss += loss.item() * args.accum_steps

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                wandb.log({
                    'train_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        r_squared = 0.0

        if val_loader is not None:
            transformer_model.eval()
            total_val_loss = 0
            val_outputs = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    longitude, latitude, stacked_tensor, metadata, oc = batch
                    elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
                    embeddings_elevation_instrument, embeddings_remaining_tensor, targets = process_batch_to_embeddings(
                        longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
                    )
                  #  embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
                   
                    targets = (targets - target_mean) / (target_std + 1e-8)
                    embeddings_elevation_instrument, embeddings_remaining_tensor,  targets = embeddings_elevation_instrument.to(torch.float32), embeddings_remaining_tensor.to(torch.float32), targets.to(torch.float32)
                    outputs =transformer_model(embeddings_elevation_instrument, embeddings_remaining_tensor)
                    loss = criterion(outputs.squeeze(), targets)
                    total_val_loss += loss.item()
                    val_outputs.extend(outputs.squeeze().cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())

            if accelerator.is_main_process:
                avg_val_loss = total_val_loss / len(val_loader.dataset)
                val_outputs = np.array(val_outputs)
                val_targets_list = np.array(val_targets_list)
                correlation = np.corrcoef(val_outputs, val_targets_list)[0, 1]
                r_squared = correlation ** 2
                mse = np.mean((val_outputs - val_targets_list) ** 2)
                rmse = np.sqrt(mse)

                wandb.log({
                    'epoch': epoch + 1,
                    'avg_train_loss': avg_train_loss,
                    'avg_val_loss': avg_val_loss,
                    'correlation': correlation,
                    'r_squared': r_squared,
                    'mse': mse,
                    'rmse': rmse,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

                if avg_val_loss < wandb.run.summary.get('best_val_loss', float('inf')):
                    wandb.run.summary['best_val_loss'] = avg_val_loss
                    best_r_squared = r_squared

                scheduler.step(avg_val_loss)
        else:
            # For no validation case, calculate R-squared on training data
            transformer_model.eval()
            train_outputs = []
            train_targets_list = []
            with torch.no_grad():
                for batch in train_loader:
                    longitude, latitude, stacked_tensor, metadata, oc = batch
                    elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
                    embeddings_elevation_instrument, embeddings_remaining_tensor, targets = process_batch_to_embeddings(
                       longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
                    )
                   # embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
                    targets = (targets - target_mean) / (target_std + 1e-8)
                    embeddings_elevation_instrument, embeddings_remaining_tensor,  targets = embeddings_elevation_instrument.to(torch.float32), embeddings_remaining_tensor.to(torch.float32), targets.to(torch.float32)
                    outputs = transformer_model(embeddings_elevation_instrument, embeddings_remaining_tensor)
                    train_outputs.extend(outputs.squeeze().cpu().numpy())
                    train_targets_list.extend(targets.cpu().numpy())

            train_outputs = np.array(train_outputs)
            train_targets_list = np.array(train_targets_list)
            correlation = np.corrcoef(train_outputs, train_targets_list)[0, 1]
            r_squared = correlation ** 2
            best_r_squared = max(best_r_squared, r_squared)

            wandb.log({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'r_squared': r_squared,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        if accelerator.is_main_process:
            checkpoint_path = os.path.join(args.output_dir, f'TemporalFusionTransformerRegressor_epoch_{epoch+1}.pth')
            accelerator.save_state(checkpoint_path)
            wandb.save(checkpoint_path)

    return transformer_model, best_r_squared

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    console = Console()
    console.print(f"Using device: {accelerator.device}")

    wandb.init(
        project="socmapping-spectralGPT-TemporalFusionTransformerRegressor",
        config={
            "model": args.model,
            "drop_path": args.drop_path,
            "nb_classes": args.nb_classes,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "accum_steps": args.accum_steps,
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "time_before": time_before,
            "bands": len(bands_list_order),
            "use_validation": args.use_validation
        }
    )

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

    train_df, val_df = create_balanced_dataset(df, use_validation=args.use_validation)
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.use_validation and val_df is not None:
        val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print( ' args.use_validation     ',args.use_validation)
    else:
        val_loader = None

    wandb.run.summary["train_size"] = len(train_df)
    if args.use_validation and val_df is not None:
        wandb.run.summary["val_size"] = len(val_df)

    vit_model = models_vit_tensor.__dict__[args.model](
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes
    ).to(accelerator.device)
    vit_model = load_model(vit_model, args, accelerator.device)

    n_parameters = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    wandb.run.summary["vit_parameters"] = n_parameters

    transformer_model, final_r_squared = train_transformer(train_loader, val_loader, vit_model, args, accelerator)

    # Format R-squared for filename
    r_squared_str = f"{final_r_squared:.4f}".replace(".", "_")
    validation_str = "FullData" if not args.use_validation else f"R2_{r_squared_str}"
    
    # Define the save path with R-squared and validation info
    mlp_path = f'spectralGPT_TemporalFusionTransformerRegressor_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}_{validation_str}.pth'

    if accelerator.is_main_process:
        accelerator.save(transformer_model.state_dict(), mlp_path)
        console.print(f"Model saved to {mlp_path}")
        wandb.run.summary["final_r_squared"] = final_r_squared

    wandb.finish()