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
from model_transformer import TransformerRegressor
from balance_dataset import create_balanced_dataset


# Define the composite loss function
def composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L1 and scaled chi-squared loss"""
    errors = targets - outputs
    l1_loss = torch.mean(torch.abs(errors))

    squared_errors = errors ** 2
    chi2_unscaled = (1/4) * squared_errors * torch.exp(-squared_errors / (2 * sigma))
    chi2_unscaled_mean = torch.mean(chi2_unscaled)

    chi2_unscaled_mean = torch.clamp(chi2_unscaled_mean, min=1e-8)
    scale_factor = l1_loss / chi2_unscaled_mean
    chi2_scaled = scale_factor * chi2_unscaled_mean

    return alpha * l1_loss + (1 - alpha) * chi2_scaled


def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L2 and scaled chi-squared loss"""
    errors = targets - outputs
    
    # L2 loss: mean squared error
    l2_loss = torch.mean(errors ** 2)
    
    # Standard chi-squared loss: errors^2 / sigma^2
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))
    
    # Ensure chi2_loss is not too small to avoid division issues
    chi2_loss = torch.clamp(chi2_loss, min=1e-8)
    
    # Scale chi2_loss to match the magnitude of l2_loss
    scale_factor = l2_loss / chi2_loss
    chi2_scaled = scale_factor * chi2_loss
    
    # Combine the losses with the weighting factor alpha
    return alpha * l2_loss + (1 - alpha) * chi2_scaled

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
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for dataloader')
    parser.add_argument('--output_dir', default='/fast/vfourel/SOCProject', type=str,
                        help='Base folder for saving model checkpoints')
    parser.add_argument('--epochs', default=num_epochs, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for Transformer')
    parser.add_argument('--accum_steps', default=8, type=int, help='Gradient accumulation steps')
    # New arguments for target transformation and loss selection
    parser.add_argument('--loss_type', type=str, default='composite_l2', choices=['composite_l1', 'l1', 'mse','composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='normalize', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
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
    B = longitude.shape[0]
    num_channels = remaining_tensor.shape[1]
    num_times = remaining_tensor.shape[2]
    all_embeddings = []
    targets = oc.to(device)
    with torch.no_grad():
        for idx in range(B):
            sample_embeddings = []
            elevation_input = elevation_instrument[idx:idx+1].unsqueeze(1).to(device).expand(1, 1, 3, 96, 96)
            elevation_emb = model.patch_embed(elevation_input).squeeze().flatten()
            sample_embeddings.append(elevation_emb)
            for channel in range(num_channels):
                for time in range(num_times):
                    x = remaining_tensor[idx, channel, time].unsqueeze(0).unsqueeze(0).to(device).expand(1, 1, 3, 96, 96)
                    emb = model.patch_embed(x).squeeze().flatten()
                    sample_embeddings.append(emb)
            sample_embeddings = torch.cat(sample_embeddings, dim=0)
            all_embeddings.append(sample_embeddings)
    all_embeddings = torch.stack(all_embeddings, dim=0)
    return all_embeddings, targets


def train_transformer(train_loader, val_loader, vit_model, args, accelerator):


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

    optimizer = optim.Adam(transformer_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_loader, transformer_model, optimizer = accelerator.prepare(train_loader, transformer_model, optimizer)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # Compute target statistics locally (could be made global if needed)
    all_targets = []
    for _, _, _, _, oc in train_loader:
        all_targets.append(oc)
    all_targets = torch.cat(all_targets)

    if args.target_transform == 'normalize':
        target_mean, target_std = all_targets.mean().item(), all_targets.std().item()
    else:
        target_mean, target_std = 0.0, 1.0
    print("  target_mean:      ", target_mean, "  target_std:   ", target_std)
    if accelerator.is_main_process:
        wandb.run.summary["target_mean"] = target_mean
        wandb.run.summary["target_std"] = target_std

    loss_type = args.loss_type
    if loss_type == 'composite_l1':
        criterion = lambda outputs, targets: composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=args.loss_alpha)
    elif loss_type == 'composite_l2':
        criterion = lambda outputs, targets: composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=args.loss_alpha)
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    best_r_squared = 0.0

    for epoch in range(args.epochs):
        transformer_model.train()
        total_train_loss = 0.0
        total_num_samples_local = 0
        optimizer.zero_grad()

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            longitude, latitude, stacked_tensor, metadata, oc = batch
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
            embeddings, targets = process_batch_to_embeddings(
                longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
            )

            embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
            if args.target_transform == 'log':
                targets = torch.log(targets + 1e-10)
            elif args.target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)

            normalized_targets = targets
            embeddings, normalized_targets = embeddings.to(torch.float32), normalized_targets.to(torch.float32)

            outputs = transformer_model(embeddings)
            loss = criterion(outputs.squeeze(), normalized_targets)
            batch_size = normalized_targets.size(0)
            accelerator.backward(loss / args.accum_steps)
            total_train_loss += loss.item() * batch_size  # Scale by batch size
            total_num_samples_local += batch_size

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                wandb.log({
                    'train_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })

        # Compute global average training loss
        total_train_loss_tensor = torch.tensor(total_train_loss).to(accelerator.device)
        total_num_samples_tensor = torch.tensor(total_num_samples_local).to(accelerator.device)
        total_train_loss_all = accelerator.gather(total_train_loss_tensor)
        total_num_samples_all = accelerator.gather(total_num_samples_tensor)
        if accelerator.is_main_process:
            global_total_train_loss = total_train_loss_all.sum().item()
            global_num_samples = total_num_samples_all.sum().item()
            avg_train_loss = global_total_train_loss / global_num_samples
        else:
            avg_train_loss = 0.0

        r_squared = 0.0

        # Validation loop
        if val_loader is not None:
            transformer_model.eval()
            val_outputs_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    longitude, latitude, stacked_tensor, metadata, oc = batch
                    elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
                    embeddings, targets = process_batch_to_embeddings(
                        longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
                    )
                    embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
                    if args.target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif args.target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)

                    normalized_targets = targets
                    embeddings, normalized_targets = embeddings.to(torch.float32), normalized_targets.to(torch.float32)
                    outputs = transformer_model(embeddings)
                    val_outputs_list.append(outputs.squeeze())
                    val_targets_list.append(normalized_targets)

            # Gather validation data across processes
            val_outputs_tensor = torch.cat(val_outputs_list, dim=0)
            val_targets_tensor = torch.cat(val_targets_list, dim=0)
            val_outputs_all = accelerator.gather_for_metrics(val_outputs_tensor)
            val_targets_all = accelerator.gather_for_metrics(val_targets_tensor)

            if accelerator.is_main_process:
                # Compute validation loss on transformed scale
                val_outputs_all_device = val_outputs_all.to(accelerator.device)
                val_targets_all_device = val_targets_all.to(accelerator.device)
                val_loss = criterion(val_outputs_all_device, val_targets_all_device).item()

                # Convert to numpy for metric computation
                val_outputs_np = val_outputs_all.cpu().numpy()
                val_targets_np = val_targets_all.cpu().numpy()

                # Inverse transform to original scale
                if args.target_transform == 'log':
                    original_val_outputs = np.exp(val_outputs_np)
                    original_val_targets = np.exp(val_targets_np)
                elif args.target_transform == 'normalize':
                    original_val_outputs = val_outputs_np * target_std + target_mean
                    original_val_targets = val_targets_np * target_std + target_mean
                else:
                    original_val_outputs = val_outputs_np
                    original_val_targets = val_targets_np

                # Compute metrics on original scale
                if len(original_val_outputs) > 1:
                    correlation = np.corrcoef(original_val_outputs, original_val_targets)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                    r_squared = correlation ** 2
                else:
                    correlation = 0.0
                    r_squared = 0.0
                mse = np.mean((original_val_outputs - original_val_targets) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(original_val_outputs - original_val_targets))
                q75, q25 = np.percentile(original_val_targets, [75, 25])
                iqr = q75 - q25
                rpiq = iqr / rmse if rmse > 0 else float('inf')

                # Log metrics to wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'avg_train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'correlation': correlation,
                    'r_squared': r_squared,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'rpiq': rpiq,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

                # Update best_r_squared and step scheduler
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                if val_loss < wandb.run.summary.get('best_val_loss', float('inf')):
                    wandb.run.summary['best_val_loss'] = val_loss
                scheduler.step(val_loss)
        else:
            # Training metrics when no validation loader (unchanged for this task)
            transformer_model.eval()
            train_outputs = []
            train_targets_list = []
            with torch.no_grad():
                for batch in train_loader:
                    longitude, latitude, stacked_tensor, metadata, oc = batch
                    elevation_instrument, remaining_tensor, metadata_strings = transform_data(stacked_tensor, metadata)
                    embeddings, targets = process_batch_to_embeddings(
                        longitude, latitude, elevation_instrument, remaining_tensor, metadata_strings, oc, vit_model, accelerator.device
                    )
                    embeddings = (embeddings - embeddings.mean(dim=1, keepdim=True)) / (embeddings.std(dim=1, keepdim=True) + 1e-8)
                    if args.target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif args.target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)
                    normalized_targets = targets
                    embeddings, normalized_targets = embeddings.to(torch.float32), normalized_targets.to(torch.float32)
                    outputs = transformer_model(embeddings)
                    train_outputs.extend(outputs.squeeze().cpu().numpy())
                    train_targets_list.extend(normalized_targets.cpu().numpy())

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
            checkpoint_path = os.path.join(args.output_dir, f'transformer_epoch_{epoch+1}.pth')
            accelerator.save_state(checkpoint_path)
            wandb.save(checkpoint_path)

    return transformer_model, best_r_squared

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    console = Console()
    console.print(f"Using device: {accelerator.device}")

    # Initialize Weights & Biases with an extended config
    wandb.init(
        project="socmapping-spectralGPT-TransformerRegressor",
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
            "use_validation": args.use_validation,
            "target_transform":args.target_transform,
            "loss_type":args.loss_type

        }
    )

    # Load and preprocess data
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
    else:
        val_loader = None

    # Log dataset sizes
    wandb.run.summary["train_size"] = len(train_df)
    if args.use_validation and val_df is not None:
        wandb.run.summary["val_size"] = len(val_df)

    # Initialize and load the ViT model
    vit_model = models_vit_tensor.__dict__[args.model](
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes
    ).to(accelerator.device)
    vit_model = load_model(vit_model, args, accelerator.device)

    n_parameters = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    wandb.run.summary["vit_parameters"] = n_parameters

    # Train the transformer model
    transformer_model, final_r_squared = train_transformer(train_loader, val_loader, vit_model, args, accelerator)

    # Construct a more informative save path
    r_squared_str = f"{final_r_squared:.4f}".replace(".", "_")
    transform_str = "log" if getattr(args, "apply_log", False) else "raw"
    loss_str = getattr(args, "loss", "mse")
    
    validation_str = f"{transform_str}_{loss_str}"
    if args.use_validation:
        validation_str += f"_R2_{r_squared_str}"
    else:
        validation_str += "_FullData"
    
    mlp_path = (
        f"spectralGPT_TransformerRegressor_MAX_OC_{MAX_OC}_"
        f"TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}_"
        f"{validation_str}.pth"
    )

    # Save the model and log final metrics
    if accelerator.is_main_process:
        accelerator.save(transformer_model.state_dict(), mlp_path)
        console.print(f"Model saved to {mlp_path}")
        wandb.run.summary["final_r_squared"] = final_r_squared

    wandb.finish()



# normalize / l2