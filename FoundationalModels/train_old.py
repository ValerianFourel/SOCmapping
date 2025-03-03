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

class MLP(nn.Module):
    def __init__(self, input_dim=768*144):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(input_dim, 2048)
        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, 1)

        # Use LayerNorm instead
        self.ln1 = nn.LayerNorm(2048)
        self.ln2 = nn.LayerNorm(512)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        return x


def validate(vit_model, mlp_model, val_loader, criterion, device, progress):
    mlp_model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    task_id = progress.add_task("[green]Validating...", total=len(val_loader))

    with torch.no_grad():
        for batch in val_loader:
            longitude, latitude, stacked_tensor, metadata, oc = batch

            # Process the batch data
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(
                stacked_tensor, metadata)

            # Get embeddings
            embeddings, targets = process_batch(
                longitude, latitude, elevation_instrument, remaining_tensor,
                metadata_strings, oc, vit_model, device
            )

            # Process through MLP
            outputs = mlp_model(embeddings)
            targets = targets.to(device).float().unsqueeze(1)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            progress.update(task_id, advance=1)

    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return total_loss / len(val_loader), r2, rmse


def process_batch(longitude, latitude, elevation_instrument, remaining_tensor, 
                 metadata_strings, oc, vit_model, device):
    batch_size = longitude.shape[0]
    all_embeddings = []

    with torch.no_grad():
        # Process elevation embeddings
        elevation_input = elevation_instrument.unsqueeze(1)  # [B, 1, 96, 96]
        elevation_input = elevation_input.to(device).expand(batch_size, 1, 3, 96, 96)
        elevation_embeddings = vit_model.patch_embed(elevation_input)  # [B, 144, 768]

        # Process channel/time embeddings
        channel_time_embeddings = []
        for channel in range(5):
            for time in range(5):
                x = remaining_tensor[:, channel, time].unsqueeze(1).unsqueeze(1)  # [B, 1, 96, 96]
                x = x.expand(batch_size, 1, 3, 96, 96)
                x = x.to(device)
                embedding = vit_model.patch_embed(x)  # [B, 144, 768]
                channel_time_embeddings.append(embedding)

        # Stack all embeddings
        channel_time_embeddings = torch.stack(channel_time_embeddings, dim=1)  # [B, 25, 144, 768]
        all_embeddings = torch.cat([elevation_embeddings.unsqueeze(1), 
                                  channel_time_embeddings], dim=1)  # [B, 26, 144, 768]

    return all_embeddings, oc

def train_epoch(vit_model, mlp_model, train_loader, criterion, optimizer, device, progress):
    mlp_model.train()
    total_loss = 0
    task_id = progress.add_task("[red]Training...", total=len(train_loader))

    for batch in train_loader:
        longitude, latitude, stacked_tensor, metadata, oc = batch

        # Process the batch data
        elevation_instrument, remaining_tensor, metadata_strings = transform_data(
            stacked_tensor, metadata)

        # Get embeddings
        embeddings, targets = process_batch(
            longitude, latitude, elevation_instrument, remaining_tensor,
            metadata_strings, oc, vit_model, device
        )
        print('embeddings.shape ',embeddings.shape)

        # Process through MLP
        optimizer.zero_grad()
        outputs = mlp_model(embeddings)
        targets = targets.to(device).float().unsqueeze(1)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.update(task_id, advance=1)

    return total_loss / len(train_loader)


def main(args):
    # Initialize wandb
    wandb.init(project="soil-organic-carbon-prediction", name="mlp-embeddings")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console = Console()
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

    # Initialize models
    vit_model = models_vit_tensor.__dict__[args.model](
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes
    ).to(device)
    vit_model = load_model(vit_model, args, device)
    vit_model.eval()

    mlp_model = MLP(input_dim=768*144*26).to(device)  # 26 = 1 elevation + 25 channel/time
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        for epoch in range(num_epochs):
            # Training
            train_loss = train_epoch(vit_model, mlp_model, train_loader, 
                                   criterion, optimizer, device, progress)

            # Validation
            val_loss, r2, rmse = validate(vit_model, mlp_model, val_loader, 
                                        criterion, device, progress)

            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'r2_score': r2,
                'rmse': rmse,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Print metrics
            console.print(f"Epoch {epoch+1}/{num_epochs}")
            console.print(f"Train Loss: {train_loss:.4f}")
            console.print(f"Val Loss: {val_loss:.4f}")
            console.print(f"R² Score: {r2:.4f}")
            console.print(f"RMSE: {rmse:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': mlp_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')


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
        elevation_input = elevation_input.to(device)
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

def save_to_parquet(train_loader, model, device, output_path='/fast/vfourel/SOCProject/embeddings.parquet'):
    all_data = []

    with torch.no_grad():
        for batch in train_loader:
            longitude, latitude, stacked_tensor, metadata, oc = batch

            # Process the batch data
            elevation_instrument, remaining_tensor, metadata_strings = transform_data(
                stacked_tensor, metadata)

            # Convert batch to DataFrame
            batch_df = process_batch_to_dataframe(
                longitude, latitude, elevation_instrument, remaining_tensor,
                metadata_strings, oc, model, device
            )

            all_data.append(batch_df)

    # Combine all batches
    final_df = pd.concat(all_data, ignore_index=True)

    # Save to parquet
    final_df.to_parquet(output_path)

    return final_df


if __name__ == "__main__":
    args = parse_args()
    df = save_to_parquet(train_loader, model, device)
    main(args)
    
