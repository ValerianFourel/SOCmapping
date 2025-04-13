import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from math import radians, sin, cos, sqrt, atan2, isnan
from pathlib import Path
import argparse
from tqdm import tqdm
from accelerate import Accelerator
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears, NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, num_epochs, NUM_HEADS, NUM_LAYERS,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
# Haversine formula (CPU or GPU compatible)
def haversine(lon1, lat1, lon2, lat2, device='cpu'):
    R = 6371  # Earth's radius in kilometers
    lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])

    if any(isnan(x) for x in [lon1, lat1, lon2, lat2]):
        return np.inf

    if device == 'cuda' and torch.cuda.is_available():
        lon1 = torch.tensor(lon1, device='cuda')
        lat1 = torch.tensor(lat1, device='cuda')
        lon2 = torch.tensor(lon2, device='cuda')
        lat2 = torch.tensor(lat2, device='cuda')
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distance = R * c
        return distance.cpu().numpy()
    else:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

# Compute minimum distance for a point to all points in another DataFrame
def compute_min_distance(row, other_df, device='cpu'):
    if other_df.empty:
        return np.inf
    min_dist = np.inf
    val_lon = float(row['GPS_LONG'])
    val_lat = float(row['GPS_LAT'])
    for _, train_row in other_df.iterrows():
        train_lon = float(train_row['GPS_LONG'])
        train_lat = float(train_row['GPS_LAT'])
        dist = haversine(val_lon, val_lat, train_lon, train_lat, device)
        min_dist = min(min_dist, dist)
    return min_dist

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True, output_dir='output', device='cpu'):
    """
    Creates balanced training and validation datasets ensuring validation points are at least 8 km
    from training points in the same bin, or selects points with maximum minimum distance if not possible.
    """
    # Convert pandas DataFrame to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=4)
    
    # Ensure necessary columns
    required_cols = ['OC', 'GPS_LAT', 'GPS_LONG']
    if not all(col in ddf.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Convert GPS columns to numeric
    ddf['GPS_LAT'] = ddf['GPS_LAT'].astype(float)
    ddf['GPS_LONG'] = ddf['GPS_LONG'].astype(float)
    ddf = ddf.dropna(subset=['GPS_LAT', 'GPS_LONG'])
    
    # Compute bins
    try:
        ddf['bin'] = dd.from_pandas(pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop'), 
                                   npartitions=ddf.npartitions)
        n_bins = len(pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop').unique())
    except Exception as e:
        print(f"Warning: qcut failed ({e}). Using factorize.")
        ddf['bin'] = dd.from_pandas(pd.factorize(df['OC'])[0], npartitions=ddf.npartitions)
        n_bins = len(np.unique(pd.factorize(df['OC'])[0]))
    
    # Compute bin counts
    bin_counts = ddf['bin'].value_counts().compute()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)

    training_dfs = []
    validation_dfs = []
    min_distances = {}  # Track minimum distances per bin

    unique_bins = bin_counts.index.tolist()

    for bin_idx in tqdm(unique_bins, desc="Processing bins"):
        bin_data = ddf[ddf['bin'] == bin_idx].compute()  # Convert to pandas for bin processing
        if bin_data.empty:
            continue

        if use_validation and len(bin_data) >= 4:
            n_val = min(13, len(bin_data) // 4) or 1
            best_val_samples = pd.DataFrame()
            best_min_distance = -np.inf

            # Try multiple random samples to find points meeting 8 km criterion
            for _ in range(10):  # Number of attempts
                val_candidates = bin_data.sample(n=min(n_val, len(bin_data)), random_state=None)
                train_candidates = bin_data.drop(val_candidates.index)

                # Compute minimum distance for each validation candidate to training set
                val_candidates['min_dist'] = val_candidates.apply(
                    lambda row: compute_min_distance(row, train_candidates, device), axis=1
                )

                # Check if all candidates satisfy 8 km criterion
                min_dist_in_sample = val_candidates['min_dist'].min()
                if min_dist_in_sample >= 8:
                    # Success: use this split
                    best_val_samples = val_candidates.drop(columns=['min_dist'], errors='ignore')
                    train_samples = train_candidates
                    best_min_distance = min_dist_in_sample
                    break
                elif min_dist_in_sample > best_min_distance:
                    # Update best split if this is the largest minimum distance so far
                    best_val_samples = val_candidates.drop(columns=['min_dist'], errors='ignore')
                    train_samples = train_candidates
                    best_min_distance = min_dist_in_sample

            if best_val_samples.empty:
                # Fallback: random split if no valid split found
                warnings.warn(f"Bin {bin_idx}: Could not find validation points >= 8 km. Using random split.")
                n_val_fallback = max(1, min(13, len(bin_data) // 4))
                best_val_samples = bin_data.sample(n=n_val_fallback)
                train_samples = bin_data.drop(best_val_samples.index)
                best_val_samples['min_dist'] = best_val_samples.apply(
                    lambda row: compute_min_distance(row, train_samples, device), axis=1
                )
                best_min_distance = best_val_samples['min_dist'].min()
                best_val_samples = best_val_samples.drop(columns=['min_dist'], errors='ignore')

            # Store minimum distance for this bin
            min_distances[bin_idx] = best_min_distance if best_min_distance != np.inf else None

            # Handle training samples
            if len(train_samples) < min_samples:
                train_samples = train_samples.sample(
                    n=min_samples, replace=(min_samples > len(train_samples)), random_state=42
                )
            training_dfs.append(train_samples)
            validation_dfs.append(best_val_samples)

        else:
            # No validation or insufficient data: all to training
            if len(bin_data) < min_samples:
                bin_data = bin_data.sample(
                    n=min_samples, replace=(min_samples > len(bin_data)), random_state=42
                )
            training_dfs.append(bin_data)

    if not training_dfs:
        raise ValueError("No training data available after binning.")

    # Concatenate results
    training_df = pd.concat(training_dfs).drop(columns=['bin'], errors='ignore')
    validation_df = pd.concat(validation_dfs).drop(columns=['bin'], errors='ignore') if validation_dfs else pd.DataFrame(columns=training_df.columns)

    # Compute overall minimum distance between final sets
    overall_min_distance = np.inf
    if not training_df.empty and not validation_df.empty:
        for _, val_row in validation_df.iterrows():
            min_dist = compute_min_distance(val_row, training_df, device)
            overall_min_distance = min(overall_min_distance, min_dist)

    # Save DataFrames
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    training_df.to_parquet(output_dir / 'training_df.parquet')
    validation_df.to_parquet(output_dir / 'validation_df.parquet')

    # Print summary
    print(f'Size of the training set:   {len(training_df)}')
    print(f'Size of the validation set: {len(validation_df)}')
    print(f'Overall minimum distance between training and validation sets (km): '
          f'{overall_min_distance:.2f}' if overall_min_distance != np.inf else 'N/A')
    print(f'Minimum distances per bin: {min_distances}')

    return training_df, validation_df

def main():
    parser = argparse.ArgumentParser(description='Create balanced dataset with spatial constraints')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output DataFrames')
    parser.add_argument('--n-bins', type=int, default=128, help='Number of bins for OC')
    parser.add_argument('--min-ratio', type=float, default=0.75, help='Minimum ratio for sampling')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for distance calculations')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    # Load your DataFrame (replace with your data loading logic)
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC) # Update with actual data source
    # Example: df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)

    # Create balanced dataset
    train_df, val_df = create_balanced_dataset(
        df,
        n_bins=args.n_bins,
        min_ratio=args.min_ratio,
        use_validation=True,
        output_dir=args.output_dir,
        device=device
    )