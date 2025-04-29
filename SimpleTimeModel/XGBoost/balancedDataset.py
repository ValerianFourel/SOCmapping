import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from accelerate import Accelerator
from tqdm import tqdm
from scipy import stats
from scipy.stats import invgamma

from config import (
    TIME_BEGINNING, TIME_END, MAX_OC,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data

def vectorized_haversine(lon1, lat1, lon2, lat2, device='cpu'):
    R = 6371
    if device == 'cuda' and torch.cuda.is_available():
        lon1 = torch.tensor(lon1, device='cuda', dtype=torch.float32)
        lat1 = torch.tensor(lat1, device='cuda', dtype=torch.float32)
        lon2 = torch.tensor(lon2, device='cuda', dtype=torch.float32)
        lat2 = torch.tensor(lat2, device='cuda', dtype=torch.float32)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distances = R * c
        return distances.cpu().numpy()
    else:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

def compute_min_distances(val_df, train_df, device='cpu'):
    if train_df.empty or val_df.empty:
        return np.full(len(val_df), np.inf)
    
    val_lons = val_df['GPS_LONG'].values.astype(float)
    val_lats = val_df['GPS_LAT'].values.astype(float)
    train_lons = train_df['GPS_LONG'].values.astype(float)
    train_lats = train_df['GPS_LAT'].values.astype(float)
    
    val_lons = np.expand_dims(val_lons, axis=1)
    val_lats = np.expand_dims(val_lats, axis=1)
    
    distances = vectorized_haversine(val_lons, val_lats, train_lons, train_lats, device)
    min_distances = np.min(distances, axis=1)
    
    invalid = np.isnan(val_lons.flatten()) | np.isnan(val_lats.flatten()) | np.isnan(min_distances)
    min_distances[invalid] = np.inf
    return min_distances

def fit_exponential_family(data):
    """
    Return the predefined Inverse Gamma distribution and its parameters.
    """
    best_dist = invgamma
    best_params = {'a': 3.5093212085018015, 'loc': -0.2207140712018134, 'scale': 55.73445932737795}
    dist_name = "Inverse Gamma"
    return best_dist, best_params, dist_name

def create_optimized_subset(df, best_dist, best_params, dist_name, target_val_ratio=0.08, output_dir='output', device='cpu', distance_threshold=1.4):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()
    df[['OC', 'GPS_LAT', 'GPS_LONG']] = df[['OC', 'GPS_LAT', 'GPS_LONG']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])
    
    total_samples = len(df)
    target_size = int(total_samples * target_val_ratio)
    initial_ratio = target_val_ratio + 0.02
    max_ratio = 0.995

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    while initial_ratio <= max_ratio:
        subset_size = int(total_samples * initial_ratio)
        
        oc_min, oc_max = df['OC'].min(), df['OC'].max()
        dist_params = list(best_params.values())
        dist_samples = best_dist.rvs(*dist_params, size=subset_size * 2)
        dist_samples = oc_min + (oc_max - oc_min) * (dist_samples - np.min(dist_samples)) / (np.max(dist_samples) - np.min(dist_samples))
        
        oc_values = df['OC'].values
        weights = best_dist.pdf(oc_values, *dist_params)
        weights = weights / weights.sum()
        
        subset_indices = np.random.choice(df.index, size=subset_size, replace=False, p=weights)
        
        subset_df = df.loc[subset_indices]
        remaining_df = df.drop(subset_indices)
        
        min_distances = compute_min_distances(subset_df, remaining_df, device)
        validation_df = subset_df[min_distances >= distance_threshold]
        points_to_flip = subset_df[min_distances < distance_threshold]
        
        if not points_to_flip.empty:
            training_df = pd.concat([remaining_df, points_to_flip])
        else:
            training_df = remaining_df
        
        val_size = len(validation_df)
        val_ratio = val_size / total_samples
        if val_size >= target_size:
            break
        else:
            initial_ratio += 0.02
            if initial_ratio > max_ratio:
                break
    
    oc_bins = pd.qcut(df['OC'], q=10, duplicates='drop')
    train_bins = pd.cut(training_df['OC'], bins=oc_bins.cat.categories)
    bin_counts = train_bins.value_counts()
    empty_bins = bin_counts[bin_counts == 0].index
    
    if not empty_bins.empty:
        for bin_label in empty_bins:
            val_in_bin = validation_df[pd.cut(validation_df['OC'], bins=oc_bins.cat.categories) == bin_label]
            if not val_in_bin.empty:
                flip_point = val_in_bin.sample(n=1)
                training_df = pd.concat([training_df, flip_point])
                validation_df = validation_df.drop(flip_point.index)
    
    validation_df.to_parquet(output_dir / 'final_validation_df.parquet')
    training_df.to_parquet(output_dir / 'final_training_df.parquet')
    
    return validation_df, training_df

def create_validation_train_sets(df=None, output_dir='output', target_val_ratio=0.08, use_gpu=False, distance_threshold=1.4):
    """
    Create optimized validation and training sets using the predefined Inverse Gamma distribution.

    Parameters:
    -----------
    df : pandas.DataFrame, optional (default=None)
        Input DataFrame to split. If None, loads data using filter_dataframe.
    output_dir : str, optional (default='output')
        Directory to save the output Parquet files.
    target_val_ratio : float, optional (default=0.08)
        Target ratio of validation set size to total samples.
    use_gpu : bool, optional (default=False)
        Whether to use GPU for computations.
    distance_threshold : float, optional (default=1.4)
        Minimum distance threshold (in km) for validation points.

    Returns:
    --------
    validation_df : pandas.DataFrame
        DataFrame containing the validation set.
    training_df : pandas.DataFrame
        DataFrame containing the training set.
    """
    accelerator = Accelerator()
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    try:
        # Load data if df is not provided
        if df is None:
            df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        if df.empty:
            raise ValueError("Empty DataFrame.")
        if 'POINTID' in df.columns:
            df = df.drop(columns=['POINTID'])
        
        oc_values = df['OC'].dropna().values
        best_dist, best_params, dist_name = fit_exponential_family(oc_values)
        
        validation_df, training_df = create_optimized_subset(
            df,
            best_dist=best_dist,
            best_params=best_params,
            dist_name=dist_name,
            target_val_ratio=target_val_ratio,
            output_dir=output_dir,
            device=device,
            distance_threshold=distance_threshold
        )
        
        return validation_df, training_df
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def resample_training_df(training_df, num_bins=128, target_fraction=0.75):
    """
    Resample training_df's 'OC' values into num_bins, ensuring each bin has at least
    target_fraction of the entries of the highest count.
    """
    oc_values = training_df['OC'].dropna()
    bins = pd.qcut(oc_values, q=num_bins, duplicates='drop')
    
    bin_counts = bins.value_counts().sort_index()
    max_count = bin_counts.max()
    target_count = int(max_count * target_fraction)
    
    print(f"Max bin count: {max_count}")
    print(f"Target count per bin (at least): {target_count}")
    
    resampled_dfs = []
    
    for bin_label in bin_counts.index:
        bin_mask = pd.cut(training_df['OC'], bins=bins.cat.categories) == bin_label
        bin_df = training_df[bin_mask]
        
        if len(bin_df) < target_count:
            additional_samples = target_count - len(bin_df)
            sampled_df = bin_df.sample(n=additional_samples, replace=True, random_state=42)
            resampled_dfs.append(pd.concat([bin_df, sampled_df]))
        else:
            resampled_dfs.append(bin_df)
    
    resampled_df = pd.concat(resampled_dfs, ignore_index=True)
    
    new_bins = pd.qcut(resampled_df['OC'], q=num_bins, duplicates='drop')
    new_bin_counts = new_bins.value_counts().sort_index()
    
    print("\nBin counts before resampling:")
    print(bin_counts)
    print("\nBin counts after resampling:")
    print(new_bin_counts)
    
    return resampled_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized validation set creation with exponential family distribution')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for validation points')
    args = parser.parse_args()

    validation_df, training_df = create_validation_train_sets(
        df=None,
        output_dir=args.output_dir,
        target_val_ratio=args.target_val_ratio,
        use_gpu=args.use_gpu,
        distance_threshold=args.distance_threshold
    )