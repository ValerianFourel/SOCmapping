import argparse
import warnings
from math import radians, sin, cos, sqrt, atan2, isnan
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime
import os

from config import (
    TIME_BEGINNING, TIME_END, MAX_OC,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataframe_loader import filter_dataframe, separate_and_add_data

# Haversine formula
def haversine(lon1, lat1, lon2, lat2, device='cpu'):
    R = 6371
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

# Compute minimum distance
def compute_min_distance(row, other_df, device='cpu'):
    if other_df.empty:
        return np.inf
    min_dist = np.inf
    val_lon = float(row['GPS_LONG'])
    val_lat = float(row['GPS_LAT'])
    if isnan(val_lon) or isnan(val_lat):
        return np.inf
    if device == 'cuda' and torch.cuda.is_available():
        val_lon_t = torch.tensor([val_lon], device='cuda')
        val_lat_t = torch.tensor([val_lat], device='cuda')
        train_lons = torch.tensor(other_df['GPS_LONG'].astype(float).values, device='cuda')
        train_lats = torch.tensor(other_df['GPS_LAT'].astype(float).values, device='cuda')
        dlon = train_lons - val_lon_t
        dlat = train_lats - val_lat_t
        a = torch.sin(dlat/2)**2 + torch.cos(val_lat_t) * torch.cos(train_lats) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        distances = 6371 * c
        min_dist = float(torch.min(distances).cpu().numpy())
    else:
        for _, train_row in other_df.iterrows():
            train_lon = float(train_row['GPS_LONG'])
            train_lat = float(train_row['GPS_LAT'])
            if isnan(train_lon) or isnan(train_lat):
                continue
            dist = haversine(val_lon, val_lat, train_lon, train_lat, device)
            min_dist = min(min_dist, dist)
    return float(min_dist)

# Visualization function
def create_visualizations(subset_df, remaining_df, save_path, iteration=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    
    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']

    # 1. Scatter plot of points
    plt.figure(figsize=(12, 10), dpi=300)
    ax = plt.gca()
    bavaria.boundary.plot(ax=ax, color='black', linewidth=1)
    plt.scatter(remaining_df['GPS_LONG'], remaining_df['GPS_LAT'], c='blue', label='Training', alpha=0.5, s=20)
    plt.scatter(subset_df['GPS_LONG'], subset_df['GPS_LAT'], c='red', label='Validation', alpha=0.5, s=20)
    plt.title(f'Spatial Distribution of Points (Iteration {iteration})', fontsize=12, pad=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f'spatial_points_{timestamp}_iter{iteration}.png'), bbox_inches='tight')
    plt.close()

    # 2. Distance distribution
    distances = []
    for _, row in subset_df.iterrows():
        dist = compute_min_distance(row, remaining_df)
        if dist != np.inf:
            distances.append(dist)
    
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=30, density=True, alpha=0.7, color='green')
    plt.title(f'Distribution of Minimum Distances (Iteration {iteration})')
    plt.xlabel('Distance (km)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'distance_distribution_{timestamp}_iter{iteration}.png'), bbox_inches='tight')
    plt.close()

    # 3. KDE plot of OC distributions
    kde_full = gaussian_kde(remaining_df['OC'].values)
    kde_subset = gaussian_kde(subset_df['OC'].values)
    oc_range = np.linspace(min(remaining_df['OC'].min(), subset_df['OC'].min()), 
                         max(remaining_df['OC'].max(), subset_df['OC'].max()), 100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(oc_range, kde_full(oc_range), label='Training', color='blue')
    plt.plot(oc_range, kde_subset(oc_range), label='Validation', color='red')
    plt.title(f'OC Distribution KDE (Iteration {iteration})')
    plt.xlabel('OC Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'oc_kde_{timestamp}_iter{iteration}.png'), bbox_inches='tight')
    plt.close()

    return kde_full, kde_subset

def create_optimized_subset(df, min_subset_ratio=0.05, max_subset_ratio=0.33, output_dir='output', device='cpu'):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()
    numeric_cols = ['OC', 'GPS_LAT', 'GPS_LONG']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])
    
    total_samples = len(df)
    min_subset_size = int(total_samples * min_subset_ratio)
    max_subset_size = int(total_samples * max_subset_ratio)

    oc_values = df['OC'].values
    kde_full = gaussian_kde(oc_values)
    
    best_subset = pd.DataFrame()
    best_remaining = pd.DataFrame()
    best_min_distance = -np.inf
    best_kde_diff = np.inf

    # Initial subset optimization
    for _ in tqdm(range(50), desc="Optimizing initial subset"):
        subset_size = np.random.randint(min_subset_size, max_subset_size + 1)
        subset_candidates = df.sample(n=subset_size, random_state=None)
        remaining_candidates = df.drop(subset_candidates.index)
        
        kde_subset = gaussian_kde(subset_candidates['OC'].values)
        oc_range = np.linspace(min(oc_values), max(oc_values), 100)
        kde_diff = np.trapz(np.abs(kde_full(oc_range) - kde_subset(oc_range)), oc_range)
        
        subset_candidates['min_dist'] = subset_candidates.apply(
            lambda row: compute_min_distance(row, remaining_candidates, device), axis=1
        ).astype(float)
        min_dist = subset_candidates['min_dist'].min()
        
        if kde_diff < best_kde_diff and min_dist > 0:
            best_subset = subset_candidates.drop(columns=['min_dist'])
            best_remaining = remaining_candidates
            best_min_distance = min_dist
            best_kde_diff = kde_diff
        elif kde_diff == best_kde_diff and min_dist > best_min_distance:
            best_subset = subset_candidates.drop(columns=['min_dist'])
            best_remaining = remaining_candidates
            best_min_distance = min_dist

    if best_subset.empty:
        warnings.warn("Could not find optimal subset. Maximizing distance only.")
        df['min_dist'] = df.apply(
            lambda row: compute_min_distance(row, df.drop(index=row.name), device), axis=1
        ).astype(float)
        best_subset = df.nlargest(max_subset_size, 'min_dist').drop(columns=['min_dist'])
        best_remaining = df.drop(best_subset.index)
        best_min_distance = best_subset.apply(
            lambda row: compute_min_distance(row, best_remaining, device), axis=1
        ).min()

    # Save initial results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    best_subset.to_parquet(output_dir / 'initial_subset_df.parquet')
    best_remaining.to_parquet(output_dir / 'initial_remaining_df.parquet')
    
    # Initial visualizations
    create_visualizations(best_subset, best_remaining, output_dir, iteration=0)

    # Flip points within 1 km
    validation_df = best_subset.copy()
    training_df = best_remaining.copy()
    
    validation_df['min_dist'] = validation_df.apply(
        lambda row: compute_min_distance(row, training_df, device), axis=1
    ).astype(float)
    
    points_to_flip = validation_df[validation_df['min_dist'] < 1.0]
    if not points_to_flip.empty:
        print(f"Flipping {len(points_to_flip)} points from validation to training set (distance < 1 km)")
        training_df = pd.concat([training_df, points_to_flip.drop(columns=['min_dist'])])
        validation_df = validation_df[validation_df['min_dist'] >= 1.0].drop(columns=['min_dist'])

    # Save updated results
    validation_df.to_parquet(output_dir / 'final_validation_df.parquet')
    training_df.to_parquet(output_dir / 'final_training_df.parquet')

    # Final visualizations with updated KDE
    final_kde_train, final_kde_val = create_visualizations(validation_df, training_df, output_dir, iteration=1)

    # Compute final metrics
    final_min_distance = validation_df.apply(
        lambda row: compute_min_distance(row, training_df, device), axis=1
    ).min()
    
    oc_range = np.linspace(min(training_df['OC'].min(), validation_df['OC'].min()),
                         max(training_df['OC'].max(), validation_df['OC'].max()), 100)
    final_kde_diff = np.trapz(np.abs(final_kde_train(oc_range) - final_kde_val(oc_range)), oc_range)

    print(f'Size of the full dataset: {total_samples}')
    print(f'Size of the final validation set: {len(validation_df)} ({(len(validation_df)/total_samples)*100:.2f}%)')
    print(f'Size of the final training set: {len(training_df)}')
    print(f'Minimum distance between validation and training sets after flipping (km): {final_min_distance:.2f}')
    print(f'Final KDE difference score (lower is better): {final_kde_diff:.4f}')

    return validation_df, training_df

def main():
    parser = argparse.ArgumentParser(description='Create optimized subset with distribution matching and distance flipping')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output DataFrames')
    parser.add_argument('--min-subset-ratio', type=float, default=0.05, help='Minimum subset size ratio')
    parser.add_argument('--max-subset-ratio', type=float, default=0.33, help='Maximum subset size ratio')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for distance calculations')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        if df.empty:
            raise ValueError("Loaded DataFrame is empty.")
        if 'POINTID' in df.columns:
            df = df.drop(columns=['POINTID'])
        print(f"Loaded DataFrame with {len(df)} rows")
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return

    try:
        validation_df, training_df = create_optimized_subset(
            df,
            min_subset_ratio=args.min_subset_ratio,
            max_subset_ratio=args.max_subset_ratio,
            output_dir=args.output_dir,
            device=device
        )
    except Exception as e:
        print(f"Error creating optimized subset: {e}")
        return

if __name__ == "__main__":
    main()