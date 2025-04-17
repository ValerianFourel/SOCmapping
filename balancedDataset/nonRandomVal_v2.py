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
from scipy.stats import gaussian_kde, expon, gamma, lognorm, weibull_min, beta, chi2, invgamma, norm

from config import (
    TIME_BEGINNING, TIME_END, MAX_OC,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)
from dataframe_loader import filter_dataframe, separate_and_add_data

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
    distributions = [
        (expon, "Exponential", ["loc", "scale"]),
        (gamma, "Gamma", ["a", "loc", "scale"]),
       (lognorm, "Lognormal", ["s", "loc", "scale"]),
        (weibull_min, "Weibull", ["c", "loc", "scale"]),
        (beta, "Beta", ["a", "b", "loc", "scale"]),
        (chi2, "Chi-Square", ["df", "loc", "scale"]),
        (invgamma, "Inverse Gamma", ["a", "loc", "scale"]),
        (norm, "Normal", ["loc", "scale"])
    ]
    
    best_dist = None
    best_params = None
    best_ks_stat = float('inf')
    
    # Scale data for Beta distribution (which requires [0,1] range)
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data_scaled = (data - data_min) / (data_max - data_min)
    else:
        data_scaled = data  # Avoid division by zero
    
    for dist, dist_name, param_names in distributions:
        try:
            # Use scaled data for Beta distribution
            fit_data = data_scaled if dist == beta else data
            params = dist.fit(fit_data)
            # Adjust parameters for Beta to map back to original scale
            if dist == beta:
                a, b, loc, scale = params
                params = (a, b, loc * (data_max - data_min) + data_min, scale * (data_max - data_min))
            ks_stat, _ = stats.ks_1samp(fit_data, dist.cdf, args=params)
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_dist = dist
                best_params = dict(zip(param_names, params))
                best_dist_name = dist_name
        except Exception as e:
            print(f"Failed to fit {dist_name}: {e}")
    
    print(f"Best fitting distribution: {best_dist_name}")
    print(f"Parameters: {best_params}")
    print(f"KS statistic: {best_ks_stat:.4f}")
    
    return best_dist, best_params, best_dist_name

def create_visualizations(subset_df, remaining_df, full_df, best_dist, best_params, dist_name, save_path, iteration=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    
    bavaria_file = 'bavaria.geojson'
    if not os.path.exists(bavaria_file):
        bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
        bavaria = bavaria[bavaria['name'] == 'Bayern']
        bavaria.to_file(bavaria_file)
    else:
        bavaria = gpd.read_file(bavaria_file)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
    
    # Scatter plot
    bavaria.boundary.plot(ax=ax1, color='black', linewidth=1)
    ax1.scatter(remaining_df['GPS_LONG'], remaining_df['GPS_LAT'], c='blue', label='Training', alpha=0.5, s=10)
    ax1.scatter(subset_df['GPS_LONG'], subset_df['GPS_LAT'], c='red', label='Validation', alpha=0.5, s=10)
    ax1.set_title(f'Spatial Distribution (Iter {iteration})')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True)
    ax1.legend()

    # Distance distribution
    distances = compute_min_distances(subset_df, remaining_df, device='cpu')
    distances = distances[np.isfinite(distances)]
    if len(distances) > 0:
        ax2.hist(distances, bins=30, density=True, alpha=0.7, color='green')
        ax2.set_title(f'Min Distances (Iter {iteration})')
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Density')
        ax2.grid(True)

    # OC distribution histogram
    ax3.hist(remaining_df['OC'].values, bins=30, density=True, alpha=0.5, color='blue', label='Training')
    ax3.hist(subset_df['OC'].values, bins=30, density=True, alpha=0.5, color='red', label='Validation')
    ax3.set_title(f'OC Distribution (Iter {iteration})')
    ax3.set_xlabel('OC Value')
    ax3.set_ylabel('Density')
    ax3.grid(True)
    ax3.legend()

    # Gaussian KDE vs Fitted Distribution
    oc_range = np.linspace(max(0, min(full_df['OC'].min(), subset_df['OC'].min())), 
                          max(full_df['OC'].max(), subset_df['OC'].max()), 100)
    
    if len(subset_df['OC']) > 1:
        kde_val = gaussian_kde(subset_df['OC'].values)
        ax4.plot(oc_range, kde_val(oc_range), 'r-', label='Validation KDE')
    
    if len(remaining_df['OC']) > 1:
        kde_train = gaussian_kde(remaining_df['OC'].values)
        ax4.plot(oc_range, kde_train(oc_range), 'b-', label='Training KDE')
    
    # Plot fitted distribution
    dist_params = list(best_params.values())
    if best_dist == beta:
        # Scale oc_range for Beta distribution
        data_min, data_max = full_df['OC'].min(), full_df['OC'].max()
        scaled_range = (oc_range - data_min) / (data_max - data_min)
        dist_pdf = best_dist.pdf(scaled_range, *dist_params[:2], loc=0, scale=1) / (data_max - data_min)
    else:
        dist_pdf = best_dist.pdf(oc_range, *dist_params)
    ax4.plot(oc_range, dist_pdf, 'g--', label=f'{dist_name} Fit')
    
    ax4.set_title(f'Gaussian KDE vs {dist_name} (Iter {iteration})')
    ax4.set_xlabel('OC Value')
    ax4.set_ylabel('Density')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'visualizations_{timestamp}_iter{iteration}.png'), bbox_inches='tight')
    plt.close()

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
        
        # Sample according to the best-fitting distribution
        oc_min, oc_max = df['OC'].min(), df['OC'].max()
        dist_params = list(best_params.values())
        dist_samples = best_dist.rvs(*dist_params, size=subset_size * 2)  # Oversample
        if best_dist == beta:
            # Rescale Beta samples to OC range
            dist_samples = oc_min + (oc_max - oc_min) * dist_samples
        else:
            dist_samples = oc_min + (oc_max - oc_min) * (dist_samples - np.min(dist_samples)) / (np.max(dist_samples) - np.min(dist_samples))  # Scale to OC range
        
        # Assign weights based on fitted distribution's PDF
        oc_values = df['OC'].values
        if best_dist == beta:
            scaled_oc = (oc_values - oc_min) / (oc_max - oc_min)
            weights = best_dist.pdf(scaled_oc, *dist_params[:2], loc=0, scale=1)
        else:
            weights = best_dist.pdf(oc_values, *dist_params)
        weights = weights / weights.sum()  # Normalize weights
        
        # Sample indices based on weights
        subset_indices = np.random.choice(df.index, size=subset_size, replace=False, p=weights)
        
        subset_df = df.loc[subset_indices]
        remaining_df = df.drop(subset_indices)
        
        subset_df.to_parquet(output_dir / 'initial_subset_df.parquet')
        remaining_df.to_parquet(output_dir / 'initial_remaining_df.parquet')
        
        create_visualizations(subset_df, remaining_df, df, best_dist, best_params, dist_name, output_dir, iteration=0)
        
        min_distances = compute_min_distances(subset_df, remaining_df, device)
        validation_df = subset_df[min_distances >= distance_threshold]
        points_to_flip = subset_df[min_distances < distance_threshold]
        
        if not points_to_flip.empty:
            print(f"Flipping {len(points_to_flip)} points (distance < {distance_threshold} km)")
            training_df = pd.concat([remaining_df, points_to_flip])
        else:
            training_df = remaining_df
        
        val_size = len(validation_df)
        val_ratio = val_size / total_samples
        if val_size >= target_size:
            break
        else:
            print(f"Validation set size {val_ratio*100:.2f}% < {target_val_ratio*100}%. Increasing ratio.")
            initial_ratio += 0.02
            if initial_ratio > max_ratio:
                print(f"Max ratio {max_ratio*100}% reached. Using current validation set.")
                break
    
    # Ensure no empty bins in training set
    oc_bins = pd.qcut(df['OC'], q=10, duplicates='drop')
    train_bins = pd.cut(training_df['OC'], bins=oc_bins.cat.categories)
    bin_counts = train_bins.value_counts()
    empty_bins = bin_counts[bin_counts == 0].index
    
    if not empty_bins.empty:
        print(f"Found {len(empty_bins)} empty bins in training set. Flipping points to fill them.")
        for bin_label in empty_bins:
            val_in_bin = validation_df[pd.cut(validation_df['OC'], bins=oc_bins.cat.categories) == bin_label]
            if not val_in_bin.empty:
                flip_point = val_in_bin.sample(n=1)
                training_df = pd.concat([training_df, flip_point])
                validation_df = validation_df.drop(flip_point.index)
                print(f"Flipped 1 point from validation to training for bin {bin_label}")
    
    validation_df.to_parquet(output_dir / 'final_validation_df.parquet')
    training_df.to_parquet(output_dir / 'final_training_df.parquet')
    
    create_visualizations(validation_df, training_df, df, best_dist, best_params, dist_name, output_dir, iteration=1)
    
    final_min_distance = compute_min_distances(validation_df, training_df, device).min()
    
    print(f'Full dataset size: {total_samples}')
    print(f'Final validation set: {len(validation_df)} ({val_ratio*100:.2f}%)')
    print(f'Final training set: {len(training_df)}')
    print(f'Minimum distance (km): {final_min_distance:.2f}')
    print(f'Validation OC distribution matched to {dist_name} with parameters: {best_params}')

    return validation_df, training_df

def main():
    parser = argparse.ArgumentParser(description='Optimized validation set creation with exponential family distribution')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        if df.empty:
            raise ValueError("Empty DataFrame.")
        if 'POINTID' in df.columns:
            df = df.drop(columns=['POINTID'])
        print(f"Loaded {len(df)} rows")
        
        # Fit exponential family distribution to OC values
        oc_values = df['OC'].dropna().values
        best_dist, best_params, dist_name = fit_exponential_family(oc_values)
        
        validation_df, training_df = create_optimized_subset(
            df,
            best_dist=best_dist,
            best_params=best_params,
            dist_name=dist_name,
            target_val_ratio=args.target_val_ratio,
            output_dir=args.output_dir,
            device=device,
            distance_threshold=1.4
        )
    except Exception as e:
        print(f"Error processing: {e}")

if __name__ == "__main__":
    main()
    #finish it by flipping bins with 0 entries in training , 