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
    
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data_scaled = (data - data_min) / (data_max - data_min)
    else:
        data_scaled = data
    
    for dist, dist_name, param_names in distributions:
        try:
            fit_data = data_scaled if dist == beta else data
            params = dist.fit(fit_data)
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

def create_visualizations(fold_dfs, training_fold_df, full_df, best_dist, best_params, dist_name, save_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    
    bavaria_file = 'bavaria.geojson'
    if not os.path.exists(bavaria_file):
        bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
        bavaria = bavaria[bavaria['name'] == 'Bayern']
        bavaria.to_file(bavaria_file)
    else:
        bavaria = gpd.read_file(bavaria_file)

    # Visualization for each fold
    for i, (val_df, train_df) in enumerate(fold_dfs):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
        
        bavaria.boundary.plot(ax=ax1, color='black', linewidth=1)
        ax1.scatter(train_df['GPS_LONG'], train_df['GPS_LAT'], c='blue', label='Training', alpha=0.5, s=10)
        ax1.scatter(val_df['GPS_LONG'], val_df['GPS_LAT'], c='red', label='Validation', alpha=0.5, s=10)
        ax1.set_title(f'Spatial Distribution Fold {i+1}')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True)
        ax1.legend()

        distances = compute_min_distances(val_df, train_df, device='cpu')
        distances = distances[np.isfinite(distances)]
        if len(distances) > 0:
            ax2.hist(distances, bins=30, density=True, alpha=0.7, color='green')
            ax2.set_title(f'Min Distances Fold {i+1}')
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Density')
            ax2.grid(True)

        ax3.hist(train_df['OC'].values, bins=30, density=True, alpha=0.5, color='blue', label='Training')
        ax3.hist(val_df['OC'].values, bins=30, density=True, alpha=0.5, color='red', label='Validation')
        ax3.set_title(f'OC Distribution Fold {i+1}')
        ax3.set_xlabel('OC Value')
        ax3.set_ylabel('Density')
        ax3.grid(True)
        ax3.legend()

        oc_range = np.linspace(max(0, min(full_df['OC'].min(), val_df['OC'].min())), 
                              max(full_df['OC'].max(), val_df['OC'].max()), 100)
        
        if len(val_df['OC']) > 1:
            kde_val = gaussian_kde(val_df['OC'].values)
            ax4.plot(oc_range, kde_val(oc_range), 'r-', label='Validation KDE')
        
        if len(train_df['OC']) > 1:
            kde_train = gaussian_kde(train_df['OC'].values)
            ax4.plot(oc_range, kde_train(oc_range), 'b-', label='Training KDE')
        
        dist_params = list(best_params.values())
        if best_dist == beta:
            data_min, data_max = full_df['OC'].min(), full_df['OC'].max()
            scaled_range = (oc_range - data_min) / (data_max - data_min)
            dist_pdf = best_dist.pdf(scaled_range, *dist_params[:2], loc=0, scale=1) / (data_max - data_min)
        else:
            dist_pdf = best_dist.pdf(oc_range, *dist_params)
        ax4.plot(oc_range, dist_pdf, 'g--', label=f'{dist_name} Fit')
        
        ax4.set_title(f'Gaussian KDE vs {dist_name} Fold {i+1}')
        ax4.set_xlabel('OC Value')
        ax4.set_ylabel('Density')
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'visualizations_{timestamp}_fold{i+1}.png'), bbox_inches='tight')
        plt.close()

    # Additional visualization for the training fold
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    bavaria.boundary.plot(ax=ax1, color='black', linewidth=1)
    ax1.scatter(training_fold_df['GPS_LONG'], training_fold_df['GPS_LAT'], c='purple', label='Training Fold', alpha=0.5, s=10)
    ax1.set_title('Spatial Distribution of Training Fold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True)
    ax1.legend()

    ax2.hist(training_fold_df['OC'].values, bins=30, density=True, alpha=0.5, color='purple', label='Training Fold')
    ax2.set_title('OC Distribution of Training Fold')
    ax2.set_xlabel('OC Value')
    ax2.set_ylabel('Density')
    ax2.grid(True)
    ax2.legend()

    oc_range = np.linspace(max(0, training_fold_df['OC'].min()), training_fold_df['OC'].max(), 100)
    if len(training_fold_df['OC']) > 1:
        kde_train_fold = gaussian_kde(training_fold_df['OC'].values)
        ax3.plot(oc_range, kde_train_fold(oc_range), 'purple', label='Training Fold KDE')
    dist_params = list(best_params.values())
    if best_dist == beta:
        data_min, data_max = full_df['OC'].min(), full_df['OC'].max()
        scaled_range = (oc_range - data_min) / (data_max - data_min)
        dist_pdf = best_dist.pdf(scaled_range, *dist_params[:2], loc=0, scale=1) / (data_max - data_min)
    else:
        dist_pdf = best_dist.pdf(oc_range, *dist_params)
    ax3.plot(oc_range, dist_pdf, 'g--', label=f'{dist_name} Fit')
    ax3.set_title(f'Gaussian KDE vs {dist_name} for Training Fold')
    ax3.set_xlabel('OC Value')
    ax3.set_ylabel('Density')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'visualizations_{timestamp}_training_fold.png'), bbox_inches='tight')
    plt.close()

def create_5fold_datasets(df, best_dist, best_params, dist_name, target_val_ratio=0.08, output_dir='output', device='cpu', distance_threshold=0.5):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()
    df[['OC', 'GPS_LAT', 'GPS_LONG']] = df[['OC', 'GPS_LAT', 'GPS_LONG']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])
    
    total_samples = len(df)
    target_size = int(total_samples * target_val_ratio)
    initial_ratio = target_val_ratio + 0.02
    max_ratio = 0.995
    n_folds = 5

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fold_dfs = []
    used_indices = set()
    
    print(f"Using distance threshold: {distance_threshold} km")
    
    for fold in range(n_folds):
        available_df = df.loc[~df.index.isin(used_indices)]
        if len(available_df) < target_size:
            print(f"Not enough samples for fold {fold+1}. Stopping at {fold} folds.")
            break
        
        while initial_ratio <= max_ratio:
            subset_size = int(len(available_df) * initial_ratio)
            
            oc_min, oc_max = df['OC'].min(), df['OC'].max()
            dist_params = list(best_params.values())
            dist_samples = best_dist.rvs(*dist_params, size=subset_size * 2)
            if best_dist == beta:
                dist_samples = oc_min + (oc_max - oc_min) * dist_samples
            else:
                dist_samples = oc_min + (oc_max - oc_min) * (dist_samples - np.min(dist_samples)) / (np.max(dist_samples) - np.min(dist_samples))
            
            oc_values = available_df['OC'].values
            if best_dist == beta:
                scaled_oc = (oc_values - oc_min) / (oc_max - oc_min)
                weights = best_dist.pdf(scaled_oc, *dist_params[:2], loc=0, scale=1)
            else:
                weights = best_dist.pdf(oc_values, *dist_params)
            weights = weights / weights.sum()
            
            subset_indices = np.random.choice(available_df.index, size=subset_size, replace=False, p=weights)
            subset_indices = [idx for idx in subset_indices if idx not in used_indices]
            if len(subset_indices) < subset_size:
                subset_size = len(subset_indices)
                if subset_size < target_size:
                    print(f"Fold {fold+1}: Not enough unique indices available. Reducing subset size to {subset_size}.")
            
            subset_df = available_df.loc[subset_indices]
            remaining_df = available_df.drop(subset_indices)
            
            subset_df.to_parquet(output_dir / f'initial_subset_df_fold{fold+1}.parquet')
            remaining_df.to_parquet(output_dir / f'initial_remaining_df_fold{fold+1}.parquet')
            
            min_distances = compute_min_distances(subset_df, remaining_df)
            validation_df = subset_df[min_distances >= distance_threshold]
            points_to_flip = subset_df[min_distances < distance_threshold]
            
            if not points_to_flip.empty:
                print(f"Fold {fold+1}: Flipping {len(points_to_flip)} points (distance < {distance_threshold} km)")
                training_df = pd.concat([remaining_df, points_to_flip])
            else:
                training_df = remaining_df
            
            val_size = len(validation_df)
            val_ratio = val_size / total_samples
            if val_size >= target_size:
                break
            else:
                print(f"Fold {fold+1}: Validation set size {val_ratio*100:.2f}% < {target_val_ratio*100}%. Increasing ratio.")
                initial_ratio += 0.02
                if initial_ratio > max_ratio:
                    print(f"Fold {fold+1}: Max ratio {max_ratio*100}% reached. Using current validation set.")
                    break
        
        used_indices.update(validation_df.index)
        
        oc_bins = pd.qcut(df['OC'], q=10, duplicates='drop')
        train_bins = pd.cut(training_df['OC'], bins=oc_bins.cat.categories)
        bin_counts = train_bins.value_counts()
        empty_bins = bin_counts[bin_counts == 0].index
        
        if not empty_bins.empty:
            print(f"Fold {fold+1}: Found {len(empty_bins)} empty bins in training set. Flipping points to fill them.")
            for bin_label in empty_bins:
                val_in_bin = validation_df[pd.cut(validation_df['OC'], bins=oc_bins.cat.categories) == bin_label]
                if val_in_bin.empty:
                    continue
                min_distances = []
                for idx, row in val_in_bin.iterrows():
                    other_df = validation_df.drop(idx)
                    if not other_df.empty:
                        distances = vectorized_haversine(
                            np.array([row['GPS_LONG']]),
                            np.array([row['GPS_LAT']]),
                            other_df['GPS_LONG'].values,
                            other_df['GPS_LAT'].values,
                            device
                        )
                        min_distance = np.min(distances)
                        min_distances.append((idx, min_distance))
                    else:
                        min_distances.append((idx, np.inf))
                if min_distances:
                    flip_idx, max_min_distance = max(min_distances, key=lambda x: x[1])
                    if max_min_distance > distance_threshold:
                        flip_point = validation_df.loc[[flip_idx]]
                        training_df = pd.concat([training_df, flip_point])
                        validation_df = validation_df.drop(flip_idx)
                        used_indices.remove(flip_idx)
                        print(f"Fold {fold+1}: Flipped 1 point from validation to training for bin {bin_label}, min_distance={max_min_distance:.2f} km")
                    else:
                        print(f"Fold {fold+1}: No isolated point found for bin {bin_label} (max min_distance={max_min_distance:.2f} km <= {distance_threshold} km), skipping flip")
                else:
                    print(f"Fold {fold+1}: No points in validation for bin {bin_label}")
        
        min_distances = compute_min_distances(validation_df, training_df, device)
        violating_points = validation_df[min_distances < distance_threshold]
        if not violating_points.empty:
            print(f"Fold {fold+1}: Reflipping {len(violating_points)} points with distances < {distance_threshold} km to training set")
            training_df = pd.concat([training_df, violating_points])
            for idx in violating_points.index:
                used_indices.remove(idx)
            validation_df = validation_df[min_distances >= distance_threshold]
        
        if any(idx in used_indices - set(validation_df.index) for idx in validation_df.index):
            raise ValueError(f"Fold {fold+1}: Validation set contains indices already used in previous folds.")
        
        validation_df.to_parquet(output_dir / f'final_validation_df_fold{fold+1}.parquet')
        training_df.to_parquet(output_dir / f'final_training_df_fold{fold+1}.parquet')
        
        fold_dfs.append((validation_df, training_df))
        used_indices.update(validation_df.index)
        
        final_min_distance = compute_min_distances(validation_df, training_df, device).min()
        
        print(f'Fold {fold+1}:')
        print(f'Full dataset size: {total_samples}')
        print(f'Validation set: {len(validation_df)} ({len(validation_df)/total_samples*100:.2f}%)')
        print(f'Training set: {len(training_df)}')
        print(f'Minimum distance (km): {final_min_distance:.2f}')
        print(f'Validation OC distribution matched to {dist_name} with parameters: {best_params}')
    
    # **New Step: Create the Training Fold**
    all_val_indices = set().union(*[set(val_df.index) for val_df, _ in fold_dfs])
    training_fold_df = df.loc[~df.index.isin(all_val_indices)]
    training_fold_df.to_parquet(output_dir / 'final_training_fold.parquet')
    print(f"Training Fold created with {len(training_fold_df)} samples ({len(training_fold_df)/total_samples*100:.2f}%)")
    
    create_visualizations(fold_dfs, training_fold_df, df, best_dist, best_params, dist_name, output_dir)
    
    total_val_ratio = sum(len(val_df) for val_df, _ in fold_dfs) / total_samples
    print(f"Total validation data across all folds: {total_val_ratio*100:.2f}%")
    
    all_val_indices_list = [set(val_df.index) for val_df, _ in fold_dfs]
    for i in range(len(all_val_indices_list)):
        for j in range(i + 1, len(all_val_indices_list)):
            overlap = all_val_indices_list[i].intersection(all_val_indices_list[j])
            if overlap:
                raise ValueError(f"Overlap detected between folds {i+1} and {j+1}: {len(overlap)} entries")
    print("No overlapping validation entries detected across folds.")
    
    return fold_dfs, training_fold_df

def main():
    parser = argparse.ArgumentParser(description='Create 5-fold datasets and a training fold with exponential family distribution')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio per fold')
    parser.add_argument('--distance-threshold', type=float, default=0.5, help='Distance threshold in km')  # **Added argument**
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
        
        oc_values = df['OC'].dropna().values
        best_dist, best_params, dist_name = fit_exponential_family(oc_values)
        
        fold_dfs, training_fold_df = create_5fold_datasets(
            df,
            best_dist=best_dist,
            best_params=best_params,
            dist_name=dist_name,
            target_val_ratio=args.target_val_ratio,
            output_dir=args.output_dir,
            device=device,
            distance_threshold=args.distance_threshold  # **Pass the configurable threshold**
        )
    except Exception as e:
        print(f"Error processing: {e}")

if __name__ == "__main__":
    main()