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
import logging

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

def compute_min_distances(test_df, train_df, device='cpu'):
    if train_df.empty or test_df.empty:
        return np.full(len(test_df), np.inf)

    test_lons = test_df['GPS_LONG'].values.astype(float)
    test_lats = test_df['GPS_LAT'].values.astype(float)
    train_lons = train_df['GPS_LONG'].values.astype(float)
    train_lats = train_df['GPS_LAT'].values.astype(float)

    test_lons = np.expand_dims(test_lons, axis=1)
    test_lats = np.expand_dims(test_lats, axis=1)

    distances = vectorized_haversine(test_lons, test_lats, train_lons, train_lats, device)
    min_distances = np.min(distances, axis=1)

    invalid = np.isnan(test_lons.flatten()) | np.isnan(test_lats.flatten()) | np.isnan(min_distances)
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

def create_optimized_subset(df, best_dist, best_params, dist_name, target_test_ratio=0.08, output_dir='output', device='cpu', distance_threshold=1.4):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()
    df[['OC', 'GPS_LAT', 'GPS_LONG']] = df[['OC', 'GPS_LAT', 'GPS_LONG']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['OC', 'GPS_LAT', 'GPS_LONG'])

    total_samples = len(df)
    target_size = int(total_samples * target_test_ratio)
    initial_ratio = target_test_ratio + 0.02
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
        test_df = subset_df[min_distances >= distance_threshold]
        points_to_flip = subset_df[min_distances < distance_threshold]

        if not points_to_flip.empty:
            training_df = pd.concat([remaining_df, points_to_flip])
        else:
            training_df = remaining_df

        test_size = len(test_df)
        test_ratio = test_size / total_samples
        if test_size >= target_size:
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
            test_in_bin = test_df[pd.cut(test_df['OC'], bins=oc_bins.cat.categories) == bin_label]
            if not test_in_bin.empty:
                flip_point = test_in_bin.sample(n=1)
                training_df = pd.concat([training_df, flip_point])
                test_df = test_df.drop(flip_point.index)

    test_df.to_parquet(output_dir / 'final_test_df.parquet')
    training_df.to_parquet(output_dir / 'final_training_df.parquet')

    return test_df, training_df

def create_test_train_sets(df=None, output_dir='output', target_test_ratio=0.08, use_gpu=False, distance_threshold=1.4):
    """
    Create optimized test and training sets using the predefined Inverse Gamma distribution.

    Parameters:
    -----------
    df : pandas.DataFrame, optional (default=None)
        Input DataFrame to split. If None, loads data using filter_dataframe.
    output_dir : str, optional (default='output')
        Directory to save the output Parquet files.
    target_test_ratio : float, optional (default=0.08)
        Target ratio of test set size to total samples.
    use_gpu : bool, optional (default=False)
        Whether to use GPU for computations.
    distance_threshold : float, optional (default=1.4)
        Minimum distance threshold (in km) for test points.

    Returns:
    --------
    test_df : pandas.DataFrame
        DataFrame containing the held-out test set.
    training_df : pandas.DataFrame
        DataFrame containing the training set.
    """
    # Set up logging
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_file = output_dir / 'split_stats.log'
    stats_file = output_dir / 'split_stats.txt'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

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

        test_df, training_df = create_optimized_subset(
            df,
            best_dist=best_dist,
            best_params=best_params,
            dist_name=dist_name,
            target_test_ratio=target_test_ratio,
            output_dir=output_dir,
            device=device,
            distance_threshold=distance_threshold
        )

        # Compute minimum distances between test and training sets
        min_distances = compute_min_distances(test_df, training_df, device)
        min_distance_stats = {
            'mean': np.mean(min_distances[~np.isinf(min_distances)]),
            'median': np.median(min_distances[~np.isinf(min_distances)]),
            'min': np.min(min_distances[~np.isinf(min_distances)]),
            'max': np.max(min_distances[~np.isinf(min_distances)]),
            'std': np.std(min_distances[~np.isinf(min_distances)])
        }

        # Compute OC distribution statistics
        test_oc = test_df['OC'].dropna().values
        train_oc = training_df['OC'].dropna().values
        oc_stats = {
            'test_mean': np.mean(test_oc),
            'test_std': np.std(test_oc),
            'train_mean': np.mean(train_oc),
            'train_std': np.std(train_oc)
        }

        # Perform Kolmogorov-Smirnov test to compare OC distributions
        ks_stat, ks_pvalue = stats.ks_2samp(test_oc, train_oc)

        # Prepare statistics output
        stats_output = [
            "Test and Training Set Split Statistics",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)",
            f"Training set size: {len(training_df)} ({len(training_df)/len(df)*100:.2f}%)",
            "Minimum Distance Statistics (km):",
        ]
        for stat_name, value in min_distance_stats.items():
            stats_output.append(f"  {stat_name.capitalize()}: {value:.4f}")
        stats_output.extend([
            "OC Distribution Statistics:",
            f"  Test Mean: {oc_stats['test_mean']:.4f}",
            f"  Test Std: {oc_stats['test_std']:.4f}",
            f"  Training Mean: {oc_stats['train_mean']:.4f}",
            f"  Training Std: {oc_stats['train_std']:.4f}",
            "Kolmogorov-Smirnov Test:",
            f"  KS Statistic: {ks_stat:.4f}",
            f"  P-value: {ks_pvalue:.4f}",
            f"Distribution Fit: {dist_name}",
            f"Distribution Parameters: {best_params}"
        ])

        # Write statistics to text file
        with open(stats_file, 'w') as f:
            f.write('\n'.join(stats_output))

        # Log statistics
        for line in stats_output:
            logger.info(line)

        return test_df, training_df, min_distance_stats
    except Exception as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        with open(stats_file, 'w') as f:
            f.write(error_msg)
        print(error_msg)
        return None, None, min_distance_stats



# Function to create balanced dataset via per-bin oversampling.
# When with_test=True a small per-bin sample is also carved out as a held-out
# slice; in the current SGT pipeline this branch is unused (test split is
# done upstream by create_test_train_sets), so train.py calls this with
# with_test=False to get the rebalanced training set only.
def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, with_test=True):
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)

    training_dfs = []
    if with_test:
        test_indices = []
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) >= 4:
                test_samples = bin_data.sample(n=min(13, len(bin_data)))
                test_indices.extend(test_samples.index)
                train_samples = bin_data.drop(test_samples.index)
                if len(train_samples) > 0:
                    if len(train_samples) < min_samples:
                        resampled = train_samples.sample(n=min_samples, replace=True)
                        training_dfs.append(resampled)
                    else:
                        training_dfs.append(train_samples)
        if not training_dfs or not test_indices:
            raise ValueError("No training or test data available after binning")
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        test_df = df.loc[test_indices].drop('bin', axis=1)
        print('Size of the training set:   ' ,len(training_df))
        print('Size of the test set:   ' ,len(test_df))
        return training_df, test_df
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        if not training_dfs:
            raise ValueError("No training data available after binning")
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        print('Size of the training set:   ' ,len(training_df))
        return training_df, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized test set creation with exponential family distribution')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-test-ratio', type=float, default=0.08, help='Target test ratio')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for test points')
    args = parser.parse_args()

    test_df, training_df , min_distance_stats = create_test_train_sets(
        df=None,
        output_dir=args.output_dir,
        target_test_ratio=args.target_test_ratio,
        use_gpu=args.use_gpu,
        distance_threshold=args.distance_threshold
    )
