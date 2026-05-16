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
        
        # Compute minimum distances between validation and training sets
        min_distances = compute_min_distances(validation_df, training_df, device)
        min_distance_stats = {
            'mean': np.mean(min_distances[~np.isinf(min_distances)]),
            'median': np.median(min_distances[~np.isinf(min_distances)]),
            'min': np.min(min_distances[~np.isinf(min_distances)]),
            'max': np.max(min_distances[~np.isinf(min_distances)]),
            'std': np.std(min_distances[~np.isinf(min_distances)])
        }
        
        # Compute OC distribution statistics
        val_oc = validation_df['OC'].dropna().values
        train_oc = training_df['OC'].dropna().values
        oc_stats = {
            'val_mean': np.mean(val_oc),
            'val_std': np.std(val_oc),
            'train_mean': np.mean(train_oc),
            'train_std': np.std(train_oc)
        }
        
        # Perform Kolmogorov-Smirnov test to compare OC distributions
        ks_stat, ks_pvalue = stats.ks_2samp(val_oc, train_oc)
        
        # Prepare statistics output
        stats_output = [
            "Validation and Training Set Split Statistics",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Validation set size: {len(validation_df)} ({len(validation_df)/len(df)*100:.2f}%)",
            f"Training set size: {len(training_df)} ({len(training_df)/len(df)*100:.2f}%)",
            "Minimum Distance Statistics (km):",
        ]
        for stat_name, value in min_distance_stats.items():
            stats_output.append(f"  {stat_name.capitalize()}: {value:.4f}")
        stats_output.extend([
            "OC Distribution Statistics:",
            f"  Validation Mean: {oc_stats['val_mean']:.4f}",
            f"  Validation Std: {oc_stats['val_std']:.4f}",
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
        
        return validation_df, training_df, min_distance_stats
    except Exception as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        with open(stats_file, 'w') as f:
            f.write(error_msg)
        print(error_msg)
        return None, None, min_distance_stats



# ---------------------------------------------------------------------------
# Stratified-spatial split (anti-overfit option 1 + 3 from the bundle)
# ---------------------------------------------------------------------------
# The legacy create_validation_train_sets picks val points from an inverse-
# gamma-weighted draw, which can yield val sets whose OC distribution is
# noticeably shifted from train (KS p-value < 0.05 in some folds). Combined
# with a small distance_threshold, that gives the model a "test set that
# looks easier than reality" failure mode.
#
# create_stratified_validation_train_sets fixes both:
#   • OC distribution match: bin by OC quantiles, sample val points
#     proportionally from EVERY bin so train/val OC histograms match.
#   • Distribution gate: after splitting, run scipy.stats.ks_2samp on the
#     OC arrays. If p < ks_pvalue_min, retry up to max_retries times with
#     a fresh random seed. Returns the LAST split if all retries fail
#     (logged warning).
#   • Spatial buffer: same haversine + distance_threshold check as the
#     legacy function.
# ---------------------------------------------------------------------------
def _kde_mode(values, lo=0.0, hi=60.0, n=300, bw=0.15):
    """OC mode via Gaussian KDE on a fixed grid. Fixed-bw for cross-split
    comparability (Scott would differ between train and test by ~10%)."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float('nan')
    k = gaussian_kde(v, bw_method=bw)
    g = np.linspace(lo, hi, n)
    return float(g[int(np.argmax(k(g)))])


def create_stratified_validation_train_sets(df=None,
                                             output_dir='output',
                                             target_val_ratio=0.08,
                                             use_gpu=False,
                                             distance_threshold=1.2,
                                             n_bins=10,
                                             ks_pvalue_min=0.0,
                                             max_retries=20,
                                             seed=None,
                                             mean_tol=None,
                                             std_tol=None,
                                             mode_tol=None,
                                             test_oc_max=None):
    """Stratified-spatial train/val split with optional KS-pvalue gate.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe (LUCAS OC + GPS_LONG + GPS_LAT). If None, filter
        from config.
    target_val_ratio : float
        Fraction of total samples to allocate to val (e.g. 0.08).
    distance_threshold : float
        Minimum km between any val point and its nearest train point.
    n_bins : int
        Number of OC quantile bins for stratification.
    ks_pvalue_min : float
        Reject splits whose ks_2samp(train_oc, val_oc).pvalue < this
        threshold. 0 disables the gate.
    max_retries : int
        How many resamples to attempt before giving up.
    seed : int or None
        Base seed; each retry uses seed + retry_idx.
    mean_tol : float or None
        Max allowed |mean_train - mean_test| / mean_global. None disables.
        0.02 means "test mean within 2% of train mean".
    std_tol : float or None
        Max allowed |std_train - std_test| / std_global. None disables.
        0.05 means "test std within 5% of train std".
    mode_tol : float or None
        Max allowed |mode_train - mode_test| in raw OC units (g/kg). None
        disables. 1.0 means "test KDE-mode within 1 g/kg of train mode".
        Modes are computed via a fixed-bandwidth KDE for cross-split
        comparability.
    test_oc_max : float or None
        Upper OC cap for the test set. When set (e.g. 50.0):
          * points with OC > test_oc_max never appear in test — they go
            straight to train (so the model still sees them during
            training but isn't graded on outlier predictions);
          * the stratified split operates ONLY on the OC ≤ test_oc_max
            subset, giving test a flatter (per-OC-bin uniform) shape
            within that range than the natural distribution — i.e. test
            has more variance within the comparable range, while keeping
            the same KDE-mode (the modal bin is still in-range);
          * mean_tol / std_tol are then evaluated against the
            train-subset-in-range, not all of train, so the comparison is
            apples-to-apples;
          * KS / mode-tol still compare train (all of it) vs test, which
            is the regime the production model is graded in.
        None (default) disables — split runs on the full filtered df.

    When mean/std/mode tolerances are given, the retry loop scores each
    candidate split by a composite metric (sum of normalized squared
    diffs). It accepts the first split that meets ALL gates (KS + the
    three tolerances), otherwise it returns the lowest-score split across
    all retries — so behaviour degrades gracefully on hard cases instead
    of throwing.

    Returns
    -------
    (val_df, train_df, min_distance_stats)
    """
    logger = logging.getLogger(__name__)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if df is None:
        df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    df = df.copy().reset_index(drop=True)

    # test_oc_max: pull OC>cap rows out of the split pool, hold them aside,
    # and reattach to train AFTER the split. This way test never contains
    # outliers but the model still trains on them.
    if test_oc_max is not None:
        outlier_df = df[df['OC'] > test_oc_max].copy()
        df = df[df['OC'] <= test_oc_max].reset_index(drop=True)
        if len(outlier_df) > 0:
            print(f"test_oc_max={test_oc_max}: {len(outlier_df)} outlier rows "
                  f"(OC>{test_oc_max}) reserved for train; stratified split "
                  f"runs on the remaining {len(df)}.")
    else:
        outlier_df = None

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    total_samples = len(df)
    target_val_size = max(1, int(round(total_samples * target_val_ratio)))

    # Stratify by OC quantiles (use n_bins or fewer if ties collapse them)
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df = df.assign(_bin=bins.values)
    unique_bins = sorted(df['_bin'].dropna().unique().astype(int))
    per_bin_target = max(1, target_val_size // max(1, len(unique_bins)))

    # Three-stage stratified-spatial split (mirrors the legacy adaptive loop):
    #   1. Oversample candidates per OC bin (pool ≫ target so the spatial
    #      buffer leaves enough survivors).
    #   2. Apply spatial buffer; candidates that fail the buffer go BACK to
    #      train (no information lost).
    #   3. From SURVIVORS, draw `per_bin_target` per bin (so the final test
    #      OC histogram matches the main distribution by construction).
    #      Excess survivors (those not picked) also go back to train.
    # Oversample factor grows each retry so a single dense pod can still hit
    # the target eventually. Behaviour matches the legacy create_validation_
    # train_sets adaptive loop except it uses OC-quantile stratification
    # instead of inverse-gamma weighting.
    # Composite scoring across retries.
    #   - KS p ≥ ks_pvalue_min                                   (hard gate)
    #   - |mean_train - mean_test| / mean_global ≤ mean_tol      (optional)
    #   - |std_train  - std_test|  / std_global  ≤ std_tol       (optional)
    #   - |mode_train - mode_test|               ≤ mode_tol      (optional, g/kg)
    # Accept-and-break the first retry that satisfies ALL active gates.
    # Else, return the lowest-composite-score split across all retries
    # (graceful degradation). The composite score is the sum of normalized
    # squared diffs, so the "least bad" split wins when no split passes.
    global_mean = float(df['OC'].mean())
    global_std  = float(df['OC'].std())
    global_mode = _kde_mode(df['OC'].values)
    best_result = None
    best_score = float('inf')
    for retry in range(max(1, max_retries)):
        rng = np.random.default_rng((seed or 0) + retry)
        # Grow oversample with each retry: 4× target, 5×, 6×, ... Capped at
        # 90% of total_samples so train always has at least 10% to learn from.
        oversample = 4 + retry
        candidate_pool = min(int(total_samples * 0.9),
                             target_val_size * oversample)
        per_bin_pool = max(per_bin_target,
                           candidate_pool // max(1, len(unique_bins)))

        # --- stage 1: stratified candidate pool --------------------------
        candidate_val_idx = []
        for b in unique_bins:
            bin_df = df[df['_bin'] == b]
            if len(bin_df) == 0:
                continue
            take = min(per_bin_pool, len(bin_df))
            picked = rng.choice(bin_df.index.values, size=take, replace=False)
            candidate_val_idx.extend(picked.tolist())
        candidate_val_idx = np.array(candidate_val_idx, dtype=np.int64)

        candidate_val = df.loc[candidate_val_idx].reset_index(drop=True)
        candidate_train = df.drop(index=candidate_val_idx).reset_index(drop=True)

        # --- stage 2: spatial buffer -------------------------------------
        # Rejected candidates rejoin the train pool later (no data thrown away).
        min_dists = compute_min_distances(candidate_val, candidate_train, device=device)
        buffer_mask = min_dists >= distance_threshold
        survivor_idx = candidate_val_idx[buffer_mask]
        survivor_dists = min_dists[buffer_mask]

        if len(survivor_idx) < target_val_size:
            # Not enough survivors with this oversample; let the retry loop
            # grow it and try again. If we're already on the last retry,
            # fall through and use what we have.
            if retry < max_retries - 1:
                continue

        # --- stage 3: stratified pick to hit target_val_size --------------
        # Within survivors, draw `per_bin_target` per bin, preferring largest
        # min_dist (cleanest spatial separation). Excess survivors + rejected
        # candidates both flow back to train.
        survivor_df = df.loc[survivor_idx].copy()
        survivor_df['_min_dist'] = survivor_dists
        final_val_idx_list = []
        final_val_dists_list = []
        for b in unique_bins:
            bin_survivors = survivor_df[survivor_df['_bin'] == b]
            bin_survivors = bin_survivors.sort_values('_min_dist', ascending=False)
            take = min(per_bin_target, len(bin_survivors))
            final_val_idx_list.extend(bin_survivors.index[:take].tolist())
            final_val_dists_list.extend(bin_survivors['_min_dist'].values[:take].tolist())

        final_val_idx = np.array(final_val_idx_list, dtype=np.int64)
        # If after per-bin truncation we're still below target (some bins
        # have very few survivors), top up by taking the next-best survivors
        # globally, regardless of bin.
        if len(final_val_idx) < target_val_size:
            remaining = survivor_df.drop(index=final_val_idx, errors='ignore')
            remaining = remaining.sort_values('_min_dist', ascending=False)
            need = target_val_size - len(final_val_idx)
            extras = remaining.index[:need].values
            extra_dists = remaining['_min_dist'].values[:need].tolist()
            final_val_idx = np.concatenate([final_val_idx, extras])
            final_val_dists_list.extend(extra_dists)

        val_df = df.loc[final_val_idx].reset_index(drop=True)
        train_df = df.drop(index=final_val_idx).reset_index(drop=True)

        # Drop helper _bin column from outputs
        val_df = val_df.drop(columns=['_bin'], errors='ignore')
        train_df = train_df.drop(columns=['_bin'], errors='ignore')

        if len(val_df) < 5 or len(train_df) < 5:
            continue

        # KS + mean/std/mode gates.
        # If test_oc_max is active, evaluate mean/std on the comparable range
        # of train (train rows with OC ≤ test_oc_max) — otherwise test is
        # unfairly penalized for not containing outliers it was forbidden
        # from sampling. KS / mode still compare against full train.
        ks_stat, ks_pvalue = stats.ks_2samp(train_df['OC'].values, val_df['OC'].values)
        result_dists = np.array(final_val_dists_list)
        if test_oc_max is not None:
            train_in_range = train_df.loc[train_df['OC'] <= test_oc_max, 'OC']
        else:
            train_in_range = train_df['OC']
        mean_diff = abs(train_in_range.mean() - val_df['OC'].mean()) / max(global_mean, 1e-9)
        std_diff  = abs(train_in_range.std()  - val_df['OC'].std())  / max(global_std,  1e-9)
        if mode_tol is not None:
            mode_train = _kde_mode(train_df['OC'].values)
            mode_val   = _kde_mode(val_df['OC'].values)
            mode_diff  = abs(mode_train - mode_val)
        else:
            mode_train = mode_val = mode_diff = float('nan')
        # Composite score (lower=better). Normalize each diff by its tol so
        # the three contribute comparably. If a tol is None, term is 0.
        score = 0.0
        if mean_tol is not None: score += (mean_diff / max(mean_tol, 1e-9)) ** 2
        if std_tol  is not None: score += (std_diff  / max(std_tol,  1e-9)) ** 2
        if mode_tol is not None: score += (mode_diff / max(mode_tol, 1e-9)) ** 2
        # Hard-gate: all active tolerances must pass
        passed = ks_pvalue >= ks_pvalue_min
        if mean_tol is not None: passed = passed and mean_diff <= mean_tol
        if std_tol  is not None: passed = passed and std_diff  <= std_tol
        if mode_tol is not None: passed = passed and mode_diff <= mode_tol
        if passed:
            best_result = (val_df, train_df, ks_stat, ks_pvalue, result_dists,
                           mean_diff, std_diff, mode_diff, mode_train, mode_val)
            break
        if score < best_score:
            best_score = score
            best_result = (val_df, train_df, ks_stat, ks_pvalue, result_dists,
                           mean_diff, std_diff, mode_diff, mode_train, mode_val)

    if best_result is None:
        raise RuntimeError("Stratified split failed to produce any valid (val, train) pair.")

    (val_df, train_df, ks_stat, ks_pvalue, val_min_dists,
     mean_diff, std_diff, mode_diff, mode_train, mode_val) = best_result
    # Reattach the OC>test_oc_max outliers to train (they were held aside
    # before the split). Train ends up with the natural bulk + all outliers;
    # test contains only OC ≤ test_oc_max points.
    if outlier_df is not None and len(outlier_df) > 0:
        # Drop the helper _bin column from outlier_df if present (it isn't)
        outlier_df_clean = outlier_df.drop(columns=['_bin'], errors='ignore')
        n_train_pre = len(train_df)
        train_df = pd.concat([train_df, outlier_df_clean], ignore_index=True)
        print(f"Reattached {len(outlier_df)} OC>{test_oc_max} outlier rows "
              f"to train: {n_train_pre} → {len(train_df)}.")
    if ks_pvalue < ks_pvalue_min:
        logger.warning(
            f"Stratified split: best KS p-value {ks_pvalue:.4f} < gate {ks_pvalue_min}; "
            f"using best of {max_retries} attempts anyway.")
    # Log all match metrics so the user can see how tight the final split is.
    # When test_oc_max is active, mean/std are evaluated on train[in-range]
    # vs test (apples-to-apples); mode is computed on full train vs test.
    cmp_label = f" [train OC≤{test_oc_max} vs test]" if test_oc_max is not None else ""
    msg_bits = [f"KS p={ks_pvalue:.4f} (gate {ks_pvalue_min})",
                f"mean_diff={mean_diff*100:.2f}%{cmp_label}"]
    if mean_tol is not None: msg_bits[-1] += f" (tol {mean_tol*100:.1f}%)"
    msg_bits.append(f"std_diff={std_diff*100:.2f}%{cmp_label}")
    if std_tol is not None:  msg_bits[-1] += f" (tol {std_tol*100:.1f}%)"
    if mode_tol is not None:
        msg_bits.append(f"mode train={mode_train:.2f} test={mode_val:.2f} "
                        f"diff={mode_diff:.3f}g/kg (tol {mode_tol:.2f})")
    print(f"Stratified split match metrics: " + " | ".join(msg_bits))

    valid_dists = val_min_dists[~np.isinf(val_min_dists) & ~np.isnan(val_min_dists)]
    min_distance_stats = {
        'mean':   float(np.mean(valid_dists))   if len(valid_dists) else float('nan'),
        'median': float(np.median(valid_dists)) if len(valid_dists) else float('nan'),
        'min':    float(np.min(valid_dists))    if len(valid_dists) else float('nan'),
        'max':    float(np.max(valid_dists))    if len(valid_dists) else float('nan'),
        'std':    float(np.std(valid_dists))    if len(valid_dists) else float('nan'),
    }

    # pyarrow refuses to write object-dtype columns with mixed Python types
    # (e.g. POINTID can be int in some rows, str in others after the
    # concat/loc indexing). Coerce all object-dtype columns to str just
    # for the parquet dump; the in-memory dataframes returned to the caller
    # keep their original dtypes.
    def _safe_to_parquet(df, path):
        df_out = df.copy()
        for col in df_out.columns:
            if df_out[col].dtype == 'object':
                df_out[col] = df_out[col].astype(str)
        df_out.to_parquet(path)

    _safe_to_parquet(val_df,   Path(output_dir) / 'final_validation_df.parquet')
    _safe_to_parquet(train_df, Path(output_dir) / 'final_training_df.parquet')
    print(f"Stratified split: val n={len(val_df)} train n={len(train_df)} "
          f"KS stat={ks_stat:.4f} p={ks_pvalue:.4f}  bins={len(unique_bins)}")
    return val_df, train_df, min_distance_stats


# ---------------------------------------------------------------------------
# Spatial K-fold splits (anti-overfit option 2 from the bundle)
# ---------------------------------------------------------------------------
# Divides Bavaria's coordinate range into K spatial blocks (latitude deciles
# by default). Each fold's val is one block; train is every other block
# minus a distance_threshold buffer at the block edges. Returns a list of
# (val_df, train_df, stats) tuples — one per fold — for the caller to loop
# over instead of independent random splits.
#
# Use n_folds=5 for the classic 80/20 spatial CV, n_folds=10 for tighter
# 90/10 with more folds.
# ---------------------------------------------------------------------------
def create_spatial_kfold_splits(df=None,
                                 output_dir='output',
                                 n_folds=5,
                                 use_gpu=False,
                                 distance_threshold=1.2,
                                 ks_pvalue_min=0.0,
                                 seed=42):
    """Build n_folds spatial-block train/val splits.

    Returns a list of length n_folds; each element is (val_df, train_df,
    min_distance_stats, fold_meta) where fold_meta is a dict with the
    fold index, latitude block edges, and KS p-value of the OC split.
    """
    logger = logging.getLogger(__name__)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if df is None:
        df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    df = df.copy().reset_index(drop=True)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    # Latitude-decile blocks (5/10/etc. equal-count slices in lat). This is
    # the simplest spatial blocking that preserves coverage of the full
    # longitude range in every fold. For multi-dimensional blocking
    # (lat × lon grid) the call site can shuffle df by some 2-D key first.
    lat_edges = np.quantile(df['GPS_LAT'].values, np.linspace(0, 1, n_folds + 1))
    folds = []
    for fold_idx in range(n_folds):
        lo, hi = lat_edges[fold_idx], lat_edges[fold_idx + 1]
        # Half-open at the lower edge, closed at the upper edge except for
        # the topmost fold which is closed at both ends.
        if fold_idx == n_folds - 1:
            fold_mask = (df['GPS_LAT'] >= lo) & (df['GPS_LAT'] <= hi)
        else:
            fold_mask = (df['GPS_LAT'] >= lo) & (df['GPS_LAT'] < hi)
        val_candidate = df[fold_mask].reset_index(drop=True)
        train_candidate = df[~fold_mask].reset_index(drop=True)
        if len(val_candidate) < 5 or len(train_candidate) < 5:
            logger.warning(f"Fold {fold_idx}: too few samples (val={len(val_candidate)}, "
                           f"train={len(train_candidate)}); skipping.")
            continue
        min_dists = compute_min_distances(val_candidate, train_candidate, device=device)
        buffer_mask = min_dists >= distance_threshold
        val_df = val_candidate[buffer_mask].reset_index(drop=True)
        train_df = pd.concat(
            [train_candidate, val_candidate[~buffer_mask]],
            ignore_index=True,
        )
        if len(val_df) < 5:
            logger.warning(f"Fold {fold_idx}: val collapsed to <5 after buffer; skipping.")
            continue
        ks_stat, ks_pvalue = stats.ks_2samp(train_df['OC'].values, val_df['OC'].values)
        if ks_pvalue_min > 0 and ks_pvalue < ks_pvalue_min:
            logger.warning(f"Fold {fold_idx}: KS p-value {ks_pvalue:.4f} below gate "
                           f"{ks_pvalue_min}; including anyway (spatial folds can't be "
                           f"resampled).")
        valid_dists = min_dists[buffer_mask]
        valid_dists = valid_dists[~np.isinf(valid_dists) & ~np.isnan(valid_dists)]
        min_distance_stats = {
            'mean':   float(np.mean(valid_dists))   if len(valid_dists) else float('nan'),
            'median': float(np.median(valid_dists)) if len(valid_dists) else float('nan'),
            'min':    float(np.min(valid_dists))    if len(valid_dists) else float('nan'),
            'max':    float(np.max(valid_dists))    if len(valid_dists) else float('nan'),
            'std':    float(np.std(valid_dists))    if len(valid_dists) else float('nan'),
        }
        fold_meta = {
            'fold_idx': fold_idx, 'lat_lo': float(lo), 'lat_hi': float(hi),
            'val_size': len(val_df), 'train_size': len(train_df),
            'ks_stat': float(ks_stat), 'ks_pvalue': float(ks_pvalue),
        }
        print(f"K-fold {fold_idx+1}/{n_folds}: lat in [{lo:.4f}, {hi:.4f}]  "
              f"val n={len(val_df)} train n={len(train_df)}  "
              f"KS p={ks_pvalue:.4f}")
        folds.append((val_df, train_df, min_distance_stats, fold_meta))
    return folds


# Function to create balanced dataset (unchanged)
def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)

    training_dfs = []
    if use_validation:
        validation_indices = []
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) >= 4:
                val_samples = bin_data.sample(n=min(13, len(bin_data)))
                validation_indices.extend(val_samples.index)
                train_samples = bin_data.drop(val_samples.index)
                if len(train_samples) > 0:
                    if len(train_samples) < min_samples:
                        resampled = train_samples.sample(n=min_samples, replace=True)
                        training_dfs.append(resampled)
                    else:
                        training_dfs.append(train_samples)
        if not training_dfs or not validation_indices:
            raise ValueError("No training or validation data available after binning")
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        validation_df = df.loc[validation_indices].drop('bin', axis=1)
        print('Size of the training set:   ' ,len(training_df))
        print('Size of the validation set:   ' ,len(validation_df))
        return training_df, validation_df
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
        return training_df, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized validation set creation with exponential family distribution')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for validation points')
    args = parser.parse_args()

    validation_df, training_df , min_distance_stats = create_validation_train_sets(
        df=None,
        output_dir=args.output_dir,
        target_val_ratio=args.target_val_ratio,
        use_gpu=args.use_gpu,
        distance_threshold=args.distance_threshold
    )