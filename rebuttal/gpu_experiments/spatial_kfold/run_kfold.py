#!/usr/bin/env python3
"""
run_kfold.py — Experiment 1 — Spatial 10-fold CV (latitude deciles).

Answers reviewer concerns R1.3, R3.6, R3.8.

This is the slimmed rewrite (2026-05-14): the training loop is now
imported directly from train.py so the recipe (cosine LR, gradient
accumulation, log/normalize inverse, R²-in-SOC) stays in sync. The CLI
mirrors train.py 1:1 for the shared flags and adds k-fold-specific
ones (--num-folds, --fold-buffer-km, --fold).

Geometry: each fold's TEST half is one latitude decile (~10% of points)
with a configurable buffer (default 1.2 km) between train and test
rows. Equal-n folds eliminate the per-fold n imbalance the old
equal-latitude-span layout had.

Outputs (all under rebuttal/gpu_experiments/spatial_kfold/):
    fold_{i}_best.pth                          (i = 0..N-1)
    fold_{i}_metrics.json                      (per-epoch dicts)
    kfold_predictions_all_folds.parquet
    kfold_results.md
    kfold_results_summary.json
    figure_kfold.png                           (300 dpi, 2 panels)
"""

from __future__ import annotations

# Force wandb into disabled mode BEFORE train.py imports wandb. With
# WANDB_MODE=disabled wandb.init/log/run.summary are stub no-ops, so the
# imported train_model from train.py runs without needing an actual
# wandb account in this k-fold context.
import os
os.environ.setdefault('WANDB_MODE', 'disabled')

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ----- Path setup ---------------------------------------------------------
# rebuttal/gpu_experiments/spatial_kfold/run_kfold.py → walk up to SOCmapping/
_THIS = Path(__file__).resolve()
SOC_CODE_DIR = _THIS.parent.parent.parent.parent          # SOCmapping/
sys.path.insert(0, str(SOC_CODE_DIR))
from _paths import (  # noqa: E402
    SOC_DATA_DIR, SOC_REBUTTAL_DIR, describe as _describe_paths,
)

SGT_DIR = SOC_CODE_DIR / 'SpatiotemporalGatedTransformer'
sys.path.insert(0, str(SGT_DIR))
sys.path.insert(0, str(SGT_DIR / 'dataloader'))

import wandb  # noqa: E402  (disabled mode — see top of file)
from accelerate import Accelerator  # noqa: E402

# Reuse train.py's recipe machinery so kfold stays in sync.
from train import (  # noqa: E402
    train_model,
    build_sgt_model,
    _resolve_accum_steps,
    compute_training_statistics_oc,
)
from dataloader.dataloaderMultiYears import (  # noqa: E402
    MultiRasterDatasetMultiYears,
    NormalizedMultiRasterDatasetMultiYears,
)
from dataloader.dataframe_loader import (  # noqa: E402
    separate_and_add_data,
)
from config import (  # noqa: E402
    bands_list_order,
    hidden_size,
    num_epochs as CONFIG_NUM_EPOCHS,
    time_before,
    window_size,
    NUM_HEADS,
    NUM_LAYERS,
)

print(_describe_paths(), flush=True)

# ----- Output paths -------------------------------------------------------
OUT_DIR = SOC_REBUTTAL_DIR / 'gpu_experiments' / 'spatial_kfold'
MODEL_READY = SOC_REBUTTAL_DIR / 'model_ready_dataset.parquet'

LUCAS_XLSX = SOC_DATA_DIR / 'LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx'
ELEV_COORDS_NPY = SOC_DATA_DIR / 'OC_LUCAS_LFU_LfL_Coordinates_v2' / 'StaticValue' / 'Elevation' / 'coordinates.npy'
ELEV_TILE_DIR = SOC_DATA_DIR / 'RasterTensorData' / 'StaticValue' / 'Elevation'

# Hardcoded reference from rebuttal_numbers.md
ORIGINAL_SINGLE_SPLIT = {
    'n_test': 1359, 'r2': 0.6258, 'rmse': 4.758, 'mae': 2.791, 'rpiq': 1.051,
}

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------
# First-run dataset bootstrap (same as before — idempotent)
# --------------------------------------------------------------------------
def _build_model_ready_dataset() -> None:
    """Build rebuttal/model_ready_dataset.parquet from the canonical xlsx
    on first run. No-op if it already exists."""
    if MODEL_READY.exists():
        return
    import re as _re
    print(f'[setup] {MODEL_READY} missing — building from {LUCAS_XLSX}', flush=True)
    if not LUCAS_XLSX.exists():
        raise FileNotFoundError(
            f'{LUCAS_XLSX} not found. Set SOC_DATA_DIR or SOC_PROJECT_ROOT.')

    raw = pd.read_excel(LUCAS_XLSX)
    raw['GPS_LONG'] = pd.to_numeric(raw['GPS_LONG'], errors='coerce')
    raw['GPS_LAT'] = pd.to_numeric(raw['GPS_LAT'], errors='coerce')
    raw['OC'] = pd.to_numeric(raw['OC'], errors='coerce')
    mask = ((raw['OC'] <= 150)
            & raw['GPS_LONG'].notna() & raw['GPS_LAT'].notna()
            & raw['OC'].notna()
            & raw['year'].between(2007, 2023, inclusive='both'))
    df = raw[mask].copy().reset_index(drop=True)
    print(f'[setup] filtered to {len(df):,} rows', flush=True)

    if not ELEV_COORDS_NPY.exists():
        raise FileNotFoundError(f'{ELEV_COORDS_NPY} missing — needed for altitude.')
    coords = np.load(ELEV_COORDS_NPY)
    elev_map = (pd.DataFrame(coords, columns=['lat', 'lon', 'id_num', 'x', 'y'])
                .drop_duplicates(['lat', 'lon']).reset_index(drop=True))
    tile_files = {int(_re.match(r'ID(\d+)', p.name).group(1)): p
                  for p in ELEV_TILE_DIR.iterdir()
                  if p.name.startswith('ID')}
    tiles: dict[int, np.ndarray] = {}
    altitude = np.empty(len(elev_map), dtype=float)
    for i, r in elev_map.iterrows():
        tid = int(r.id_num)
        if tid not in tiles:
            tiles[tid] = np.load(tile_files[tid])
        altitude[i] = float(tiles[tid][int(r.x), int(r.y)])
    elev_map['altitude'] = altitude

    df = df.merge(elev_map[['lat', 'lon', 'altitude']],
                  left_on=['GPS_LAT', 'GPS_LONG'],
                  right_on=['lat', 'lon'], how='left').drop(columns=['lat', 'lon'])
    n_missing = int(df['altitude'].isna().sum())
    if n_missing:
        print(f'[setup] {n_missing} rows missing altitude — dropping', flush=True)
        df = df.dropna(subset=['altitude']).copy()
    df['altitude'] = df['altitude'].astype(float)
    df['year'] = df['year'].astype(int)

    if 'season' not in df.columns:
        sd = pd.to_datetime(df['survey_date'], errors='coerce')
        month = sd.dt.month.fillna(0).astype(int)
        season_of_month = month.map(
            {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring',
             5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
             9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter'}
        ).fillna('winter')
        df['season'] = df['year'].astype(str) + '_' + season_of_month

    keep = [c for c in ('POINTID', 'GPS_LONG', 'GPS_LAT', 'year', 'OC',
                        'survey_date', 'season', 'bin', 'dataset_type',
                        'altitude') if c in df.columns]
    out = df[keep].copy()
    for c in ('POINTID', 'season'):
        if c in out.columns:
            out[c] = out[c].astype(str)

    MODEL_READY.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MODEL_READY)
    print(f'[setup] wrote {MODEL_READY} ({len(out):,} rows)', flush=True)


# --------------------------------------------------------------------------
# Per-bin OC rebalancing (same as create_balanced_dataset(with_test=False))
# --------------------------------------------------------------------------
def rebalance_by_oc_bin(df: pd.DataFrame, n_bins: int = 128,
                        min_ratio: float = 0.75) -> pd.DataFrame:
    """qcut OC into n_bins quantile bins; upsample rare bins to
    max_bin_count × min_ratio. No-op if min_ratio<=0."""
    if min_ratio <= 0 or n_bins < 2 or len(df) < n_bins:
        return df.reset_index(drop=True)
    df = df.copy()
    df['_bin'] = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    bc = df['_bin'].value_counts()
    if bc.empty:
        return df.drop(columns=['_bin']).reset_index(drop=True)
    max_n = int(bc.max())
    min_n = max(int(max_n * min_ratio), 5)
    parts = []
    for bin_id in bc.index:
        rows = df[df['_bin'] == bin_id]
        if len(rows) == 0:
            continue
        if len(rows) < min_n:
            parts.append(rows.sample(n=min_n, replace=True,
                                     random_state=int(bin_id)))
        else:
            parts.append(rows)
    return pd.concat(parts, ignore_index=True).drop(columns=['_bin'])


# --------------------------------------------------------------------------
# Spatial fold construction
# --------------------------------------------------------------------------
def haversine_km_matrix(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1)); lon1 = np.deg2rad(np.asarray(lon1))
    lat2 = np.deg2rad(np.asarray(lat2)); lon2 = np.deg2rad(np.asarray(lon2))
    dlat = lat2[np.newaxis, :] - lat1[:, np.newaxis]
    dlon = lon2[np.newaxis, :] - lon1[:, np.newaxis]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1)[:, np.newaxis] * np.cos(lat2)[np.newaxis, :]
         * np.sin(dlon / 2) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def min_distance_to_set_km(t_lat, t_lon, r_lat, r_lon, chunk=2048):
    n = len(t_lat)
    out = np.full(n, np.inf, dtype=float)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = haversine_km_matrix(t_lat[s:e], t_lon[s:e], r_lat, r_lon)
        out[s:e] = d.min(axis=1)
    return out


def build_folds_latitude_deciles(df: pd.DataFrame, n_folds: int = 10,
                                  buffer_km: float = 1.2) -> list[dict]:
    """Latitude-strip folds with EQUAL N per fold.

    The strip boundaries are quantiles of GPS_LAT (np.quantile at
    np.linspace(0, 1, n_folds+1)) rather than equal-latitude-span splits,
    so each fold's test half holds ≈ len(df) / n_folds rows regardless
    of the spatial density of samples.

    Each train pool then has a 1.2 km buffer applied: any train-candidate
    within buffer_km of any test row is excluded from train (kept in
    buffer_idx). Buffer rows are not scored on, just dropped from train.

    Returns list[dict] with fold_id, lat_lo, lat_hi, test_idx, train_idx,
    buffer_idx (np int arrays into df.index)."""
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError('build_folds requires df.index == RangeIndex')

    lats = df['GPS_LAT'].to_numpy(dtype=float)
    qs = np.linspace(0, 1, n_folds + 1)
    edges = np.quantile(lats, qs)
    edges[-1] = lats.max() + 1e-9                          # inclusive top edge
    edges[0] = lats.min() - 1e-9                           # inclusive bottom edge

    folds = []
    for i in range(n_folds):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_strip = (lats >= lo) & (lats < hi)
        test_idx = df.index[in_strip].to_numpy()
        train_pool_idx = df.index[~in_strip].to_numpy()

        t_lat = df.loc[test_idx, 'GPS_LAT'].to_numpy(dtype=float)
        t_lon = df.loc[test_idx, 'GPS_LONG'].to_numpy(dtype=float)
        p_lat = df.loc[train_pool_idx, 'GPS_LAT'].to_numpy(dtype=float)
        p_lon = df.loc[train_pool_idx, 'GPS_LONG'].to_numpy(dtype=float)
        d = min_distance_to_set_km(p_lat, p_lon, t_lat, t_lon)
        keep = d >= buffer_km
        train_idx = train_pool_idx[keep]
        buffer_idx = train_pool_idx[~keep]

        folds.append({
            'fold_id': i, 'lat_lo': lo, 'lat_hi': hi,
            'test_idx': test_idx, 'train_idx': train_idx,
            'buffer_idx': buffer_idx,
        })
    return folds


# --------------------------------------------------------------------------
# Dataset builders
# --------------------------------------------------------------------------
def _flatten(lst):
    out = []
    for x in lst:
        if isinstance(x, list):
            out.extend(_flatten(x))
        else:
            out.append(x)
    return out


def make_dataset(df: pd.DataFrame, feature_means=None, feature_stds=None):
    sample_paths, data_paths = separate_and_add_data()
    sample_paths = list(dict.fromkeys(_flatten(sample_paths)))
    data_paths = list(dict.fromkeys(_flatten(data_paths)))
    ds = MultiRasterDatasetMultiYears(
        samples_coordinates_array_subfolders=sample_paths,
        data_array_subfolders=data_paths,
        dataframe=df.reset_index(drop=True),
        time_before=time_before,
    )
    if feature_means is not None and feature_stds is not None:
        ds = _NormalizingWrapper(ds, feature_means, feature_stds)
    return ds


class _NormalizingWrapper(Dataset):
    def __init__(self, base, means: torch.Tensor, stds: torch.Tensor):
        self.base = base
        self.means = means.float()
        self.stds = torch.clamp(stds.float(), min=1e-8)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        lon, lat, features, oc = self.base[idx]
        features = (features - self.means[:, None, None]) / self.stds[:, None, None]
        return lon, lat, features, oc


_FULL_FEATURE_STATS_CACHE = None


def compute_full_feature_statistics():
    """Feature mean/std from a NormalizedMultiRasterDatasetMultiYears over
    the full 16,514-row df — same convention as train.py."""
    global _FULL_FEATURE_STATS_CACHE
    if _FULL_FEATURE_STATS_CACHE is not None:
        return _FULL_FEATURE_STATS_CACHE
    df = pd.read_parquet(MODEL_READY)
    sample_paths, data_paths = separate_and_add_data()
    sample_paths = list(dict.fromkeys(_flatten(sample_paths)))
    data_paths = list(dict.fromkeys(_flatten(data_paths)))
    norm_ds = NormalizedMultiRasterDatasetMultiYears(sample_paths, data_paths, df)
    fm = norm_ds.get_feature_means()
    fs = norm_ds.get_feature_stds()
    print(f'Computed full-df feature stats once: '
          f'means.shape={tuple(fm.shape)}  stds.shape={tuple(fs.shape)}',
          flush=True)
    _FULL_FEATURE_STATS_CACHE = (fm, fs)
    return fm, fs


# --------------------------------------------------------------------------
# Per-fold training (delegates to train.py's train_model)
# --------------------------------------------------------------------------
def train_one_fold(args, fold: dict, df: pd.DataFrame,
                   accelerator: Accelerator, feature_means, feature_stds,
                   target_mean: float, target_std: float) -> dict:
    fold_id = fold['fold_id']
    seed = args.seed_base + fold_id
    torch.manual_seed(seed); np.random.seed(seed)

    print(f'\n=== Fold {fold_id} | lat [{fold["lat_lo"]:.4f}, '
          f'{fold["lat_hi"]:.4f}) | seed={seed} ===', flush=True)

    train_df_raw = df.loc[fold['train_idx']].reset_index(drop=True)
    test_df = df.loc[fold['test_idx']].reset_index(drop=True)
    train_df = rebalance_by_oc_bin(train_df_raw, n_bins=args.rebalance_n_bins,
                                    min_ratio=args.rebalance_min_ratio)

    print(f'n_train_raw={len(train_df_raw)}  n_train_rebalanced={len(train_df)}  '
          f'n_test={len(test_df)}  n_buffer_excluded={len(fold["buffer_idx"])}',
          flush=True)
    print(f'Test SOC: mean={test_df.OC.mean():.2f} std={test_df.OC.std():.2f} '
          f'max={test_df.OC.max():.1f} %>50={100*(test_df.OC>50).mean():.2f}%',
          flush=True)

    train_ds = make_dataset(train_df, feature_means, feature_stds)
    test_ds = make_dataset(test_df, feature_means, feature_stds)
    num_workers = int(os.environ.get('SOC_KFOLD_NUM_WORKERS', 0))
    train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.per_gpu_batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # Resolve gradient accumulation steps once we know num_processes
    accum_steps = _resolve_accum_steps(args, accelerator.num_processes)
    effective_batch = accelerator.num_processes * args.per_gpu_batch_size * accum_steps
    if accelerator.is_main_process:
        print(f'[grad-accum] num_gpus={accelerator.num_processes}  '
              f'per_gpu_batch={args.per_gpu_batch_size}  accum_steps={accum_steps}  '
              f'effective_batch={effective_batch}', flush=True)

    model = build_sgt_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {type(model).__name__}  ({n_params:,} trainable params)',
          flush=True)

    # Dummy wandb run for the imported train_model (disabled mode = no-op).
    wandb_run = wandb.init(
        project='socmapping-kfold',
        name=f'fold_{fold_id}_seed_{seed}',
        config={'fold_id': fold_id, **vars(args)},
        reinit=True,
    )

    # Delegate to the canonical training loop in train.py — same cosine LR,
    # same gradient accumulation, same R²-in-SOC inverse-transform path.
    (model, _test_outputs, _test_targets, best_state, best_r2, epoch_metrics
     ) = train_model(
        model, train_loader, test_loader,
        target_mean=target_mean, target_std=target_std,
        num_epochs=args.num_epochs,
        accelerator=accelerator,
        lr=args.lr,
        loss_type=args.loss_type,
        target_transform=args.target_transform,
        min_r2=-float('inf'),               # always save the best, even if low
        use_test=True,
        accum_steps=accum_steps,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_gamma=args.lr_gamma,
        lr_restart_T0=args.lr_restart_T0,
        plateau_monitor=args.plateau_monitor,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
    )

    wandb_run.finish()

    # ----- Save fold artefacts -------------------------------------------
    model_config = {
        'input_channels': len(bands_list_order),
        'height': window_size, 'width': window_size,
        'time_steps': time_before, 'd_model': args.hidden_size,
        'num_heads': args.num_heads, 'num_layers': args.num_layers,
        'dropout': args.dropout_rate, 'model_size': args.model_size,
    }
    pth_path = OUT_DIR / f'fold_{fold_id}_best.pth'
    accelerator.save({
        'model_state_dict': best_state,
        'model_config': model_config,
        'fold_id': fold_id, 'best_r2': float(best_r2),
        'n_train': len(train_df), 'n_test': len(test_df),
        'feature_means': feature_means, 'feature_stds': feature_stds,
        'target_mean': target_mean, 'target_std': target_std,
        'target_transform': args.target_transform,
        'effective_batch_size': effective_batch,
        'accum_steps': accum_steps, 'args': vars(args),
    }, pth_path)
    (OUT_DIR / f'fold_{fold_id}_metrics.json').write_text(
        json.dumps(epoch_metrics, indent=2, default=str))
    print(f'Saved {pth_path}', flush=True)

    # ----- Re-evaluate with best weights to harvest predictions ----------
    model.load_state_dict(best_state)
    model.eval()
    eps = 1e-10
    all_lon, all_lat, all_pred, all_act = [], [], [], []
    with torch.no_grad():
        for lon, lat, x, y in test_loader:
            x = x.to(accelerator.device, non_blocking=True)
            y = y.cpu().numpy().astype(float)
            pred = model(x).float().cpu().numpy()
            if args.target_transform == 'log':
                pred = np.exp(pred)
            elif args.target_transform == 'normalize':
                pred = pred * target_std + target_mean
            all_pred.append(pred)
            all_act.append(y)
            all_lon.append(np.asarray(lon))
            all_lat.append(np.asarray(lat))
    test_pred = np.concatenate(all_pred)
    test_actual = np.concatenate(all_act)
    test_lon = np.concatenate(all_lon)
    test_lat = np.concatenate(all_lat)

    pearson_r = float(np.corrcoef(test_pred, test_actual)[0, 1])
    pearson_r2 = pearson_r ** 2
    ss_res = float(np.sum((test_actual - test_pred) ** 2))
    ss_tot = float(np.sum((test_actual - test_actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((test_pred - test_actual) ** 2)))
    mae = float(np.mean(np.abs(test_pred - test_actual)))
    bias = float((test_pred - test_actual).mean())
    q1, q3 = np.percentile(test_actual, [25, 75])
    rpiq = float((q3 - q1) / rmse) if rmse > 0 else float('inf')

    return {
        'fold_id': fold_id,
        'lat_lo': fold['lat_lo'], 'lat_hi': fold['lat_hi'],
        'n_test': int(len(test_df)),
        'n_train': int(len(train_df)),
        'n_train_raw': int(len(train_df_raw)),
        'n_buffer': int(len(fold['buffer_idx'])),
        'accum_steps': accum_steps,
        'effective_batch_size': effective_batch,
        'test_oc_mean': float(test_df.OC.mean()),
        'test_oc_std': float(test_df.OC.std()),
        'test_oc_max': float(test_df.OC.max()),
        'test_pct_gt_50': float(100 * (test_df.OC > 50).mean()),
        'r2': r2, 'pearson_r2': pearson_r2, 'pearson_r': pearson_r,
        'rmse': rmse, 'mae': mae, 'bias': bias, 'rpiq': rpiq,
        'best_epoch_r2_during_training': float(best_r2),
        '_predictions': {
            'lon': test_lon, 'lat': test_lat,
            'pred': test_pred, 'actual': test_actual,
            'year': test_df['year'].to_numpy(),
            'altitude': (test_df['altitude'].to_numpy()
                          if 'altitude' in test_df.columns
                          else np.full(len(test_df), np.nan)),
        },
    }


# --------------------------------------------------------------------------
# Reporting (mostly unchanged from previous version)
# --------------------------------------------------------------------------
def write_results(fold_results: list[dict], args):
    rows = []
    for r in fold_results:
        rows.append({k: v for k, v in r.items() if not k.startswith('_')})
    summary_df = pd.DataFrame(rows)

    def ci95(values):
        m = float(np.mean(values))
        s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        half = 1.96 * s / np.sqrt(len(values))
        return m, s, (m - half, m + half)

    r2_m, r2_s, r2_ci = ci95(summary_df['r2'].to_numpy())
    rmse_m, rmse_s, rmse_ci = ci95(summary_df['rmse'].to_numpy())
    mae_m, mae_s, mae_ci = ci95(summary_df['mae'].to_numpy())
    rpiq_m, rpiq_s, rpiq_ci = ci95(summary_df['rpiq'].to_numpy())

    summary_json = {
        'fold_results': rows,
        'across_folds': {
            'r2_mean': r2_m, 'r2_std': r2_s, 'r2_ci95': r2_ci,
            'rmse_mean': rmse_m, 'rmse_std': rmse_s, 'rmse_ci95': rmse_ci,
            'mae_mean': mae_m, 'mae_std': mae_s, 'mae_ci95': mae_ci,
            'rpiq_mean': rpiq_m, 'rpiq_std': rpiq_s, 'rpiq_ci95': rpiq_ci,
        },
        'original_single_split': ORIGINAL_SINGLE_SPLIT,
        'n_folds': args.num_folds,
        'distance_threshold_km': args.fold_buffer_km,
        'fold_geometry': 'latitude_deciles_equal_n',
        'recipe': {
            'lr': args.lr, 'loss_type': args.loss_type,
            'target_transform': args.target_transform,
            'num_epochs': args.num_epochs,
            'per_gpu_batch_size': args.per_gpu_batch_size,
            'effective_batch_size': args.effective_batch_size,
            'lr_scheduler': args.lr_scheduler, 'lr_min': args.lr_min,
            'num_heads': args.num_heads, 'num_layers': args.num_layers,
            'rebalance_n_bins': args.rebalance_n_bins,
            'rebalance_min_ratio': args.rebalance_min_ratio,
        },
    }
    (OUT_DIR / 'kfold_results_summary.json').write_text(
        json.dumps(summary_json, indent=2, default=str))

    md = []
    md.append(f'# Spatial {args.num_folds}-fold CV — latitude deciles (equal n)')
    md.append('')
    md.append(f'Each fold\'s test half is one decile of GPS_LAT '
              f'(~{int(100/args.num_folds)}% of points). Train pool is the '
              f'complement, minus a {args.fold_buffer_km} km buffer zone. '
              f'Per-fold train half rebalanced via 128-qcut OC bins to '
              f'≥ 75% of the densest bin. EnhancedSGT (heads={args.num_heads}, '
              f'layers={args.num_layers}, ~{1_120_546 if args.num_heads==4 else "?"} '
              f'params) trained for {args.num_epochs} epochs with '
              f'`{args.lr_scheduler}` LR schedule from {args.lr} → '
              f'{args.lr_min if args.lr_scheduler in ("cosine","cosine_warm_restarts") else "n/a"}, '
              f'Adam, {args.loss_type.upper()} on {args.target_transform}'
              f'-transformed target.')
    md.append('')
    md.append('| Fold | Lat range | n_test | n_train | R² | RMSE (g/kg) | MAE (g/kg) | RPIQ |')
    md.append('|------|-----------|--------|---------|-----|-------------|------------|------|')
    for r in fold_results:
        md.append(f'| {r["fold_id"]} | [{r["lat_lo"]:.4f}, {r["lat_hi"]:.4f}) | '
                  f'{r["n_test"]} | {r["n_train"]} | {r["r2"]:.4f} | {r["rmse"]:.3f} | '
                  f'{r["mae"]:.3f} | {r["rpiq"]:.3f} |')
    md.append(f'| **Mean ± std** | — | — | — | '
              f'{r2_m:.4f} ± {r2_s:.4f} | {rmse_m:.3f} ± {rmse_s:.3f} | '
              f'{mae_m:.3f} ± {mae_s:.3f} | {rpiq_m:.3f} ± {rpiq_s:.3f} |')
    md.append(f'| **95% CI** | — | — | — | [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}] | '
              f'[{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}] | '
              f'[{mae_ci[0]:.3f}, {mae_ci[1]:.3f}] | '
              f'[{rpiq_ci[0]:.3f}, {rpiq_ci[1]:.3f}] |')
    md.append(f'| *Original single-split* | — | {ORIGINAL_SINGLE_SPLIT["n_test"]} | — | '
              f'{ORIGINAL_SINGLE_SPLIT["r2"]:.3f} | '
              f'{ORIGINAL_SINGLE_SPLIT["rmse"]:.3f} | '
              f'{ORIGINAL_SINGLE_SPLIT["mae"]:.3f} | '
              f'{ORIGINAL_SINGLE_SPLIT["rpiq"]:.3f} |')
    md.append('')
    (OUT_DIR / 'kfold_results.md').write_text('\n'.join(md))


def write_predictions_parquet(fold_results: list[dict]):
    frames = []
    for r in fold_results:
        p = r['_predictions']
        frames.append(pd.DataFrame({
            'GPS_LAT': p['lat'], 'GPS_LONG': p['lon'],
            'OC_actual': p['actual'], 'OC_predicted': p['pred'],
            'fold_id': r['fold_id'], 'year': p['year'],
            'altitude': p['altitude'],
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT_DIR / 'kfold_predictions_all_folds.parquet')


def make_figure(fold_results, folds_meta, df_master, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_folds = len(folds_meta)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_folds, 1)))
    buffer_all = np.concatenate([f['buffer_idx'] for f in folds_meta]) \
        if folds_meta else np.array([], dtype=int)
    if len(buffer_all):
        bufr = df_master.loc[np.unique(buffer_all)]
        ax.scatter(bufr['GPS_LONG'], bufr['GPS_LAT'], s=3,
                   c='lightgrey', alpha=0.6, label='Buffer-excluded')
    for f, color in zip(folds_meta, colors):
        idx = f['test_idx']
        ax.scatter(df_master.loc[idx, 'GPS_LONG'],
                   df_master.loc[idx, 'GPS_LAT'],
                   s=5, color=color, label=f'Fold {f["fold_id"]}', alpha=0.7)
        ax.axhline(f['lat_hi'], color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'Bavaria — {n_folds} latitude-decile folds '
                 f'(buffer {args.fold_buffer_km} km)')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[1]
    x_groups = np.arange(n_folds)
    width = 0.25
    r2s = np.array([r['r2'] for r in fold_results])
    rmses = np.array([r['rmse'] for r in fold_results]) / 10.0
    maes = np.array([r['mae'] for r in fold_results]) / 10.0
    ax.bar(x_groups - width, r2s, width, label='R²', color='#1f77b4')
    ax.bar(x_groups, rmses, width, label='RMSE / 10', color='#ff7f0e')
    ax.bar(x_groups + width, maes, width, label='MAE / 10', color='#2ca02c')
    ax.axhline(ORIGINAL_SINGLE_SPLIT['r2'], color='#1f77b4', linestyle='--', linewidth=1)
    ax.axhline(ORIGINAL_SINGLE_SPLIT['rmse'] / 10, color='#ff7f0e', linestyle='--', linewidth=1)
    ax.axhline(ORIGINAL_SINGLE_SPLIT['mae'] / 10, color='#2ca02c', linestyle='--', linewidth=1)
    ax.set_xticks(x_groups)
    ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
    ax.set_ylabel('Value')
    ax.set_title('Per-fold metrics (dashed lines = original single-split)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'Experiment 1 — Spatial {n_folds}-fold CV (EnhancedSGT)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = OUT_DIR / 'figure_kfold.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}', flush=True)


# --------------------------------------------------------------------------
# CLI — mirrors train.py for the shared flags
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Spatial k-fold CV — latitude deciles, single-GPU sequential.')

    # ----- Mirrored from train.py -----
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_heads', type=int, default=NUM_HEADS)
    p.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    p.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mse'])
    p.add_argument('--target_transform', type=str, default='normalize',
                   choices=['none', 'log', 'normalize'])
    p.add_argument('--hidden_size', type=int, default=hidden_size)
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--model-size', type=str, default='big',
                   choices=['small', 'big'])
    p.add_argument('--per-gpu-batch-size', type=int, default=256)
    p.add_argument('--effective-batch-size', type=int, default=2048)
    p.add_argument('--accum-steps', type=int, default=0)
    p.add_argument('--num-epochs', type=int, default=CONFIG_NUM_EPOCHS)
    p.add_argument('--lr-scheduler', type=str, default='none',
                   choices=['none', 'cosine', 'cosine_warm_restarts', 'exponential', 'plateau'])
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--lr-gamma', type=float, default=0.99)
    p.add_argument('--lr-restart-T0', type=int, default=50)
    p.add_argument('--plateau-monitor', type=str, default='pearson_r2',
                   choices=['pearson_r2', 'r_squared', 'test_loss', 'rmse', 'mae'])
    p.add_argument('--plateau-patience', type=int, default=20)
    p.add_argument('--plateau-factor', type=float, default=0.5)

    # ----- K-fold-specific -----
    p.add_argument('--num-folds', type=int, default=10,
                   help='Number of latitude-decile folds.')
    p.add_argument('--fold-buffer-km', type=float, default=1.2,
                   help='Train/test buffer-zone distance in km.')
    p.add_argument('--fold', type=int, default=None,
                   help='Run a single fold by ID (0..num_folds-1). '
                        'Default: run all folds sequentially.')
    p.add_argument('--seed-base', type=int, default=42,
                   help='Per-fold seed = seed_base + fold_id.')
    p.add_argument('--rebalance-n-bins', type=int, default=128,
                   help='Number of OC quantile bins for per-bin oversample.')
    p.add_argument('--rebalance-min-ratio', type=float, default=0.75,
                   help='Per-bin floor as fraction of densest bin. 0=disable.')
    p.add_argument('--skip-figure', action='store_true',
                   help='Skip matplotlib figure (faster, lighter deps).')
    return p.parse_args()


# --------------------------------------------------------------------------
# Main — sequential, single GPU, one Accelerator for all folds
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _build_model_ready_dataset()

    df = pd.read_parquet(MODEL_READY).reset_index(drop=True)
    folds_meta = build_folds_latitude_deciles(
        df, n_folds=args.num_folds, buffer_km=args.fold_buffer_km)
    print(f'Loaded {len(df)} rows from {MODEL_READY}', flush=True)
    print(f'num_folds={args.num_folds}  buffer={args.fold_buffer_km} km', flush=True)
    for f in folds_meta:
        print(f'Fold {f["fold_id"]}: lat [{f["lat_lo"]:.4f}, {f["lat_hi"]:.4f}) '
              f'| n_test={len(f["test_idx"])} n_train={len(f["train_idx"])} '
              f'n_buffer={len(f["buffer_idx"])}', flush=True)

    feature_means, feature_stds = compute_full_feature_statistics()
    target_mean, target_std = compute_training_statistics_oc()
    print(f'target_mean={target_mean:.4f}  target_std={target_std:.4f}', flush=True)

    accelerator = Accelerator()
    print(f'Accelerator: device={accelerator.device}  '
          f'num_processes={accelerator.num_processes}', flush=True)

    # Allow running a single fold via --fold N (handy for debugging).
    if args.fold is not None:
        folds_to_run = [f for f in folds_meta if f['fold_id'] == args.fold]
        if not folds_to_run:
            raise SystemExit(f'--fold {args.fold} out of range [0, {args.num_folds - 1}]')
    else:
        folds_to_run = folds_meta

    t_start = time.time()
    fold_results = []
    for f in folds_to_run:
        result = train_one_fold(args, f, df, accelerator,
                                feature_means, feature_stds,
                                target_mean, target_std)
        fold_results.append(result)
        elapsed = time.time() - t_start
        print(f'>>> Fold {f["fold_id"]} done. Cumulative elapsed {elapsed/60:.1f} min',
              flush=True)

    elapsed = time.time() - t_start
    print(f'\nAll {len(fold_results)} fold(s) finished in {elapsed/60:.1f} min.',
          flush=True)

    if args.fold is None:
        # Only write the cross-fold aggregates when ALL folds ran.
        write_results(fold_results, args)
        write_predictions_parquet(fold_results)
        if not args.skip_figure:
            with contextlib.suppress(Exception):
                make_figure(fold_results, folds_meta, df, args)
        print('Experiment 1 complete.', flush=True)
    else:
        print(f'Single-fold mode: fold {args.fold} done. '
              f'Run without --fold to write cross-fold aggregates.', flush=True)


if __name__ == '__main__':
    main()
