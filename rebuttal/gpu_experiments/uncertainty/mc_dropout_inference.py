#!/usr/bin/env python3
"""
mc_dropout_inference.py — Experiment 2 — MC Dropout uncertainty map.

Answers reviewer concerns R3.9, R4.4.

Loads Model B (the operational mapping model trained on all 16,514 samples)
and runs N_PASSES = 30 stochastic forward passes per Bavaria-grid point,
keeping BatchNorm in eval mode and dropout in train mode (selective
activation). Welford online accumulator tracks mean and variance in the
*original* SOC units (g/kg) by inverting the log-transform inside the loop.

Outputs (under rebuttal/gpu_experiments/uncertainty/):
    SGT_1mil_2023_mean_mc30.tif       (mean prediction, float32, EPSG:32632)
    SGT_1mil_2023_std_mc30.tif        (per-point SD, float32, EPSG:32632)
    mc_dropout_points.parquet         (lon, lat, mean, std per grid point)
    mc_dropout_metadata.json
    validation_check.txt              (Pearson r vs the single-pass map)

DO NOT RUN until ASSUMPTION blocks have been reviewed.
"""

# ASSUMPTION: rasterio is installed (it isn't yet — task does no exec).
# To install: pip install rasterio.
# If rasterio is unavailable on the GPU host the script still writes the
# parquet (lon, lat, mean, std) and the .npy backups; only the GeoTIFF
# output is skipped.

# ASSUMPTION: the project's existing 2023 mapping output is a directory of
# .npy chunks under SOCmapping/Maps/, SOCmapping/SGT_Maps_MixedProcessing/,
# or SOCmapping/AllResultsMappingTogether*. The validation check searches
# those locations; if no single-pass map is found the check prints a
# WARNING and continues. The MC mean prediction should still match
# Figures 14/15 visually.

# ASSUMPTION: the inference grid Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv
# has columns ['longitude', 'latitude']. Confirmed: 1,300,000 rows, 2 columns.

# ASSUMPTION: Model B was trained with target_transform='log' (LOSS_l1).
# Inference must apply np.exp() per the original running.py:apply_inverse_transform.
# NOT torch.expm1. Switching to expm1 would shift every predicted SOC value
# down by ~e^(1e-10) ≈ 1 g/kg — small but enough to drift from Figures 14/15.

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ----- Path setup ---------------------------------------------------------
_THIS = Path(__file__).resolve()
SOC_CODE_DIR = _THIS.parent.parent.parent.parent       # SOCmapping/
sys.path.insert(0, str(SOC_CODE_DIR))
from _paths import (  # noqa: E402
    SOC_DATA_DIR, SOC_WEIGHTS_DIR, SOC_REBUTTAL_DIR, describe as _describe_paths,
)

SGT_DIR = SOC_CODE_DIR / 'SpatiotemporalGatedTransformer'
sys.path.insert(0, str(SGT_DIR))

from EnhancedSGT import EnhancedSGT  # noqa: E402
from dataloader.dataloaderMapping import (  # noqa: E402
    NormalizedMultiRasterDataset1MilMultiYears,
)
from dataloader.dataframe_loader import (  # noqa: E402
    filter_dataframe, separate_and_add_data, separate_and_add_data_1mil_inference,
)
from dataloader.dataloaderMultiYears import (  # noqa: E402
    NormalizedMultiRasterDatasetMultiYears,
)
from config import (  # noqa: E402
    INFERENCE_TIME,
    MAX_OC,
    TIME_BEGINNING,
    TIME_END,
    bands_list_order,
    hidden_size,
    time_before,
    window_size,
    file_path_coordinates_Bavaria_1mil,
)

# ----- Configuration ------------------------------------------------------
MODEL_B_PATH = SOC_WEIGHTS_DIR / (
    'TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/'
    'TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_'
    'TRANSFORM_log_LOSS_l1_R2_1.0000.pth'
)
OUT_DIR = SOC_REBUTTAL_DIR / 'gpu_experiments' / 'uncertainty'
N_PASSES = 30
BATCH_SIZE = 256
NUM_WORKERS = 4
TARGET_TRANSFORM = 'log'
TARGET_RESOLUTION_M = 250        # raster pixel size
TARGET_CRS = 'EPSG:32632'        # UTM Zone 32N (Bavaria)


# --------------------------------------------------------------------------
# Model loading (mirrors running.py:load_model_with_metadata but stripped
# of Accelerate so MC sampling stays single-process)
# --------------------------------------------------------------------------
def load_model_b(device: torch.device) -> tuple[nn.Module, dict]:
    print(f'Loading Model B: {MODEL_B_PATH}', flush=True)
    ckpt = torch.load(MODEL_B_PATH, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise RuntimeError('Checkpoint does not have the expected wrapper '
                           'dict — re-check the file.')
    cfg = ckpt.get('model_config', {})
    model = EnhancedSGT(
        input_channels=cfg.get('input_channels', len(bands_list_order)),
        height=cfg.get('height', window_size),
        width=cfg.get('width', window_size),
        time_steps=cfg.get('time_steps', time_before),
        d_model=cfg.get('d_model', hidden_size),
        num_heads=4,                  # ASSUMPTION: see Step 0 summary
        dropout=0.3,
        num_encoder_layers=3,
        expansion_factor=4,
    )
    sd = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device)

    # Selective dropout activation — keep BatchNorm in eval mode
    model.eval()
    n_dropout_enabled = 0
    module_types = set()
    for m in model.modules():
        module_types.add(type(m).__name__)
        if isinstance(m, nn.Dropout):
            m.train()
            n_dropout_enabled += 1
    print(f'Module types present: {sorted(module_types)}', flush=True)
    print(f'Dropout layers re-enabled for MC sampling: {n_dropout_enabled}',
          flush=True)
    # Sanity: confirm BN modules stayed in eval
    n_bn_eval = sum(1 for m in model.modules()
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                    and not m.training)
    n_bn_total = sum(1 for m in model.modules()
                      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))
    print(f'BatchNorm modules in eval: {n_bn_eval}/{n_bn_total}', flush=True)

    return model, ckpt


# --------------------------------------------------------------------------
# Normalisation stats (must match what Model B was trained with)
# --------------------------------------------------------------------------
def load_normalization_stats(ckpt: dict):
    """Read the saved per-channel feature_means/feature_stds + target
    mean/std from the checkpoint. Fall back to a fresh compute if absent."""
    ns = ckpt.get('normalization_stats') or {}
    feature_means = ns.get('feature_means')
    feature_stds = ns.get('feature_stds')
    target_mean = ns.get('target_mean')
    target_std = ns.get('target_std')

    if feature_means is None or feature_stds is None:
        # ASSUMPTION: this branch is only used as fallback. The 2 checkpoints
        # I inspected (Models A, B) both carried normalization_stats inline.
        print('No normalization_stats in checkpoint; recomputing from '
              'training data — this is a ~5 min one-off.', flush=True)
        df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        sp, dp = separate_and_add_data()

        def flatten(lst):
            o = []
            for x in lst:
                if isinstance(x, list):
                    o.extend(flatten(x))
                else:
                    o.append(x)
            return o
        sp = list(dict.fromkeys(flatten(sp)))
        dp = list(dict.fromkeys(flatten(dp)))
        ds = NormalizedMultiRasterDatasetMultiYears(sp, dp, df_train)
        feature_means = ds.get_feature_means()
        feature_stds = ds.get_feature_stds()
        target_mean = float(df_train['OC'].mean())
        target_std = float(df_train['OC'].std())

    if not isinstance(feature_means, torch.Tensor):
        feature_means = torch.as_tensor(feature_means, dtype=torch.float32)
    if not isinstance(feature_stds, torch.Tensor):
        feature_stds = torch.as_tensor(feature_stds, dtype=torch.float32)
    return feature_means.cpu(), feature_stds.cpu(), target_mean, target_std


# --------------------------------------------------------------------------
# Inference data + MC sampling
# --------------------------------------------------------------------------
def build_inference_loader(feature_means, feature_stds, start_idx=0, end_idx=None,
                           indices=None):
    """Match the dataset construction in running.py exactly.

    If `indices` (1-D np.ndarray of int row positions) is provided it wins
    over (start_idx, end_idx). Used by the orchestrator to hand each
    worker a non-contiguous slice (e.g. every k-th row of the CSV)."""
    df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
    if indices is not None:
        df_sub = df_full.iloc[indices].copy().reset_index(drop=True)
    else:
        if end_idx is None:
            end_idx = len(df_full)
        df_sub = df_full.iloc[start_idx:end_idx].copy()
    sp, dp = separate_and_add_data_1mil_inference()

    def flatten(lst):
        o = []
        for x in lst:
            if isinstance(x, list):
                o.extend(flatten(x))
            else:
                o.append(x)
        return o
    sp = list(dict.fromkeys(flatten(sp)))
    dp = list(dict.fromkeys(flatten(dp)))
    ds = NormalizedMultiRasterDataset1MilMultiYears(
        samples_coordinates_array_path=sp,
        data_array_path=dp,
        df=df_sub,
        feature_means=feature_means,
        feature_stds=feature_stds,
        time_before=time_before,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return df_sub, loader, len(ds)


def invert_log(p: torch.Tensor) -> torch.Tensor:
    """Apply np.exp inverse (matches running.py:apply_inverse_transform for
    target_transform='log')."""
    return torch.exp(p)


def run_mc_dropout(model, loader, n_total_points, device):
    """Welford accumulators over MC passes, in original SOC units."""
    means_all = np.empty(n_total_points, dtype=np.float32)
    stds_all = np.empty(n_total_points, dtype=np.float32)
    lons_all = np.empty(n_total_points, dtype=np.float64)
    lats_all = np.empty(n_total_points, dtype=np.float64)

    cursor = 0
    t_start = time.time()
    batches = list(loader)
    print(f'Total batches: {len(batches)} (≈ {n_total_points} points)',
          flush=True)
    for b_idx, batch in enumerate(batches):
        lon, lat, x = batch
        lon = lon.cpu().numpy()
        lat = lat.cpu().numpy()
        x = x.to(device, non_blocking=True).float()
        bsz = x.shape[0]
        running_mean = torch.zeros(bsz, device=device, dtype=torch.float32)
        running_M2 = torch.zeros(bsz, device=device, dtype=torch.float32)
        with torch.no_grad():
            for i in range(N_PASSES):
                pred_log = model(x).squeeze()
                pred = invert_log(pred_log).float()
                # Welford update
                delta = pred - running_mean
                running_mean = running_mean + delta / (i + 1)
                delta2 = pred - running_mean
                running_M2 = running_M2 + delta * delta2
        mean_pred = running_mean.cpu().numpy().astype(np.float32)
        var_pred = (running_M2 / max(N_PASSES, 1)).cpu().numpy().astype(np.float32)
        std_pred = np.sqrt(np.maximum(var_pred, 0.0))

        means_all[cursor:cursor + bsz] = mean_pred
        stds_all[cursor:cursor + bsz] = std_pred
        lons_all[cursor:cursor + bsz] = lon
        lats_all[cursor:cursor + bsz] = lat
        cursor += bsz

        if (b_idx + 1) % 50 == 0 or b_idx == len(batches) - 1:
            t = time.time() - t_start
            done = cursor
            rate = done / max(t, 1e-6)
            remain = (n_total_points - done) / max(rate, 1e-6)
            print(f'  batch {b_idx+1}/{len(batches)}  point {done}/{n_total_points}  '
                  f'elapsed {t:.1f}s  ETA {remain:.0f}s', flush=True)

    return lons_all[:cursor], lats_all[:cursor], means_all[:cursor], stds_all[:cursor]


# --------------------------------------------------------------------------
# GeoTIFF writing
# --------------------------------------------------------------------------
def points_to_geotiffs(lons_deg, lats_deg, mean_vals, std_vals):
    """Rasterise scattered point predictions onto the Bavaria UTM grid."""
    try:
        import rasterio
        from rasterio.transform import from_origin
        from pyproj import Transformer
    except ImportError as e:
        print(f'WARNING: rasterio/pyproj not available ({e}); '
              'skipping GeoTIFF output. Parquet output still written.',
              flush=True)
        return None

    # Reproject lon/lat to UTM
    tr = Transformer.from_crs('EPSG:4326', TARGET_CRS, always_xy=True)
    x_utm, y_utm = tr.transform(lons_deg, lats_deg)

    # Grid extent
    x_min = float(np.floor(x_utm.min() / TARGET_RESOLUTION_M)
                  * TARGET_RESOLUTION_M)
    x_max = float(np.ceil(x_utm.max() / TARGET_RESOLUTION_M)
                  * TARGET_RESOLUTION_M)
    y_min = float(np.floor(y_utm.min() / TARGET_RESOLUTION_M)
                  * TARGET_RESOLUTION_M)
    y_max = float(np.ceil(y_utm.max() / TARGET_RESOLUTION_M)
                  * TARGET_RESOLUTION_M)
    width = int((x_max - x_min) / TARGET_RESOLUTION_M)
    height = int((y_max - y_min) / TARGET_RESOLUTION_M)
    transform = from_origin(x_min, y_max, TARGET_RESOLUTION_M,
                            TARGET_RESOLUTION_M)

    def rasterise(values):
        grid = np.full((height, width), np.nan, dtype=np.float32)
        col = ((x_utm - x_min) / TARGET_RESOLUTION_M).astype(int)
        row = ((y_max - y_utm) / TARGET_RESOLUTION_M).astype(int)
        # Clip to grid in case of edge rounding
        m = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        grid[row[m], col[m]] = values[m]
        return grid

    mean_grid = rasterise(mean_vals)
    std_grid = rasterise(std_vals)

    profile = dict(
        driver='GTiff', height=height, width=width, count=1,
        dtype='float32', crs=TARGET_CRS, transform=transform,
        nodata=np.nan, compress='LZW',
        tiled=True, blockxsize=256, blockysize=256,
    )
    for name, grid in (('SGT_1mil_2023_mean_mc30.tif', mean_grid),
                        ('SGT_1mil_2023_std_mc30.tif', std_grid)):
        with rasterio.open(OUT_DIR / name, 'w', **profile) as dst:
            dst.write(grid, 1)
        print(f'Wrote {OUT_DIR / name}  ({grid.shape[0]}×{grid.shape[1]})',
              flush=True)
    return mean_grid, std_grid, profile


# --------------------------------------------------------------------------
# Optional: validation check vs the existing single-pass map
# --------------------------------------------------------------------------
def validation_check(lons, lats, mc_mean):
    """Search known locations for an existing single-pass 2023 prediction
    file. Returns (r, n_compared, note)."""
    # ASSUMPTION: the single-pass results live in one of these directories
    # as .npy chunks (running.py writes coordinates_*.npy + predictions_*.npy).
    # If a GeoTIFF version exists somewhere else, edit the search list.
    candidates = [
        PROJECT_ROOT / 'SOCmapping' / 'Maps',
        PROJECT_ROOT / 'SOCmapping' / 'SGT_Maps_MixedProcessing',
        PROJECT_ROOT / 'SOCmapping' / 'SGT_Maps_Rescaled_NoTitles',
        PROJECT_ROOT / 'SOCmapping' / 'AllResultsMappingTogether_June20th_2025',
        PROJECT_ROOT / 'SOCmapping' / 'AllResultsMappingTogether_NoTitles_June20th_2025',
    ]
    # Look for chunked .npy outputs
    pred_files = []
    coord_files = []
    for d in candidates:
        if not d.exists():
            continue
        for p in d.rglob('predictions_*.npy'):
            pred_files.append(p)
        for p in d.rglob('coordinates_*.npy'):
            coord_files.append(p)
    if not pred_files:
        return None, 0, ('Single-pass GeoTIFF/.npy not located — '
                         'skipping validation. Mean prediction should '
                         'match Figures 14/15 visually.')

    # Match each coord file with its prediction sibling and merge
    coord_map = {p.name.split('_', 1)[1]: p for p in coord_files}
    arr_pred, arr_coord = [], []
    for p in pred_files:
        suffix = p.name.split('_', 1)[1]
        c = coord_map.get(suffix)
        if c is None:
            continue
        try:
            arr_pred.append(np.load(p))
            arr_coord.append(np.load(c))
        except Exception:
            continue
    if not arr_pred:
        return None, 0, 'No matched coord/pred pairs found.'
    sp_pred = np.concatenate(arr_pred)
    sp_coord = np.concatenate(arr_coord)
    if sp_coord.shape[1] >= 2:
        sp_lon, sp_lat = sp_coord[:, 0], sp_coord[:, 1]
    else:
        return None, 0, 'Coord file shape unexpected; skipping check.'

    # Match on rounded coordinates (degrees, 5-decimal precision)
    df_mc = pd.DataFrame({'lon': np.round(lons, 5),
                          'lat': np.round(lats, 5),
                          'mc': mc_mean})
    df_sp = pd.DataFrame({'lon': np.round(sp_lon, 5),
                          'lat': np.round(sp_lat, 5),
                          'sp': sp_pred})
    merged = df_mc.merge(df_sp, on=['lon', 'lat'], how='inner')
    if len(merged) < 100:
        return None, len(merged), 'Too few matches for a stable check.'
    r = float(np.corrcoef(merged['mc'].to_numpy(),
                          merged['sp'].to_numpy())[0, 1])
    return r, int(len(merged)), 'ok'


# --------------------------------------------------------------------------
# Worker mode — runs MC dropout on a single contiguous shard of the grid.
# Invoked as a subprocess by the orchestrator with CUDA_VISIBLE_DEVICES
# already restricted to one physical GPU.
# --------------------------------------------------------------------------
def _worker_run_shard(start_idx: int, end_idx: int, shard_id: int,
                      indices_file: Path | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if indices_file is not None:
        indices = np.load(indices_file)
        print(f'[worker shard={shard_id}] device={device} '
              f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")} '
              f'indices-file={indices_file.name}  n_points={len(indices):,}',
              flush=True)
    else:
        indices = None
        print(f'[worker shard={shard_id}] device={device} '
              f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")} '
              f'range=[{start_idx}, {end_idx})  '
              f'n_points={end_idx - start_idx}', flush=True)

    # Per-shard RNG offset so MC sampling streams don't repeat across GPUs.
    torch.manual_seed(20260513 + shard_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260513 + shard_id)

    model, ckpt = load_model_b(device)
    feature_means, feature_stds, _, _ = load_normalization_stats(ckpt)
    _, loader, n_pts = build_inference_loader(
        feature_means, feature_stds,
        start_idx=start_idx, end_idx=end_idx, indices=indices)
    print(f'[worker shard={shard_id}] points in shard: {n_pts:,}', flush=True)

    lons, lats, mean_pred, std_pred = run_mc_dropout(model, loader, n_pts, device)

    out = OUT_DIR / f'_shard_{shard_id:02d}.pkl'
    with open(out, 'wb') as f:
        pickle.dump({
            'shard_id': shard_id,
            'start_idx': start_idx, 'end_idx': end_idx,
            'lons': lons, 'lats': lats,
            'mean_pred': mean_pred, 'std_pred': std_pred,
        }, f)
    print(f'[worker shard={shard_id}] wrote {out}', flush=True)


# --------------------------------------------------------------------------
# Orchestrator — shards the 1.3 M point grid across visible GPUs and
# launches one worker subprocess per shard, in parallel.
# --------------------------------------------------------------------------
def _orchestrate_shards(gpu_ids: list[int], max_points: int = 0,
                        stride: int = 1, indices_npy: Path | None = None) -> list[dict]:
    """Spawn one worker per GPU, each handling an equal-sized slice of
    the (possibly sub-sampled) inference grid.

    Sub-sampling rules (applied in this order — first match wins):
      • indices_npy != None  → use those exact row indices
      • stride > 1           → take every `stride`-th row of the full CSV
      • max_points > 0       → uniform stride = ceil(total / max_points)
      • otherwise            → use all rows

    Workers receive contiguous slices of the resulting *sub-sampled* index
    array via a per-shard indices.npy file (saved alongside the shard
    pickles), so each shard knows exactly which CSV rows it owns.
    """
    df_full = pd.read_csv(file_path_coordinates_Bavaria_1mil)
    total = len(df_full)

    # Build the array of CSV row indices the experiment should actually use
    if indices_npy is not None:
        sel_idx = np.load(indices_npy).astype(np.int64)
        if sel_idx.max() >= total or sel_idx.min() < 0:
            raise SystemExit(f'indices file {indices_npy} out of range '
                             f'[0, {total}).')
        print(f'Using {len(sel_idx):,} pre-selected indices from {indices_npy}',
              flush=True)
    elif stride > 1:
        sel_idx = np.arange(0, total, stride, dtype=np.int64)
        print(f'Stride {stride} → {len(sel_idx):,} / {total:,} grid points',
              flush=True)
    elif max_points and max_points < total:
        eff_stride = int(np.ceil(total / max_points))
        sel_idx = np.arange(0, total, eff_stride, dtype=np.int64)
        print(f'max_points={max_points:,} → stride={eff_stride} → '
              f'{len(sel_idx):,} / {total:,} grid points', flush=True)
    else:
        sel_idx = np.arange(total, dtype=np.int64)
        print(f'Using full grid: {total:,} points', flush=True)

    n_used = int(len(sel_idx))
    n_shards = len(gpu_ids)
    # Save the per-shard slices of sel_idx to disk so workers can mmap them
    chunk = n_used // n_shards
    shard_index_files = []
    for i in range(n_shards):
        s = i * chunk
        e = (i + 1) * chunk if i < n_shards - 1 else n_used
        shard_idx = sel_idx[s:e]
        path = OUT_DIR / f'_shard_{i:02d}_indices.npy'
        np.save(path, shard_idx)
        shard_index_files.append(path)
    print(f'Total selected grid points: {n_used:,}  shards: {n_shards}',
          flush=True)
    for i, path in enumerate(shard_index_files):
        n_i = len(np.load(path))
        print(f'  shard {i:>2}  GPU {gpu_ids[i]}  n={n_i:>9,}  '
              f'(from {path.name})', flush=True)

    log_dir = OUT_DIR / 'worker_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[int, subprocess.Popen, Path]] = []
    t0 = time.time()
    for shard_id, (gpu_id, idx_path) in enumerate(zip(gpu_ids, shard_index_files)):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['SOC_MC_WORKER'] = '1'
        log_path = log_dir / f'shard_{shard_id:02d}_gpu_{gpu_id}.log'
        cmd = [sys.executable, str(Path(__file__).resolve()),
               '--worker',
               '--shard-id', str(shard_id),
               '--indices-file', str(idx_path)]
        p = subprocess.Popen(cmd, env=env,
                             stdout=open(log_path, 'w'),
                             stderr=subprocess.STDOUT)
        running.append((shard_id, p, log_path))
        print(f'[orchestrator] launched shard {shard_id} on GPU {gpu_id} '
              f'(pid={p.pid}, log={log_path.name})', flush=True)

    # Wait for them all
    failed = []
    while running:
        time.sleep(5)
        still_running = []
        for shard_id, p, log in running:
            rc = p.poll()
            if rc is None:
                still_running.append((shard_id, p, log))
            else:
                elapsed = time.time() - t0
                tag = 'OK' if rc == 0 else f'FAILED rc={rc}'
                print(f'[orchestrator t+{elapsed/60:.1f}m] shard {shard_id} → {tag}',
                      flush=True)
                if rc != 0:
                    failed.append(shard_id)
        running = still_running

    if failed:
        raise RuntimeError(
            f'Worker(s) for shard(s) {failed} exited non-zero. '
            f'Check {log_dir}/shard_<id>_gpu_<g>.log for details.')

    # Load the shard pickles
    shards = []
    for shard_id in range(n_shards):
        with open(OUT_DIR / f'_shard_{shard_id:02d}.pkl', 'rb') as f:
            shards.append(pickle.load(f))
    return shards


def _concat_shards(shards: list[dict]) -> tuple[np.ndarray, ...]:
    lons = np.concatenate([s['lons'] for s in shards])
    lats = np.concatenate([s['lats'] for s in shards])
    mean_pred = np.concatenate([s['mean_pred'] for s in shards]).astype(np.float32)
    std_pred = np.concatenate([s['std_pred'] for s in shards]).astype(np.float32)
    return lons, lats, mean_pred, std_pred


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='MC Dropout uncertainty map (multi-GPU).')
    parser.add_argument('--worker', action='store_true',
                        help='Internal: run ONE shard as a worker subprocess.')
    parser.add_argument('--shard-id', type=int, default=None)
    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--indices-file', type=str, default=None,
                        help='Internal: per-shard .npy of CSV row indices.')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use, e.g. "0,1,2,3". '
                             'Default: all visible CUDA devices.')
    parser.add_argument('--sequential', action='store_true',
                        help='Force the legacy single-GPU loop.')
    parser.add_argument('--keep-shards', action='store_true',
                        help='Keep the _shard_*.pkl + _shard_*_indices.npy '
                             'files after concat.')
    parser.add_argument('--max-points', type=int,
                        default=int(os.environ.get('SOC_MAX_INFERENCE_POINTS', 0)),
                        help='Cap the inference grid to this many points using '
                             'uniform stride sampling (0 = use all 1.3 M). '
                             'Default: $SOC_MAX_INFERENCE_POINTS env var or 0.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Take every Nth row of the CSV. Overrides --max-points '
                             'when > 1. Default: 1 (no striding).')
    parser.add_argument('--indices-npy', type=str, default=None,
                        help='Path to a pre-built .npy of int row indices into '
                             'the 1.3 M CSV; overrides --max-points / --stride.')
    args = parser.parse_args()

    # ----- Worker mode -----
    if args.worker:
        if args.shard_id is None:
            raise SystemExit('--shard-id is required for --worker.')
        if args.indices_file is not None:
            _worker_run_shard(0, 0, args.shard_id, Path(args.indices_file))
        elif args.start_idx is not None and args.end_idx is not None:
            _worker_run_shard(args.start_idx, args.end_idx, args.shard_id)
        else:
            raise SystemExit('worker needs either --indices-file or both '
                             '--start-idx and --end-idx.')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(_describe_paths(), flush=True)

    # ----- Pick GPUs -----
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(',') if g.strip() != '']
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
    use_parallel = (not args.sequential) and len(gpu_ids) >= 2
    print(f'GPUs visible: {gpu_ids}  parallel={use_parallel}', flush=True)

    t0 = time.time()

    indices_npy = Path(args.indices_npy) if args.indices_npy else None

    # ----- Parallel sharded mode -----
    if use_parallel:
        shards = _orchestrate_shards(gpu_ids,
                                     max_points=args.max_points,
                                     stride=args.stride,
                                     indices_npy=indices_npy)
        lons, lats, mean_pred, std_pred = _concat_shards(shards)
        # Re-derive normalization stats for the metadata block by loading
        # the checkpoint once on CPU
        ckpt = torch.load(MODEL_B_PATH, map_location='cpu', weights_only=False)
        feature_means, feature_stds, t_mean, t_std = load_normalization_stats(ckpt)

    # ----- Sequential fallback -----
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Sequential mode on device={device}', flush=True)
        model, ckpt = load_model_b(device)
        feature_means, feature_stds, t_mean, t_std = load_normalization_stats(ckpt)
        # Apply the same sub-sample rules used in the parallel orchestrator
        if indices_npy is not None:
            sel = np.load(indices_npy).astype(np.int64)
        elif args.stride > 1:
            df_tmp = pd.read_csv(file_path_coordinates_Bavaria_1mil)
            sel = np.arange(0, len(df_tmp), args.stride, dtype=np.int64)
        elif args.max_points:
            df_tmp = pd.read_csv(file_path_coordinates_Bavaria_1mil)
            eff = int(np.ceil(len(df_tmp) / args.max_points))
            sel = np.arange(0, len(df_tmp), eff, dtype=np.int64)
        else:
            sel = None
        _, loader, n_pts = build_inference_loader(feature_means, feature_stds,
                                                   indices=sel)
        print(f'Inference grid: {n_pts:,} points', flush=True)
        lons, lats, mean_pred, std_pred = run_mc_dropout(model, loader, n_pts, device)

    elapsed = time.time() - t0
    print(f'\nMC sampling done in {elapsed/60:.1f} min '
          f'(mode={"parallel" if use_parallel else "sequential"}).',
          flush=True)

    # ----- write parquet and metadata -----
    parq = pd.DataFrame({'longitude': lons, 'latitude': lats,
                         'mean_pred_g_per_kg': mean_pred,
                         'std_pred_g_per_kg': std_pred,
                         'cv_pct': np.clip(100 * std_pred / np.maximum(mean_pred, 1e-6),
                                           0, 100)})
    parq.to_parquet(OUT_DIR / 'mc_dropout_points.parquet')
    print(f'Wrote {OUT_DIR / "mc_dropout_points.parquet"}', flush=True)
    np.save(OUT_DIR / 'mc_dropout_mean.npy', mean_pred)
    np.save(OUT_DIR / 'mc_dropout_std.npy', std_pred)
    np.save(OUT_DIR / 'mc_dropout_coords.npy',
            np.column_stack([lons, lats]))

    points_to_geotiffs(lons, lats, mean_pred, std_pred)

    r, n_cmp, note = validation_check(lons, lats, mean_pred)
    txt = []
    txt.append(f'MC mean prediction summary')
    txt.append(f'  n_points: {len(mean_pred)}')
    txt.append(f'  mean: {mean_pred.mean():.3f} g/kg')
    txt.append(f'  std (across grid): {mean_pred.std():.3f} g/kg')
    txt.append(f'  min/max: {mean_pred.min():.3f} / {mean_pred.max():.3f}')
    txt.append('')
    txt.append(f'MC std prediction summary')
    txt.append(f'  mean: {std_pred.mean():.3f} g/kg')
    txt.append(f'  median: {np.median(std_pred):.3f}')
    txt.append(f'  p95: {np.percentile(std_pred, 95):.3f}')
    txt.append(f'  min/max: {std_pred.min():.3f} / {std_pred.max():.3f}')
    txt.append('')
    if r is None:
        txt.append(f'Validation: {note}')
    else:
        flag = 'OK' if r > 0.95 else 'WARNING (r < 0.95)'
        txt.append(f'Validation [{flag}]: Pearson r (mean_mc vs single_pass) '
                   f'= {r:.6f}  (n_compared={n_cmp})')
    (OUT_DIR / 'validation_check.txt').write_text('\n'.join(txt))
    print('\n'.join(txt), flush=True)

    meta = {
        'model_b_path': str(MODEL_B_PATH),
        'n_passes': N_PASSES,
        'batch_size': BATCH_SIZE,
        'inference_time': INFERENCE_TIME,
        'time_before': time_before,
        'window_size': window_size,
        'target_transform': TARGET_TRANSFORM,
        'target_resolution_m': TARGET_RESOLUTION_M,
        'target_crs': TARGET_CRS,
        'n_grid_points': int(len(mean_pred)),
        'feature_means_shape': list(feature_means.shape),
        'feature_stds_shape': list(feature_stds.shape),
        'target_mean': float(t_mean) if t_mean is not None else None,
        'target_std': float(t_std) if t_std is not None else None,
        'validation_pearson_r': r,
        'validation_n_compared': n_cmp,
        'multi_gpu_mode': 'parallel' if use_parallel else 'sequential',
        'gpus_used': gpu_ids,
        'wall_time_seconds': float(elapsed),
        'max_points_requested': int(args.max_points),
        'stride_requested': int(args.stride),
        'indices_npy_requested': args.indices_npy,
    }
    (OUT_DIR / 'mc_dropout_metadata.json').write_text(
        json.dumps(meta, indent=2, default=str))
    print(f'Wrote {OUT_DIR / "mc_dropout_metadata.json"}', flush=True)

    # Clean up the per-shard pickles + index files unless asked to keep them
    if use_parallel and not args.keep_shards:
        for pattern in ('_shard_*.pkl', '_shard_*_indices.npy'):
            for p in OUT_DIR.glob(pattern):
                p.unlink(missing_ok=True)

    print('\nExperiment 2 complete.', flush=True)


if __name__ == '__main__':
    main()
