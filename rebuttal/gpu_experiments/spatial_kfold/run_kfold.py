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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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

# Single source of truth for the --bands-list flag.
sys.path.insert(0, str(_THIS.parent))
from band_subsets import get_band_indices, band_suffix  # noqa: E402

print(_describe_paths(), flush=True)


# --------------------------------------------------------------------------
# Generic model factory — train SGT, 3DCNN, CNNLSTM, or SimpleTransformer
# under the SAME spatial-CV pipeline. All four take (B, C, H, W, T) input
# from the SGT dataloader and return a (B,) regression scalar.
# Sibling architectures are imported lazily inside the factory so a plain
# --model-family sgt run doesn't pay their import cost.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 43-band two-path band encoder
# --------------------------------------------------------------------------
# When we go from the 20-band stack to the 43-band stack (full_extended:
# 20 revision bands + Tier 1/2/3 covariates), dumping all 43 channels into
# the inner model's first conv has two issues:
#   1. SimpleTransformerV2's d_model = C·H·W·T grows from 2500 (20-band) to
#      5375 (43-band), which is no longer divisible by num_heads=4 and
#      causes a wall-time blow-up.
#   2. The 23 new bands are heterogeneous (Landsat SRC, multi-scale terrain,
#      climate/phenology) and adding them raw dilutes the gradient signal
#      for the core 20 bands.
#
# This wrapper preserves the original 20 channels untouched and learns a
# small per-time-step Conv2d that reduces the 23 extended channels to
# n_ext_reduced (default 8). Output is (B, 28, H, W, T), then the inner
# model is built with input_channels=28. Cheap, drop-in, and brings
# SimpleTransformer's d_model back to 3500 (= 4×heads-divisible).
N_CORE_BANDS = 20  # first 20 entries of bands_list_order are the "full_20" set


class _BandTwoPathEncoder(nn.Module):
    def __init__(self, n_core: int, n_ext: int, n_ext_reduced: int):
        super().__init__()
        self.n_core = n_core
        self.n_ext = n_ext
        self.n_ext_reduced = n_ext_reduced
        self.reduce = nn.Conv2d(n_ext, n_ext_reduced, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(n_ext_reduced)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, T) with C == n_core + n_ext
        B, C, H, W, T = x.shape
        if C != self.n_core + self.n_ext:
            raise RuntimeError(
                f'_BandTwoPathEncoder expects {self.n_core + self.n_ext} '
                f'input channels, got {C}.')
        core = x[:, :self.n_core]                         # (B, n_core, H, W, T)
        ext = x[:, self.n_core:]                          # (B, n_ext, H, W, T)
        # Conv2d wants (N, C, H, W); fold T into the batch dim.
        ext_p = ext.permute(0, 4, 1, 2, 3).contiguous()   # (B, T, n_ext, H, W)
        ext_p = ext_p.view(B * T, self.n_ext, H, W)
        ext_p = F.relu(self.bn(self.reduce(ext_p)))       # (B*T, n_ext_reduced, H, W)
        ext_back = (ext_p.view(B, T, self.n_ext_reduced, H, W)
                          .permute(0, 2, 3, 4, 1)
                          .contiguous())                  # (B, n_ext_reduced, H, W, T)
        return torch.cat([core, ext_back], dim=1)         # (B, n_core+n_ext_reduced, H, W, T)


class _TwoPathBandWrapper(nn.Module):
    def __init__(self, inner: nn.Module, n_core: int, n_ext: int, n_ext_reduced: int):
        super().__init__()
        self.encoder = _BandTwoPathEncoder(n_core, n_ext, n_ext_reduced)
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.encoder(x))


def _build_model(args):
    family = getattr(args, 'model_family', 'sgt')
    # Honor --bands-list when building the model: input_channels must
    # match the dataset's channel-dim after subsetting.
    band_indices = get_band_indices(
        getattr(args, 'bands_list', 'full_20'),
        list(bands_list_order),
    )
    n_bands = len(band_indices)
    # Spatial window edge length fed to the model. Defaults to config
    # window_size; --window-size overrides it (and the dataset crop) per run.
    ws = getattr(args, 'window_size', window_size)

    # Two-path 43-band encoder — only active when explicitly requested AND
    # the band stack actually has more than the 20 core bands.
    band_arch = getattr(args, 'band_arch', 'none')
    use_two_path = (band_arch == 'two_path' and n_bands > N_CORE_BANDS)
    if use_two_path:
        n_ext_reduced = int(getattr(args, 'ext_reduced', 8))
        eff_in_channels = N_CORE_BANDS + n_ext_reduced
    else:
        n_ext_reduced = 0
        eff_in_channels = n_bands

    def _wrap(inner: nn.Module) -> nn.Module:
        if use_two_path:
            return _TwoPathBandWrapper(
                inner, N_CORE_BANDS, n_bands - N_CORE_BANDS, n_ext_reduced)
        return inner

    if family == 'sgt':
        if use_two_path:
            # Bypass build_sgt_model — it hardcodes
            # input_channels=len(bands_list_order) (=43 now), which is wrong
            # after the encoder reduces to eff_in_channels (=28 by default).
            from SimpleSGT import SimpleSGT
            if args.model_size == 'small':
                inner = SimpleSGT(
                    input_channels=eff_in_channels,
                    height=ws, width=ws, time_steps=time_before,
                    d_model=args.hidden_size,
                    num_heads=args.num_heads,
                    dropout=args.dropout_rate,
                )
            else:
                from EnhancedSGT import EnhancedSGT
                inner = EnhancedSGT(
                    input_channels=eff_in_channels,
                    height=ws, width=ws, time_steps=time_before,
                    d_model=args.hidden_size,
                    num_heads=args.num_heads,
                    dropout=args.dropout_rate,
                    num_encoder_layers=args.num_layers,
                    expansion_factor=4,
                )
            return _wrap(inner)
        return build_sgt_model(args)

    if family == '3dcnn':
        sib = SOC_CODE_DIR / '3DCNN'
        if str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        from modelCNNMultiYear import Small3DCNN
        return _wrap(Small3DCNN(
            input_channels=eff_in_channels,
            input_height=ws,
            input_width=ws,
            input_time=time_before,
            dropout_rate=args.dropout_rate,
        ))

    if family == 'cnnlstm':
        sib = SOC_CODE_DIR / 'CNNLSTM'
        if str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        from models import RefittedCovLSTM
        # RefittedCovLSTM has its own internal CNN → fixed feature size of
        # 128 going into the LSTM (see models.py line ~100: x_cnn.view
        # (..., 128)). lstm_input_size must therefore be 128. Hidden size
        # is the LSTM hidden state — we mirror args.hidden_size for that.
        return _wrap(RefittedCovLSTM(
            num_channels=eff_in_channels,
            lstm_input_size=128,
            lstm_hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
        ))

    if family == 'simpletransformer':
        sib = SOC_CODE_DIR / 'SimpleTransformer'
        if str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        from modelSimpleTransformerNew import SimpleTransformerV2
        return _wrap(SimpleTransformerV2(
            input_channels=eff_in_channels,
            input_height=ws,
            input_width=ws,
            input_time=time_before,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
        ))

    if family == 'vanilla_transformer':
        # Same input convention and architecture as SimpleSGT (d_model is
        # a free knob, decoupled from input shape) but the gated residual
        # network is replaced by a plain Linear → fair-comparison ablation
        # for the "is SGT's gating worth it?" question.
        from VanillaSpatiotemporalTransformer import VanillaSpatiotemporalTransformer
        return _wrap(VanillaSpatiotemporalTransformer(
            input_channels=eff_in_channels,
            height=ws,
            width=ws,
            time_steps=time_before,
            d_model=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
        ))

    if family == 'lightweight_transformer':
        # Pure-transformer baseline — same head + same d_model/heads/layers
        # contract as vanilla_transformer, but the spatial Conv2d+AvgPool
        # block is replaced by a flat Linear projection. Isolates "what
        # does the CNN spatial encoder buy you?" at matched parameter
        # scale (vanilla and lightweight both honour --hidden_size and land
        # in the 85k-370k param band, unlike SimpleTransformerV2 at 11M).
        from LightweightTransformer import LightweightTransformer
        return _wrap(LightweightTransformer(
            input_channels=eff_in_channels,
            height=ws,
            width=ws,
            time_steps=time_before,
            d_model=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
        ))

    raise ValueError(f'Unknown --model-family: {family!r}. '
                     f'Choose from: sgt, 3dcnn, cnnlstm, simpletransformer, '
                     f'vanilla_transformer, lightweight_transformer.')


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


def assign_balanced_clusters(lat, lon, k: int, seed: int = 42) -> np.ndarray:
    """K spatially-coherent clusters of (near-)equal size.

    K-Means on equirectangular coordinates (longitude scaled by cos(mean lat)
    so a degree of lon ≈ a degree of lat in distance) gives the cluster
    centres; a capacity-constrained nearest-centre assignment then forces every
    cluster to hold floor(N/k) or ceil(N/k) points (equal ±1). Points are
    assigned in order of how strongly they prefer their nearest centre (largest
    gap to the 2nd-nearest first), so boundary distortion from the equal-size
    constraint is minimised. Returns an int label array of length N."""
    from sklearn.cluster import KMeans
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    n = len(lat)
    xy = np.column_stack([lon * np.cos(np.deg2rad(lat.mean())), lat])
    centres = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(xy).cluster_centers_
    d = np.linalg.norm(xy[:, None, :] - centres[None, :, :], axis=2)   # (n, k)
    cap = int(np.ceil(n / k))
    srt = np.sort(d, axis=1)
    gap = (srt[:, 1] - srt[:, 0]) if k > 1 else np.zeros(n)
    order = np.argsort(-gap)
    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(k, dtype=int)
    for i in order:
        for c in np.argsort(d[i]):
            if counts[c] < cap:
                labels[i] = c; counts[c] += 1; break
    return labels


def build_folds_spatial_deciles(df: pd.DataFrame, n_folds: int = 10,
                                 buffer_km: float = 1.2,
                                 axis: str = 'lat', seed: int = 42) -> list[dict]:
    """Spatial folds with EQUAL N per fold.

    axis='lat'     → quantile strips of GPS_LAT  (south↔north bands);
    axis='lon'     → quantile strips of GPS_LONG (west↔east bands);
    axis='cluster' → K spatially-coherent, equal-size K-Means clusters
                     (see assign_balanced_clusters).

    Strip boundaries are quantiles of the split coordinate, so each fold's test
    half holds ≈ len(df) / n_folds rows regardless of sample density; clusters
    are balanced to the same ±1 tolerance.

    Each train pool then has a buffer_km great-circle buffer applied: any
    train-candidate within buffer_km of any test row is excluded from train
    (kept in buffer_idx, not scored). The buffer is identical regardless of how
    the test set was carved out.

    Returns list[dict] with fold_id, split_axis, test_idx, train_idx,
    buffer_idx, plus edge_lo/edge_hi (strips) or centroid_lat/centroid_lon
    (cluster). For clusters edge_lo/edge_hi are NaN."""
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError('build_folds requires df.index == RangeIndex')
    if axis not in ('lat', 'lon', 'cluster'):
        raise ValueError(f"axis must be 'lat', 'lon' or 'cluster', got {axis!r}")

    lat_all = df['GPS_LAT'].to_numpy(dtype=float)
    lon_all = df['GPS_LONG'].to_numpy(dtype=float)

    test_masks, extra = [], []
    if axis == 'cluster':
        labels = assign_balanced_clusters(lat_all, lon_all, n_folds, seed)
        for i in range(n_folds):
            m = labels == i
            test_masks.append(m)
            extra.append({'centroid_lat': float(lat_all[m].mean()),
                          'centroid_lon': float(lon_all[m].mean())})
    else:
        coord = lat_all if axis == 'lat' else lon_all
        edges = np.quantile(coord, np.linspace(0, 1, n_folds + 1))
        edges[-1] = coord.max() + 1e-9
        edges[0] = coord.min() - 1e-9
        for i in range(n_folds):
            lo, hi = float(edges[i]), float(edges[i + 1])
            test_masks.append((coord >= lo) & (coord < hi))
            extra.append({'edge_lo': lo, 'edge_hi': hi})

    folds = []
    for i in range(n_folds):
        in_test = test_masks[i]
        test_idx = df.index[in_test].to_numpy()
        train_pool_idx = df.index[~in_test].to_numpy()
        d = min_distance_to_set_km(lat_all[~in_test], lon_all[~in_test],
                                   lat_all[in_test], lon_all[in_test])
        keep = d >= buffer_km
        folds.append({
            'fold_id': i, 'split_axis': axis,
            'edge_lo': float('nan'), 'edge_hi': float('nan'),
            **extra[i],
            'test_idx': test_idx,
            'train_idx': train_pool_idx[keep],
            'buffer_idx': train_pool_idx[~keep],
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


def make_dataset(df: pd.DataFrame, feature_means=None, feature_stds=None,
                 band_indices=None, window_size=window_size):
    """Build the per-sample dataset.

    band_indices, if provided, restricts the channel dim to that subset
    AFTER normalization (so the caller can pass the full 20-channel
    feature_means/feature_stds and let the wrapper slice the output).

    window_size sets the H×W spatial window cropped per sample (default from
    config). The stored raster tiles are far larger than any window, so a
    bigger window is just a larger crop — no data regeneration needed.
    """
    sample_paths, data_paths = separate_and_add_data()
    sample_paths = list(dict.fromkeys(_flatten(sample_paths)))
    data_paths = list(dict.fromkeys(_flatten(data_paths)))
    ds = MultiRasterDatasetMultiYears(
        samples_coordinates_array_subfolders=sample_paths,
        data_array_subfolders=data_paths,
        dataframe=df.reset_index(drop=True),
        time_before=time_before,
        window_size=window_size,
    )
    if feature_means is not None and feature_stds is not None:
        ds = _NormalizingWrapper(ds, feature_means, feature_stds)
    if band_indices is not None and len(band_indices) < len(bands_list_order):
        ds = _BandSubsetWrapper(ds, band_indices)
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


class _BandSubsetWrapper(Dataset):
    """Slice the channel dimension of each sample's features tensor to a
    chosen subset of bands. Channel ordering follows bands_list_order from
    SpatiotemporalGatedTransformer/config.py. Applied AFTER normalization
    so the project-wide 20-channel feature_means/stds remain valid; the
    cost is a few extra disk reads (the MultiRaster dataset still fetches
    all 20 bands), but the simplicity is worth it at this dataset scale."""
    def __init__(self, base, band_indices):
        self.base = base
        self.band_indices = torch.as_tensor(list(band_indices), dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        lon, lat, features, oc = self.base[idx]
        # features: (C, ...) — index along dim 0 to keep only the chosen
        # channels. Works for both (C, T, H, W) and (C, H, W) layouts.
        features = features.index_select(0, self.band_indices)
        return lon, lat, features, oc


class _AugmentingWrapper(Dataset):
    """D4 spatial augmentation for the train loader only.

    Picks a random element of the dihedral group on the spatial (H, W) plane
    of `features` — dims (-3, -2) of the (C, H, W, T) tensor, NOT the last two
    (which are W, T): 4 rotations × 2 flips = 8 equivalence
    classes. Soil patches have no orientation prior, so this is
    label-preserving. Different draws of the same row give different views,
    which is the point — combined with WeightedRandomSampler oversampling,
    rare-OC rows stop being literal duplicates.
    """
    def __init__(self, base, seed: int = 0):
        self.base = base
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        lon, lat, features, oc = self.base[idx]
        # features layout is (C, H, W, T). The spatial plane is dims (-3, -2).
        # rot90/flip on (-2, -1) would mix the WIDTH and TIME axes — harmless on
        # shape only while window_size == time_before, but when they differ it
        # transposes W↔T and makes per-sample shapes inconsistent (collate fails).
        k = int(self._rng.integers(0, 4))
        if k:
            features = torch.rot90(features, k=k, dims=[-3, -2])
        if self._rng.integers(0, 2):
            features = torch.flip(features, dims=[-2])
        return lon, lat, features, oc


def compute_density_weights(oc_values: np.ndarray, alpha: float = 0.5,
                             eps: float = 1e-6) -> np.ndarray:
    """Per-sample weights ∝ kde(log(OC))^(-alpha), normalized to mean 1.

    log(OC) because SOC is approximately log-normal. KDE bandwidth from
    Scott's rule (scipy default) — not tuned. alpha=0 → uniform,
    alpha=1 → full inverse-frequency, alpha=0.5 → sqrt-frequency (the
    standard imbalance-regression compromise; Yang et al. ICML 2021).
    """
    from scipy.stats import gaussian_kde
    log_oc = np.log(np.maximum(oc_values.astype(float), eps))
    kde = gaussian_kde(log_oc)           # Scott's rule by default
    density = kde(log_oc)
    w = np.power(np.maximum(density, eps), -alpha)
    w = w * (len(w) / w.sum())           # mean weight = 1
    return w.astype(np.float64)


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

    _where = (f'centroid ({fold["centroid_lat"]:.3f}, {fold["centroid_lon"]:.3f})'
              if fold["split_axis"] == 'cluster'
              else f'[{fold["edge_lo"]:.4f}, {fold["edge_hi"]:.4f})')
    print(f'\n=== Fold {fold_id} | {fold["split_axis"]} {_where} | seed={seed} ===',
          flush=True)

    train_df_raw = df.loc[fold['train_idx']].reset_index(drop=True)
    test_df = df.loc[fold['test_idx']].reset_index(drop=True)

    # Sampler mode: kde (KDE-density weighted, no duplication) or qcut (legacy
    # quantile-bin upsampling with literal row duplication).
    sample_weights: np.ndarray | None = None
    if args.sampler_mode == 'kde':
        train_df = train_df_raw
        sample_weights = compute_density_weights(
            train_df['OC'].to_numpy(), alpha=args.alpha_density)
        print(f'KDE sampler: alpha={args.alpha_density}  '
              f'w in [{sample_weights.min():.3f}, {sample_weights.max():.3f}]  '
              f'mean={sample_weights.mean():.3f}', flush=True)
    else:
        train_df = rebalance_by_oc_bin(
            train_df_raw, n_bins=args.rebalance_n_bins,
            min_ratio=args.rebalance_min_ratio)

    print(f'n_train_raw={len(train_df_raw)}  n_train={len(train_df)}  '
          f'n_test={len(test_df)}  n_buffer_excluded={len(fold["buffer_idx"])}',
          flush=True)
    print(f'Test SOC: mean={test_df.OC.mean():.2f} std={test_df.OC.std():.2f} '
          f'max={test_df.OC.max():.1f} %>50={100*(test_df.OC>50).mean():.2f}%',
          flush=True)

    # Resolve --bands-list once and pass to make_dataset for slicing.
    _band_indices = get_band_indices(
        getattr(args, 'bands_list', 'full_20'),
        list(bands_list_order),
    )
    train_ds = make_dataset(train_df, feature_means, feature_stds,
                             band_indices=_band_indices,
                             window_size=args.window_size)
    test_ds = make_dataset(test_df, feature_means, feature_stds,
                            band_indices=_band_indices,
                            window_size=args.window_size)
    if args.augment_train:
        train_ds = _AugmentingWrapper(train_ds, seed=seed)
        print('Train augmentation: D4 spatial (rot90 × flip)', flush=True)

    num_workers = int(os.environ.get('SOC_KFOLD_NUM_WORKERS', 0))
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.per_gpu_batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # Resolve gradient accumulation steps once we know num_processes
    accum_steps = _resolve_accum_steps(args, accelerator.num_processes)
    effective_batch = accelerator.num_processes * args.per_gpu_batch_size * accum_steps
    if accelerator.is_main_process:
        print(f'[grad-accum] num_gpus={accelerator.num_processes}  '
              f'per_gpu_batch={args.per_gpu_batch_size}  accum_steps={accum_steps}  '
              f'effective_batch={effective_batch}', flush=True)

    model = _build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {type(model).__name__} '
          f'(family={getattr(args, "model_family", "sgt")}, '
          f'{n_params:,} trainable params)', flush=True)

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
        loss_alpha=getattr(args, 'loss_alpha', 1.0),
        chi2_weight=getattr(args, 'chi2_weight', 0.1),
    )

    wandb_run.finish()

    # ----- Save fold artefacts -------------------------------------------
    model_config = {
        'input_channels': len(bands_list_order),
        'height': args.window_size, 'width': args.window_size,
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

    test_year = test_df['year'].to_numpy()
    test_alt = (test_df['altitude'].to_numpy() if 'altitude' in test_df.columns
                else np.full(len(test_df), np.nan))

    # Persist per-fold predictions so --aggregate-only can reconstruct the
    # cross-fold tables even when folds were launched as separate processes
    # (via run_folds_parallel.py).
    per_fold_pred_path = OUT_DIR / f'fold_{fold_id}_predictions.parquet'
    pd.DataFrame({
        'GPS_LAT': test_lat, 'GPS_LONG': test_lon,
        'OC_actual': test_actual, 'OC_predicted': test_pred,
        'fold_id': fold_id, 'year': test_year, 'altitude': test_alt,
    }).to_parquet(per_fold_pred_path)
    print(f'Saved {per_fold_pred_path}', flush=True)

    per_fold_metrics = {
        'fold_id': fold_id,
        'split_axis': fold['split_axis'],
        'edge_lo': fold['edge_lo'], 'edge_hi': fold['edge_hi'],
        'centroid_lat': fold.get('centroid_lat'),
        'centroid_lon': fold.get('centroid_lon'),
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
    }
    (OUT_DIR / f'fold_{fold_id}_summary.json').write_text(
        json.dumps(per_fold_metrics, indent=2, default=str))

    return {
        **per_fold_metrics,
        '_predictions': {
            'lon': test_lon, 'lat': test_lat,
            'pred': test_pred, 'actual': test_actual,
            'year': test_year, 'altitude': test_alt,
        },
    }


# --------------------------------------------------------------------------
# Reporting (mostly unchanged from previous version)
# --------------------------------------------------------------------------
OC_BANDS_GKG: list[tuple[float, float]] = [
    (0.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, float('inf'))
]


def _band_label(lo: float, hi: float) -> str:
    return f'[{lo:g}, {hi:g})' if np.isfinite(hi) else f'[{lo:g}, ∞)'


def _metrics_for(pred: np.ndarray, actual: np.ndarray) -> dict:
    """R²/RMSE/MAE/RPIQ on a (pred, actual) pair. NaN-safe."""
    n = len(actual)
    if n == 0 or len(pred) != n:
        return {'n': 0, 'r2': float('nan'), 'pearson_r2': float('nan'),
                'rmse': float('nan'), 'mae': float('nan'),
                'bias': float('nan'), 'rpiq': float('nan')}
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(ss_res / n))
    mae = float(np.mean(np.abs(pred - actual)))
    bias = float((pred - actual).mean())
    if n >= 2 and np.std(pred) > 0 and np.std(actual) > 0:
        pr = float(np.corrcoef(pred, actual)[0, 1])
    else:
        pr = float('nan')
    q1, q3 = np.percentile(actual, [25, 75]) if n >= 4 else (np.nan, np.nan)
    rpiq = float((q3 - q1) / rmse) if rmse > 0 and np.isfinite(q1) else float('nan')
    return {'n': int(n), 'r2': r2, 'pearson_r2': pr ** 2 if pr == pr else float('nan'),
            'rmse': rmse, 'mae': mae, 'bias': bias, 'rpiq': rpiq}


def _pooled_predictions(fold_results: list[dict]) -> pd.DataFrame:
    frames = []
    for r in fold_results:
        p = r['_predictions']
        frames.append(pd.DataFrame({
            'GPS_LAT': p['lat'], 'GPS_LONG': p['lon'],
            'OC_actual': p['actual'], 'OC_predicted': p['pred'],
            'fold_id': r['fold_id'], 'year': p['year'],
            'altitude': p['altitude'],
        }))
    return pd.concat(frames, ignore_index=True)


def stratified_band_metrics(pooled: pd.DataFrame) -> list[dict]:
    """Compute per-OC-band metrics on the pooled (across-fold) predictions.

    Reports both the regime-restricted metrics (essential for the rebuttal —
    answers 'is the headline R² coming from the easy mineral-soil middle?')
    and an 'all rows' row spanning the full kept range.
    """
    actual = pooled['OC_actual'].to_numpy()
    pred = pooled['OC_predicted'].to_numpy()
    out = []
    for lo, hi in OC_BANDS_GKG:
        mask = (actual >= lo) & (actual < hi)
        m = _metrics_for(pred[mask], actual[mask])
        m['band'] = _band_label(lo, hi)
        m['lo'] = lo
        m['hi'] = hi
        out.append(m)
    m_all = _metrics_for(pred, actual)
    m_all['band'] = 'all'
    m_all['lo'] = float(actual.min()) if len(actual) else float('nan')
    m_all['hi'] = float(actual.max()) if len(actual) else float('nan')
    out.append(m_all)
    return out


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

    pooled = _pooled_predictions(fold_results)
    band_rows = stratified_band_metrics(pooled)

    sampler_mode = getattr(args, 'sampler_mode', 'qcut')
    alpha_density = getattr(args, 'alpha_density', None)
    augment_train = getattr(args, 'augment_train', False)
    max_oc = getattr(args, 'max_oc', None)

    # Full config manifest fields — make every run self-documenting so the
    # summarizer can group/filter by any dimension (bands, fold geometry, …).
    bands_list = getattr(args, 'bands_list', None)
    n_bands = getattr(args, 'n_bands', None)
    if n_bands is None and bands_list:
        try:
            n_bands = len(get_band_indices(bands_list, list(bands_list_order)))
        except Exception:
            n_bands = None

    summary_json = {
        'fold_results': rows,
        'across_folds': {
            'r2_mean': r2_m, 'r2_std': r2_s, 'r2_ci95': r2_ci,
            'rmse_mean': rmse_m, 'rmse_std': rmse_s, 'rmse_ci95': rmse_ci,
            'mae_mean': mae_m, 'mae_std': mae_s, 'mae_ci95': mae_ci,
            'rpiq_mean': rpiq_m, 'rpiq_std': rpiq_s, 'rpiq_ci95': rpiq_ci,
        },
        'pooled_by_oc_band': band_rows,
        'original_single_split': ORIGINAL_SINGLE_SPLIT,
        'n_folds': args.num_folds,
        'distance_threshold_km': args.fold_buffer_km,
        'split_axis': args.split_axis,
        'fold_geometry': ('balanced_kmeans_clusters_equal_n'
                          if args.split_axis == 'cluster'
                          else f'{"latitude" if args.split_axis == "lat" else "longitude"}_deciles_equal_n'),
        'recipe': {
            'bands_list': bands_list, 'n_bands': n_bands,
            'band_arch': getattr(args, 'band_arch', 'none'),
            'ext_reduced': getattr(args, 'ext_reduced', None),
            'model_family': getattr(args, 'model_family', None),
            'model_size': getattr(args, 'model_size', None),
            'hidden_size': getattr(args, 'hidden_size', None),
            'dropout_rate': getattr(args, 'dropout_rate', None),
            'seed_base': getattr(args, 'seed_base', None),
            'lr': args.lr, 'loss_type': args.loss_type,
            'target_transform': args.target_transform,
            'window_size': args.window_size,
            'num_epochs': args.num_epochs,
            'per_gpu_batch_size': args.per_gpu_batch_size,
            'effective_batch_size': args.effective_batch_size,
            'lr_scheduler': args.lr_scheduler, 'lr_min': args.lr_min,
            'num_heads': args.num_heads, 'num_layers': args.num_layers,
            'max_oc': max_oc,
            'sampler_mode': sampler_mode,
            'alpha_density': alpha_density,
            'augment_train': augment_train,
            'rebalance_n_bins': args.rebalance_n_bins,
            'rebalance_min_ratio': args.rebalance_min_ratio,
        },
    }
    (OUT_DIR / 'kfold_results_summary.json').write_text(
        json.dumps(summary_json, indent=2, default=str))

    if sampler_mode == 'kde':
        sampler_desc = (f'KDE-weighted sampling on log(OC) with α={alpha_density} '
                        f'(no row duplication; rare-OC samples drawn ~k× more often '
                        f'per epoch, each draw a fresh augmented patch).')
    else:
        sampler_desc = (f'Legacy qcut rebalancing: {args.rebalance_n_bins} OC '
                        f'quantile bins upsampled (by duplication) to '
                        f'≥ {args.rebalance_min_ratio:.0%} of the densest bin.')
    aug_desc = ('D4 spatial augmentation (rot90 × flip) applied to train patches.'
                if augment_train else 'No train augmentation.')

    if args.split_axis == 'cluster':
        geom_title = 'balanced K-Means clusters'
        geom_sent = (f"Each fold's test set is one of {args.num_folds} equal-size "
                     f"balanced K-Means spatial clusters "
                     f"(~{int(100/args.num_folds)}% of points each).")
        col_header = 'Cluster centroid (lat, lon)'
        def _range_str(r):
            cl, co = r.get('centroid_lat'), r.get('centroid_lon')
            return f'({cl:.3f}, {co:.3f})' if cl is not None else '—'
    else:
        axis_word = 'latitude' if args.split_axis == 'lat' else 'longitude'
        coord_col = 'GPS_LAT' if args.split_axis == 'lat' else 'GPS_LONG'
        geom_title = f'{axis_word} deciles'
        geom_sent = (f"Each fold's test half is one decile of {coord_col} "
                     f"(~{int(100/args.num_folds)}% of points).")
        col_header = f'{axis_word.capitalize()} range'
        def _range_str(r):
            return f'[{r["edge_lo"]:.4f}, {r["edge_hi"]:.4f})'
    md = []
    md.append(f'# Spatial {args.num_folds}-fold CV — {geom_title} (equal n)')
    md.append('')
    md.append(f'{geom_sent} Train pool is the complement, minus a '
              f'{args.fold_buffer_km} km buffer zone. Spatial window fed to the '
              f'model: {args.window_size}×{args.window_size}. **Modeling domain: '
              f'OC ≤ {max_oc} g/kg** (non-histosol soils, per WRB). {sampler_desc} '
              f'{aug_desc} EnhancedSGT (heads={args.num_heads}, '
              f'layers={args.num_layers}) trained for {args.num_epochs} epochs with '
              f'`{args.lr_scheduler}` LR schedule from {args.lr} → '
              f'{args.lr_min if args.lr_scheduler in ("cosine","cosine_warm_restarts") else "n/a"}, '
              f'Adam, {args.loss_type.upper()} on {args.target_transform}'
              f'-transformed target.')
    md.append('')
    md.append('## Per-fold metrics')
    md.append('')
    md.append(f'| Fold | {col_header} | n_test | n_train | R² | RMSE (g/kg) | MAE (g/kg) | RPIQ |')
    md.append('|------|-----------|--------|---------|-----|-------------|------------|------|')
    for r in fold_results:
        md.append(f'| {r["fold_id"]} | {_range_str(r)} | '
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

    md.append('## Pooled metrics by OC band (across all folds)')
    md.append('')
    md.append('Predictions concatenated across all 10 fold test sets, then binned '
              'by actual OC. This is the regime-by-regime breakdown — answers the '
              'reviewer question "where does the R² come from?"')
    md.append('')
    md.append('| OC band (g/kg) | n | R² | RMSE (g/kg) | MAE (g/kg) | Bias | RPIQ |')
    md.append('|-----------------|----|-----|-------------|------------|------|------|')
    for b in band_rows:
        if b['n'] == 0:
            continue
        md.append(f'| {b["band"]} | {b["n"]} | {b["r2"]:.4f} | '
                  f'{b["rmse"]:.3f} | {b["mae"]:.3f} | '
                  f'{b["bias"]:+.3f} | {b["rpiq"]:.3f} |')
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
    split_axis = folds_meta[0].get('split_axis', 'lat') if folds_meta else 'lat'
    axis_word = {'lat': 'latitude-decile', 'lon': 'longitude-decile',
                 'cluster': 'balanced-cluster'}.get(split_axis, split_axis)
    for f, color in zip(folds_meta, colors):
        idx = f['test_idx']
        ax.scatter(df_master.loc[idx, 'GPS_LONG'],
                   df_master.loc[idx, 'GPS_LAT'],
                   s=5, color=color, label=f'Fold {f["fold_id"]}', alpha=0.7)
        if split_axis == 'lat':
            ax.axhline(f['edge_hi'], color='black', linestyle='--', linewidth=0.5)
        elif split_axis == 'lon':
            ax.axvline(f['edge_hi'], color='black', linestyle='--', linewidth=0.5)
        # cluster mode: the coloured points already show the partition
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'Bavaria — {n_folds} {axis_word} folds '
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
    p.add_argument('--loss_type', type=str, default='l1',
                   choices=['l1', 'mse', 'chi2', 'composite_l1', 'composite_l2'],
                   help='composite_l1/composite_l2 add a Pearson chi-square term '
                        '(in original g/kg space): '
                        'loss = loss_alpha × base + chi2_weight × chi-square.')
    p.add_argument('--loss-alpha', type=float, default=1.0,
                   help='Weight on the base term in composite losses (default 1.0).')
    p.add_argument('--chi2-weight', type=float, default=0.1,
                   help='Weight on the chi-square term in composite losses '
                        '(default 0.1). Ignored for plain l1/mse.')
    p.add_argument('--target_transform', type=str, default='normalize',
                   choices=['none', 'log', 'normalize'])
    p.add_argument('--hidden_size', type=int, default=hidden_size)
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--window-size', type=int, default=window_size,
                   help='Edge length (pixels) of the square spatial window fed '
                        'to the model AND the crop size the dataset extracts. '
                        f'Default {window_size} (config). The stored raster tiles '
                        'are far larger than any window, so a bigger value '
                        '(e.g. 7 or 9) is just a larger crop — no data '
                        'regeneration needed. One flag keeps model H×W and the '
                        'data crop consistent.')
    p.add_argument('--model-size', type=str, default='big',
                   choices=['small', 'big'])
    p.add_argument('--model-family', type=str, default='sgt',
                   choices=['sgt', '3dcnn', 'cnnlstm', 'simpletransformer',
                            'vanilla_transformer', 'lightweight_transformer'],
                   help='Architecture to train. "sgt" uses the EnhancedSGT/'
                        'SimpleSGT variants (selected by --model-size). The '
                        'other four are 20-channel ports of sibling models '
                        'at the same (5×5×5) spatiotemporal window. '
                        '"vanilla_transformer" is the SimpleSGT-minus-GRN '
                        'fair-comparison ablation.')
    p.add_argument('--band-arch', type=str, default='none',
                   choices=['none', 'two_path'],
                   help='Optional band-input wrapper. "none" = feed all '
                        'channels straight into the inner model (default, '
                        'matches every prior run). "two_path" = only active '
                        'when len(bands) > 20: keep the 20 core bands raw '
                        'and learn a Conv2d that reduces the remaining '
                        '(--ext-reduced default 8) extended bands per time '
                        'step; concat → 28-channel input. Designed to keep '
                        '43-band runs from blowing up SimpleTransformer\'s '
                        'd_model and to give the new Tier 1/2/3 bands a '
                        'dedicated representation pathway.')
    p.add_argument('--ext-reduced', type=int, default=8,
                   help='[--band-arch two_path only] Channel-count after '
                        'compressing the extended (non-core) bands. Default 8.')
    p.add_argument('--bands-list', type=str, default='full_20',
                   choices=['full_20', 'original_6', 'full_extended', 'full_extended_s2'],
                   help='Covariate-stack subset. "full_20" = the 20 revision '
                        'bands (default). "full_extended" = 20 + Tier 1/2/3 '
                        'covariates (Landsat SRC, multi-scale terrain, '
                        'climate/phenology). "original_6" restricts to the 6 bands '
                        'used in the original submission '
                        '(Elevation, LAI, LST, MODIS_NPP, SoilEvaporation, '
                        'TotalEvapotranspiration) for direct comparison. '
                        'The dataset still fetches all 20 bands; the wrapper '
                        'slices the channel dim after normalization.')
    p.add_argument('--per-gpu-batch-size', type=int, default=256)
    p.add_argument('--effective-batch-size', type=int, default=2048)
    p.add_argument('--accum-steps', type=int, default=0)
    p.add_argument('--num-epochs', type=int, default=CONFIG_NUM_EPOCHS)
    p.add_argument('--lr-scheduler', type=str, default='none',
                   choices=['none', 'cosine', 'cosine_warm_restarts', 'exponential'])
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--lr-gamma', type=float, default=0.99)
    p.add_argument('--lr-restart-T0', type=int, default=50)

    # ----- K-fold-specific -----
    p.add_argument('--num-folds', type=int, default=10,
                   help='Number of spatial folds (e.g. 5 or 10).')
    p.add_argument('--split-axis', type=str, default='lat',
                   choices=['lat', 'lon', 'cluster'],
                   help='How the equal-size folds are carved: "lat" = '
                        'south↔north latitude bands (original), "lon" = '
                        'west↔east longitude bands, "cluster" = equal-size '
                        'balanced K-Means spatial clusters. Buffer is '
                        'great-circle in every case.')
    p.add_argument('--fold-buffer-km', type=float, default=1.2,
                   help='Train/test buffer-zone distance in km.')
    p.add_argument('--fold', type=int, default=None,
                   help='Run a single fold by ID (0..num_folds-1). '
                        'Default: run all folds sequentially.')
    p.add_argument('--seed-base', type=int, default=42,
                   help='Per-fold seed = seed_base + fold_id.')
    # ----- Modeling-domain cap (the rebuttal anchor) -----
    p.add_argument('--max-oc', type=float, default=120.0,
                   help='Upper OC cap in g/kg applied after loading the parquet. '
                        '120 = WRB histosol threshold (default; non-histosol soils only). '
                        'Pass 150 for the legacy no-cap behavior.')
    # ----- Sampler / rebalancing -----
    p.add_argument('--sampler-mode', type=str, default='kde',
                   choices=['kde', 'qcut'],
                   help='Train sampler: kde = WeightedRandomSampler on '
                        'kde(log(OC))^(-alpha), no duplication (recommended). '
                        'qcut = legacy quantile-bin row duplication.')
    p.add_argument('--alpha-density', type=float, default=0.5,
                   help='KDE sampler exponent. 0=uniform, 1=full inverse-frequency. '
                        '0.5 = sqrt-frequency (Yang et al. ICML 2021).')
    p.add_argument('--augment-train', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Apply D4 spatial augmentation (rot90 × flip) to train '
                        'patches. Soil has no orientation prior; this is '
                        'label-preserving. Use --no-augment-train to disable.')
    p.add_argument('--rebalance-n-bins', type=int, default=128,
                   help='[qcut mode only] Number of OC quantile bins.')
    p.add_argument('--rebalance-min-ratio', type=float, default=0.75,
                   help='[qcut mode only] Per-bin floor as fraction of densest bin.')
    p.add_argument('--aggregate-only', action='store_true',
                   help='Skip training. Read all fold_<i>_predictions.parquet '
                        'from OUT_DIR and write the cross-fold summary + '
                        'stratified-by-band tables.')
    p.add_argument('--out-subdir', type=str, default=None,
                   help='If set, redirect all kfold outputs to OUT_DIR/<subdir>. '
                        'Used by the architecture sweep to keep config results separate.')
    p.add_argument('--skip-figure', action='store_true',
                   help='Skip matplotlib figure (faster, lighter deps).')
    return p.parse_args()


# --------------------------------------------------------------------------
# Aggregate-only mode — reconstructs fold_results from per-fold parquets
# --------------------------------------------------------------------------
def aggregate_from_disk(args) -> int:
    pred_paths = sorted(OUT_DIR.glob('fold_*_predictions.parquet'))
    if not pred_paths:
        print(f'[aggregate-only] no fold_*_predictions.parquet in {OUT_DIR}',
              file=sys.stderr)
        return 1

    fold_results = []
    for pp in pred_paths:
        fid = int(pp.stem.split('_')[1])
        p = pd.read_parquet(pp)
        summary_p = OUT_DIR / f'fold_{fid}_summary.json'
        if summary_p.exists():
            meta = json.loads(summary_p.read_text())
        else:
            actual = p['OC_actual'].to_numpy()
            pred = p['OC_predicted'].to_numpy()
            is_cluster = args.split_axis == 'cluster'
            coord = p['GPS_LAT'] if args.split_axis == 'lat' else p['GPS_LONG']
            meta = {'fold_id': fid,
                    'split_axis': args.split_axis,
                    'edge_lo': float('nan') if is_cluster else float(coord.min()),
                    'edge_hi': float('nan') if is_cluster else float(coord.max()),
                    'centroid_lat': float(p['GPS_LAT'].mean()) if is_cluster else None,
                    'centroid_lon': float(p['GPS_LONG'].mean()) if is_cluster else None,
                    'n_test': len(p), 'n_train': 0, 'n_train_raw': 0,
                    'n_buffer': 0, 'accum_steps': 0, 'effective_batch_size': 0,
                    'test_oc_mean': float(actual.mean()),
                    'test_oc_std': float(actual.std()),
                    'test_oc_max': float(actual.max()),
                    'test_pct_gt_50': float(100 * (actual > 50).mean()),
                    'best_epoch_r2_during_training': float('nan'),
                    **_metrics_for(pred, actual)}
        fold_results.append({
            **meta,
            '_predictions': {
                'lon': p['GPS_LONG'].to_numpy(),
                'lat': p['GPS_LAT'].to_numpy(),
                'pred': p['OC_predicted'].to_numpy(),
                'actual': p['OC_actual'].to_numpy(),
                'year': p['year'].to_numpy() if 'year' in p.columns
                        else np.zeros(len(p), dtype=int),
                'altitude': (p['altitude'].to_numpy() if 'altitude' in p.columns
                             else np.full(len(p), np.nan)),
            },
        })
    fold_results.sort(key=lambda r: r['fold_id'])
    print(f'[aggregate-only] loaded {len(fold_results)} folds, '
          f'{sum(len(r["_predictions"]["actual"]) for r in fold_results)} '
          f'pooled predictions.', flush=True)

    write_results(fold_results, args)
    write_predictions_parquet(fold_results)
    print(f'[aggregate-only] wrote kfold_results.md + summary.json + '
          f'kfold_predictions_all_folds.parquet to {OUT_DIR}', flush=True)
    return 0


# --------------------------------------------------------------------------
# Main — sequential, single GPU, one Accelerator for all folds
# --------------------------------------------------------------------------
def main():
    global OUT_DIR
    args = parse_args()
    if args.out_subdir:
        OUT_DIR = OUT_DIR / args.out_subdir
        print(f'[out-subdir] kfold outputs redirected to {OUT_DIR}', flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        rc = aggregate_from_disk(args)
        sys.exit(rc)

    _build_model_ready_dataset()

    df = pd.read_parquet(MODEL_READY).reset_index(drop=True)
    if args.max_oc is not None and args.max_oc > 0:
        n_before = len(df)
        df = df[df['OC'] <= args.max_oc].reset_index(drop=True)
        print(f'Applied --max-oc {args.max_oc:.1f} g/kg: kept {len(df):,}/{n_before:,} '
              f'({100*len(df)/n_before:.2f}%)  '
              f'OC max in set = {df["OC"].max():.1f}', flush=True)
    folds_meta = build_folds_spatial_deciles(
        df, n_folds=args.num_folds, buffer_km=args.fold_buffer_km,
        axis=args.split_axis, seed=args.seed_base)
    print(f'Loaded {len(df)} rows from {MODEL_READY}', flush=True)
    print(f'num_folds={args.num_folds}  buffer={args.fold_buffer_km} km  '
          f'split_axis={args.split_axis}  window_size={args.window_size}', flush=True)
    for f in folds_meta:
        where = (f'cluster centroid ({f["centroid_lat"]:.3f}, {f["centroid_lon"]:.3f})'
                 if args.split_axis == 'cluster'
                 else f'[{f["edge_lo"]:.4f}, {f["edge_hi"]:.4f})')
        print(f'Fold {f["fold_id"]}: {f["split_axis"]} {where} '
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
