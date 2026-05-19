#!/usr/bin/env python3
"""
run_baselines.py — Random Forest + XGBoost baselines on the SAME 10-fold
latitude-decile splits used by the SGT k-fold experiment.

Why this exists
---------------
The rebuttal needs honest baselines on the same spatial-CV geometry. Without
them, "SGT reaches R² = 0.16 on spatial CV" is unanchored. A tree-ensemble
on per-band statistics is the canonical SOC-mapping baseline (Hengl et al.,
Padarian et al., many others), and if SGT doesn't clearly beat it under the
same evaluation we should say so.

Features
--------
For each LUCAS sample we have a (C=20 bands, T=5 years, H=5, W=5 pixel)
spatiotemporal cube. We collapse it to 80 features per sample:

    for each of 20 bands → {mean, std, min, max} over the (T, H, W) cube

Tree models don't benefit from normalization (scale-invariant), so we use
the raw cube values, not the feature-normalized ones the SGT sees.

Targets are log-transformed by default (matches SGT) so the loss attends to
relative errors. Predictions are inverse-transformed before metric
computation, so R²/RMSE/MAE/RPIQ are reported in the original g/kg scale —
directly comparable to the SGT numbers in sweep_ranking.md.

Folds: reuses build_folds_latitude_deciles + the MAX_OC filter from
run_kfold.py. Outputs land in sweep/baseline_<model>_<tag>/ in the same
fold_<i>_predictions.parquet + kfold_results_summary.json format as the
SGT runs, so sweep_summarize.py ranks everything together.

Usage
-----
    python rebuttal/gpu_experiments/spatial_kfold/run_baselines.py \
        --models xgb,rf --max-oc 90 --target-transform log

    # GPU XGBoost (default if --device cuda and a GPU is visible)
    python run_baselines.py --models xgb --device cuda

    # CPU sklearn RF (always available; cuML auto-detected if installed)
    python run_baselines.py --models rf

Use sweep_summarize.py afterwards to rank baselines against SGT configs.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Reuse run_kfold's path setup, fold construction, dataset wrapper, and
# metric helpers. This pulls in accelerate/wandb at import time (because
# run_kfold imports them) but doesn't instantiate them — we don't need
# Accelerator for the baselines.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kfold import (  # noqa: E402
    MODEL_READY, OUT_DIR, _build_model_ready_dataset,
    build_folds_latitude_deciles, make_dataset,
    _metrics_for, write_results, write_predictions_parquet,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _aggregate_cube(features: torch.Tensor) -> np.ndarray:
    """Reduce (C, T, H, W) to (C*4,) of per-band {mean, std, min, max}."""
    x = features.detach().cpu().numpy().astype(np.float32)
    C = x.shape[0]
    x_flat = x.reshape(C, -1)
    stats = np.empty((C, 4), dtype=np.float32)
    stats[:, 0] = x_flat.mean(axis=1)
    stats[:, 1] = x_flat.std(axis=1)
    stats[:, 2] = x_flat.min(axis=1)
    stats[:, 3] = x_flat.max(axis=1)
    return stats.reshape(-1)


def extract_features_for_df(df: pd.DataFrame, cache_path: Path | None = None
                             ) -> tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """Extract (X, y, lon, lat) for every row in df.

    Reads from the raster tiles via run_kfold.make_dataset (without
    normalization — tree models are scale-invariant). Slow first time
    (~30 s/1000 rows depending on disk), so we cache to .npz on disk
    when cache_path is provided.
    """
    if cache_path and cache_path.exists():
        z = np.load(cache_path)
        print(f'[baseline] cache hit: {cache_path} '
              f'(X.shape={z["X"].shape})', flush=True)
        return z['X'], z['y'], z['lon'], z['lat']

    print(f'[baseline] extracting features for {len(df):,} samples …',
          flush=True)
    # No feature normalization — tree models are scale-invariant.
    ds = make_dataset(df, feature_means=None, feature_stds=None)
    n = len(ds)
    X = np.empty((n, 80), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    lons = np.empty(n, dtype=np.float64)
    lats = np.empty(n, dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        lon, lat, feats, oc = ds[i]
        X[i] = _aggregate_cube(feats)
        y[i] = float(oc)
        lons[i] = float(lon)
        lats[i] = float(lat)
        if (i + 1) % 1000 == 0:
            print(f'  [{i+1:>6}/{n:>6}] elapsed {time.time()-t0:.1f}s',
                  flush=True)
    print(f'[baseline] done in {time.time()-t0:.1f}s', flush=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, y=y, lon=lons, lat=lats)
        print(f'[baseline] cached to {cache_path}', flush=True)
    return X, y, lons, lats


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_xgb(args):
    import xgboost as xgb
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f'[baseline] XGBoost device={device}', flush=True)
    return xgb.XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist',
        device=device,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def make_rf(args):
    """Random Forest — cuML on GPU if available, sklearn on CPU otherwise."""
    use_cuml = (args.device == 'cuda' and torch.cuda.is_available())
    if use_cuml:
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
            print(f'[baseline] cuML RandomForest on GPU', flush=True)
            return CumlRF(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth if args.rf_max_depth > 0 else 16,
                max_features='sqrt',
                random_state=42,
            )
        except ImportError:
            print('[baseline] cuML not installed; falling back to sklearn RF (CPU)',
                  flush=True)
    from sklearn.ensemble import RandomForestRegressor
    print(f'[baseline] sklearn RandomForest on CPU '
          f'(n_jobs=-1, {os.cpu_count()} cores)', flush=True)
    return RandomForestRegressor(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Target transform
# ---------------------------------------------------------------------------
def transform_y(y: np.ndarray, mode: str, eps: float = 1e-10) -> np.ndarray:
    if mode == 'log':
        return np.log(np.maximum(y, eps))
    if mode == 'normalize':
        return y     # std-normalization applied per-fold below
    return y


def inverse_y(y_pred: np.ndarray, mode: str,
              mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    if mode == 'log':
        return np.exp(y_pred)
    if mode == 'normalize':
        return y_pred * std + mean
    return y_pred


# ---------------------------------------------------------------------------
# Per-fold train + evaluate
# ---------------------------------------------------------------------------
def run_fold(model_name: str, args, fold: dict,
             X_all: np.ndarray, y_all: np.ndarray,
             lon_all: np.ndarray, lat_all: np.ndarray,
             out_dir: Path) -> dict:
    fid = fold['fold_id']
    train_idx = np.asarray(fold['train_idx'])
    test_idx = np.asarray(fold['test_idx'])

    X_train = X_all[train_idx]
    y_train_raw = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test_raw = y_all[test_idx]
    lons_test = lon_all[test_idx]
    lats_test = lat_all[test_idx]

    # Target transform happens per fold (mean/std are train-only).
    if args.target_transform == 'normalize':
        mu, sd = float(y_train_raw.mean()), float(y_train_raw.std() or 1.0)
        y_train = (y_train_raw - mu) / sd
    elif args.target_transform == 'log':
        mu, sd = 0.0, 1.0
        y_train = transform_y(y_train_raw, 'log')
    else:
        mu, sd = 0.0, 1.0
        y_train = y_train_raw

    print(f'\n=== {model_name} fold {fid}  '
          f'n_train={len(X_train)}  n_test={len(X_test)} ===', flush=True)
    if model_name == 'xgb':
        model = make_xgb(args)
    elif model_name == 'rf':
        model = make_rf(args)
    else:
        raise ValueError(f'unknown model: {model_name}')

    t0 = time.time()
    model.fit(X_train, y_train)
    t_fit = time.time() - t0

    pred_raw = model.predict(X_test)
    # cuML returns numpy already; ensure type
    pred_raw = np.asarray(pred_raw, dtype=np.float64).reshape(-1)
    pred = inverse_y(pred_raw, args.target_transform, mean=mu, std=sd)
    pred = np.clip(pred, 0.0, None)  # SOC is non-negative

    m = _metrics_for(pred, y_test_raw.astype(np.float64))
    print(f'  R²={m["r2"]:+.4f}  RMSE={m["rmse"]:.3f}  '
          f'MAE={m["mae"]:.3f}  RPIQ={m["rpiq"]:.3f}  '
          f'(fit {t_fit:.1f}s)', flush=True)

    # Save in run_kfold-compatible format for sweep_summarize.py
    pd.DataFrame({
        'GPS_LAT': lats_test, 'GPS_LONG': lons_test,
        'OC_actual': y_test_raw.astype(np.float64),
        'OC_predicted': pred,
        'fold_id': fid,
        'year': np.zeros(len(test_idx), dtype=int),   # unused for baselines
        'altitude': np.full(len(test_idx), np.nan),
    }).to_parquet(out_dir / f'fold_{fid}_predictions.parquet')

    per_fold_meta = {
        'fold_id': fid,
        'lat_lo': float(fold['lat_lo']), 'lat_hi': float(fold['lat_hi']),
        'n_test': int(len(test_idx)),
        'n_train': int(len(train_idx)),
        'n_train_raw': int(len(train_idx)),
        'n_buffer': int(len(fold['buffer_idx'])),
        'accum_steps': 0, 'effective_batch_size': 0,
        'test_oc_mean': float(y_test_raw.mean()),
        'test_oc_std': float(y_test_raw.std()),
        'test_oc_max': float(y_test_raw.max()),
        'test_pct_gt_50': float(100 * (y_test_raw > 50).mean()),
        'best_epoch_r2_during_training': float('nan'),
        'fit_seconds': float(t_fit),
        **m,
    }
    (out_dir / f'fold_{fid}_summary.json').write_text(
        json.dumps(per_fold_meta, indent=2, default=str))
    return per_fold_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--models', type=str, default='xgb,rf',
                   help='Comma-separated list: xgb, rf. Default both.')
    p.add_argument('--num-folds', type=int, default=10)
    p.add_argument('--fold-buffer-km', type=float, default=1.2)
    p.add_argument('--max-oc', type=float, default=90.0,
                   help='Match the SGT sweep default (90).')
    p.add_argument('--target-transform', type=str, default='log',
                   choices=['none', 'log', 'normalize'])
    p.add_argument('--device', type=str, default='cuda',
                   help='cuda or cpu (XGBoost device; RF uses cuML if cuda).')
    p.add_argument('--tag-suffix', type=str, default='default',
                   help='Output subdir is baseline_<model>_<tag-suffix>.')
    p.add_argument('--output-subdir', type=str, default='sweep',
                   help='Output goes to OUT_DIR/<output-subdir>/'
                        'baseline_<model>_<tag-suffix>/. Default "sweep". '
                        'For max-oc sensitivity: pass "sweep/oc120" etc.')
    p.add_argument('--bands-list', type=str, default='full_20',
                   choices=['full_20', 'original_6'],
                   help='Covariate subset. Tree features are 80-d per sample '
                        '(20 bands × {mean, std, min, max}); under '
                        'original_6 we extract the full 80-d vector once '
                        'and keep only the 24 columns belonging to the 6 '
                        'original bands. Cache is shared across runs that '
                        'use the same max-oc.')
    p.add_argument('--cache-features', action='store_true', default=True,
                   help='Cache extracted features to .npz so re-runs are fast.')
    p.add_argument('--no-cache-features', dest='cache_features',
                   action='store_false')
    # XGBoost knobs
    p.add_argument('--xgb-n-estimators', type=int, default=2000)
    p.add_argument('--xgb-max-depth', type=int, default=6)
    p.add_argument('--xgb-lr', type=float, default=0.05)
    # RF knobs
    p.add_argument('--rf-n-estimators', type=int, default=500)
    p.add_argument('--rf-max-depth', type=int, default=0,
                   help='0 = unbounded (sklearn) or 16 (cuML).')
    return p.parse_args()


def main():
    args = parse()
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for m in models:
        if m not in ('xgb', 'rf'):
            raise SystemExit(f'unknown model: {m}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _build_model_ready_dataset()
    df = pd.read_parquet(MODEL_READY).reset_index(drop=True)

    if args.max_oc is not None and args.max_oc > 0:
        n_before = len(df)
        df = df[df['OC'] <= args.max_oc].reset_index(drop=True)
        print(f'Applied --max-oc {args.max_oc:.1f}: '
              f'kept {len(df):,}/{n_before:,}', flush=True)

    folds_meta = build_folds_latitude_deciles(
        df, n_folds=args.num_folds, buffer_km=args.fold_buffer_km)
    print(f'Built {args.num_folds} latitude-decile folds '
          f'(buffer {args.fold_buffer_km} km).', flush=True)

    cache_path = (OUT_DIR / 'baseline_features'
                  / f'feats_max_oc_{int(args.max_oc)}.npz'
                  ) if args.cache_features else None
    X, y, lon, lat = extract_features_for_df(df, cache_path=cache_path)
    print(f'[baseline] X={X.shape} y={y.shape}  '
          f'OC range [{y.min():.2f}, {y.max():.2f}]', flush=True)

    # --- Optional band-subset slicing ---
    # The cache always stores 80-feature vectors (20 bands × {mean,std,min,max}).
    # When --bands-list=original_6 we keep the 24 columns belonging to the
    # first 6 bands (Elevation, LAI, LST, MODIS_NPP, SoilEvaporation,
    # TotalEvapotranspiration), which sit at indices 0..5 of bands_list_order
    # by construction (see SpatiotemporalGatedTransformer/config.py).
    from band_subsets import get_band_indices    # noqa: E402  (already on sys.path via HERE)
    from config import bands_list_order          # noqa: E402  (via SGT sys.path side-effect from run_kfold)

    band_indices = get_band_indices(args.bands_list, list(bands_list_order))
    if len(band_indices) < len(bands_list_order):
        # Map each kept band to its 4 columns in X (mean, std, min, max).
        col_indices = []
        for b in band_indices:
            col_indices.extend([b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3])
        X = X[:, col_indices]
        print(f'[baseline] --bands-list={args.bands_list}: sliced X to '
              f'{X.shape} ({len(band_indices)} bands × 4 stats)', flush=True)

    for model_name in models:
        tag = f'baseline_{model_name}_{args.tag_suffix}'
        # output_subdir is a relative path like 'sweep' or 'sweep/oc120'
        out_dir = OUT_DIR / args.output_subdir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n========== {tag}  →  {out_dir} ==========', flush=True)

        fold_results = []
        for f in folds_meta:
            res = run_fold(model_name, args, f, X, y, lon, lat, out_dir)
            fold_results.append({**res, '_predictions': {
                # write_results uses these for stratified-band metrics
                'lon': lon[np.asarray(f['test_idx'])],
                'lat': lat[np.asarray(f['test_idx'])],
                'pred': pd.read_parquet(out_dir / f'fold_{f["fold_id"]}_predictions.parquet')
                          ['OC_predicted'].to_numpy(),
                'actual': y[np.asarray(f['test_idx'])].astype(np.float64),
                'year': np.zeros(len(f['test_idx']), dtype=int),
                'altitude': np.full(len(f['test_idx']), np.nan),
            }})

        # Build a fake args namespace mirroring run_kfold's CLI so write_results
        # picks up sensible recipe metadata.
        recipe_args = argparse.Namespace(
            num_folds=args.num_folds,
            fold_buffer_km=args.fold_buffer_km,
            lr=float('nan'),
            loss_type=f'{model_name}-mse',
            target_transform=args.target_transform,
            num_epochs=-1,
            per_gpu_batch_size=-1,
            effective_batch_size=-1,
            lr_scheduler='none', lr_min=-1.0,
            num_heads=-1, num_layers=-1,
            sampler_mode='none', alpha_density=None,
            augment_train=False,
            rebalance_n_bins=-1, rebalance_min_ratio=-1,
            max_oc=args.max_oc,
        )
        # Write outputs into the baseline subdir specifically — patch OUT_DIR
        # locally so write_results writes to the baseline dir, not the parent.
        import run_kfold
        saved_OUT_DIR = run_kfold.OUT_DIR
        run_kfold.OUT_DIR = out_dir
        try:
            write_results(fold_results, recipe_args)
            write_predictions_parquet(fold_results)
        finally:
            run_kfold.OUT_DIR = saved_OUT_DIR
        print(f'[{tag}] wrote kfold_results_summary.json + .md to {out_dir}',
              flush=True)


if __name__ == '__main__':
    main()
