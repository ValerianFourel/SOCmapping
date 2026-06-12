#!/usr/bin/env python3
"""
rebuttal/final_models/train_full_baselines.py — train RF / XGBoost on
the entire LUCAS dataset for production mapping.

Mirrors train_full.py but for tree ensembles. Reuses the per-band
statistic feature extraction from run_baselines.py (80 features per
sample: {mean, std, min, max} × 20 bands over the 5×5×5 cube).

Outputs (under rebuttal/final_models/checkpoints/<run-name>/):
    final_model.{joblib | json}   trained estimator
    stats.json                     features metadata, n_train, max-oc
    config.json                    full args

Run:
    python rebuttal/final_models/train_full_baselines.py \\
        --run-name rf_default \\
        --model rf --max-oc 150 --target-transform log

    python rebuttal/final_models/train_full_baselines.py \\
        --run-name xgb_shallow \\
        --model xgb --xgb-max-depth 4 --xgb-n-estimators 2000 \\
        --max-oc 150 --target-transform log
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

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]
sys.path.insert(0, str(SOC_ROOT))
from _paths import SOC_REBUTTAL_DIR  # noqa: E402

# Reuse the feature-extraction helper from run_baselines.py
KFOLD_DIR = SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'
sys.path.insert(0, str(KFOLD_DIR))
from run_kfold import MODEL_READY, _build_model_ready_dataset, compute_density_weights  # noqa: E402
from run_baselines import extract_features_for_df, transform_y, inverse_y  # noqa: E402
from band_subsets import get_band_indices, band_suffix  # noqa: E402

CHECKPOINTS_ROOT = HERE / 'checkpoints'


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', type=str, required=True)
    p.add_argument('--model', type=str, default='rf', choices=['rf', 'xgb'])
    p.add_argument('--max-oc', type=float, default=120.0)
    p.add_argument('--target-transform', type=str, default='log',
                   choices=['none', 'log', 'normalize'])
    # XGB knobs
    p.add_argument('--xgb-n-estimators', type=int, default=2000)
    p.add_argument('--xgb-max-depth', type=int, default=6)
    p.add_argument('--xgb-lr', type=float, default=0.05)
    p.add_argument('--xgb-device', type=str, default='cuda')
    # RF knobs
    p.add_argument('--rf-n-estimators', type=int, default=500)
    p.add_argument('--rf-max-depth', type=int, default=0,
                   help='0 = unbounded')
    p.add_argument('--bands-list', type=str, default='full_20',
                   choices=['full_20', 'original_6', 'full_extended', 'full_extended_s2'],
                   help='Covariate subset (default full_20). Run-name '
                        'auto-appends "_6band" / "_20band" / "_extband". '
                        'full_extended keeps all 43 bands → 172 features '
                        '(no slice); predict_tree handles the 172-wide cube.')
    p.add_argument('--sampler-mode', type=str, default='none',
                   choices=['none', 'kde'],
                   help='Training sample weighting. "none" (default): each '
                        'training row contributes equally (raw LUCAS SOC '
                        'distribution). "kde": fit with sample_weight equal '
                        'to KDE-inverse-density on log(SOC) — upweights '
                        'high-SOC rare-tail rows so the production map '
                        "covers Bavaria's organic-rich regions. Both RF and "
                        'XGB support sample_weight natively (no row '
                        'duplication needed).')
    p.add_argument('--sampler-alpha', type=float, default=0.5,
                   help='[kde mode only] KDE inversion exponent. Default 0.5 '
                        '= sqrt-inverse-density (Yang et al. ICML 2021).')
    return p.parse_args()


def main():
    args = parse()
    # Auto-append band suffix; use 'in' rather than endswith so a
    # rebal-suffixed name is not double-suffixed.
    suf = band_suffix(args.bands_list)
    if '_20band' not in args.run_name and '_6band' not in args.run_name:
        args.run_name = args.run_name + suf
    # Auto-append _rebal when KDE sample-weighting is on, so rebalanced
    # and non-rebalanced tree models coexist under checkpoints/.
    if args.sampler_mode == 'kde' and not args.run_name.endswith('_rebal'):
        args.run_name = args.run_name + '_rebal'
    out_dir = CHECKPOINTS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[baseline-final] run_name = {args.run_name}  '
          f'(bands_list={args.bands_list})', flush=True)
    print(f'[baseline-final] output dir = {out_dir}', flush=True)

    _build_model_ready_dataset()
    df = pd.read_parquet(MODEL_READY).reset_index(drop=True)
    if args.max_oc and args.max_oc > 0:
        n_before = len(df)
        df = df[df['OC'] <= args.max_oc].reset_index(drop=True)
        print(f'[baseline-final] max-oc {args.max_oc:.1f}: '
              f'kept {len(df):,}/{n_before:,}', flush=True)

    # Cached feature extraction (shared with the k-fold baseline pipeline).
    cache_path = (SOC_REBUTTAL_DIR / 'gpu_experiments' / 'spatial_kfold'
                  / 'baseline_features' / f'feats_max_oc_{int(args.max_oc)}.npz')
    X, y, lon, lat = extract_features_for_df(df, cache_path=cache_path)
    print(f'[baseline-final] X={X.shape}  y range [{y.min():.2f}, {y.max():.2f}]',
          flush=True)

    # Optional band-subset slicing — same recipe as run_baselines.py
    from config import bands_list_order        # noqa: E402
    band_indices = get_band_indices(args.bands_list, list(bands_list_order))
    if len(band_indices) < len(bands_list_order):
        col_indices = []
        for b in band_indices:
            col_indices.extend([b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3])
        X = X[:, col_indices]
        print(f'[baseline-final] --bands-list={args.bands_list}: sliced X '
              f'to {X.shape} ({len(band_indices)} bands × 4 stats)', flush=True)

    # Guard against a stale feature cache: full_extended needs all 172 columns
    # (43 bands × 4). An old cache extracted at 80 (20-band) would silently
    # train on the wrong width and fail at inference, so fail loudly here.
    expected_feats = 4 * len(band_indices)
    if X.shape[1] != expected_feats:
        raise SystemExit(
            f'\n[ERROR] feature width {X.shape[1]} != expected {expected_feats} '
            f'for --bands-list={args.bands_list}. The cached feature matrix is '
            f'likely a stale 20-band (80-col) extraction. Delete it and re-run '
            f'so it re-extracts all {len(bands_list_order)} bands:\n'
            f'  rm {cache_path}\n')

    # Target transform — log/normalize/none. Targets are tracked in train
    # space; predictions inverse-transformed before any reporting.
    if args.target_transform == 'normalize':
        mu, sd = float(y.mean()), float(y.std() or 1.0)
        y_train = (y.astype(np.float64) - mu) / sd
    elif args.target_transform == 'log':
        mu, sd = 0.0, 1.0
        y_train = transform_y(y.astype(np.float64), 'log')
    else:
        mu, sd = 0.0, 1.0
        y_train = y.astype(np.float64)

    # ---- Sample weights (rebalanced fit) ----
    # Both XGBRegressor.fit and RandomForestRegressor.fit accept the
    # sample_weight kwarg directly — no row duplication needed. Same
    # KDE-inverse-density recipe as the NN path so the two pipelines
    # rebalance the SOC distribution identically.
    sample_weight = None
    if args.sampler_mode == 'kde':
        sample_weight = compute_density_weights(y, alpha=args.sampler_alpha)
        print(f'[baseline-final] sampler=kde  alpha={args.sampler_alpha}  '
              f'n={len(y)}  w in [{sample_weight.min():.3f}, '
              f'{sample_weight.max():.3f}]  mean={sample_weight.mean():.3f}',
              flush=True)
    else:
        print(f'[baseline-final] sampler=none  (raw LUCAS distribution)',
              flush=True)

    # ---- Fit ----
    t0 = time.time()
    if args.model == 'xgb':
        import xgboost as xgb
        device = 'cuda' if args.xgb_device == 'cuda' else 'cpu'
        try:
            import torch
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
        except ImportError:
            device = 'cpu'
        model = xgb.XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_lr,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', device=device, n_jobs=-1,
            random_state=42, verbosity=0,
        )
        print(f'[baseline-final] XGBoost device={device}  n_est={args.xgb_n_estimators}  '
              f'depth={args.xgb_max_depth}  lr={args.xgb_lr}', flush=True)
        model.fit(X, y_train, sample_weight=sample_weight)
        out_model_path = out_dir / 'final_model.json'
        model.save_model(str(out_model_path))
    else:
        from sklearn.ensemble import RandomForestRegressor
        max_depth = args.rf_max_depth if args.rf_max_depth > 0 else None
        model = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=max_depth,
            max_features='sqrt', min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        print(f'[baseline-final] RandomForest n_est={args.rf_n_estimators}  '
              f'max_depth={max_depth}  (CPU, n_jobs=-1)', flush=True)
        model.fit(X, y_train, sample_weight=sample_weight)
        out_model_path = out_dir / 'final_model.joblib'
        import joblib
        joblib.dump(model, out_model_path)
    elapsed = time.time() - t0
    print(f'[baseline-final] training done in {elapsed/60:.1f} min', flush=True)
    print(f'[baseline-final] saved {out_model_path}', flush=True)

    # Training-set sanity: report in-sample fit quality (NOT a generalization
    # metric, just confirms the model converged).
    pred_raw = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
    pred = inverse_y(pred_raw, args.target_transform, mean=mu, std=sd)
    pred = np.clip(pred, 0.0, None)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f'[baseline-final] in-sample R²={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}  '
          f'(not a generalization estimate)', flush=True)

    stats = {
        'family': args.model,
        'target_transform': args.target_transform,
        'target_mean_for_normalize': mu,
        'target_std_for_normalize': sd,
        'max_oc': float(args.max_oc),
        'n_train': int(len(y)),
        'n_features': int(X.shape[1]),
        'in_sample_r2': r2,
        'in_sample_rmse': rmse,
        'in_sample_mae': mae,
        'fit_seconds': float(elapsed),
    }
    (out_dir / 'stats.json').write_text(json.dumps(stats, indent=2, default=str))
    (out_dir / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str))


if __name__ == '__main__':
    main()
