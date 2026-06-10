"""blend_predictions.py — ensemble per-fold predictions from multiple k-fold runs.

Blends several spatial-kfold runs (different model families and/or seeds) by
aligning them on a sample key and averaging OC_predicted, then recomputes the
metrics the sweep reports: per-fold mean R2 (the headline 'R2 mean'), pooled
R2, RMSE, MAE. No retraining — operates on the
`kfold_predictions_all_folds.parquet` files run_kfold.py already wrote.

Join key (--key):
  * sampleyear (default): (GPS_LAT, GPS_LONG, year). Use for transformer<->
    transformer blends (seed or architecture ensembles) — they predict per
    (location, year).
  * latlon: (GPS_LAT, GPS_LONG). Use for CROSS-FAMILY blends — the tree
    baselines (RF/XGB) store year=0 and predict per location, so the
    transformer's per-year predictions are mean-collapsed to per-location to
    match. Recommended for transformer + RF + XGB.

Duplicate rows per key within a run are mean-aggregated before merging; runs
are inner-joined, so a cap mismatch just shrinks the common set (reported n).

Usage:
    # transformer seed/arch ensemble (same granularity)
    python scripts/blend_predictions.py --runs A B C
    # cross-family blend (transformer + trees)
    python scripts/blend_predictions.py --key latlon --runs SGT RF XGB
    python scripts/blend_predictions.py --key latlon --runs SGT RF XGB --nnls
A run is a directory containing kfold_predictions_all_folds.parquet, or the
parquet file itself.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = {'sampleyear': ['GPS_LAT', 'GPS_LONG', 'year'],
        'latlon': ['GPS_LAT', 'GPS_LONG']}


def load_run(p):
    p = Path(p)
    if p.is_dir():
        p = p / 'kfold_predictions_all_folds.parquet'
    if not p.is_file():
        raise SystemExit(f'no predictions parquet at {p}')
    df = pd.read_parquet(p)
    missing = {'GPS_LAT', 'GPS_LONG', 'fold_id', 'OC_actual', 'OC_predicted'} - set(df.columns)
    if missing:
        raise SystemExit(f'{p} missing columns {missing}')
    return df


def collapse(df, key):
    """One row per key: mean prediction/actual, first fold_id."""
    return df.groupby(key, as_index=False).agg(
        OC_actual=('OC_actual', 'mean'),
        OC_predicted=('OC_predicted', 'mean'),
        fold_id=('fold_id', 'first'))


def metrics(df, pred_col):
    a = df['OC_actual'].to_numpy(float)
    pr = df[pred_col].to_numpy(float)
    r2s = []
    for _, g in df.groupby('fold_id'):
        ga = g['OC_actual'].to_numpy(float)
        gp = g[pred_col].to_numpy(float)
        ss_tot = float(((ga - ga.mean()) ** 2).sum())
        if ss_tot > 0:
            r2s.append(1.0 - float(((ga - gp) ** 2).sum()) / ss_tot)
    ss_tot = float(((a - a.mean()) ** 2).sum())
    pooled = 1.0 - float(((a - pr) ** 2).sum()) / ss_tot if ss_tot > 0 else float('nan')
    return dict(
        r2_mean=float(np.mean(r2s)) if r2s else float('nan'),
        r2_std=float(np.std(r2s)) if r2s else float('nan'),
        r2_pooled=pooled,
        rmse=float(np.sqrt(((a - pr) ** 2).mean())),
        n=len(df), n_folds=len(r2s),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='run dirs (each with kfold_predictions_all_folds.parquet) or parquet files')
    ap.add_argument('--key', choices=list(KEYS), default='sampleyear',
                    help='join granularity (sampleyear=transformer ensembles; latlon=cross-family)')
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--weights', nargs='*', type=float, default=None)
    ap.add_argument('--nnls', action='store_true',
                    help='fit non-negative weights on pooled actual (OPTIMISTIC upper bound)')
    ap.add_argument('--out', default=None, help='write the blended parquet here')
    a = ap.parse_args()

    key = KEYS[a.key]
    names = a.names or [Path(r).name if Path(r).is_dir() else Path(r).parent.name for r in a.runs]
    if len(names) != len(a.runs):
        raise SystemExit('--names length must match --runs')
    runs = [collapse(load_run(r), key) for r in a.runs]

    merged = runs[0][key + ['fold_id', 'OC_actual']].copy()
    pred_cols = []
    for nm, df in zip(names, runs):
        col = f'pred__{nm}'
        merged = merged.merge(df[key + ['OC_predicted']].rename(columns={'OC_predicted': col}),
                              on=key, how='inner')
        pred_cols.append(col)
    if merged.empty:
        raise SystemExit('no common samples — different split/cap, or try --key latlon for cross-family')

    print(f'[blend] key={a.key}  aligned {len(merged)} common samples across {len(a.runs)} runs\n')
    hdr = f'{"component":42s} {"R2_mean":>8s} {"R2_std":>7s} {"R2_pool":>8s} {"RMSE":>7s}'
    print(hdr); print('-' * len(hdr))
    for nm, col in zip(names, pred_cols):
        m = metrics(merged, col)
        print(f'{nm:42s} {m["r2_mean"]:8.4f} {m["r2_std"]:7.4f} {m["r2_pooled"]:8.4f} {m["rmse"]:7.3f}')

    P = merged[pred_cols].to_numpy(float)
    y = merged['OC_actual'].to_numpy(float)
    if a.nnls:
        from scipy.optimize import nnls
        w, _ = nnls(P, y)
        w = w if w.sum() > 0 else np.ones(len(pred_cols))
        wlabel = 'nnls (optimistic)'
    elif a.weights:
        if len(a.weights) != len(a.runs):
            raise SystemExit('--weights length must match --runs')
        w = np.array(a.weights, float); wlabel = 'manual'
    else:
        w = np.ones(len(pred_cols)); wlabel = 'equal'
    w = w / w.sum()
    merged['pred__blend'] = P @ w

    mb = metrics(merged, 'pred__blend')
    print('-' * len(hdr))
    print(f'{"BLEND (" + wlabel + ")":42s} {mb["r2_mean"]:8.4f} {mb["r2_std"]:7.4f} '
          f'{mb["r2_pooled"]:8.4f} {mb["rmse"]:7.3f}')
    print('[blend] weights: ' + ', '.join(f'{n}={x:.3f}' for n, x in zip(names, w)))
    print('\n[blend] R2_mean is the sweep headline. Equal-weight is the honest number; '
          '--nnls is an optimistic upper bound (weights fit on the same data).')

    if a.out:
        merged.to_parquet(a.out)
        print(f'[blend] wrote {a.out}')


if __name__ == '__main__':
    main()
