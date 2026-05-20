#!/usr/bin/env python3
"""
validation_set_stats.py — T2.4 (R1.3, R3.3)

Addresses the R²/RMSE paradox: the random-split validation set shows
**higher** R² *and* higher RMSE than the spatial-CV validation set,
which is counterintuitive at first glance. The reviewers asked for
descriptive statistics on both validation sets to explain the
mechanism.

Mechanism (to confirm with this script):
  * R² = 1 - SS_res / SS_tot.
  * Random-split validation retains a representative slice of the
    full SOC distribution, including the heavy tail (peat/fen soils
    at 100-150 g/kg).
  * The heavy tail dramatically inflates the variance denominator
    (SS_tot), letting R² climb even when residuals are large.
  * The same heavy-tail samples also produce large absolute
    residuals (ceiling-effect underestimates), inflating RMSE.
  * Spatial-CV validation, blocking on latitude deciles + a 1.2 km
    buffer, holds out specific *geographies* — typically a single
    fold dominated by Alpine carbon-rich soils. The other 9 folds
    have less heavy-tail representation, so their RMSE is lower but
    so is their R² (lower SS_tot too).

This script computes:
  1. Random 91/9 split (fixed seed) descriptive stats: n, mean, SD,
     median, IQR, max, % > 50 g/kg, % > 120 g/kg.
  2. Per-fold spatial-CV split (latitude-decile) descriptive stats:
     same fields, plus the fold's lat range.
  3. The "across-folds aggregate" — what the kfold_results_summary
     averages over.

Tabulates side-by-side; computes the implied R²/RMSE for an
idealized "predict the training mean" baseline on each set to
illustrate how variance composition drives the paradox.

Outputs:
    rebuttal/reviewer_responses/results/validation_set_stats.{json,md}
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np
import pandas as pd

from _common import MODEL_READY, write_pair, banner


def desc_stats(s: pd.Series) -> dict:
    """Standard descriptive stats on an SOC vector."""
    s = s.astype(float)
    return {
        'n': int(s.size),
        'mean': float(s.mean()),
        'sd': float(s.std(ddof=1)) if s.size > 1 else 0.0,
        'median': float(s.median()),
        'q1': float(s.quantile(0.25)),
        'q3': float(s.quantile(0.75)),
        'iqr': float(s.quantile(0.75) - s.quantile(0.25)),
        'min': float(s.min()),
        'max': float(s.max()),
        'pct_above_50': float((s > 50).mean() * 100),
        'pct_above_120': float((s > 120).mean() * 100),
    }


def split_random(df: pd.DataFrame, val_frac: float, seed: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 91/9 partition. The original paper used val_frac ≈ 0.09."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(len(df) * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def split_spatial_kfold(df: pd.DataFrame, n_folds: int = 10
                          ) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Replicate the run_kfold latitude-decile partition.

    Returns list of (train_df, val_df, info) per fold, where info
    contains the fold's latitude range. The 1.2 km buffer between
    train and val is NOT applied here — this is purely the latitude-
    decile assignment used for fold splitting; the buffer is enforced
    inside run_kfold at training time by dropping rows near the
    decile boundary.
    """
    sorted_df = df.sort_values('GPS_LAT').reset_index(drop=True)
    n = len(sorted_df)
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds = []
    for f in range(n_folds):
        val_mask = np.zeros(n, dtype=bool)
        val_mask[edges[f]:edges[f + 1]] = True
        train_df = sorted_df[~val_mask].reset_index(drop=True)
        val_df = sorted_df[val_mask].reset_index(drop=True)
        info = {
            'fold': f,
            'lat_min': float(val_df['GPS_LAT'].min()),
            'lat_max': float(val_df['GPS_LAT'].max()),
            'n_train': len(train_df),
            'n_val': len(val_df),
        }
        folds.append((train_df, val_df, info))
    return folds


def baseline_metrics(train_y: np.ndarray, val_y: np.ndarray) -> dict:
    """If the model predicted train_y.mean() for every val sample, what
    would the metrics look like? Demonstrates the variance composition."""
    yhat = train_y.mean()
    resid = val_y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((val_y - val_y.mean()) ** 2))
    return {
        'mean_only_R2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        'mean_only_rmse': float(np.sqrt(ss_res / val_y.size)),
        'mean_only_mae': float(np.mean(np.abs(resid))),
        'val_ss_tot': ss_tot,
        'val_variance': float(np.var(val_y, ddof=1)) if val_y.size > 1 else 0.0,
    }


def render_markdown(rand_info, spatial_info, baseline) -> str:
    md = ['# Validation-set descriptive statistics — random vs spatial-CV', '']
    md.append('Addresses reviewer comments R1.3 (single split weaker than CV) and '
              'R3.3 (R²/RMSE paradox).')
    md.append('')
    md.append('## Random 91/9 split (fixed seed = 42)')
    md.append('')
    md.append('| stat | training (91%) | validation (9%) |')
    md.append('|---|---|---|')
    for k, label in [('n', 'n samples'),
                      ('mean', 'mean SOC (g/kg)'),
                      ('sd', 'std SOC (g/kg)'),
                      ('median', 'median SOC'),
                      ('iqr', 'IQR'),
                      ('max', 'max SOC'),
                      ('pct_above_50', '% > 50 g/kg'),
                      ('pct_above_120', '% > 120 g/kg')]:
        tr = rand_info['train_stats'][k]
        vl = rand_info['val_stats'][k]
        prec = 0 if k in ('n',) else 2
        md.append(f'| {label} | {tr:.{prec}f} | {vl:.{prec}f} |')
    md.append('')
    md.append('## Spatial-CV (10 latitude-decile folds) — aggregated across folds')
    md.append('')
    md.append('| stat | training (across folds) | validation (across folds) |')
    md.append('|---|---|---|')
    sp = spatial_info['aggregate']
    for k, label in [('n_mean', 'mean n samples per fold'),
                      ('soc_mean_of_means', 'fold-averaged mean SOC'),
                      ('soc_sd_of_sds', 'fold-averaged SD SOC'),
                      ('soc_median_of_medians', 'fold-avg median SOC'),
                      ('soc_pct50_of_pcts', 'fold-avg % > 50 g/kg'),
                      ('soc_pct120_of_pcts', 'fold-avg % > 120 g/kg')]:
        tr = sp['train'][k]
        vl = sp['val'][k]
        prec = 0 if k == 'n_mean' else 2
        md.append(f'| {label} | {tr:.{prec}f} | {vl:.{prec}f} |')
    md.append('')
    md.append('## Per-fold validation breakdown')
    md.append('')
    md.append('| fold | lat range | n val | mean | SD | max | % > 50 | % > 120 |')
    md.append('|---|---|---|---|---|---|---|---|')
    for f in spatial_info['folds']:
        st = f['val_stats']
        md.append(f'| {f["fold"]} | [{f["lat_min"]:.2f}, {f["lat_max"]:.2f}] '
                  f'| {st["n"]} | {st["mean"]:.2f} | {st["sd"]:.2f} | '
                  f'{st["max"]:.2f} | {st["pct_above_50"]:.2f} | {st["pct_above_120"]:.2f} |')
    md.append('')
    md.append('## The R²/RMSE paradox, explained')
    md.append('')
    md.append('The mean-only baseline (predict `mean(y_train)` for every '
              'validation sample) yields:')
    md.append('')
    md.append('| Split | val SS_tot | val variance | mean-only RMSE | mean-only R² |')
    md.append('|---|---|---|---|---|')
    md.append(f'| **Random 91/9** | '
              f'{baseline["random"]["val_ss_tot"]:.1f} | '
              f'{baseline["random"]["val_variance"]:.2f} | '
              f'{baseline["random"]["mean_only_rmse"]:.2f} | '
              f'{baseline["random"]["mean_only_R2"]:+.4f} |')
    for fb in baseline['spatial_per_fold']:
        md.append(f'| Spatial fold {fb["fold"]} | '
                  f'{fb["val_ss_tot"]:.1f} | '
                  f'{fb["val_variance"]:.2f} | '
                  f'{fb["mean_only_rmse"]:.2f} | '
                  f'{fb["mean_only_R2"]:+.4f} |')
    md.append('')
    md.append('**Reading**: a validation set with higher variance (more heavy-tail '
              'samples) inflates SS_tot, which makes R² = 1 − SS_res/SS_tot larger '
              'for the *same* residual sum of squares. The random split typically '
              'has higher variance (it samples the long SOC tail uniformly), so its '
              'R² appears higher even when its residuals are larger. The R²/RMSE '
              'paradox is therefore not a sign of better model behavior under random '
              'splitting — it is a sign that R² is *not invariant* to the validation '
              'distribution. Spatial CV reports the model behavior on '
              'distribution-shifted hold-outs (specific geographies), which is '
              'the more relevant operational claim.')
    return '\n'.join(md)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-frac', type=float, default=0.09,
                   help='Random-split val fraction (default 0.09).')
    p.add_argument('--n-folds', type=int, default=10)
    p.add_argument('--max-oc', type=float, default=150.0)
    a = p.parse_args()

    print(banner('VALIDATION-SET DESCRIPTIVE STATISTICS'))
    if not MODEL_READY.exists():
        print(f'[error] model-ready dataset not found at {MODEL_READY}',
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(MODEL_READY)
    if a.max_oc > 0:
        df = df[df['OC'] <= a.max_oc].reset_index(drop=True)
    print(f'  Loaded {len(df)} samples')

    # ---- Random 91/9 split ----
    train_df, val_df = split_random(df, a.val_frac, a.seed)
    rand_info = {
        'train_stats': desc_stats(train_df['OC']),
        'val_stats': desc_stats(val_df['OC']),
    }

    # ---- Spatial 10-fold CV ----
    folds = split_spatial_kfold(df, n_folds=a.n_folds)
    spatial_info = {'folds': []}
    for tr, vl, info in folds:
        spatial_info['folds'].append({
            **info,
            'train_stats': desc_stats(tr['OC']),
            'val_stats': desc_stats(vl['OC']),
        })
    # Aggregate fold-wise stats
    def mean_of(field, kind):
        return float(np.mean([f[f'{kind}_stats'][field] for f in spatial_info['folds']]))
    spatial_info['aggregate'] = {
        'train': {
            'n_mean': mean_of('n', 'train'),
            'soc_mean_of_means': mean_of('mean', 'train'),
            'soc_sd_of_sds': mean_of('sd', 'train'),
            'soc_median_of_medians': mean_of('median', 'train'),
            'soc_pct50_of_pcts': mean_of('pct_above_50', 'train'),
            'soc_pct120_of_pcts': mean_of('pct_above_120', 'train'),
        },
        'val': {
            'n_mean': mean_of('n', 'val'),
            'soc_mean_of_means': mean_of('mean', 'val'),
            'soc_sd_of_sds': mean_of('sd', 'val'),
            'soc_median_of_medians': mean_of('median', 'val'),
            'soc_pct50_of_pcts': mean_of('pct_above_50', 'val'),
            'soc_pct120_of_pcts': mean_of('pct_above_120', 'val'),
        },
    }

    # ---- Mean-only baseline metrics — illustrate the paradox ----
    baseline = {
        'random': baseline_metrics(train_df['OC'].to_numpy(),
                                    val_df['OC'].to_numpy()),
        'spatial_per_fold': [],
    }
    for f, (tr, vl, info) in zip(spatial_info['folds'], folds):
        b = baseline_metrics(tr['OC'].to_numpy(), vl['OC'].to_numpy())
        baseline['spatial_per_fold'].append({'fold': info['fold'], **b})

    md = render_markdown(rand_info, spatial_info, baseline)
    print()
    print(md)
    write_pair('validation_set_stats',
               {'random': rand_info, 'spatial': spatial_info, 'baseline': baseline,
                'meta': {'seed': a.seed, 'val_frac': a.val_frac,
                          'n_folds': a.n_folds, 'max_oc': a.max_oc,
                          'n_total': int(len(df))}},
               md)


if __name__ == '__main__':
    main()
