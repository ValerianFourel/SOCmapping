#!/usr/bin/env python3
"""
bootstrap_ci.py — T2.1 (R1.3, R3.6, R4 implied)

Addresses the reviewers' request for confidence intervals on the
performance metrics. Two complementary bootstrap procedures:

  1. **Bootstrap over folds** (default): resample the 10 spatial-CV fold
     R² / RMSE / MAE values with replacement; recompute the mean each
     time. The percentile CI on those B bootstrapped means is the
     uncertainty on the cross-fold aggregate. This is the standard
     uncertainty quantification for k-fold cross-validation (Bouckaert
     2003, Efron & Tibshirani 1993 §8). Cheap, runs on the existing
     kfold_results_summary.json — no per-prediction data needed.

  2. **Bootstrap over predictions** (--prediction-bootstrap): pool all
     ~16,000 held-out predictions across the 10 folds; bootstrap-resample
     with replacement at the prediction level; recompute R²/RMSE/MAE.
     Tighter intervals; needs per-fold prediction parquets in
     sweep/<group>/<tag>/fold_*_predictions.parquet. Falls back to
     mode 1 if those aren't present.

Reports 95% CIs (percentile method) for each architecture in the
flagship comparison set (8 rows), B = 10,000 by default.

Outputs:
    rebuttal/reviewer_responses/results/bootstrap_ci.{json,md}
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

from _common import (FLAGSHIP, COMPARISONS, all_entries,
                     load_summary, load_fold_predictions,
                     write_pair, banner)


def _bootstrap_over_folds(fold_values: list[float], n_boot: int,
                           rng: np.random.Generator) -> tuple[float, float, float]:
    """Resample fold_values with replacement n_boot times; return
    (mean, ci_lo, ci_hi) at the 2.5 / 97.5 percentile."""
    arr = np.array([v for v in fold_values
                    if v is not None and isinstance(v, (int, float)) and v == v])
    if arr.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_over_predictions(predicted: np.ndarray, actual: np.ndarray,
                                  n_boot: int, rng: np.random.Generator
                                  ) -> dict[str, tuple[float, float, float]]:
    """Bootstrap-resample (pred, actual) pairs with replacement; recompute
    R², RMSE, MAE each time. Returns dict of metric → (mean, lo, hi)."""
    n = predicted.size
    if n == 0:
        nan = (float('nan'), float('nan'), float('nan'))
        return {'r2': nan, 'rmse': nan, 'mae': nan}

    # Pre-compute the point estimates from the original predictions
    resid = predicted - actual
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2_point = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse_point = float(np.sqrt(ss_res / n))
    mae_point = float(np.mean(np.abs(resid)))

    # Bootstrap
    r2s = np.empty(n_boot); rmses = np.empty(n_boot); maes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = predicted[idx]; a = actual[idx]
        r = p - a
        sr = np.sum(r * r); st = np.sum((a - a.mean()) ** 2)
        r2s[b] = 1.0 - sr / st if st > 0 else float('nan')
        rmses[b] = np.sqrt(sr / n)
        maes[b] = np.mean(np.abs(r))
    return {
        'r2':   (r2_point, float(np.nanpercentile(r2s, 2.5)),  float(np.nanpercentile(r2s, 97.5))),
        'rmse': (rmse_point, float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))),
        'mae':  (mae_point,  float(np.percentile(maes, 2.5)),  float(np.percentile(maes, 97.5))),
    }


def bootstrap_one(entry, n_boot: int, rng: np.random.Generator,
                    use_predictions: bool) -> dict:
    """Run the bootstrap for a single architecture entry. Returns a dict
    suitable for table rendering and downstream JSON dump."""
    summary = load_summary(entry)
    if summary is None:
        return {'tag': entry.tag, 'group': entry.group, 'family': entry.family,
                'mode': 'unavailable', 'r2': None, 'rmse': None, 'mae': None}

    fold_r2s   = [f.get('r2')   for f in summary.get('fold_results', [])]
    fold_rmses = [f.get('rmse') for f in summary.get('fold_results', [])]
    fold_maes  = [f.get('mae')  for f in summary.get('fold_results', [])]
    n_folds = sum(1 for v in fold_r2s
                  if isinstance(v, (int, float)) and v == v)

    # Default: bootstrap over folds
    over_folds = {
        'r2':   _bootstrap_over_folds(fold_r2s,   n_boot, rng),
        'rmse': _bootstrap_over_folds(fold_rmses, n_boot, rng),
        'mae':  _bootstrap_over_folds(fold_maes,  n_boot, rng),
    }
    out = {
        'tag': entry.tag, 'group': entry.group, 'family': entry.family,
        'nice': entry.nice, 'is_flagship': entry.is_flagship,
        'mode': 'over_folds', 'n_folds': n_folds, 'n_boot': n_boot,
        'r2': over_folds['r2'], 'rmse': over_folds['rmse'], 'mae': over_folds['mae'],
    }

    # If requested, also do the (tighter) prediction-level bootstrap.
    if use_predictions:
        df = load_fold_predictions(entry)
        if df is None or df.empty:
            out['prediction_bootstrap'] = 'predictions_unavailable'
        else:
            pred_col = next((c for c in df.columns
                             if c.lower() in ('predicted', 'prediction', 'pred', 'predicted_soc')), None)
            actual_col = next((c for c in df.columns
                               if c.lower() in ('actual', 'oc', 'y_true', 'target', 'soc')), None)
            if pred_col is None or actual_col is None:
                out['prediction_bootstrap'] = f'columns_unknown:{list(df.columns)}'
            else:
                pb = _bootstrap_over_predictions(
                    df[pred_col].to_numpy(dtype=float),
                    df[actual_col].to_numpy(dtype=float),
                    n_boot, rng)
                out['prediction_bootstrap'] = {
                    'mode': 'over_predictions',
                    'n_predictions': int(len(df)),
                    'r2': pb['r2'], 'rmse': pb['rmse'], 'mae': pb['mae'],
                }
    return out


def render_markdown(rows: list[dict], mode_label: str) -> str:
    md = [f'# Bootstrap 95% CIs ({mode_label})', '']
    md.append('Addresses reviewer comments R1.3, R3.6, and R4 (implied).')
    md.append('B = ' + str(rows[0].get('n_boot', '?')) + ' bootstrap resamples; '
              'percentile method for confidence intervals.')
    md.append('')
    md.append('| Architecture | R² mean [95% CI] | RMSE mean [95% CI] | MAE mean [95% CI] |')
    md.append('|---|---|---|---|')
    def fmt(t, p=3):
        if t is None or not isinstance(t, (list, tuple)):
            return '—'
        m, lo, hi = t
        return f'{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]'
    def fmt_pos(t, p=3):
        if t is None or not isinstance(t, (list, tuple)):
            return '—'
        m, lo, hi = t
        return f'{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]'
    for r in rows:
        pf = '**' if r.get('is_flagship') else ''
        nice = r.get('nice', r['tag'])
        md.append(f'| {pf}{nice}{pf} | {fmt(r["r2"])} | {fmt_pos(r["rmse"])} | {fmt_pos(r["mae"])} |')
    md.append('')
    md.append('## Reading')
    md.append('')
    md.append('A 95% CI that *excludes zero* on R² mean is the formal statement '
              'that the model generalizes (mean R² > 0 with 95% confidence). Two '
              'architectures whose R² CIs *overlap substantially* are not '
              'statistically distinguishable on this evidence; see "Vanilla vs SGT" '
              'below.')
    md.append('')
    md.append('## Vanilla vs SGT — directly comparable claim')
    fl = next((r for r in rows if r.get('is_flagship')), None)
    sgt = next((r for r in rows if r.get('family') == 'sgt-small'), None)
    if fl and sgt:
        def lohi(t):
            if t is None: return (None, None)
            return t[1], t[2]
        flo, fhi = lohi(fl['r2'])
        slo, shi = lohi(sgt['r2'])
        if flo is not None and slo is not None:
            overlap_lo = max(flo, slo)
            overlap_hi = min(fhi, shi)
            if overlap_lo <= overlap_hi:
                md.append('')
                md.append(f'- Vanilla R² CI: [{flo:+.3f}, {fhi:+.3f}]')
                md.append(f'- SGT R² CI:     [{slo:+.3f}, {shi:+.3f}]')
                md.append(f'- Overlap:       [{overlap_lo:+.3f}, {overlap_hi:+.3f}]')
                md.append('')
                md.append('The CIs overlap — Vanilla and SGT are statistically '
                          'indistinguishable on the spatial-CV evidence. Combined '
                          'with the 40% parameter reduction (215k vs 363k), the '
                          'recommendation flips to Vanilla on parsimony grounds.')
    return '\n'.join(md)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-boot', type=int, default=10000,
                   help='Bootstrap resamples (default 10000).')
    p.add_argument('--prediction-bootstrap', action='store_true',
                   help='Also bootstrap at the prediction level (tighter CIs; '
                        'needs per-fold prediction parquets).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-broken', action='store_true')
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    entries = all_entries(include_broken=not a.no_broken)

    print(banner(f'BOOTSTRAP CIs  (B={a.n_boot})'))
    rows = []
    for e in entries:
        print(f'  {e.tag} @ {e.group} ...', flush=True)
        rows.append(bootstrap_one(e, a.n_boot, rng, a.prediction_bootstrap))

    mode_label = 'over folds (and predictions where available)' if a.prediction_bootstrap else 'over folds'
    md = render_markdown(rows, mode_label)
    print()
    print(md)
    write_pair('bootstrap_ci', {'rows': rows, 'meta': {
        'n_boot': a.n_boot, 'seed': a.seed,
        'prediction_bootstrap': a.prediction_bootstrap,
    }}, md)


if __name__ == '__main__':
    main()
