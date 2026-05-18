#!/usr/bin/env python3
"""
inspect_sweep.py — diagnostic for the architecture sweep.

For each sweep config, for each fold, prints:
  best_ep / total: which epoch had the highest R² (coefficient of det.) and what value
  first / last R²: trajectory shape
  saved R²:        what made it into the per-fold summary (i.e. what the saved
                    best_state's predictions actually score on test)

If best R² >> saved R², the best-state tracking is buggy. If best ≈ saved and
both are negative, the architecture genuinely never reaches positive R² on that
fold — that's geography + capacity, not a code bug.

Per-epoch JSON uses key 'r_squared' (coefficient of determination). Don't confuse
with 'pearson_r2' (Pearson r squared) which is also in there.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', default='rebuttal/gpu_experiments/spatial_kfold/sweep',
                    help='Root sweep directory.')
    ap.add_argument('--tags', default=None,
                    help='Comma-separated list of tags to inspect (default: all).')
    ap.add_argument('--show-peak-epoch', action='store_true',
                    help='Also print the epoch index where R² peaked (for early-stop guidance).')
    a = ap.parse_args()

    sweep = Path(a.sweep_dir)
    if not sweep.exists():
        print(f'[inspect] {sweep} not found', file=sys.stderr); sys.exit(1)

    tags = a.tags.split(',') if a.tags else sorted(
        d.name for d in sweep.iterdir()
        if d.is_dir() and d.name not in ('sbatch', 'slurm_logs')
        and (d / 'kfold_results_summary.json').exists())

    if not tags:
        print(f'[inspect] no completed config dirs under {sweep}', file=sys.stderr)
        sys.exit(1)

    for tag in tags:
        cfg_dir = sweep / tag
        sumf = cfg_dir / 'kfold_results_summary.json'
        if not sumf.exists():
            print(f'\n=== {tag} ===  (no summary yet)')
            continue
        summary = json.loads(sumf.read_text())
        rows_by_fold = {r['fold_id']: r for r in summary.get('fold_results', [])}
        recipe = summary.get('recipe', {})
        print(f'\n=== {tag} ===  epochs={recipe.get("num_epochs")}  '
              f'variant={recipe.get("model_size", "?")}  '
              f'lr={recipe.get("lr")}  dropout={recipe.get("dropout_rate", "?")}')
        hdr = (f'{"fold":>4}  {"best_ep":>8}/{"tot":<4}  {"best R²":>9}  '
               f'{"first R²":>9}  {"last R²":>9}  {"saved R²":>9}  '
               f'{"Δ saved-best":>13}')
        print(hdr)
        for fid in range(10):
            mpath = cfg_dir / f'fold_{fid}_metrics.json'
            if not mpath.exists():
                print(f'{fid:>4}  (no metrics file)')
                continue
            try:
                ms = json.loads(mpath.read_text())
            except Exception as e:
                print(f'{fid:>4}  (parse error: {e})')
                continue
            # epoch_metrics is a list of dicts with 'r_squared' key
            if not isinstance(ms, list):
                print(f'{fid:>4}  (unexpected structure: {type(ms).__name__})')
                continue
            r2s = []
            for m in ms:
                if isinstance(m, dict) and 'r_squared' in m:
                    v = m['r_squared']
                    if isinstance(v, (int, float)) and v == v:  # not nan
                        r2s.append(v)
            if not r2s:
                # Show what keys are actually present in the first epoch
                ks = list(ms[0].keys()) if ms and isinstance(ms[0], dict) else []
                print(f'{fid:>4}  (no r_squared values; first-epoch keys: {ks})')
                continue
            best_i = max(range(len(r2s)), key=lambda i: r2s[i])
            saved = rows_by_fold.get(fid, {}).get('r2', float('nan'))
            delta = saved - r2s[best_i]
            print(f'{fid:>4}  {best_i+1:>8}/{len(r2s):<4}  '
                  f'{r2s[best_i]:+9.4f}  '
                  f'{r2s[0]:+9.4f}  '
                  f'{r2s[-1]:+9.4f}  '
                  f'{saved:+9.4f}  '
                  f'{delta:+13.4f}')

        # Summary line: when did most folds peak?
        if a.show_peak_epoch:
            peaks = []
            for fid in range(10):
                mpath = cfg_dir / f'fold_{fid}_metrics.json'
                if not mpath.exists(): continue
                try: ms = json.loads(mpath.read_text())
                except Exception: continue
                r2s = [m['r_squared'] for m in ms
                       if isinstance(m, dict) and 'r_squared' in m
                       and isinstance(m['r_squared'], (int, float))
                       and m['r_squared'] == m['r_squared']]
                if r2s:
                    peaks.append(max(range(len(r2s)), key=lambda i: r2s[i]) + 1)
            if peaks:
                print(f'  peak epochs across folds: {peaks}  '
                      f'(median={sorted(peaks)[len(peaks)//2]})')


if __name__ == '__main__':
    main()
