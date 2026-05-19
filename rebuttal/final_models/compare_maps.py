#!/usr/bin/env python3
"""
rebuttal/final_models/compare_maps.py — assemble the cross-architecture
comparison figure for the rebuttal pivot.

After submit_finals.py has trained each architecture on the full
dataset and inferred over the Bavaria 1mil grid, every run leaves a
predictions parquet under rebuttal/final_models/maps/<run-name>/. This
script reads all of them and produces:

    maps_comparison_<year>.png       N-panel side-by-side map figure
                                      (one panel per architecture)
    maps_comparison_<year>.md        per-model summary table +
                                      pairwise correlation matrix
    maps_comparison_<year>.json      machine-readable summary

Run from SOCmapping/ (no GPU needed, ~10 s):
    python rebuttal/final_models/compare_maps.py
    python rebuttal/final_models/compare_maps.py --year 2007
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE / 'maps'

# Display order + nice labels for the figure.
# (run_name, display label, family_kind)
DEFAULT_RUNS = [
    ('sgt_d128_h4_L1',                'SimpleSGT — gated (363k)',         'sgt'),
    ('vanilla_transformer_d128_h4_L1', 'Vanilla transformer — no gate (215k)', 'vanilla'),
    ('simpletransformer_d64_h4_L1',   'SimpleTransformerV2 (11.2M)',      'transformer'),
    ('cnnlstm_d64_h4_L1',             'CNNLSTM (93k)',                    'cnnlstm'),
    ('xgb_shallow',                   'XGBoost (per-band stats)',         'tree'),
    ('rf_default',                    'RandomForest (per-band stats)',    'tree'),
]

# Auto-expand each run_name across both band-variants so we don't need to
# repeat the (run_name, label, kind) triples manually for _6band / _20band.
DEFAULT_RUNS = [
    (rn + sfx, lbl + f' [{sfx[1:]}]', kind)
    for rn, lbl, kind in DEFAULT_RUNS
    for sfx in ('_20band', '_6band')
]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run_names to include (default: '
                        'all entries in DEFAULT_RUNS that have outputs).')
    return p.parse_args()


def load_run(run_name: str, year: int):
    pq = MAPS_ROOT / run_name / f'bavaria_{year}_predictions.parquet'
    if not pq.exists():
        return None, None
    df = pd.read_parquet(pq)
    summary_p = MAPS_ROOT / run_name / f'bavaria_{year}_summary.json'
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    return df, summary


def main():
    a = parse()
    wanted = set(a.runs.split(',')) if a.runs else None

    runs_with_data = []
    for run_name, label, kind in DEFAULT_RUNS:
        if wanted and run_name not in wanted:
            continue
        df, summary = load_run(run_name, a.year)
        if df is None:
            print(f'[compare] skip {run_name}: no map output yet for year {a.year}',
                  file=sys.stderr)
            continue
        runs_with_data.append((run_name, label, kind, df, summary))

    if not runs_with_data:
        print(f'[compare] no map outputs found under {MAPS_ROOT} for year {a.year}',
              file=sys.stderr)
        sys.exit(1)

    print(f'[compare] {len(runs_with_data)} run(s) loaded:')
    for run_name, label, kind, df, summary in runs_with_data:
        print(f'   {run_name:>32}  ({label})  n={len(df):>7}  '
              f'mean={summary.get("mean", float("nan")):.2f}', flush=True)

    # ---- Cross-model pairwise correlation (Pearson, on common grid points) ----
    # Align all maps on (round(GPS_LONG, 5), round(GPS_LAT, 5)).
    aligned = {}
    base_keys = None
    for run_name, label, kind, df, _ in runs_with_data:
        keys = list(zip(np.round(df.GPS_LONG, 5), np.round(df.GPS_LAT, 5)))
        s = pd.Series(df.predicted_soc.values, index=keys, name=run_name)
        aligned[run_name] = s
    aligned_df = pd.concat(aligned.values(), axis=1, join='inner')
    aligned_df.columns = [r[0] for r in runs_with_data]
    print(f'[compare] common grid points across all models: {len(aligned_df):,}',
          flush=True)

    corr = aligned_df.corr().round(3)
    diff_stats = {}     # pairwise mean abs difference
    for i in range(len(runs_with_data)):
        for j in range(i + 1, len(runs_with_data)):
            a_, b_ = runs_with_data[i][0], runs_with_data[j][0]
            diff = aligned_df[a_].values - aligned_df[b_].values
            diff_stats[f'{a_}__vs__{b_}'] = {
                'mean_abs_diff': float(np.nanmean(np.abs(diff))),
                'mean_signed_diff': float(np.nanmean(diff)),
                'std_diff': float(np.nanstd(diff)),
                'pearson_r': float(corr.loc[a_, b_]),
            }

    # ---- Figure: side-by-side maps ----
    n_runs = len(runs_with_data)
    ncols = min(n_runs, 3)
    nrows = (n_runs + ncols - 1) // ncols
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 5 * nrows),
                                  squeeze=False)
        # Shared color range across panels for fair visual comparison.
        all_preds = np.concatenate([df.predicted_soc.values
                                      for _, _, _, df, _ in runs_with_data])
        vmin = max(0.0, float(np.nanpercentile(all_preds, 2)))
        vmax = float(np.nanpercentile(all_preds, 98))
        for k, (run_name, label, kind, df, summary) in enumerate(runs_with_data):
            r, c = k // ncols, k % ncols
            ax = axes[r][c]
            sc = ax.scatter(df.GPS_LONG, df.GPS_LAT, c=df.predicted_soc,
                              s=1, cmap='YlOrBr', vmin=vmin, vmax=vmax,
                              alpha=0.85)
            mean_str = f'{summary.get("mean", float("nan")):.2f}'
            ax.set_title(f'{label}\nmean = {mean_str} g/kg', fontsize=10)
            ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
            ax.set_aspect('equal', adjustable='box')
        # Turn off unused subplots
        for k in range(len(runs_with_data), nrows * ncols):
            r, c = k // ncols, k % ncols
            axes[r][c].axis('off')
        # Shared colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.20, 0.012, 0.62])
        fig.colorbar(sc, cax=cbar_ax, label='Predicted SOC (g/kg)')
        fig.suptitle(f'Bavaria-wide SOC predictions, target year {a.year} '
                      f'— full-data trained, identical inference grid',
                      fontsize=12, fontweight='bold', y=1.01)
        fig.tight_layout()
        out_png = HERE / f'maps_comparison_{a.year}.png'
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[compare] saved {out_png}', flush=True)
    except Exception as e:
        print(f'[compare] map figure failed: {e}', file=sys.stderr)

    # ---- Markdown summary ----
    md = [f'# Cross-architecture SOC map comparison — Bavaria {a.year}', '']
    md.append('All models trained on the **full** LUCAS/LfL/LfU dataset '
              '(no spatial holdout; 5% random monitor used only for '
              'best-state tracking) and applied to the 1mil Bavaria '
              'reference grid with the natural 5-year window '
              f'{{{a.year-4}, …, {a.year}}}.')
    md.append('')
    md.append('## Per-model summary')
    md.append('')
    md.append('| Run | Label | n_valid | mean | std | p05 | p50 | p95 |')
    md.append('|-----|-------|---------|------|-----|-----|-----|-----|')
    for run_name, label, kind, df, summary in runs_with_data:
        v = df.predicted_soc.values
        v_valid = v[np.isfinite(v)]
        md.append(f'| `{run_name}` | {label} | {len(v_valid)} | '
                  f'{summary.get("mean", float("nan")):.2f} | '
                  f'{summary.get("std", float("nan")):.2f} | '
                  f'{summary.get("p05", float("nan")):.2f} | '
                  f'{summary.get("p50", float("nan")):.2f} | '
                  f'{summary.get("p95", float("nan")):.2f} |')
    md.append('')
    md.append('## Pairwise Pearson correlation (across-grid)')
    md.append('')
    md.append('|       | ' + ' | '.join(corr.columns) + ' |')
    md.append('|-------|' + '|'.join(['-' * 12] * len(corr.columns)) + '|')
    for idx, row in corr.iterrows():
        md.append(f'| **{idx}** | ' + ' | '.join(f'{v:.3f}' for v in row.values) + ' |')
    md.append('')
    md.append('## Pairwise difference statistics (a − b)')
    md.append('')
    md.append('| Pair | mean abs diff | mean signed diff | std diff | Pearson r |')
    md.append('|------|---------------|------------------|----------|-----------|')
    for pair, s in diff_stats.items():
        md.append(f'| {pair.replace("__vs__", " vs ")} | '
                  f'{s["mean_abs_diff"]:.2f} | {s["mean_signed_diff"]:+.3f} | '
                  f'{s["std_diff"]:.2f} | {s["pearson_r"]:.3f} |')
    md.append('')
    md.append('## Interpretation')
    md.append('')
    md.append('- High pairwise correlation (>0.7) indicates models agree on '
              '*relative* spatial structure even where mean SOC magnitudes differ.')
    md.append('- Large mean abs difference (>5 g/kg) between two models indicates '
              'they disagree on *level* — typically tree models predict toward '
              'the training-pool central tendency, while neural models extrapolate '
              'further. Pair this with the spatial-CV fold-0 result (tree '
              'baselines uniformly negative on the Alpine extrapolation fold; '
              'SGT uniformly positive).')
    md.append('- The architecture-driven differences here are *not* artifacts of '
              'overfitting to a particular evaluation split — they reflect '
              'genuinely different inductive biases applied to identical training '
              'data and identical inference geometry.')
    md.append('')
    (HERE / f'maps_comparison_{a.year}.md').write_text('\n'.join(md))
    print(f'[compare] saved {HERE / f"maps_comparison_{a.year}.md"}', flush=True)

    (HERE / f'maps_comparison_{a.year}.json').write_text(
        json.dumps({
            'year': a.year,
            'runs': [r[0] for r in runs_with_data],
            'pairwise_correlation': corr.to_dict(),
            'pairwise_differences': diff_stats,
            'per_run_summary': {r[0]: r[4] for r in runs_with_data},
        }, indent=2, default=str))


if __name__ == '__main__':
    main()
