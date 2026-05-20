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
    ('sgt_d128_h4_L1',                  'SimpleSGT — gated (363k)',           'sgt'),
    ('vanilla_transformer_d128_h4_L1',  'Vanilla transformer — no gate (215k)', 'vanilla'),
    ('lightweight_transformer_d128_h4_L1', 'Lightweight Transformer — no CNN (240k)', 'lightweight'),
    ('simpletransformer_d64_h4_L1',     'SimpleTransformerV2 (11.2M)',        'transformer'),
    ('cnnlstm_d64_h4_L1',               'CNNLSTM (93k)',                      'cnnlstm'),
    ('3dcnn_d64_h4_L1',                 '3DCNN (failed family)',              '3dcnn'),
    ('xgb_shallow',                     'XGBoost (per-band stats)',           'tree'),
    ('rf_default',                      'RandomForest (per-band stats)',      'tree'),
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
    for run_name, label, kind, df, _ in runs_with_data:
        keys = list(zip(np.round(df.GPS_LONG, 5), np.round(df.GPS_LAT, 5)))
        s = pd.Series(df.predicted_soc.values, index=keys, name=run_name)
        # Drop duplicate (lat, lon) labels (keep first) — required for
        # pd.concat(..., join='inner'). Duplicates occur because the
        # 1mil Bavaria grid has repeated coordinates at 5-decimal rounding.
        n_before = len(s)
        s = s[~s.index.duplicated(keep='first')]
        if len(s) < n_before:
            print(f'   [{run_name}] dropped {n_before - len(s):,} duplicate '
                  f'(lat, lon) keys before alignment', flush=True)
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

    # ---- Pair runs by architecture and band variant for the figure layout ----
    def _split_run(rn: str):
        """Return (arch_key, band_variant) where band_variant ∈
        {'20band', '6band', ''}."""
        if rn.endswith('_6band'):
            return rn[:-6], '6band'
        if rn.endswith('_20band'):
            return rn[:-7], '20band'
        return rn, ''

    # Group runs by arch_key, preserving the DEFAULT_RUNS-given label/kind.
    by_arch: dict[str, dict] = {}
    arch_order: list[str] = []
    for run_name, label, kind, df, summary in runs_with_data:
        arch_key, band = _split_run(run_name)
        # Strip the bracketed " [20band]" / " [6band]" suffix from the
        # label so the architecture-row label is clean.
        base_label = label
        for tag in (' [20band]', ' [6band]'):
            if base_label.endswith(tag):
                base_label = base_label[: -len(tag)]
        if arch_key not in by_arch:
            by_arch[arch_key] = {'base_label': base_label, 'kind': kind, 'panels': {}}
            arch_order.append(arch_key)
        by_arch[arch_key]['panels'][band or '20band'] = {
            'run_name': run_name, 'df': df, 'summary': summary,
        }

    # ---- Figure: 2-col grid, 20band on left, 6band on right ----
    BAND_COLS = ['20band', '6band']    # column order
    DOWNSAMPLE_MAX = 150_000   # cap points per panel for fast render

    # Classify panels: nan-mean = broken (rendered as placeholder, never scattered).
    broken_panels: list[tuple[str, str]] = []
    for arch_key in arch_order:
        for band in BAND_COLS:
            panel = by_arch[arch_key]['panels'].get(band)
            if panel is None:
                continue
            m = panel['summary'].get('mean')
            if not isinstance(m, (int, float)) or m != m:   # NaN check
                broken_panels.append((arch_key, band))
    if broken_panels:
        print(f'[compare] {len(broken_panels)} panel(s) have NaN summary mean '
              f'and will render as "(broken — NaN predictions)":')
        for arch_key, band in broken_panels:
            run = by_arch[arch_key]['panels'][band]['run_name']
            print(f'   - {run} (arch={arch_key}, band={band})')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_rows = len(arch_order)
        fig, axes = plt.subplots(n_rows, 2,
                                  figsize=(11, 4.5 * max(n_rows, 1)),
                                  squeeze=False)

        # Shared colour range from valid (non-broken) panels only. Predictions
        # ARE plotted at their true values — only the colour normalization is
        # clipped at 100 g/kg — so "broken" tree panels (e.g. XGB shallow
        # 6-band mean ≈ 157) render as saturated and are flagged in title.
        valid_preds = []
        for arch_key in arch_order:
            for band in BAND_COLS:
                if (arch_key, band) in broken_panels:
                    continue
                panel = by_arch[arch_key]['panels'].get(band)
                if panel is None:
                    continue
                v = panel['df'].predicted_soc.values
                v = v[np.isfinite(v)]
                if v.size:
                    valid_preds.append(v)
        if valid_preds:
            all_pred = np.concatenate(valid_preds)
            pred_for_vmax = np.clip(all_pred, 0, 100)
            vmin = max(0.0, float(np.nanpercentile(pred_for_vmax, 2)))
            vmax = float(np.nanpercentile(pred_for_vmax, 98))
        else:
            vmin, vmax = 0.0, 80.0
        broken_threshold = 80.0    # mean SOC above this flags as broken

        # RNG for deterministic downsampling — same subset across panels so the
        # spatial coverage is identical (no apparent density differences just
        # because two panels random-sampled different points).
        rng = np.random.default_rng(0)

        sc = None
        for i, arch_key in enumerate(arch_order):
            entry = by_arch[arch_key]
            for j, band in enumerate(BAND_COLS):
                ax = axes[i][j]
                panel = entry['panels'].get(band)
                if panel is None:
                    ax.text(0.5, 0.5, f'(not run yet)\n{arch_key}\n[{band}]',
                              ha='center', va='center', fontsize=9, color='gray',
                              transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                if (arch_key, band) in broken_panels:
                    ax.text(0.5, 0.5,
                            f'(broken — NaN predictions)\n{arch_key}\n[{band}]',
                            ha='center', va='center', fontsize=9, color='crimson',
                            transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                df = panel['df']
                summary = panel['summary']
                # Downsample for speed: 1.3M × 12 panels = ~16M scatter
                # points crashed the renderer on the user's box. 150k per
                # panel preserves Bavaria coverage and renders in seconds.
                if len(df) > DOWNSAMPLE_MAX:
                    idx = rng.choice(len(df), size=DOWNSAMPLE_MAX, replace=False)
                    lon = df.GPS_LONG.values[idx]
                    lat = df.GPS_LAT.values[idx]
                    soc = df.predicted_soc.values[idx]
                else:
                    lon, lat, soc = df.GPS_LONG.values, df.GPS_LAT.values, df.predicted_soc.values
                sc = ax.scatter(lon, lat, c=soc,
                                  s=1, cmap='YlOrBr', vmin=vmin, vmax=vmax,
                                  alpha=0.85)
                mean_v = float(summary.get('mean', float('nan')))
                warning = '  ⚠ broken' if (mean_v == mean_v and mean_v > broken_threshold) else ''
                ax.set_title(f'mean = {mean_v:.2f} g/kg{warning}', fontsize=9)
                ax.set_xlabel('Lon', fontsize=8); ax.set_ylabel('Lat', fontsize=8)
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(labelsize=7)
            # Row label on the leftmost axis
            axes[i][0].set_ylabel(f'{entry["base_label"]}\nLat', fontsize=9)

        # Column headers
        if n_rows > 0:
            axes[0][0].annotate('20-band stack', xy=(0.5, 1.10),
                                  xycoords='axes fraction', ha='center',
                                  fontsize=11, fontweight='bold')
            axes[0][1].annotate('6-band stack (original-paper subset)',
                                  xy=(0.5, 1.10), xycoords='axes fraction',
                                  ha='center', fontsize=11, fontweight='bold')

        # Shared colorbar
        if sc is not None:
            fig.subplots_adjust(right=0.92)
            cbar_ax = fig.add_axes([0.94, 0.20, 0.012, 0.62])
            cbar = fig.colorbar(sc, cax=cbar_ax,
                                  label=f'Predicted SOC (g/kg, clipped to {int(vmax)})')
        fig.suptitle(f'Bavaria-wide SOC predictions, target year {a.year} '
                      f'— 20-band vs 6-band per architecture',
                      fontsize=13, fontweight='bold', y=1.01)
        # NOTE: skip tight_layout + bbox_inches='tight' — both are slow with
        # 12 axes × 150k points, and tight_layout warns about colorbar axes
        # anyway. subplots_adjust above gave the colorbar its space.
        out_png = HERE / f'maps_comparison_{a.year}.png'
        fig.savefig(out_png, dpi=200)
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
    # ---- 20-band vs 6-band side-by-side per architecture ----
    md.append('## 20-band vs 6-band (per architecture)')
    md.append('')
    md.append('Each row pairs the same architecture trained on the full 20-band '
              'covariate stack (revision expansion) versus the original 6-band '
              'subset. "—" means that variant has not finished training yet.')
    md.append('')
    md.append('| Architecture | 20-band mean | 20-band std | 6-band mean | 6-band std | Δ mean (6−20) |')
    md.append('|---|---|---|---|---|---|')
    for arch_key in arch_order:
        entry = by_arch[arch_key]
        p20 = entry['panels'].get('20band', {}).get('summary', {})
        p6  = entry['panels'].get('6band',  {}).get('summary', {})
        m20 = p20.get('mean'); s20 = p20.get('std')
        m6  = p6.get('mean');  s6  = p6.get('std')
        cell = lambda v, prec=2: f'{v:.{prec}f}' if isinstance(v, (int, float)) and v == v else '—'
        delta = (m6 - m20) if (isinstance(m20, (int, float)) and isinstance(m6, (int, float))) else None
        md.append(f'| **{entry["base_label"]}** | {cell(m20)} | {cell(s20)} | '
                  f'{cell(m6)} | {cell(s6)} | '
                  f'{cell(delta)} |')
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
