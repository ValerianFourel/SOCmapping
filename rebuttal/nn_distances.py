#!/usr/bin/env python3
"""
nn_distances.py — Task 4 of the Geoderma rebuttal.

Computes pairwise nearest-neighbour distances between every sample point
in the LUCAS+LfU+LfL Bavaria OC dataset (the master sample file declared
in SOCmapping/balancedDataset/config.py). Plots the histogram and reports
distance summary statistics.

Outputs:
    rebuttal/nn_distance_histogram.png
    rebuttal/nn_distances.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')           # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SAMPLE_PATH = Path('/home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx')
OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal')
EARTH_RADIUS_M = 6_371_008.8     # mean Earth radius in metres, GRS80


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(SAMPLE_PATH)
    print(f'Loaded {len(df):,} rows from {SAMPLE_PATH.name}')
    print(f'Columns: {list(df.columns)}')

    # Many surveys revisit identical coordinates. NN distance is a property of
    # the set of unique sampling locations, so deduplicate first.
    locs = (df[['GPS_LAT', 'GPS_LONG']]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True))
    print(f'Unique (lat, lon) pairs (NaN dropped): {len(locs):,}')

    lat_rad = np.deg2rad(locs['GPS_LAT'].to_numpy())
    lon_rad = np.deg2rad(locs['GPS_LONG'].to_numpy())
    coords = np.column_stack([lat_rad, lon_rad])

    # k=2 to skip the self-match (distance 0)
    tree = BallTree(coords, metric='haversine')
    dists_rad, _ = tree.query(coords, k=2)
    nn_dist_m = dists_rad[:, 1] * EARTH_RADIUS_M

    summary = {
        'sample_file': str(SAMPLE_PATH),
        'n_rows_in_file': int(len(df)),
        'n_unique_locations': int(len(locs)),
        'nn_distance_metres': {
            'min': float(np.min(nn_dist_m)),
            'p5': float(np.percentile(nn_dist_m, 5)),
            'p25': float(np.percentile(nn_dist_m, 25)),
            'median': float(np.median(nn_dist_m)),
            'mean': float(np.mean(nn_dist_m)),
            'p75': float(np.percentile(nn_dist_m, 75)),
            'p95': float(np.percentile(nn_dist_m, 95)),
            'p99': float(np.percentile(nn_dist_m, 99)),
            'max': float(np.max(nn_dist_m)),
            'std': float(np.std(nn_dist_m, ddof=1)),
        },
        'tail_share': {
            'frac_below_100m': float(np.mean(nn_dist_m < 100.0)),
            'frac_below_500m': float(np.mean(nn_dist_m < 500.0)),
            'frac_below_1000m': float(np.mean(nn_dist_m < 1000.0)),
            'frac_below_2000m': float(np.mean(nn_dist_m < 2000.0)),
        },
    }

    # ---- Histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(nn_dist_m, bins=80, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Nearest-neighbour distance (m)')
    axes[0].set_ylabel('Count of sample locations')
    axes[0].set_title('Linear scale')
    axes[0].axvline(summary['nn_distance_metres']['median'], color='crimson',
                    linestyle='--', label=f'median = {summary["nn_distance_metres"]["median"]:.0f} m')
    axes[0].axvline(summary['nn_distance_metres']['mean'], color='orange',
                    linestyle='--', label=f'mean = {summary["nn_distance_metres"]["mean"]:.0f} m')
    axes[0].legend()

    bins_log = np.logspace(
        np.log10(max(1.0, summary['nn_distance_metres']['min'])),
        np.log10(summary['nn_distance_metres']['max']),
        80)
    axes[1].hist(nn_dist_m, bins=bins_log, color='steelblue', edgecolor='white')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Nearest-neighbour distance (m, log scale)')
    axes[1].set_ylabel('Count of sample locations')
    axes[1].set_title('Log scale')
    axes[1].axvline(summary['nn_distance_metres']['median'], color='crimson',
                    linestyle='--')
    axes[1].axvline(summary['nn_distance_metres']['mean'], color='orange',
                    linestyle='--')

    fig.suptitle(f'Nearest-neighbour distance distribution — '
                 f'{len(locs):,} unique sample locations in Bavaria')
    fig.tight_layout()
    fig_path = OUT_DIR / 'nn_distance_histogram.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    # ---- Markdown ----
    s = summary['nn_distance_metres']
    md = [
        '# Nearest-neighbour distances — LUCAS+LfU+LfL Bavaria sample set',
        '',
        f'_Source: `{SAMPLE_PATH}`_',
        '',
        f'- Rows in source file: **{summary["n_rows_in_file"]:,}** '
        '(many revisited POINTIDs share identical coordinates).',
        f'- Unique sample locations used for NN distance: **{summary["n_unique_locations"]:,}**.',
        f'- Distances computed with `sklearn.neighbors.BallTree(metric="haversine")` × Earth radius '
        f'{EARTH_RADIUS_M:.1f} m.',
        '',
        '## Summary statistics (metres)',
        '',
        '| stat | metres |',
        '|------|--------|',
        f'| min | {s["min"]:.2f} |',
        f'| 5th percentile | {s["p5"]:.2f} |',
        f'| 25th percentile | {s["p25"]:.2f} |',
        f'| median | {s["median"]:.2f} |',
        f'| mean | {s["mean"]:.2f} |',
        f'| 75th percentile | {s["p75"]:.2f} |',
        f'| 95th percentile | {s["p95"]:.2f} |',
        f'| 99th percentile | {s["p99"]:.2f} |',
        f'| max | {s["max"]:.2f} |',
        f'| std (ddof=1) | {s["std"]:.2f} |',
        '',
        '## Tail shares',
        '',
        '| threshold | fraction of locations whose NN is closer |',
        '|-----------|-----------------------------------------|',
        f'| < 100 m | {summary["tail_share"]["frac_below_100m"] * 100:.2f}% |',
        f'| < 500 m | {summary["tail_share"]["frac_below_500m"] * 100:.2f}% |',
        f'| < 1 km  | {summary["tail_share"]["frac_below_1000m"] * 100:.2f}% |',
        f'| < 2 km  | {summary["tail_share"]["frac_below_2000m"] * 100:.2f}% |',
        '',
        f'_Plot:_ `{fig_path.relative_to(OUT_DIR)}`',
        '',
    ]
    md_path = OUT_DIR / 'nn_distances.md'
    md_path.write_text('\n'.join(md))
    print(f'Wrote {md_path}')

    # Persist a JSON for the master rebuttal_numbers.md aggregator
    (OUT_DIR / 'nn_distances.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
