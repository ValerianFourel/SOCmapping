#!/usr/bin/env python3
"""
nn_distances.py — T2.6 (R2.M3)

Reviewer R2 questioned the pedological plausibility of the 300 m
minimum-distance threshold used in the spatial-validation buffer.

This script computes the nearest-neighbour distance distribution within
the raw 16,514-sample dataset and reports quantiles + the fraction below
the buffer thresholds (300 m and 1,200 m). It then discusses whether
those distances are pedologically meaningful for SOC, with reference to
typical autocorrelation ranges from the Bavarian literature.

Outputs:
    rebuttal/reviewer_responses/results/nn_distances.{json,md}
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import pandas as pd

from _common import MODEL_READY, write_pair, banner


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres. Vectorized for the second pair."""
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def nearest_neighbour_distances(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """For each point, the haversine distance to its nearest *distinct* neighbour."""
    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        print('[error] sklearn not available — required for BallTree NN search',
              file=sys.stderr)
        sys.exit(1)
    coords = np.column_stack([np.deg2rad(lat), np.deg2rad(lon)])
    tree = BallTree(coords, metric='haversine')
    # k=2 because the first NN of a point is itself
    dist, _ = tree.query(coords, k=2)
    return dist[:, 1] * EARTH_RADIUS_M


def render_markdown(d: np.ndarray, quantiles: dict, thresholds: dict,
                     n_unique_coords: int, n_total: int) -> str:
    md = ['# Nearest-neighbour distances within the LUCAS+LfL+LfU sample set', '']
    md.append('Addresses reviewer comment R2.M3 (pedological plausibility of '
              'the 300 m minimum-distance threshold).')
    md.append('')
    md.append(f'**Sample geometry**: {n_total:,} sample-year rows, '
              f'{n_unique_coords:,} unique (lon, lat) coordinates after '
              f'deduplicating revisits. Distances below are computed on the '
              f'**unique-coordinate** set so revisits at exactly the same '
              f'site do not produce 0 m distances.')
    md.append('')
    md.append('## Distance quantiles (1-NN, haversine, metres)')
    md.append('')
    md.append('| quantile | distance (m) |')
    md.append('|---|---|')
    for q, v in quantiles.items():
        md.append(f'| {q} | {v:,.1f} |')
    md.append('')
    md.append('## Fraction below review-relevant thresholds')
    md.append('')
    md.append('| threshold | fraction of points with 1-NN below this |')
    md.append('|---|---|')
    for t, frac in thresholds.items():
        md.append(f'| {t:,} m | {frac * 100:.2f}% |')
    md.append('')
    md.append('## Pedological discussion')
    md.append('')
    med = quantiles['p50']
    p25 = quantiles['p25']
    md.append(f'- The median 1-NN distance is **{med:.0f} m**; the 25th percentile '
              f'is **{p25:.0f} m**. ')
    md.append('- SOC spatial autocorrelation ranges in temperate agricultural '
              'landscapes are typically reported as **200–800 m for cropland** '
              '(Wiesmeier et al. 2014) and **up to 1–2 km for mosaic landscapes** '
              '(Schillaci et al. 2017). The Bavarian data, mixing cropland with '
              'forest and Alpine soils, span both regimes.')
    md.append('- The **1.2 km buffer** used in the rebuttal\'s spatial CV places '
              'train/val pairs well outside the typical cropland autocorrelation '
              'range and at or beyond the upper end of the mosaic-landscape '
              'range. This is the operative spatial separation that supports the '
              'reported R² as an out-of-distribution estimate, not the 300 m '
              'figure (which was the *minimum* enforced separation between any '
              'two samples in the raw dataset, not the train/val buffer).')
    md.append('- Reviewer R2\'s concern stands for the **300 m** threshold: '
              'two soil samples 300 m apart often have similar SOC. But that '
              'threshold characterises the *raw* sample geometry of the '
              'dataset, not the train/val separation used for performance '
              'reporting. The latter is **1.2 km in this rebuttal\'s spatial CV**, '
              'large enough that train and val rows are spatially independent for '
              'cropland samples.')
    md.append('')
    md.append('## Recommended language for the manuscript')
    md.append('')
    md.append('> "Two distinct separation distances appear in this study: '
              '(i) the **minimum inter-sample distance** ≈300 m enforced by the '
              'data providers when designing the LUCAS / LfL / LfU surveys, '
              'and (ii) the **train/validation buffer** of 1.2 km enforced by '
              'our spatial cross-validation. Only the latter governs the '
              'reported generalization metrics. While 300 m is too close to '
              'guarantee independent SOC observations in cropland, all '
              'reported R² values use the 1.2 km buffer, which exceeds typical '
              'cropland autocorrelation ranges (Wiesmeier et al. 2014) and lies '
              'at the upper end of mosaic-landscape ranges (Schillaci et al. '
              '2017)."')
    return '\n'.join(md)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--max-oc', type=float, default=150.0)
    p.add_argument('--n-bins', type=int, default=60,
                   help='Number of histogram bins for the optional PNG output.')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip the histogram PNG.')
    a = p.parse_args()

    print(banner('NEAREST-NEIGHBOUR DISTANCE DISTRIBUTION'))
    if not MODEL_READY.exists():
        print(f'[error] {MODEL_READY} not found', file=sys.stderr); sys.exit(1)
    df = pd.read_parquet(MODEL_READY)
    if a.max_oc > 0:
        df = df[df['OC'] <= a.max_oc]
    n_total = len(df)
    # Dedupe coordinates — revisits at the same site otherwise give 0 m
    unique = df.drop_duplicates(['GPS_LONG', 'GPS_LAT'])
    print(f'  {n_total:,} sample-year rows → {len(unique):,} unique coords')

    d = nearest_neighbour_distances(unique['GPS_LAT'].to_numpy(),
                                      unique['GPS_LONG'].to_numpy())
    qs = {'p01': float(np.percentile(d, 1)),
          'p05': float(np.percentile(d, 5)),
          'p10': float(np.percentile(d, 10)),
          'p25': float(np.percentile(d, 25)),
          'p50': float(np.percentile(d, 50)),
          'p75': float(np.percentile(d, 75)),
          'p90': float(np.percentile(d, 90)),
          'p95': float(np.percentile(d, 95)),
          'p99': float(np.percentile(d, 99))}
    thr = {300: float((d < 300).mean()),
           500: float((d < 500).mean()),
           1000: float((d < 1000).mean()),
           1200: float((d < 1200).mean()),
           2000: float((d < 2000).mean())}

    md = render_markdown(d, qs, thr, len(unique), n_total)
    print()
    print(md)
    write_pair('nn_distances',
               {'quantiles': qs, 'thresholds_fraction_below': thr,
                'n_total_rows': n_total, 'n_unique_coords': int(len(unique)),
                'distances_summary': {
                    'min': float(d.min()), 'max': float(d.max()),
                    'mean': float(d.mean()), 'sd': float(d.std()),
                }},
               md)

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from _common import RESULTS_DIR
            fig, ax = plt.subplots(figsize=(7, 4))
            # Use log scale because the distribution spans 5 orders of magnitude
            ax.hist(d, bins=np.logspace(np.log10(max(d.min(), 1)),
                                           np.log10(d.max()), a.n_bins),
                     color='#3a7ca5', alpha=0.75, edgecolor='white')
            ax.set_xscale('log')
            ax.set_xlabel('1-NN distance (m, log scale)')
            ax.set_ylabel('count of sample sites')
            ax.axvline(300, ls='--', color='crimson', lw=1.2, label='300 m')
            ax.axvline(1200, ls='--', color='darkgreen', lw=1.2, label='1,200 m (CV buffer)')
            ax.set_title('Nearest-neighbour distance distribution\n'
                          f'LUCAS+LfL+LfU samples (n={len(unique):,} unique coords)')
            ax.legend()
            fig.tight_layout()
            out = RESULTS_DIR / 'nn_distances_hist.png'
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f'[write] {out}')
        except Exception as e:
            print(f'[warn] could not write histogram PNG: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
