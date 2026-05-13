#!/usr/bin/env python3
"""
split_comparison.py — Task 5 of the Geoderma rebuttal.

The five Bavaria 1-million-point TFT runs in the repository all use the
*spatial* split logic from `balancedDataset.create_validation_train_sets`
(min-distance threshold). No purely-random parquet survives. To answer
the reviewers' request for a random-vs-spatial comparison, this script:

  1. Loads the actual spatial-split validation rows from the Task 1
     parquet (TFT 1mil composite_l2_v2 run 1).
  2. Draws a matched-size random validation set from the same total
     pool of rows (same parquet, but ignoring dataset_type), with a
     fixed seed.
  3. Computes the same descriptive stats on both and prints a
     side-by-side comparison.

If you find an old parquet whose validation set was actually random,
point this script at it via the SPATIAL_PARQUET / RANDOM_PARQUET
variables — the descriptive function works on any OC vector.

Outputs:
    rebuttal/split_comparison.md
    rebuttal/split_comparison.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPATIAL_PARQUET = Path(
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/'
    'TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/'
    'train_val_data_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_'
    'TRANSFORM_normalize_LOSS_composite_l2.parquet'
)
OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal')
RANDOM_SEED = 20260513      # rebuttal deadline date


def descriptive(oc: pd.Series) -> dict:
    s = pd.to_numeric(oc, errors='coerce').dropna().to_numpy()
    if s.size == 0:
        return {'n': 0}
    q25, q50, q75 = np.percentile(s, [25, 50, 75])
    return {
        'n': int(s.size),
        'mean': float(np.mean(s)),
        'sd': float(np.std(s, ddof=1)),
        'median': float(q50),
        'q25': float(q25),
        'q75': float(q75),
        'iqr': float(q75 - q25),
        'min': float(s.min()),
        'max': float(s.max()),
        'pct_gt_50': float(100.0 * (s > 50.0).mean()),
        'pct_gt_120': float(100.0 * (s > 120.0).mean()),
        'n_gt_50': int((s > 50.0).sum()),
        'n_gt_120': int((s > 120.0).sum()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SPATIAL_PARQUET)
    spatial_val = df[df['dataset_type'] == 'val'].copy()
    n_val = len(spatial_val)
    print(f'Parquet rows: {len(df):,} '
          f'(spatial-val n={n_val}, spatial-train n={(df.dataset_type=="train").sum()})')

    # Random split of equal size, drawn from the full pool of rows
    rng = np.random.default_rng(RANDOM_SEED)
    random_val_idx = rng.choice(len(df), size=n_val, replace=False)
    random_val = df.iloc[random_val_idx].copy()
    print(f'Random val of size {n_val} drawn with seed={RANDOM_SEED}')

    sp_desc = descriptive(spatial_val['OC'])
    rd_desc = descriptive(random_val['OC'])

    # Spatial extents — just to show whether one fills the bounding box more uniformly
    geom = {
        'spatial_val': {
            'lat_min': float(spatial_val['GPS_LAT'].min()),
            'lat_max': float(spatial_val['GPS_LAT'].max()),
            'lon_min': float(spatial_val['GPS_LONG'].min()),
            'lon_max': float(spatial_val['GPS_LONG'].max()),
            'unique_locs': int(spatial_val[['GPS_LAT', 'GPS_LONG']].drop_duplicates().shape[0]),
        },
        'random_val': {
            'lat_min': float(random_val['GPS_LAT'].min()),
            'lat_max': float(random_val['GPS_LAT'].max()),
            'lon_min': float(random_val['GPS_LONG'].min()),
            'lon_max': float(random_val['GPS_LONG'].max()),
            'unique_locs': int(random_val[['GPS_LAT', 'GPS_LONG']].drop_duplicates().shape[0]),
        }
    }

    # Per-year breakdown
    sp_year = (spatial_val.groupby('year').size().rename('n_spatial')).reset_index()
    rd_year = (random_val.groupby('year').size().rename('n_random')).reset_index()
    per_year = sp_year.merge(rd_year, on='year', how='outer').fillna(0)
    per_year['n_spatial'] = per_year['n_spatial'].astype(int)
    per_year['n_random'] = per_year['n_random'].astype(int)

    out = {
        'spatial_parquet': str(SPATIAL_PARQUET),
        'random_seed': RANDOM_SEED,
        'spatial_val_desc': sp_desc,
        'random_val_desc': rd_desc,
        'geometry': geom,
        'per_year': per_year.to_dict(orient='records'),
    }

    # ---- markdown ----
    md = []
    md.append('# Spatial vs random validation split — descriptive comparison')
    md.append('')
    md.append(f'_Spatial split source:_ `{SPATIAL_PARQUET}`')
    md.append('')
    md.append(f'_Random split:_ synthetic, drawn here with seed = `{RANDOM_SEED}`. '
              f'Same size ({sp_desc["n"]} rows) sampled without replacement '
              f'from the full pool of {len(df)} rows in the same parquet.')
    md.append('')
    md.append('> **Caveat.** Every train/val parquet I found in the repository '
              '(`residualModels1mil_normalize_*`, `_log_*`, `_normalize_mse`) uses '
              'the same `create_validation_train_sets(distance_threshold=1.2)` '
              'spatial-split logic. There is no surviving parquet from a purely '
              'random split. The "random" column below is therefore a synthetic '
              'reference draw — it shows what a same-size random validation set '
              'looks like on this dataset, which is the closest you can get to '
              'an apples-to-apples random-vs-spatial comparison without retraining.')
    md.append('')

    md.append('## Side-by-side descriptive statistics')
    md.append('')
    md.append('| metric | spatial val | random val | Δ (random − spatial) |')
    md.append('|--------|-------------|------------|----------------------|')
    rows = [
        ('n', '{:d}', False),
        ('mean (g/kg)', '{:.3f}', True),
        ('sd (g/kg)', '{:.3f}', True),
        ('median (g/kg)', '{:.3f}', True),
        ('q25 (g/kg)', '{:.3f}', True),
        ('q75 (g/kg)', '{:.3f}', True),
        ('IQR (g/kg)', '{:.3f}', True),
        ('min (g/kg)', '{:.3f}', True),
        ('max (g/kg)', '{:.3f}', True),
        ('n > 50 g/kg', '{:d}', False),
        ('% > 50 g/kg', '{:.2f}%', True),
        ('n > 120 g/kg', '{:d}', False),
        ('% > 120 g/kg', '{:.2f}%', True),
    ]
    key_map = {
        'n': 'n', 'mean (g/kg)': 'mean', 'sd (g/kg)': 'sd',
        'median (g/kg)': 'median', 'q25 (g/kg)': 'q25', 'q75 (g/kg)': 'q75',
        'IQR (g/kg)': 'iqr', 'min (g/kg)': 'min', 'max (g/kg)': 'max',
        'n > 50 g/kg': 'n_gt_50', '% > 50 g/kg': 'pct_gt_50',
        'n > 120 g/kg': 'n_gt_120', '% > 120 g/kg': 'pct_gt_120',
    }
    for label, fmt, numeric_diff in rows:
        key = key_map[label]
        s = sp_desc[key]; r = rd_desc[key]
        sf, rf = fmt.format(s), fmt.format(r)
        if numeric_diff and isinstance(s, (int, float)):
            diff_val = r - s
            df_label = (f'{diff_val:+.3f}' if isinstance(diff_val, float)
                        else f'{diff_val:+d}')
            md.append(f'| {label} | {sf} | {rf} | {df_label} |')
        else:
            md.append(f'| {label} | {sf} | {rf} | — |')
    md.append('')

    md.append('## Geographic extent')
    md.append('')
    md.append('| split | lat range | lon range | unique locations |')
    md.append('|-------|-----------|-----------|------------------|')
    for split_name, g in geom.items():
        md.append(f'| {split_name} | [{g["lat_min"]:.4f}, {g["lat_max"]:.4f}] | '
                  f'[{g["lon_min"]:.4f}, {g["lon_max"]:.4f}] | {g["unique_locs"]} |')
    md.append('')

    md.append('## Per-year sample count in each validation set')
    md.append('')
    md.append('| year | spatial val n | random val n |')
    md.append('|------|---------------|--------------|')
    for r in per_year.itertuples(index=False):
        md.append(f'| {int(r.year)} | {r.n_spatial} | {r.n_random} |')
    md.append('')

    (OUT_DIR / 'split_comparison.md').write_text('\n'.join(md))
    (OUT_DIR / 'split_comparison.json').write_text(json.dumps(out, indent=2))
    print(f'Wrote {OUT_DIR / "split_comparison.md"}')
    print(f'Wrote {OUT_DIR / "split_comparison.json"}')


if __name__ == '__main__':
    main()
