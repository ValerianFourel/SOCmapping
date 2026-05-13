#!/usr/bin/env python3
"""
temporal_regression.py — Task 3 of the Geoderma rebuttal.

Reproduces the paper's Eq. 1 style temporal regression of SOC on year
(and altitude) using the LUCAS+LfU+LfL Bavaria master sample table, then
runs three sensitivity variants requested by the reviewers:

  Model 1  OLS  SOC ~ year + altitude                            (all years)
  Model 2  OLS  SOC ~ year + altitude              (exclude year >= 2022)   T2.2
  Model 3  WLS  SOC ~ year + altitude  weights = 1/n_year                   T2.3
  Model 4  OLS  SOC ~ year + altitude   restricted to years where n >= 100  T2.3b

Altitude is joined from
  Data/OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation/coordinates.npy
which gives, for each sample location, the tile id and (x, y) pixel
indices used to read the elevation value from the matching
RasterTensorData/StaticValue/Elevation/ID*.npy tile.

Outputs:
    rebuttal/temporal_regression.md
    rebuttal/temporal_regression.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SAMPLE_PATH = Path('/home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx')
ELEV_COORDS_NPY = Path('/home/valerian/SGTPublication/Data/'
                       'OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation/coordinates.npy')
ELEV_TILE_DIR = Path('/home/valerian/SGTPublication/Data/'
                     'RasterTensorData/StaticValue/Elevation')
OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal')

MAX_OC = 150     # paper's filter cap


# --------------------------------------------------------------------------
# Altitude lookup
# --------------------------------------------------------------------------
def load_elevations() -> pd.DataFrame:
    """Return a per-unique-location dataframe with columns
    [lat, lon, id_num, x, y, elevation]."""
    coords = np.load(ELEV_COORDS_NPY)
    elev_map = (pd.DataFrame(coords, columns=['lat', 'lon', 'id_num', 'x', 'y'])
                .drop_duplicates(['lat', 'lon'])
                .reset_index(drop=True))

    tile_files = {int(re.match(r'ID(\d+)', f).group(1)):
                  ELEV_TILE_DIR / f
                  for f in os.listdir(ELEV_TILE_DIR) if f.startswith('ID')}
    tiles: dict[int, np.ndarray] = {}
    elevations = np.empty(len(elev_map), dtype=float)
    for i, r in elev_map.iterrows():
        tid = int(r.id_num)
        if tid not in tiles:
            tiles[tid] = np.load(tile_files[tid])
        elevations[i] = float(tiles[tid][int(r.x), int(r.y)])
    elev_map['elevation'] = elevations
    return elev_map


def fit_summary(model_res, label: str) -> dict:
    coefs = {name: {'coef': float(model_res.params[name]),
                    'std_err': float(model_res.bse[name]),
                    'p_value': float(model_res.pvalues[name]),
                    't': float(model_res.tvalues[name]),
                    'ci_lo': float(model_res.conf_int().loc[name, 0]),
                    'ci_hi': float(model_res.conf_int().loc[name, 1])}
             for name in model_res.params.index}
    return {
        'label': label,
        'n_obs': int(model_res.nobs),
        'r_squared': float(model_res.rsquared),
        'adj_r_squared': float(getattr(model_res, 'rsquared_adj', float('nan'))),
        'f_p': float(getattr(model_res, 'f_pvalue', float('nan'))),
        'aic': float(model_res.aic),
        'coefs': coefs,
    }


def fmt_p(p):
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(SAMPLE_PATH)
    df = df.dropna(subset=['GPS_LAT', 'GPS_LONG', 'year', 'OC']).copy()
    print(f'LUCAS rows (after dropping NaN year/OC/coords): {len(df):,}')

    # Replicate the paper's MAX_OC filter
    df = df[df['OC'] <= MAX_OC].copy()
    print(f'After MAX_OC=={MAX_OC} cap: {len(df):,}')

    elev_map = load_elevations()
    df = df.merge(elev_map[['lat', 'lon', 'elevation']],
                  left_on=['GPS_LAT', 'GPS_LONG'],
                  right_on=['lat', 'lon'], how='left').drop(columns=['lat', 'lon'])
    n_missing_elev = int(df['elevation'].isna().sum())
    print(f'Rows missing elevation after merge: {n_missing_elev}')
    df = df.dropna(subset=['elevation']).copy()
    df['elevation'] = df['elevation'].astype(float)
    df['year_int'] = df['year'].astype(int)
    print(f'Final regression dataframe: {len(df):,} rows over '
          f'{df["year_int"].min()}–{df["year_int"].max()}')

    # Per-year counts
    year_counts = df.groupby('year_int').size().rename('n').reset_index()
    print('\nPer-year sample count:')
    for _, r in year_counts.iterrows():
        print(f'  {r.year_int}: {r.n}')

    # ---------------- Model 1: OLS full ----------------
    X1 = sm.add_constant(df[['year_int', 'elevation']])
    y = df['OC']
    m1 = sm.OLS(y, X1).fit()
    s1 = fit_summary(m1, 'OLS  SOC ~ year + altitude  (all years)')

    # ---------------- Model 2: OLS, drop year >= 2022 ----------------
    sub2 = df[df['year_int'] < 2022].copy()
    X2 = sm.add_constant(sub2[['year_int', 'elevation']])
    m2 = sm.OLS(sub2['OC'], X2).fit()
    s2 = fit_summary(m2, 'OLS  SOC ~ year + altitude  (year < 2022)')

    # ---------------- Model 3: WLS with weights = 1/n_year ----------------
    w = df['year_int'].map(year_counts.set_index('year_int')['n'])
    weights = 1.0 / w.astype(float)
    X3 = sm.add_constant(df[['year_int', 'elevation']])
    m3 = sm.WLS(df['OC'], X3, weights=weights).fit()
    s3 = fit_summary(m3, 'WLS  SOC ~ year + altitude  (weights = 1 / n_year)')

    # ---------------- Model 4: OLS restricted to years where n >= 100 ----------------
    good_years = set(year_counts.loc[year_counts['n'] >= 100, 'year_int'].tolist())
    sub4 = df[df['year_int'].isin(good_years)].copy()
    X4 = sm.add_constant(sub4[['year_int', 'elevation']])
    m4 = sm.OLS(sub4['OC'], X4).fit()
    s4 = fit_summary(m4, 'OLS  SOC ~ year + altitude  (years with n ≥ 100)')

    all_summaries = [s1, s2, s3, s4]
    out = {
        'sample_file': str(SAMPLE_PATH),
        'elev_coords_file': str(ELEV_COORDS_NPY),
        'n_obs_total_after_filter': int(len(df)),
        'year_min': int(df['year_int'].min()),
        'year_max': int(df['year_int'].max()),
        'per_year_counts': {int(r.year_int): int(r.n)
                            for _, r in year_counts.iterrows()},
        'good_years_n_ge_100': sorted(good_years),
        'models': all_summaries,
    }

    # ---------------- Markdown ----------------
    md = []
    md.append('# Temporal regression — SOC ~ year + altitude (T2.2 / T2.3)')
    md.append('')
    md.append(f'_Master sample file:_ `{SAMPLE_PATH}`')
    md.append('')
    md.append(f'_Altitude join:_ `{ELEV_COORDS_NPY}` × tiles in `{ELEV_TILE_DIR}`')
    md.append('')
    md.append(f'After dropping rows with NaN coordinates/year/OC and applying the paper’s '
              f'`MAX_OC ≤ {MAX_OC}` cap (and dropping any rows still missing elevation), '
              f'the regression dataframe has **{len(df):,}** observations spanning '
              f'**{df["year_int"].min()}–{df["year_int"].max()}**.')
    md.append('')
    md.append('## Per-year sample counts')
    md.append('')
    md.append('| year | n | included in n≥100 OLS? |')
    md.append('|------|---|------------------------|')
    for _, r in year_counts.iterrows():
        ok = '✔' if int(r.n) >= 100 else '—'
        md.append(f'| {int(r.year_int)} | {int(r.n)} | {ok} |')
    md.append('')

    md.append('## Coefficient table — four models')
    md.append('')
    md.append('| model | n | R² | adj. R² | β_year | SE_year | p_year | 95% CI on β_year | β_altitude | p_altitude |')
    md.append('|-------|---|----|---------|--------|---------|--------|------------------|------------|------------|')
    for s in all_summaries:
        b_y = s['coefs']['year_int']
        b_a = s['coefs']['elevation']
        md.append(
            f'| {s["label"]} | {s["n_obs"]} | {s["r_squared"]:.4f} | '
            f'{s["adj_r_squared"]:.4f} | {b_y["coef"]:+.4f} | {b_y["std_err"]:.4f} | '
            f'{fmt_p(b_y["p_value"])} | [{b_y["ci_lo"]:+.4f}, {b_y["ci_hi"]:+.4f}] | '
            f'{b_a["coef"]:+.5f} | {fmt_p(b_a["p_value"])} |'
        )
    md.append('')
    md.append('Interpretation:')
    md.append('')
    md.append('- β_year is the change in SOC (g/kg) per additional calendar year, '
              'holding altitude constant.')
    md.append('- β_altitude is the change in SOC (g/kg) per 1 m of elevation.')
    md.append('')

    # Drop in full coefficient tables for transparency
    for s in all_summaries:
        md.append(f'### `{s["label"]}`')
        md.append('')
        md.append(f'- n = {s["n_obs"]}, R² = {s["r_squared"]:.4f}, '
                  f'adj. R² = {s["adj_r_squared"]:.4f}, F p-value = {fmt_p(s["f_p"])}')
        md.append('')
        md.append('| term | coef | std err | t | p | 95% CI lo | 95% CI hi |')
        md.append('|------|------|---------|---|---|-----------|-----------|')
        for name, c in s['coefs'].items():
            md.append(f'| `{name}` | {c["coef"]:+.6f} | {c["std_err"]:.6f} | '
                      f'{c["t"]:+.3f} | {fmt_p(c["p_value"])} | {c["ci_lo"]:+.6f} | '
                      f'{c["ci_hi"]:+.6f} |')
        md.append('')

    (OUT_DIR / 'temporal_regression.md').write_text('\n'.join(md))
    (OUT_DIR / 'temporal_regression.json').write_text(json.dumps(out, indent=2))
    print(f'\nWrote {OUT_DIR / "temporal_regression.md"}')
    print(f'Wrote {OUT_DIR / "temporal_regression.json"}')


if __name__ == '__main__':
    main()
