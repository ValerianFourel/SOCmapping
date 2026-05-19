#!/usr/bin/env python3
"""
covariate_temporal_stats.py — Task T2.5 (R2.M5).

R2.M5 asked: "did the climate patterns really change in your study region?"
We compute Bavaria-wide annual statistics (mean, std, percentiles) for each
dynamic covariate (LAI, LST, MODIS_NPP, SoilEvaporation, TotalEvapotranspiration)
across 2007-2023, sampled at the 16,514 LUCAS/LfL/LfU soil-sample locations
(i.e. at the same points the model evaluates).

We don't sample over a Bavaria-wide grid — the model only ever sees the
sample-location distribution, so that's the distribution whose temporal
drift matters for interpreting the +0.751 g/kg/yr coefficient. If the
covariates are stable over 2007-2023, the temporal coefficient cannot be
"explained" by a real climate-driven SOC change, and the sampling-bias
interpretation is reinforced.

Outputs (under rebuttal/):
    covariate_temporal_stats.json     full per-band per-year statistics
    covariate_temporal_stats.md       markdown summary table
    covariate_temporal_trends.png     5-panel figure, one per band

Run on the cluster from SOCmapping/:
    python rebuttal/covariate_temporal_stats.py

Compute cost: ~5 min (single CPU pass over 5 bands × 17 years × 16,514
samples = ~1.4M pixel reads, all from in-memory raster cache).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Path resolution — match the rebuttal-folder convention
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parent
sys.path.insert(0, str(SOC_ROOT))
from _paths import SOC_DATA_DIR, SOC_REBUTTAL_DIR    # noqa: E402

SGT_DIR = SOC_ROOT / 'SpatiotemporalGatedTransformer'
sys.path.insert(0, str(SGT_DIR))
sys.path.insert(0, str(SGT_DIR / 'dataloader'))

from dataloaderMultiYears import RasterTensorDataset      # noqa: E402

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_READY = SOC_REBUTTAL_DIR / 'model_ready_dataset.parquet'
RASTER_DIR = SOC_DATA_DIR / 'RasterTensorData' / 'YearlyValue'
COORDS_DIR_TEMPLATE = SOC_DATA_DIR / 'OC_LUCAS_LFU_LfL_Coordinates_v2' / 'YearlyValue' / '{band}' / '{year}'
YEARS = list(range(2007, 2024))     # 2007..2023 inclusive

# The five dynamic covariates the paper's Model A uses.
# (Elevation is static, no temporal variation, so excluded.)
DYNAMIC_BANDS = ['LAI', 'LST', 'MODIS_NPP',
                 'SoilEvaporation', 'TotalEvapotranspiration']


def load_coordinates_for(band: str, year: int) -> dict:
    """(lat, lon) → (id_num, x, y) lookup for one band/year directory."""
    coord_path = SOC_DATA_DIR / 'OC_LUCAS_LFU_LfL_Coordinates_v2' / 'YearlyValue' / band / str(year) / 'coordinates.npy'
    if not coord_path.exists():
        return {}
    coords = np.load(coord_path)
    return {
        (round(float(r[0]), 9), round(float(r[1]), 9)): (int(r[2]), int(r[3]), int(r[4]))
        for r in coords
    }


def sample_band_for_year(band: str, year: int, lats: np.ndarray, lons: np.ndarray):
    """Return the centre-pixel value of `band` for `year` at every (lat, lon).

    Returns array of shape (n_samples,) with NaN for any missing tile/coord.
    """
    raster_path = RASTER_DIR / band / str(year)
    if not raster_path.exists():
        print(f'  [warn] {raster_path} missing — skipping', file=sys.stderr)
        return np.full(len(lats), np.nan)
    coord_lookup = load_coordinates_for(band, year)
    if not coord_lookup:
        return np.full(len(lats), np.nan)
    rds = RasterTensorDataset(str(raster_path))
    out = np.full(len(lats), np.nan, dtype=np.float64)
    miss = 0
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        key = (round(float(lat), 9), round(float(lon), 9))
        triple = coord_lookup.get(key)
        if triple is None:
            miss += 1
            continue
        id_num, x, y = triple
        try:
            tile = rds.data_cache.get(id_num)
            if tile is None:
                miss += 1
                continue
            # Centre-pixel value: the raster window is window_size×window_size
            # but for descriptive stats we only need the value at (x, y).
            if 0 <= x < tile.shape[0] and 0 <= y < tile.shape[1]:
                out[i] = float(tile[x, y])
            else:
                miss += 1
        except Exception:
            miss += 1
    if miss:
        print(f'  [info] {band} {year}: {miss}/{len(lats)} samples missing tile data', flush=True)
    return out


def describe(values: np.ndarray) -> dict:
    """Mean / std / quantile summary, NaN-safe."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {'n': 0, 'mean': np.nan, 'std': np.nan,
                'p10': np.nan, 'p50': np.nan, 'p90': np.nan,
                'min': np.nan, 'max': np.nan}
    return {
        'n': int(v.size),
        'mean': float(v.mean()),
        'std': float(v.std(ddof=1)) if v.size > 1 else 0.0,
        'p10': float(np.percentile(v, 10)),
        'p50': float(np.percentile(v, 50)),
        'p90': float(np.percentile(v, 90)),
        'min': float(v.min()),
        'max': float(v.max()),
    }


def main():
    if not MODEL_READY.exists():
        print(f'ERROR: {MODEL_READY} not found. '
              f'Run rebuttal/gpu_experiments/spatial_kfold/run_kfold.py '
              f'first to build it.', file=sys.stderr)
        sys.exit(2)

    df = pd.read_parquet(MODEL_READY)
    lats = df['GPS_LAT'].to_numpy(dtype=np.float64)
    lons = df['GPS_LONG'].to_numpy(dtype=np.float64)
    print(f'Loaded {len(df):,} LUCAS sample locations from {MODEL_READY}', flush=True)
    print(f'Sampling {len(DYNAMIC_BANDS)} bands × {len(YEARS)} years '
          f'= {len(DYNAMIC_BANDS) * len(YEARS)} (band, year) combinations.\n',
          flush=True)

    results = {}     # results[band][year] = stats dict
    for band in DYNAMIC_BANDS:
        results[band] = {}
        print(f'=== {band} ===', flush=True)
        for year in YEARS:
            vals = sample_band_for_year(band, year, lats, lons)
            stats = describe(vals)
            results[band][year] = stats
            print(f'  {year}: n={stats["n"]:>5}  '
                  f'mean={stats["mean"]:>8.3f}  '
                  f'std={stats["std"]:>7.3f}  '
                  f'p10={stats["p10"]:>8.3f}  '
                  f'p90={stats["p90"]:>8.3f}', flush=True)

    # ---- Save JSON ----
    out_json = HERE / 'covariate_temporal_stats.json'
    out_json.write_text(json.dumps(results, indent=2))
    print(f'\nSaved {out_json}', flush=True)

    # ---- Trend tests (linear regression of annual mean against year) ----
    trend_rows = []
    for band in DYNAMIC_BANDS:
        years_arr = np.array(YEARS, dtype=float)
        means = np.array([results[band][y]['mean'] for y in YEARS], dtype=float)
        mask = np.isfinite(means)
        if mask.sum() < 3:
            slope, intercept, r2 = np.nan, np.nan, np.nan
        else:
            x = years_arr[mask]; y = means[mask]
            slope, intercept = np.polyfit(x, y, 1)
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        trend_rows.append({
            'band': band,
            'slope_per_year': float(slope),
            'intercept': float(intercept),
            'r2_linear': float(r2) if r2 == r2 else np.nan,
            'mean_2007': float(results[band][2007]['mean']),
            'mean_2023': float(results[band][2023]['mean']),
            'pct_change_2007_to_2023':
                float(100 * (results[band][2023]['mean'] - results[band][2007]['mean'])
                      / abs(results[band][2007]['mean']))
                if results[band][2007]['mean'] != 0 else np.nan,
        })

    # ---- Markdown summary ----
    md = ['# Covariate temporal statistics (Bavaria, 2007–2023)', '']
    md.append('Annual means and trend tests of the five dynamic covariates that '
              'feed Model A, sampled at the 16,514 LUCAS/LfL/LfU soil-sample '
              'locations. This is the distribution Model A actually evaluates; '
              'if these covariates do not drift meaningfully over 2007–2023, '
              'the +0.751 g/kg/yr temporal SOC coefficient cannot be attributed '
              'to climate change in the predictors.')
    md.append('')
    md.append('## Linear trend tests (annual mean ~ year)')
    md.append('')
    md.append('| Band | slope per year | R² (linear) | mean 2007 | mean 2023 | % change |')
    md.append('|------|----------------|-------------|-----------|-----------|----------|')
    for row in trend_rows:
        md.append(f'| {row["band"]} | {row["slope_per_year"]:+.4f} | '
                  f'{row["r2_linear"]:.3f} | '
                  f'{row["mean_2007"]:.3f} | '
                  f'{row["mean_2023"]:.3f} | '
                  f'{row["pct_change_2007_to_2023"]:+.2f}% |')
    md.append('')
    md.append('## Per-band annual statistics')
    md.append('')
    for band in DYNAMIC_BANDS:
        md.append(f'### {band}')
        md.append('')
        md.append('| Year | n | mean | std | p10 | p50 | p90 |')
        md.append('|------|---|------|-----|-----|-----|-----|')
        for year in YEARS:
            s = results[band][year]
            md.append(f'| {year} | {s["n"]} | {s["mean"]:.3f} | {s["std"]:.3f} | '
                      f'{s["p10"]:.3f} | {s["p50"]:.3f} | {s["p90"]:.3f} |')
        md.append('')
    out_md = HERE / 'covariate_temporal_stats.md'
    out_md.write_text('\n'.join(md))
    print(f'Saved {out_md}', flush=True)

    # ---- Figure ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(DYNAMIC_BANDS),
                                  figsize=(4 * len(DYNAMIC_BANDS), 3.2),
                                  sharex=True)
        for ax, band, row in zip(axes, DYNAMIC_BANDS, trend_rows):
            years_arr = np.array(YEARS, dtype=float)
            means = np.array([results[band][y]['mean'] for y in YEARS])
            stds = np.array([results[band][y]['std'] for y in YEARS])
            mask = np.isfinite(means)
            ax.errorbar(years_arr[mask], means[mask], yerr=stds[mask],
                        fmt='o-', capsize=3, linewidth=1.5, markersize=4)
            # Trend line
            if np.isfinite(row['slope_per_year']):
                line_x = years_arr[mask]
                line_y = row['slope_per_year'] * line_x + row['intercept']
                ax.plot(line_x, line_y, '--', color='red', alpha=0.6, linewidth=1)
            ax.set_title(f'{band}\nslope = {row["slope_per_year"]:+.4f}/yr  '
                          f'R²_linear = {row["r2_linear"]:.2f}', fontsize=9)
            ax.set_xlabel('Year')
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('Annual mean (across LUCAS samples)')
        fig.suptitle('Dynamic covariate temporal stability, Bavaria 2007–2023',
                      fontsize=11, fontweight='bold')
        fig.tight_layout()
        out_png = HERE / 'covariate_temporal_trends.png'
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        print(f'Saved {out_png}', flush=True)
    except Exception as e:
        print(f'[warn] figure generation failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
