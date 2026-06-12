#!/usr/bin/env python3
"""
figFD_flagship_map.py — wall-to-wall Bavaria 2023 SOC prediction maps for the
flagship SGT run and the vanilla-transformer comparison run (the two runs that
drive the figure-selection choice in the Geoderma revision).

Reads <run>_2023_predictions.parquet (cols: longitude, latitude, predicted_soc)
from --maps-dir (default fd.MAPS_DIR_DEFAULT), plots each via fs.bavaria_scatter
on the shared SOC colour scale (YlOrBr, 0-80 g/kg) with scale bar + N arrow, and
writes ONE figure per run:
  - figFD_flagship_sgt   (sgt_d32_h2_L1_extband)
  - figFD_vanilla        (vanilla_transformer_d64_h4_L1_extband)

The canonical map parquets live on JUPITER only; locally --maps-dir is empty, so
each missing run is Manifest.block-ed (with its exact path) and skipped. The
SMOKE TEST fabricates a synthetic parquet so the rendering path is exercised.

CLI: --maps-dir --out-dir --sweep-dir --axis --bands --max-oc
(the sweep flags only steer the provenance/title; metrics are read from disk).
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs  # noqa: E402
import figdata as fd   # noqa: E402

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fs.setup()

OUT_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'

# run-name -> (canonical family key, output figure basename)
RUNS = {
    'sgt_d32_h2_L1_extband':                 ('sgt',     'figFD_flagship_sgt'),
    'vanilla_transformer_d64_h4_L1_extband': ('vanilla', 'figFD_vanilla'),
}

# Columns we accept for the prediction value / coordinates (robust to schema).
_VAL_COLS = ['predicted_soc', 'OC_predicted', 'prediction', 'pred']
_LON_COLS = ['longitude', 'GPS_LONG', 'lon', 'GPS_LONG_x']
_LAT_COLS = ['latitude', 'GPS_LAT', 'lat', 'GPS_LAT_x']


def _pick(df, cands, what):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f'no {what} column in parquet (tried {cands}; have '
                   f'{list(df.columns)})')


def _read_map(parquet_path):
    df = pd.read_parquet(parquet_path)
    lon = df[_pick(df, _LON_COLS, 'longitude')].to_numpy(dtype=float)
    lat = df[_pick(df, _LAT_COLS, 'latitude')].to_numpy(dtype=float)
    val = df[_pick(df, _VAL_COLS, 'predicted_soc')].to_numpy(dtype=float)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    return lon[ok], lat[ok], val[ok]


def render_run(run, fam, basename, maps_dir, out_dir, args, manifest):
    parquet = Path(maps_dir) / f'{run}_2023_predictions.parquet'
    if not parquet.exists():
        manifest.block(f'figFD/{run}',
                       'map parquet missing (JUPITER-only; pass --maps-dir there)',
                       str(parquet))
        return None
    lon, lat, val = _read_map(parquet)
    if val.size == 0:
        manifest.block(f'figFD/{run}', 'parquet has no finite predictions',
                       str(parquet))
        return None

    label, color, _marker = fs.fam_style(fam)
    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    sc = fs.bavaria_scatter(ax, lon, lat, val, s=2,
                            vmin=fs.SOC_VMIN, vmax=fs.SOC_VMAX, cmap=fs.SOC_CMAP)
    ax.set_title(f'{label}\nBavaria 2023 SOC prediction  (run: {run})',
                 color=color, fontsize=9.5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, extend='max')
    cb.set_label('Predicted SOC (g kg$^{-1}$)')
    mean_v = float(np.mean(val))
    ax.annotate(f'n = {val.size:,}\nmean = {mean_v:.1f} g/kg',
                xy=(0.03, 0.03), xycoords='axes fraction', fontsize=8,
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7',
                          alpha=0.85))
    pdf, png = fs.save(fig, out_dir, basename, script='figFD_flagship_map.py')
    manifest.done(f'figFD/{run}', [pdf.name, png.name], str(parquet),
                  f'n={val.size}, mean={mean_v:.2f} g/kg, '
                  f'axis={args.axis}/{args.bands}b/oc{args.max_oc}')
    print(f'OK {pdf}')
    return pdf


def _synthetic_dir():
    """Fabricate a tiny synthetic SGT parquet in a temp dir for the smoke test
    so the rendering branch runs even though the real maps dir is empty."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix='figFD_synth_'))
    rng = np.random.default_rng(42)
    n = 4000
    lon = rng.uniform(9.0, 13.8, n)
    lat = rng.uniform(47.3, 50.5, n)
    soc = np.clip(20 + 25 * (lat - 47.3) / 3.2 + rng.normal(0, 6, n), 0, 80)
    df = pd.DataFrame({'longitude': lon, 'latitude': lat,
                       'GPS_LONG': lon, 'GPS_LAT': lat,
                       'predicted_soc': soc, 'year': 2023})
    df.to_parquet(d / 'sgt_d32_h2_L1_extband_2023_predictions.parquet')
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--maps-dir', default=str(fd.MAPS_DIR_DEFAULT))
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT))
    ap.add_argument('--axis', default='lon')
    ap.add_argument('--bands', default='43')
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--synthetic', action='store_true',
                    help='fabricate a tiny SGT parquet (smoke test)')
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir)
    if args.synthetic:
        maps_dir = _synthetic_dir()
        print(f'[synthetic] maps-dir = {maps_dir}')

    manifest = fd.Manifest()

    parquets = list(maps_dir.glob('*_2023_predictions.parquet')) \
        if maps_dir.exists() else []
    if not parquets:
        manifest.block('figFD_flagship_map',
                       'no *_2023_predictions.parquet in maps dir '
                       '(canonical maps are JUPITER-only)', str(maps_dir))

    any_ok = False
    for run, (fam, basename) in RUNS.items():
        pdf = render_run(run, fam, basename, maps_dir, args.out_dir, args,
                         manifest)
        any_ok = any_ok or (pdf is not None)

    manifest.write(Path(args.out_dir) / 'figFD_flagship_map_manifest.md')

    if not any_ok:
        print('BLOCKED no map parquet rendered (maps dir is empty/JUPITER-only); '
              f'looked in {maps_dir}')
        # Exit 0: a blocked-but-correct run is the EXPECTED local state.
    return 0


if __name__ == '__main__':
    sys.exit(main())
