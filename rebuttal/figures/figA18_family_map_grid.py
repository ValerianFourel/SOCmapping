#!/usr/bin/env python3
"""
figA18_family_map_grid.py — 7-panel grid of wall-to-wall Bavaria 2023 SOC maps,
one panel per production run-name, all on a SHARED 0-80 g/kg YlOrBr scale with a
single shared colorbar. Each panel is titled with the family label/colour and
annotated with its own spatial mean.

Production run-names (panel order):
  sgt_d32_h2_L1_extband, vanilla_transformer_d64_h4_L1_extband,
  simpletransformer_d64_h4_L1_extband, cnnlstm_d64_h4_L1_extband,
  3dcnn_d64_h4_L1_extband, rf_deep_20band, xgb_deep_20band

Reads <run>_2023_predictions.parquet from --maps-dir (default fd.MAPS_DIR_DEFAULT;
cols longitude/latitude/predicted_soc). The tree models (rf/xgb) are only drawn
if their parquet exists AND is non-degenerate (non-constant predictions); a
constant or missing panel is Manifest.block-ed (with path) and left blank.

The canonical maps live on JUPITER only; locally --maps-dir is empty, so the
grid is Manifest.block-ed. The SYNTHETIC smoke test fabricates 3 tiny parquets
in a temp dir and confirms a grid PDF is produced.

CLI: --maps-dir --out-dir --sweep-dir --axis --bands --max-oc
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
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize   # noqa: E402

fs.setup()

OUT_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'

# (run-name, canonical family key) in panel order. RF/XGB are the tree models.
PANELS = [
    ('sgt_d32_h2_L1_extband',                 'sgt'),
    ('vanilla_transformer_d64_h4_L1_extband', 'vanilla'),
    ('simpletransformer_d64_h4_L1_extband',   'simpletransformer'),
    ('cnnlstm_d64_h4_L1_extband',             'cnnlstm'),
    ('3dcnn_d64_h4_L1_extband',               '3dcnn'),
    ('rf_deep_20band',                        'rf'),
    ('xgb_deep_20band',                       'xgb'),
]
TREE_FAMS = {'rf', 'xgb'}

_VAL_COLS = ['predicted_soc', 'OC_predicted', 'prediction', 'pred']
_LON_COLS = ['longitude', 'GPS_LONG', 'lon']
_LAT_COLS = ['latitude', 'GPS_LAT', 'lat']


def _pick(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def _read_map(parquet_path):
    """Return (lon, lat, val) finite arrays or None if unreadable/empty."""
    df = pd.read_parquet(parquet_path)
    lc, ac, vc = (_pick(df, _LON_COLS), _pick(df, _LAT_COLS),
                  _pick(df, _VAL_COLS))
    if not (lc and ac and vc):
        return None
    lon = df[lc].to_numpy(dtype=float)
    lat = df[ac].to_numpy(dtype=float)
    val = df[vc].to_numpy(dtype=float)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    if ok.sum() == 0:
        return None
    return lon[ok], lat[ok], val[ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--maps-dir', default=str(fd.MAPS_DIR_DEFAULT))
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT))
    ap.add_argument('--axis', default='lon')
    ap.add_argument('--bands', default='43')
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--synthetic', action='store_true',
                    help='fabricate 3 tiny parquets (smoke test)')
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir)
    if args.synthetic:
        maps_dir = _synthetic_dir()
        print(f'[synthetic] maps-dir = {maps_dir}')

    manifest = fd.Manifest()

    parquets = list(maps_dir.glob('*_2023_predictions.parquet')) \
        if maps_dir.exists() else []
    if not parquets:
        manifest.block('figA18_family_map_grid',
                       'no *_2023_predictions.parquet in maps dir '
                       '(canonical maps are JUPITER-only)', str(maps_dir))
        manifest.write(Path(args.out_dir) / 'figA18_family_map_grid_manifest.md')
        print('BLOCKED no map parquets found; canonical grid needs the JUPITER '
              f'maps dir. Looked in {maps_dir}')
        return 0

    # 7 panels -> 3 rows x 3 cols (last 2 cells unused but kept for a clean grid)
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 13.0))
    axes = axes.ravel()
    drawn = 0
    panel_numbers = []
    for ax, (run, fam) in zip(axes, PANELS):
        label, color, _m = fs.fam_style(fam)
        parquet = maps_dir / f'{run}_2023_predictions.parquet'
        if not parquet.exists():
            manifest.block(f'figA18/{run}', 'map parquet missing', str(parquet))
            _blank(ax, label, color, 'no parquet')
            continue
        data = _read_map(parquet)
        if data is None:
            manifest.block(f'figA18/{run}', 'parquet unreadable / no finite preds',
                           str(parquet))
            _blank(ax, label, color, 'no valid data')
            continue
        lon, lat, val = data
        # Trees only if non-degenerate (non-constant) predictions.
        if fam in TREE_FAMS and float(np.nanstd(val)) < 1e-6:
            manifest.block(f'figA18/{run}',
                           'tree predictions are constant (degenerate) -> skipped',
                           str(parquet))
            _blank(ax, label, color, 'constant preds')
            continue

        fs.bavaria_scatter(ax, lon, lat, val, s=1.5,
                           vmin=fs.SOC_VMIN, vmax=fs.SOC_VMAX, cmap=fs.SOC_CMAP,
                           scalebar=(drawn == 0))  # one scale bar is enough
        mean_v = float(np.mean(val))
        ax.set_title(label, color=color, fontsize=9)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.annotate(f'mean = {mean_v:.1f} g/kg\nn = {val.size:,}',
                    xy=(0.03, 0.03), xycoords='axes fraction', fontsize=7.5,
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7',
                              alpha=0.85))
        panel_numbers.append((run, val.size, mean_v))
        drawn += 1

    # Hide the 2 unused grid cells.
    for ax in axes[len(PANELS):]:
        ax.axis('off')

    if drawn == 0:
        plt.close(fig)
        manifest.block('figA18_family_map_grid',
                       'every panel blocked (no valid parquet)', str(maps_dir))
        manifest.write(Path(args.out_dir) / 'figA18_family_map_grid_manifest.md')
        print('BLOCKED no family-map panel could be drawn; maps dir is empty/'
              f'JUPITER-only. Looked in {maps_dir}')
        return 0

    sm = ScalarMappable(norm=Normalize(fs.SOC_VMIN, fs.SOC_VMAX), cmap=fs.SOC_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02,
                      extend='max', shrink=0.6)
    cb.set_label('Predicted SOC (g kg$^{-1}$)  — shared 0–80 scale')
    fig.suptitle('Bavaria 2023 SOC predictions by model family  '
                 f'(axis={args.axis}, {args.bands}-band, oc{int(args.max_oc)})',
                 fontsize=11, y=0.995)

    pdf, png = fs.save(fig, args.out_dir, 'figA18_family_map_grid',
                       script='figA18_family_map_grid.py')
    nums = '; '.join(f'{r}:mean={m:.1f}(n={n})' for r, n, m in panel_numbers)
    manifest.done('figA18_family_map_grid', [pdf.name, png.name], str(maps_dir),
                  f'{drawn}/{len(PANELS)} panels drawn — {nums}')
    manifest.write(Path(args.out_dir) / 'figA18_family_map_grid_manifest.md')
    print(f'OK {pdf}')
    if drawn < len(PANELS):
        print(f'NOTE only {drawn}/{len(PANELS)} panels had data; real run needs '
              'the JUPITER maps dir for the full family grid.')
    return 0


def _blank(ax, label, color, msg):
    ax.set_title(label, color=color, fontsize=9)
    ax.text(0.5, 0.5, f'BLOCKED\n{msg}', transform=ax.transAxes, ha='center',
            va='center', fontsize=9, color='0.55')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def _synthetic_dir():
    """Fabricate 3 tiny synthetic parquets (sgt, vanilla, cnnlstm) for the
    smoke test so the multi-panel rendering path is exercised."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix='figA18_synth_'))
    rng = np.random.default_rng(7)
    n = 3000
    lon = rng.uniform(9.0, 13.8, n)
    lat = rng.uniform(47.3, 50.5, n)
    base = 20 + 25 * (lat - 47.3) / 3.2
    for run, jitter in [('sgt_d32_h2_L1_extband', 5.0),
                        ('vanilla_transformer_d64_h4_L1_extband', 7.0),
                        ('cnnlstm_d64_h4_L1_extband', 9.0)]:
        soc = np.clip(base + rng.normal(0, jitter, n), 0, 80)
        pd.DataFrame({'longitude': lon, 'latitude': lat,
                      'GPS_LONG': lon, 'GPS_LAT': lat,
                      'predicted_soc': soc, 'year': 2023}).to_parquet(
            d / f'{run}_2023_predictions.parquet')
    return d


if __name__ == '__main__':
    sys.exit(main())
