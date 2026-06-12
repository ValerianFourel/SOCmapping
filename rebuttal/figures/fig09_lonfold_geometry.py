#!/usr/bin/env python3
"""
fig09_lonfold_geometry.py — map of the spatial-CV fold blocks + buffer.

Resolves the best config at the requested protocol (default: lon-split,
43-band 'full_extended', oc150, 10-fold — the CANONICAL result that lives only
on JUPITER), reads the ACTUAL per-fold boundaries via fd.fold_boundaries(), and
draws the N fold bands as strips across Bavaria's extent:
  - lon-split  -> vertical strips (west -> east)
  - lat-split  -> horizontal strips (south -> north)
Strips are coloured by fs.FOLD_COLORS; the buffer_km and geometry label are
annotated. If a pooled fold-prediction parquet exists, the sample points are
overlaid coloured by fold_id (same palette), confirming the block assignment.

All geometry/metrics are read from disk — nothing is hardcoded. If the config
or its kfold summary is missing, the script prints BLOCKED <path> and exits.

CANONICAL run is JUPITER-only: pass --sweep-dir there. The LOCAL bundle
(SOCrebuttal_HF/sweep) is the STALE lat / 20-band experiment (schema-identical,
wrong numbers) — that is what the smoke test exercises:

    python fig09_lonfold_geometry.py \
        --sweep-dir /home/valerian/SGTPublication/SOCrebuttal_HF/sweep \
        --axis lat --bands 20 --max-oc 150
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs  # noqa: E402
import figdata as fd   # noqa: E402

import numpy as np  # noqa: E402

fs.setup()

OUT_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'

# Bavaria bounding extent (lon/lat) used to draw the cross-axis span of strips.
BAVARIA_LON = (8.9, 13.9)
BAVARIA_LAT = (47.2, 50.6)


def _to_xy(lon, lat, utm):
    """Project to UTM 32N (m) if available, else return lon/lat as-is."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if utm:
        xy = fs._utm32n(lon, lat)
        if xy is not None:
            return xy[0], xy[1], 'm'
    return lon, lat, 'deg'


def resolve_config(rows, axis, bands, max_oc):
    """Best (highest-score) config at the protocol; prefer SGT, else any family."""
    cand = fd.select(rows, axis=axis, bands=str(bands), max_oc=max_oc, family='sgt')
    note = 'sgt'
    if not cand:
        cand = fd.select(rows, axis=axis, bands=str(bands), max_oc=max_oc)
        note = 'any-family (no sgt at protocol)'
    if not cand:
        return None, note
    return max(cand, key=lambda r: r['score']), note


def build_figure(fb, df, *, axis, geom, buffer_km, n_folds, run_tag,
                 sweep_dir, utm=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    folds = sorted(fb['folds'], key=lambda f: (f['lo'] if f['lo'] is not None else 0))

    # cross-axis extent of every strip (the dimension NOT being split)
    if axis == 'lon':
        cross = BAVARIA_LAT
        split_label, cross_label = 'Longitude (split axis)', 'Latitude'
    else:
        cross = BAVARIA_LON
        split_label, cross_label = 'Latitude (split axis)', 'Longitude'

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # ----- draw fold strips as rectangles in projected coords --------------
    # Build rectangle corners in lon/lat then project the corners.
    n = len(folds)
    handles = []
    for i, f in enumerate(folds):
        lo, hi = f['lo'], f['hi']
        if lo is None or hi is None:
            continue
        col = fs.FOLD_COLORS[i % len(fs.FOLD_COLORS)]
        if axis == 'lon':
            corners_lon = [lo, hi, hi, lo]
            corners_lat = [cross[0], cross[0], cross[1], cross[1]]
        else:
            corners_lat = [lo, hi, hi, lo]
            corners_lon = [cross[0], cross[0], cross[1], cross[1]]
        cx, cy, unit = _to_xy(corners_lon, corners_lat, utm)
        poly = plt.Polygon(np.column_stack([cx, cy]), closed=True,
                           facecolor=col, edgecolor='white', linewidth=0.8,
                           alpha=0.30, zorder=1)
        ax.add_patch(poly)
        r2 = f.get('r2')
        r2s = f'{r2:+.2f}' if r2 is not None else 'n/a'
        handles.append(Line2D([0], [0], marker='s', linestyle='',
                              markerfacecolor=col, markeredgecolor='none',
                              markersize=7,
                              label=f"fold {f['fold_id']}: R²={r2s}  (n={f.get('n_test')})"))

    # ----- overlay sample points coloured by fold (if predictions exist) ---
    pts_note = 'no fold-prediction parquet (strips only)'
    if df is not None and {'GPS_LONG', 'GPS_LAT', 'fold_id'}.issubset(df.columns):
        px, py, _ = _to_xy(df['GPS_LONG'].values, df['GPS_LAT'].values, utm)
        fids = df['fold_id'].astype(int).values
        cols = np.array([fs.FOLD_COLORS[k % len(fs.FOLD_COLORS)] for k in fids])
        ax.scatter(px, py, s=2.0, c=cols, alpha=0.55, linewidths=0,
                   rasterized=True, zorder=2)
        pts_note = f'{len(df):,} hold-out samples coloured by fold'

    ax.set_aspect('equal', adjustable='box')
    if utm and fs._utm32n([10.0], [49.0]) is not None:
        ax.set_xlabel('Easting (UTM 32N, km)')
        ax.set_ylabel('Northing (km)')
        ax.xaxis.set_major_formatter(lambda v, _p: f'{v/1000:.0f}')
        ax.yaxis.set_major_formatter(lambda v, _p: f'{v/1000:.0f}')
    else:
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    bk = f'{buffer_km:g} km' if buffer_km is not None else 'n/a'
    ax.set_title(f'Spatial-CV fold geometry ({axis}-split): {n} blocks, '
                 f'buffer {bk}', fontsize=10.5)

    # caption block: geometry + provenance
    cap = (f'geometry: {geom or "n/a"}   |   split: {split_label.split(" (")[0]}'
           f'   |   buffer = {bk}\n'
           f'run: {run_tag}   |   {pts_note}')
    fig.text(0.5, 0.005, cap, ha='center', va='bottom', fontsize=7.0,
             color='#444444')

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=6.6, title=f'{n} spatial folds', title_fontsize=7.4,
              frameon=False, handletextpad=0.4, labelspacing=0.35)
    fig.subplots_adjust(right=0.74, bottom=0.10)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                    help='sweep root (JUPITER for canonical lon/43 results)')
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat'],
                    help="split axis of the run to plot (default lon = canonical)")
    ap.add_argument('--bands', default='43',
                    help="band count of the run (default 43 = full_extended)")
    ap.add_argument('--max-oc', type=float, default=150.0,
                    help='max_oc cap of the run (default 150 = oc150)')
    ap.add_argument('--n-folds', type=int, default=10)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--no-utm', action='store_true',
                    help='plot in lon/lat instead of UTM 32N')
    a = ap.parse_args()

    man = fd.Manifest()
    sweep_dir = Path(a.sweep_dir)
    if not sweep_dir.exists():
        man.block('fig09 fold geometry',
                  f'sweep dir missing', str(sweep_dir))
        man.write(Path(a.out_dir) / 'fig09_manifest.md')
        print(f'BLOCKED sweep-dir not found: {sweep_dir}')
        return

    rows = fd.load_ranking(sweep_dir)
    row, sel_note = resolve_config(rows, a.axis, a.bands, a.max_oc)
    if row is None:
        reason = (f'no config at axis={a.axis} bands={a.bands} '
                  f'max_oc={a.max_oc:g} in ranking')
        man.block('fig09 fold geometry', reason, str(sweep_dir))
        man.write(Path(a.out_dir) / 'fig09_manifest.md')
        print(f'BLOCKED {reason}')
        return

    cd = Path(row['config_dir'])
    fb = fd.fold_boundaries(cd)
    if fb is None:
        path = str(cd / 'kfold_results_summary.json')
        man.block('fig09 fold geometry', 'kfold_results_summary.json missing', path)
        man.write(Path(a.out_dir) / 'fig09_manifest.md')
        print(f'BLOCKED kfold_results_summary.json not found: {path}')
        return

    # axis comes from the summary itself — trust disk over the CLI hint
    axis = fb['axis'] or a.axis
    df = fd.load_fold_predictions(cd)   # may be None -> strips only

    fig = build_figure(
        fb, df, axis=axis, geom=fb['geometry'], buffer_km=fb['buffer_km'],
        n_folds=fb['n_folds'], run_tag=f"{row['sweep_group']}/{row['tag']}",
        sweep_dir=str(sweep_dir), utm=not a.no_utm)
    pdf, png = fs.save(fig, a.out_dir, 'fig09_lonfold_geometry',
                       script='fig09_lonfold_geometry.py')

    bk = fb['buffer_km']
    nums = (f"axis={axis}; {fb['n_folds']} folds; "
            f"buffer_km={bk:g}; geom={fb['geometry']}; "
            f"sel={sel_note}; n_pts={0 if df is None else len(df)}")
    man.done('fig09 fold geometry', [str(pdf), str(png)],
             f"fold_boundaries({cd})", nums)
    man.write(Path(a.out_dir) / 'fig09_manifest.md')
    print(f'OK {pdf}')


if __name__ == '__main__':
    main()
