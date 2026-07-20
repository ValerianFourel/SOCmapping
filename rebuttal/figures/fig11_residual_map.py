#!/usr/bin/env python3
"""fig11_residual_map.py — Bavaria map of the flagship SGT's spatial-CV residuals.

Reads the pooled hold-out fold predictions of the flagship config (best SGT row
at the canonical protocol: split-axis lon, 43-band 'full_extended', oc150) and
maps residual = OC_predicted - OC_actual over Bavaria with a symmetric diverging
colour scale (robust percentile vlim). Title carries the mean +/- SD residual.

CANONICAL results live ONLY on JUPITER -> pass --sweep-dir, and the defaults
(--axis lon --bands 43 --max-oc 150) select them. The LOCAL bundle
(SOCrebuttal_HF/sweep) is STALE (lat / 20-band); smoke-test with
  --sweep-dir /home/.../SOCrebuttal_HF/sweep --axis lat --bands 20 --max-oc 150
(it renders a PDF+PNG; the numbers are stale, as expected).

Every metric is read from disk via figdata; nothing is hardcoded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs
import figdata as fd
fs.setup()

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / 'out'
RESID_CMAP = 'RdBu_r'   # +resid (over-prediction) -> red, -resid -> blue


def resolve_flagship(rows, *, axis, bands, max_oc):
    """Best-score SGT row at the requested protocol (None if absent)."""
    cand = fd.select(rows, axis=axis, bands=str(bands), max_oc=max_oc,
                     family=fs.FLAGSHIP)
    if not cand:
        return None
    return max(cand, key=lambda r: r['score'])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                    help='sweep root (JUPITER for canonical lon/43/oc150)')
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat', 'cluster'],
                    help='spatial split axis (canonical: lon)')
    ap.add_argument('--bands', default='43',
                    help="band stack key (canonical: 43)")
    ap.add_argument('--max-oc', type=float, default=150.0,
                    help='max OC cap (canonical: 150)')
    ap.add_argument('--pct', type=float, default=98.0,
                    help='robust percentile for symmetric colour limit')
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    ap.add_argument('--name', default='fig11_residual_map')
    a = ap.parse_args()

    man = fd.Manifest()
    sweep_dir = Path(a.sweep_dir)

    rows = fd.load_ranking(sweep_dir)
    flag = resolve_flagship(rows, axis=a.axis, bands=a.bands, max_oc=a.max_oc)
    if flag is None:
        reason = (f'no SGT flagship row at axis={a.axis} bands={a.bands} '
                  f'max_oc={a.max_oc}')
        man.block('fig11_residual_map', reason, str(sweep_dir))
        man.write(Path(a.out_dir) / 'fig11_residual_map_manifest.md')
        print(f'BLOCKED {reason}')
        return 1

    cd = Path(flag['config_dir'])
    df = fd.load_pooled_predictions(flag)   # 3-seed ensemble for the seed-avg flagship
    if df is None:
        reason = 'load_fold_predictions returned None (no fold parquet)'
        man.block('fig11_residual_map', reason, str(cd))
        man.write(Path(a.out_dir) / 'fig11_residual_map_manifest.md')
        print(f'BLOCKED {reason}: {cd}')
        return 1

    # residual = predicted - actual  (positive = over-prediction)
    resid = (df['OC_predicted'].to_numpy(float)
             - df['OC_actual'].to_numpy(float))
    lon = df['GPS_LONG'].to_numpy(float)
    lat = df['GPS_LAT'].to_numpy(float)
    ok = np.isfinite(resid) & np.isfinite(lon) & np.isfinite(lat)
    resid, lon, lat = resid[ok], lon[ok], lat[ok]
    n = resid.size

    mu = float(np.mean(resid))
    sd = float(np.std(resid, ddof=1)) if n > 1 else 0.0
    # symmetric, robust colour limit (ignore extreme tails)
    vlim = float(np.percentile(np.abs(resid), a.pct))
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = float(np.max(np.abs(resid))) or 1.0

    import figgeo as fg
    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    sc = fs.bavaria_scatter(ax, lon, lat, resid, s=6,
                            vmin=-vlim, vmax=vlim, cmap=RESID_CMAP,
                            scalebar=False, north=True, utm=False)
    # show ALL hold-out points across the 10 folds (some fall just outside the
    # simplified border — that is expected); draw the Bavaria border on top.
    fg.plot_boundary(ax, lw=1.6, zorder=7)
    fg.clip_axes_to_bavaria(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual (predicted - actual), g/kg')

    title = (f'{fs.fam_label(fs.FLAGSHIP)}\n'
             f'Spatial-CV hold-out residuals over Bavaria\n'
             f'mean = {mu:+.2f}  SD = {sd:.2f} g/kg  '
             f'(n = {n}, |resid| {a.pct:.0f}th pct = {vlim:.1f})')
    ax.set_title(title, fontsize=9)

    pdf, png = fs.save(fig, a.out_dir, a.name,
                       script='fig11_residual_map.py')

    man.done(
        'fig11_residual_map',
        files=[str(pdf), str(png)],
        source=str(cd),
        numbers=(f'mean_resid={mu:+.3f} sd_resid={sd:.3f} n={n} '
                 f'vlim(p{a.pct:.0f})={vlim:.2f} '
                 f'flagship={flag["tag"]} r2_mean={flag["r2_mean"]:.4f}'))
    man.write(Path(a.out_dir) / 'fig11_residual_map_manifest.md')

    print(f'flagship={flag["tag"]} cd={cd}')
    print(f'mean_resid={mu:+.3f} sd_resid={sd:.3f} n={n} vlim={vlim:.2f}')
    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
