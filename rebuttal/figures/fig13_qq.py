#!/usr/bin/env python3
"""fig13_qq.py — normal Q-Q plot of the flagship SGT's pooled CV residuals.

From the flagship config's pooled hold-out fold predictions (canonical protocol:
split-axis lon, 43-band 'full_extended', oc150), residual = OC_predicted -
OC_actual. We compare the residual distribution to a normal via
scipy.stats.probplot (with its least-squares reference line) and annotate the
sample skew and (excess) kurtosis plus a normality test, so the reader can judge
how heavy-tailed / skewed the errors are.

CANONICAL data is JUPITER-only -> pass --sweep-dir; defaults select lon/43/oc150.
Smoke-test with the STALE local bundle:
  --sweep-dir /home/.../SOCrebuttal_HF/sweep --axis lat --bands 20 --max-oc 150
Every metric is read from disk via figdata; nothing hardcoded.
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
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / 'out'


def resolve_flagship(rows, *, axis, bands, max_oc):
    cand = fd.select(rows, axis=axis, bands=str(bands), max_oc=max_oc,
                     family=fs.FLAGSHIP)
    if not cand:
        return None
    return max(cand, key=lambda r: r['score'])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT))
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat', 'cluster'])
    ap.add_argument('--bands', default='43')
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--standardize', action='store_true',
                    help='z-score residuals before the Q-Q (default: raw g/kg)')
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    ap.add_argument('--name', default='fig13_qq')
    a = ap.parse_args()

    man = fd.Manifest()
    sweep_dir = Path(a.sweep_dir)

    rows = fd.load_ranking(sweep_dir)
    flag = resolve_flagship(rows, axis=a.axis, bands=a.bands, max_oc=a.max_oc)
    if flag is None:
        reason = (f'no SGT flagship row at axis={a.axis} bands={a.bands} '
                  f'max_oc={a.max_oc}')
        man.block('fig13_qq', reason, str(sweep_dir))
        man.write(Path(a.out_dir) / 'fig13_qq_manifest.md')
        print(f'BLOCKED {reason}')
        return 1

    cd = Path(flag['config_dir'])
    df = fd.load_fold_predictions(cd)
    if df is None:
        reason = 'load_fold_predictions returned None (no fold parquet)'
        man.block('fig13_qq', reason, str(cd))
        man.write(Path(a.out_dir) / 'fig13_qq_manifest.md')
        print(f'BLOCKED {reason}: {cd}')
        return 1

    resid = (df['OC_predicted'].to_numpy(float)
             - df['OC_actual'].to_numpy(float))
    resid = resid[np.isfinite(resid)]
    n = resid.size

    mu = float(np.mean(resid))
    sd = float(np.std(resid, ddof=1)) if n > 1 else 0.0
    skew = float(stats.skew(resid))
    exkurt = float(stats.kurtosis(resid))   # Fisher: excess kurtosis (0=normal)

    # normality test (D'Agostino-Pearson; robust on large n)
    try:
        nt_stat, nt_p = stats.normaltest(resid)
        nt_stat, nt_p = float(nt_stat), float(nt_p)
    except Exception:
        nt_stat, nt_p = float('nan'), float('nan')

    z = (resid - mu) / sd if (a.standardize and sd > 0) else resid
    unit = 'standardized residual' if a.standardize else 'residual (g/kg)'

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    # probplot draws the points + a least-squares reference line on ax
    (osm, osr), (slope, intercept, r) = stats.probplot(z, dist='norm', plot=ax)

    # restyle probplot's default artists to our palette
    col = fs.fam_style(fs.FLAGSHIP)[1]
    pts, line = ax.get_lines()[0], ax.get_lines()[1]
    pts.set_marker('o'); pts.set_markersize(3); pts.set_alpha(0.35)
    pts.set_markerfacecolor(col); pts.set_markeredgecolor('none')
    pts.set_linestyle('none')
    line.set_color('k'); line.set_linewidth(1.8); line.set_linestyle('--')

    ax.set_title(f'{fs.fam_label(fs.FLAGSHIP)} — normal Q-Q of CV residuals',
                 fontsize=10)
    ax.set_xlabel('Theoretical quantiles (normal)')
    ax.set_ylabel(f'Sample quantiles — {unit}')

    txt = (f'n = {n}\n'
           f'mean = {mu:+.2f}  SD = {sd:.2f}\n'
           f'skew = {skew:+.3f}\n'
           f'excess kurtosis = {exkurt:+.3f}\n'
           f'fit $R^2$ = {r**2:.3f}\n'
           f"normaltest p = {nt_p:.1e}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5,
            bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.9))

    fig.tight_layout()
    pdf, png = fs.save(fig, a.out_dir, a.name, script='fig13_qq.py')

    man.done(
        'fig13_qq',
        files=[str(pdf), str(png)],
        source=str(cd),
        numbers=(f'n={n} mean={mu:+.3f} sd={sd:.3f} skew={skew:+.3f} '
                 f'exkurt={exkurt:+.3f} normaltest_p={nt_p:.2e} '
                 f'flagship={flag["tag"]} r2_mean={flag["r2_mean"]:.4f}'))
    man.write(Path(a.out_dir) / 'fig13_qq_manifest.md')

    print(f'flagship={flag["tag"]} cd={cd}')
    print(f'n={n} mean={mu:+.3f} sd={sd:.3f} skew={skew:+.3f} '
          f'exkurt={exkurt:+.3f} normaltest_p={nt_p:.2e}')
    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
