#!/usr/bin/env python3
"""
fig07_altitude_year_soc.py — clean 2D replacement for the old 3D
altitude / year / SOC scatter (Figure 7, Geoderma revision GEODER-D-26-01032).

DATA, in priority order (no blind import of the old script):
  1. The committed measured-sample table that figure7_replacement.py produced,
     rebuttal/figure7_data.csv  (cols: POINTID, GPS_LONG, GPS_LAT, year, OC,
     elevation, altitude_bin, year_bin) — measured SOC, true source of the panel.
  2. Fallback: the flagship SGT pooled fold predictions, which carry 'altitude'
     and 'year' columns; we then plot measured SOC (OC_actual) vs altitude /
     year over those hold-out points.
  3. Neither present -> Manifest.block (no fabrication).

Two panels, both 2D and honest:
  (left)  SOC vs survey year, coloured by altitude band, with per-band OLS trend
          + 95% slope CI in the legend (so a flat trend reads as flat).
  (right) SOC vs altitude, coloured by year band, with per-band OLS trend.

All slopes/intercepts/counts computed FROM the loaded table; nothing hardcoded.

Smoke test (uses local csv; numbers are real measured-sample numbers):
  python fig07_altitude_year_soc.py \
    --sweep-dir /home/valerian/SGTPublication/SOCrebuttal_HF/sweep \
    --axis lat --bands 20 --max-oc 150
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs  # noqa: E402
import figdata as fd   # noqa: E402

import argparse        # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np     # noqa: E402
import pandas as pd    # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fs.setup()

OUT_DIR_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'
FIG7_CSV = Path('/home/valerian/SGTPublication/SOCmapping/rebuttal/figure7_data.csv')

ALT_EDGES = [0, 300, 700, 1500, np.inf]
ALT_LABELS = ['<300 m', '300-700 m', '700-1500 m', '>1500 m']
ALT_COLORS = ['#0072B2', '#009E73', '#E69F00', '#D55E00']

YEAR_EDGES = [2006, 2011, 2016, 2024]
YEAR_LABELS = ['2007-2011', '2012-2016', '2017-2023']
YEAR_COLORS = ['#7570b3', '#1b9e77', '#d95f02']


def _fit_line(x, y):
    """OLS slope, intercept, n, and 95% CI half-width on the slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3 or x.std() == 0:
        return float('nan'), float('nan'), int(n), float('nan')
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    dof = n - 2
    s_err = np.sqrt(np.sum((y - yhat) ** 2) / dof)
    sxx = np.sum((x - x.mean()) ** 2)
    se_slope = s_err / np.sqrt(sxx) if sxx > 0 else float('nan')
    ci = 1.96 * se_slope  # ~95% normal approx (large n)
    return float(slope), float(intercept), int(n), float(ci)


def _load_table(args):
    """Return (df, soc_col, alt_col, source_str) or (None, None, None, None)."""
    # Priority 1: committed measured-sample csv from figure7_replacement.py.
    if FIG7_CSV.exists():
        df = pd.read_csv(FIG7_CSV)
        if {'OC', 'elevation', 'year'}.issubset(df.columns):
            return df, 'OC', 'elevation', str(FIG7_CSV)

    # Priority 2: flagship fold predictions (carry altitude + year).
    rows = fd.load_ranking(args.sweep_dir)
    cand = fd.select(rows, axis=args.axis, bands=args.bands,
                     max_oc=args.max_oc, family=args.family)
    if cand:
        best = max(cand, key=lambda r: r['score'])
        cfg = Path(best['config_dir'])
        fp = fd.load_fold_predictions(cfg)
        if fp is not None and not fp.empty and \
                {'OC_actual', 'altitude', 'year'}.issubset(fp.columns):
            return fp, 'OC_actual', 'altitude', f'{cfg} (best={best["tag"]})'
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                    help='sweep root (only used for the fold-prediction fallback)')
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat'])
    ap.add_argument('--bands', default='43')
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--family', default=fs.FLAGSHIP)
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    man = fd.Manifest()
    name = 'fig07_altitude_year_soc'

    df, soc_col, alt_col, source = _load_table(args)
    if df is None:
        reason = ('no measured-sample csv and no fold predictions with '
                  'altitude/year columns')
        man.block(name, reason,
                  f'{FIG7_CSV} OR {args.sweep_dir}')
        man.write(Path(args.out_dir) / f'{name}_manifest.md')
        print(f'BLOCKED {reason}')
        return 1

    df = df.copy()
    df = df[np.isfinite(df[soc_col]) & np.isfinite(df[alt_col]) &
            np.isfinite(df['year'])]
    df['alt_bin'] = pd.cut(df[alt_col], bins=ALT_EDGES, labels=ALT_LABELS,
                           right=False)
    df['year_bin'] = pd.cut(df['year'], bins=YEAR_EDGES, labels=YEAR_LABELS,
                            right=True, include_lowest=True)

    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # ---- Left: SOC vs year, coloured by altitude band ----
    ax = axes[0]
    left_trends = []
    for i, lab in enumerate(ALT_LABELS):
        sub = df[df['alt_bin'] == lab]
        if sub.empty:
            continue
        jit = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(sub['year'] + jit, sub[soc_col], s=5, alpha=0.16,
                   color=ALT_COLORS[i], linewidths=0)
    for i, lab in enumerate(ALT_LABELS):
        sub = df[df['alt_bin'] == lab]
        slope, intercept, n, ci = _fit_line(sub['year'], sub[soc_col])
        if not np.isfinite(slope):
            continue
        yrs = np.array([df['year'].min(), df['year'].max()], float)
        ax.plot(yrs, slope * yrs + intercept, color=ALT_COLORS[i], lw=2.4,
                label=f'{lab} (n={n:,}, {slope:+.2f}'
                      f'{"" if not np.isfinite(ci) else f"±{ci:.2f}"} g/kg/yr)')
        left_trends.append((lab, n, slope, ci))
    ax.set_xlabel('Survey year')
    ax.set_ylabel(f'SOC (g/kg){"" if soc_col == "OC" else " — measured"}')
    ax.set_title('SOC vs year, by altitude band', fontsize=10)
    ax.legend(loc='upper left', fontsize=7.2)

    # ---- Right: SOC vs altitude, coloured by year band ----
    ax = axes[1]
    for i, lab in enumerate(YEAR_LABELS):
        sub = df[df['year_bin'] == lab]
        if sub.empty:
            continue
        ax.scatter(sub[alt_col], sub[soc_col], s=5, alpha=0.14,
                   color=YEAR_COLORS[i], linewidths=0)
    for i, lab in enumerate(YEAR_LABELS):
        sub = df[df['year_bin'] == lab]
        slope, intercept, n, ci = _fit_line(sub[alt_col], sub[soc_col])
        if not np.isfinite(slope):
            continue
        elev = np.array([df[alt_col].min(), df[alt_col].max()], float)
        ci100 = ci * 100 if np.isfinite(ci) else float('nan')
        ax.plot(elev, slope * elev + intercept, color=YEAR_COLORS[i], lw=2.4,
                label=f'{lab} (n={n:,}, {slope * 100:+.2f}'
                      f'{"" if not np.isfinite(ci100) else f"±{ci100:.2f}"} '
                      f'g/kg/100m)')
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('SOC (g/kg)')
    ax.set_title('SOC vs altitude, by year band', fontsize=10)
    ax.legend(loc='upper right', fontsize=7.2)

    src_short = 'measured samples' if soc_col == 'OC' \
        else 'flagship hold-out points (measured SOC)'
    fig.suptitle(f'SOC vs altitude and year — n = {len(df):,} {src_short}',
                 fontsize=11, y=1.0)
    fig.tight_layout()

    pdf, png = fs.save(fig, args.out_dir, name, script='fig07_altitude_year_soc.py')

    nums = '; '.join(f'{lab}:slope={s:+.2f}/yr(n={n})'
                     for lab, n, s, _ in left_trends) or 'n/a'
    man.done(name, [pdf.name, png.name], source,
             f'n={len(df)} year-trends[{nums}]')
    man.write(Path(args.out_dir) / f'{name}_manifest.md')

    print(f'[altitude_year_soc] source={source} n={len(df)} '
          f'soc_col={soc_col} alt_col={alt_col}')
    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
