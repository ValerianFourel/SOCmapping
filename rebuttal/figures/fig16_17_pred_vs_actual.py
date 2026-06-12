#!/usr/bin/env python3
"""
fig16_17_pred_vs_actual.py — Predicted vs actual SOC for the flagship SGT,
pooled spatial-CV hold-out (Geoderma revision, GEODER-D-26-01032).

Reads the pooled out-of-fold predictions (all 10 lon-blocked folds concatenated)
for the BEST SGT config at the canonical protocol (lon / 43-band / oc150) via
fd.load_fold_predictions, then draws a predicted-vs-actual figure:

  * hexbin density of every hold-out point (per-point density = log counts),
  * 1:1 identity line + OLS fit line,
  * pooled R2 / RMSE / MAE annotated, computed FROM the parquet (never hardcoded).

HONESTY: we only have hold-out (out-of-fold) predictions on disk; train-fit
predictions are NOT stored, so we render the hold-out panel only and print a
note. We do NOT fabricate a train panel.

CANONICAL data is JUPITER-only: pass --sweep-dir there. The local
SOCrebuttal_HF/sweep bundle is schema-identical but STALE (lat / 20-band /
different numbers) — smoke-test only.

Smoke test (numbers will be stale/wrong, that is expected):
  python fig16_17_pred_vs_actual.py \
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

fs.setup()

OUT_DIR_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'


def _metrics(actual, pred):
    """Pooled R2 / RMSE / MAE computed straight from the arrays."""
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[m], pred[m]
    n = actual.size
    if n < 2:
        return dict(n=n, r2=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan)
    resid = pred - actual
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    return dict(n=n, r2=r2, rmse=rmse, mae=mae, bias=bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                    help='sweep root (JUPITER for canonical; local bundle is stale)')
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat'],
                    help="spatial-CV split axis (canonical 'lon')")
    ap.add_argument('--bands', default='43',
                    help="band stack id (canonical '43' full_extended)")
    ap.add_argument('--max-oc', type=float, default=150.0,
                    help='max OC cap (canonical 150)')
    ap.add_argument('--family', default=fs.FLAGSHIP,
                    help="model family (canonical flagship 'sgt')")
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    ap.add_argument('--gridsize', type=int, default=45,
                    help='hexbin grid resolution')
    args = ap.parse_args()

    man = fd.Manifest()
    name = 'fig16_17_pred_vs_actual'

    rows = fd.load_ranking(args.sweep_dir)
    cand = fd.select(rows, axis=args.axis, bands=args.bands,
                     max_oc=args.max_oc, family=args.family)
    if not cand:
        reason = (f'no {args.family} config at axis={args.axis} '
                  f'bands={args.bands} max_oc={args.max_oc}')
        man.block(name, reason, args.sweep_dir)
        man.write(Path(args.out_dir) / f'{name}_manifest.md')
        print(f'BLOCKED {reason} in {args.sweep_dir}')
        return 1

    best = max(cand, key=lambda r: r['score'])
    cfg = Path(best['config_dir'])
    df = fd.load_fold_predictions(cfg)
    if df is None or df.empty:
        reason = 'no pooled fold predictions parquet'
        man.block(name, reason, str(cfg))
        man.write(Path(args.out_dir) / f'{name}_manifest.md')
        print(f'BLOCKED {reason} at {cfg}')
        return 1

    actual = df['OC_actual'].to_numpy(float)
    pred = df['OC_predicted'].to_numpy(float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    M = _metrics(actual, pred)

    # OLS fit line (predicted ~ actual) for visual reference alongside the 1:1.
    slope, intercept = np.polyfit(actual, pred, 1)

    # Axis range: shared, padded, includes 1:1 line domain.
    lo = float(min(actual.min(), pred.min()))
    hi = float(max(actual.max(), pred.max()))
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    hb = ax.hexbin(actual, pred, gridsize=args.gridsize, cmap='viridis',
                   bins='log', mincnt=1, linewidths=0.0,
                   extent=(lo, hi, lo, hi), rasterized=True)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('Hold-out point count (log scale)')

    # 1:1 identity
    ax.plot([lo, hi], [lo, hi], color='k', lw=1.4, ls='--',
            label='1:1', zorder=5)
    # OLS fit
    xx = np.array([lo, hi])
    ax.plot(xx, slope * xx + intercept, color=fs.fam_style(args.family)[1],
            lw=1.8, label=f'OLS fit (slope={slope:.2f})', zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Measured SOC (g/kg)')
    ax.set_ylabel('Predicted SOC (g/kg)')

    fam_label = fs.fam_label(args.family)
    ax.set_title(f'{fam_label.split(" (")[0]} — pooled spatial-CV hold-out\n'
                 f'(axis={args.axis}, {args.bands}-band, max_oc={args.max_oc:.0f}, '
                 f"{best.get('n_folds', '?')} folds)",
                 fontsize=9.5)

    # Metrics box — read entirely from the parquet.
    txt = (f"$R^2$ = {M['r2']:.3f}\n"
           f"RMSE = {M['rmse']:.2f} g/kg\n"
           f"MAE = {M['mae']:.2f} g/kg\n"
           f"bias = {M['bias']:+.2f} g/kg\n"
           f"n = {M['n']:,}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.4', fc='white',
                                  ec='0.6', alpha=0.9))

    # Honesty note: hold-out only; train-fit not stored.
    ax.text(0.98, 0.04,
            'Hold-out (out-of-fold) only;\ntrain-fit predictions not stored.',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=7.5,
            color='0.35', style='italic')

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.13), fontsize=8)

    fig.tight_layout()
    pdf, png = fs.save(fig, args.out_dir, name, script='fig16_17_pred_vs_actual.py')

    man.done(name, [pdf.name, png.name], str(cfg),
             f"R2={M['r2']:.3f} RMSE={M['rmse']:.2f} MAE={M['mae']:.2f} "
             f"n={M['n']} (best={best['tag']})")
    man.write(Path(args.out_dir) / f'{name}_manifest.md')

    print(f"[pred_vs_actual] best={best['tag']} axis={args.axis} "
          f"bands={args.bands} oc={args.max_oc} -> R2={M['r2']:.3f} "
          f"RMSE={M['rmse']:.2f} MAE={M['mae']:.2f} n={M['n']}")
    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
