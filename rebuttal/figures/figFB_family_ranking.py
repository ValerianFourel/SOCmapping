#!/usr/bin/env python3
"""
figFB_family_ranking.py — best-of-each-family hold-out R2 as horizontal bars at
a matched protocol (canonical: lon-blocked, 43-band full_extended, oc150,
10-fold). One bar per model family, error bar = per-fold SD (margins are
~0.4 fold-SD; the SD bar keeps us from overclaiming), parameter count annotated
beside each bar (computed from param_counts for the NN families; '-' for
gradient-boosted trees). The 0 line is emphasized because a failed family
(e.g. 3D-CNN) can sit at negative R2.

Order/grouping (top to bottom in the figure):
  attention  : sgt (flagship, highlighted), vanilla, simpletransformer, lightweight
  trees      : rf, xgb
  weak/failed: cnnlstm, 3dcnn

This makes the CNN-frontend story legible: vanilla (CNN+Transformer) > rf >
simpletransformer (no CNN). Numbers are READ FROM DISK via figdata; nothing is
hardcoded. A family missing from the sweep at the requested protocol is
logged BLOCKED and skipped (no fabricated bar).

CANONICAL run lives only on JUPITER: pass --sweep-dir there.
Smoke-test locally with the STALE bundle:
  python figFB_family_ranking.py \
    --sweep-dir /home/valerian/SGTPublication/SOCrebuttal_HF/sweep \
    --axis lat --bands 20 --max-oc 150
(the layout is correct; the NUMBERS will be the stale lat/20-band values.)
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs
import figdata as fd

fs.setup()

OUT_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'
NAME = 'figFB_family_ranking'

# Display order: attention block, tree block, weak/failed block.
ORDER = ['sgt', 'vanilla', 'simpletransformer', 'lightweight',
         'rf', 'xgb', 'cnnlstm', '3dcnn']
GROUP_OF = {
    'sgt': 'attention', 'vanilla': 'attention',
    'simpletransformer': 'attention', 'lightweight': 'attention',
    'rf': 'trees', 'xgb': 'trees',
    'cnnlstm': 'weak/failed', '3dcnn': 'weak/failed',
}
TREE_FAMILIES = {'rf', 'xgb'}

# canonical family key -> param_counts.build() family name.
# 'simpletransformer' is the pure-transformer (no-CNN) class; in param_counts
# the buildable pure-transformer family is 'lightweight_transformer', which is
# the same constructor signature, so we map both there for a budget estimate.
_PC_FAMILY = {
    'sgt': 'sgt',
    'vanilla': 'vanilla_transformer',
    'simpletransformer': 'lightweight_transformer',
    'lightweight': 'lightweight_transformer',
    'cnnlstm': 'cnnlstm',
    '3dcnn': '3dcnn',
}


def _param_count(fam, row, C, ws, T):
    """Trainable param count for a NN family/config, or None if not derivable.

    Trees return None (annotated '-'). NN families build the actual torch model
    via param_counts; any build failure (e.g. a module not importable in this
    env) degrades to None rather than a fabricated number.
    """
    if fam in TREE_FAMILIES:
        return None
    pc_fam = _PC_FAMILY.get(fam)
    if pc_fam is None:
        return None
    try:
        import param_counts as pc
    except Exception as e:
        print(f'[warn] param_counts import failed ({type(e).__name__}); '
              f'param annotations disabled', file=sys.stderr)
        return None
    d = int(row.get('d_model') or 0) or 128
    h = int(row.get('num_heads') or 0) or 4
    L = int(row.get('num_layers') or 0) or 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = pc.build(pc_fam, C, d, h, L, ws, T)
            return int(pc.n_params(m))
    except Exception as e:
        print(f'[warn] param build failed for {fam} ({pc_fam}) '
              f'd{d}_h{h}_L{L}: {type(e).__name__}: {e}', file=sys.stderr)
        return None


def _fmt_params(n):
    if n is None:
        return '-'
    if n >= 1_000_000:
        return f'{n/1e6:.2f}M'
    return f'{n/1e3:.0f}k'


def build_figure(rows, axis, bands, max_oc, C, ws, T, manifest):
    best = fd.best_per_family(rows, axis=axis, bands=bands, max_oc=max_oc,
                              families=ORDER)

    # Keep ORDER but drop families absent at this protocol (log BLOCKED).
    fams = []
    for fam in ORDER:
        if fam in best:
            fams.append(fam)
        else:
            manifest.block(
                f'family {fam}',
                f'no config at axis={axis} bands={bands} max_oc={max_oc}',
                str(fd.SWEEP_DIR_DEFAULT))
    if not fams:
        return None, best, []

    # Assemble per-family numbers from disk.
    items = []
    for fam in fams:
        r = best[fam]
        r2 = float(r['r2_mean'])
        sd = float(r.get('r2_std') or 0.0)
        npar = _param_count(fam, r, C, ws, T)
        items.append(dict(fam=fam, r2=r2, sd=sd, npar=npar, row=r))

    # y positions: top family at top. Insert a small gap between groups.
    ys = []
    labels = []
    y = 0.0
    prev_group = None
    for it in items:
        g = GROUP_OF[it['fam']]
        if prev_group is not None and g != prev_group:
            y -= 0.7  # group gap
        ys.append(y)
        labels.append(fs.fam_label(it['fam']))
        prev_group = g
        y -= 1.0
    ys = np.array(ys)

    fig_h = max(3.4, 0.62 * len(items) + 1.6)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    r2s = np.array([it['r2'] for it in items])
    sds = np.array([it['sd'] for it in items])

    for it, yy in zip(items, ys):
        _, color, _ = fs.fam_style(it['fam'])
        is_flag = (it['fam'] == fs.FLAGSHIP)
        ax.barh(yy, it['r2'], height=0.7,
                color=color, alpha=0.95 if is_flag else 0.78,
                edgecolor='black' if is_flag else color,
                linewidth=1.8 if is_flag else 0.4,
                zorder=3)
        # per-fold SD error bar (one-sided away from 0 looks misleading; show
        # full symmetric bar — that is the honest fold-to-fold spread).
        ax.errorbar(it['r2'], yy, xerr=it['sd'], fmt='none',
                    ecolor='#222222', elinewidth=1.1, capsize=3, zorder=4)
        # value + param annotation. For positive bars: just past the bar end
        # (and its SD whisker). For negative bars: just right of the 0 line, in
        # the empty positive space, so it never collides with the y-tick label.
        ann = f"R²={it['r2']:.3f}±{it['sd']:.3f}   [{_fmt_params(it['npar'])}]"
        if it['r2'] >= 0:
            x_txt = it['r2'] + it['sd'] + 0.006
        else:
            x_txt = 0.012
        ax.text(x_txt, yy, ann, va='center', ha='left', fontsize=7.6,
                fontweight='bold' if is_flag else 'normal',
                color='#111111')

    # Emphasize the 0 line.
    ax.axvline(0.0, color='black', linewidth=1.4, zorder=2)
    ax.text(0.0, ys.max() + 0.85, 'R² = 0\n(predicts the mean)',
            ha='center', va='bottom', fontsize=7.0, color='#555555')

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_ylim(ys.min() - 0.9, ys.max() + 1.7)

    # x-limits: leave room for the longest annotation on the right.
    lo = min(0.0, float((r2s - sds).min())) - 0.04
    hi = float((r2s + sds).max())
    ax.set_xlim(lo, hi + 0.34 * (hi - lo + 0.01) + 0.18)
    ax.set_xlabel('Spatial 10-fold hold-out R²  (mean ± per-fold SD)')

    # Group separators / labels on the right margin.
    # (light dividers between groups)
    seen = []
    for it, yy in zip(items, ys):
        g = GROUP_OF[it['fam']]
        if g not in seen:
            seen.append(g)

    prot = (f'axis={axis} · {bands}-band · max_oc={max_oc:g} · '
            f'{items[0]["row"].get("n_folds")}-fold')
    ax.set_title('Best configuration per model family (matched protocol)',
                 fontsize=10.5, pad=18)
    ax.text(0.5, 1.012, prot, transform=ax.transAxes, ha='center',
            va='bottom', fontsize=7.6, color='#555555')

    ax.grid(axis='x', alpha=0.25)
    ax.grid(axis='y', visible=False)

    # Flagship callout — plotted star marker (font-safe, unlike a unicode glyph).
    if fs.FLAGSHIP in [it['fam'] for it in items]:
        fi = [it['fam'] for it in items].index(fs.FLAGSHIP)
        yf = ys[fi]
        ax.plot([lo + 0.012], [yf], marker='*', markersize=13,
                color=fs.fam_style(fs.FLAGSHIP)[1],
                markeredgecolor='white', markeredgewidth=0.8,
                clip_on=False, zorder=6)

    fig.tight_layout()
    return fig, best, items


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                   help='sweep root (JUPITER for canonical 43-band/lon runs).')
    p.add_argument('--axis', default='lon', choices=['lon', 'lat'],
                   help='spatial split axis of the runs to plot (default lon).')
    p.add_argument('--bands', default='43',
                   help="band count tag ('43'=full_extended, default).")
    p.add_argument('--max-oc', type=float, default=150.0,
                   help='max OC cap of the runs (default 150).')
    p.add_argument('--channels', type=int, default=None,
                   help='input_channels for param counts (default = --bands).')
    p.add_argument('--window-size', type=int, default=5)
    p.add_argument('--time-before', type=int, default=5)
    p.add_argument('--out-dir', default=OUT_DEFAULT)
    return p.parse_args()


def main():
    a = parse()
    C = a.channels if a.channels is not None else int(a.bands)
    manifest = fd.Manifest()

    rows = fd.load_ranking(a.sweep_dir)
    rows = [r for r in rows if str(r.get('bands')) == str(a.bands)
            and r.get('axis') == a.axis]
    if not rows:
        print(f'BLOCKED no rows at axis={a.axis} bands={a.bands} in '
              f'{a.sweep_dir}')
        manifest.block(NAME, f'no rows axis={a.axis} bands={a.bands}',
                       a.sweep_dir)
        manifest.write(Path(a.out_dir) / f'{NAME}_manifest.md')
        return 1

    fig, best, items = build_figure(
        rows, a.axis, a.bands, a.max_oc, C, a.window_size, a.time_before,
        manifest)
    if fig is None:
        print(f'BLOCKED no families at axis={a.axis} bands={a.bands} '
              f'max_oc={a.max_oc}')
        manifest.write(Path(a.out_dir) / f'{NAME}_manifest.md')
        return 1

    pdf, png = fs.save(fig, a.out_dir, NAME, script=NAME)

    nums = '; '.join(f'{it["fam"]}={it["r2"]:.3f}±{it["sd"]:.3f}'
                     f'({_fmt_params(it["npar"])})' for it in items)
    manifest.done(NAME, [pdf.name, png.name],
                  f'load_ranking({a.sweep_dir}) best_per_family '
                  f'axis={a.axis} bands={a.bands} max_oc={a.max_oc}', nums)
    manifest.write(Path(a.out_dir) / f'{NAME}_manifest.md')

    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
