#!/usr/bin/env python3
"""
figFA_param_scaling.py — the headline "smaller is better" figure for the
Geoderma revision (GEODER-D-26-01032).

Mean spatial-CV R^2 (y, with +/-1 per-fold SD error bars) vs trainable
parameters (x, log scale), coloured by model family. Every metric is read
from disk via figdata; param counts come from the same constructors the
sweep used (rebuttal/gpu_experiments/spatial_kfold/param_counts.build +
n_params, with SimpleTransformerV2 counted directly because param_counts
cannot build it). Trees (rf/xgb) have no parameter x-coordinate and are
omitted. Two never-run configs (sgt d192_h4_L1, lightweight d192_h4_L1)
are shown as HOLLOW markers at their param-x and the family plateau level,
explicitly annotated "not run (projected)" — no fabricated R^2.

The CANONICAL results are 43-band full_extended, oc150, split-axis 'lon',
10-fold, and live ONLY on JUPITER (pass --sweep-dir). Defaults below target
that protocol. The local SOCrebuttal_HF/sweep bundle is schema-identical but
STALE (axis='lat', bands='20'); smoke-test with --axis lat --bands 20.

Usage:
    python figFA_param_scaling.py \
        --sweep-dir /path/on/jupiter/sweep --axis lon --bands 43 --max-oc 150

    # smoke test (stale local bundle; numbers will be wrong, that's expected):
    python figFA_param_scaling.py \
        --sweep-dir /home/valerian/SGTPublication/SOCrebuttal_HF/sweep \
        --axis lat --bands 20 --max-oc 150
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs
import figdata as fd
fs.setup()

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]          # rebuttal/figures -> rebuttal -> SOCmapping
KFOLD_DIR = SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'
sys.path.insert(0, str(KFOLD_DIR))
# Model constructors used by param_counts.build import repo-root helpers
# (_bands) and per-family `config` modules. param_counts.main() adds the
# sibling dirs lazily; replicate that here (plus SOC_ROOT for _bands) so the
# builds resolve regardless of cwd.
for _p in (SOC_ROOT, SOC_ROOT / 'SpatiotemporalGatedTransformer',
           SOC_ROOT / 'SimpleTransformer', SOC_ROOT / '3DCNN',
           SOC_ROOT / 'CNNLSTM'):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
import param_counts as pc  # build(family,C,d,h,L,ws,T) + n_params(model)
import figparams as fp      # shared param utility (same numbers across figures)

# canonical SGT stack geometry (window 5x5, 5 years)
WS, T = 5, 5

# canonical-family-key -> param_counts.build family name. Trees have no
# parameter coordinate; simpletransformer is built directly (param_counts
# cannot construct SimpleTransformerV2).
_BUILD_NAME = {
    'sgt': 'sgt',
    'vanilla': 'vanilla_transformer',
    'lightweight': 'lightweight_transformer',
    'cnnlstm': 'cnnlstm',
    '3dcnn': '3dcnn',
}
_NO_PARAM = {'rf', 'xgb'}            # tree families: skip (no x)
FLAGSHIP_PARAMS = 84_449            # sgt d32_h2_L1 @ C=43 (annotated landmark)

# Configs the sweep never ran but the paper references as projections. We do
# NOT invent an R^2: we place them at their param-x and the family plateau and
# label them "not run (projected)".
NOT_RUN = [
    {'family': 'sgt',         'd': 192, 'h': 4, 'L': 1, 'label': 'SGT d192 (not run)'},
    {'family': 'lightweight', 'd': 192, 'h': 4, 'L': 1, 'label': 'Lightweight d192 (not run)'},
]


def _simpletransformer_params(C, h, L):
    """Trainable params of SimpleTransformerV2 (the model the sweep used for
    family 'simpletransformer'); param_counts.build cannot construct it.

    modelSimpleTransformerNew does `from config import NUM_LAYERS, NUM_HEADS`,
    and several siblings ship their own `config.py`. Force the SimpleTransformer
    dir to the front and evict any already-cached conflicting `config`/model
    modules so the right one resolves regardless of import order."""
    import importlib
    st_dir = str(SOC_ROOT / 'SimpleTransformer')
    sys.path.insert(0, st_dir)
    for mod in ('config', 'modelSimpleTransformerNew'):
        sys.modules.pop(mod, None)
    try:
        m = importlib.import_module('modelSimpleTransformerNew')
        model = m.SimpleTransformerV2(
            input_channels=C, input_height=WS, input_width=WS, input_time=T,
            num_heads=h or 4, num_layers=L or 1, dropout_rate=0.5)
        return pc.n_params(model)
    finally:
        sys.modules.pop('config', None)   # don't poison later sibling imports


def count_params(fam, C, d, h, L):
    """Trainable params for a sweep row's architecture, or None if the family
    has no parameter coordinate (rf/xgb) or cannot be built. Delegates to the
    shared figparams utility so figFA and figFB never disagree."""
    return fp.params_for(fam, C, d, h, L, WS, T)


def best_unique_configs(rows, axis, bands, max_oc):
    """Best-scoring row per (family,d,h,L) at the fixed protocol, so each
    architecture appears once at the right size."""
    sel = fd.select(rows, axis=axis, bands=bands, max_oc=max_oc)
    by_cfg = {}
    for r in sel:
        key = (r['family'], r.get('d_model'), r.get('num_heads'),
               r.get('num_layers'))
        if key not in by_cfg or r['score'] > by_cfg[key]['score']:
            by_cfg[key] = r
    return list(by_cfg.values())


def fold_sd(row):
    """+/-1 SD of per-fold R^2 (recompute from r2_per_fold if present, else
    fall back to the stored r2_std). Never overclaim margins."""
    rpf = row.get('r2_per_fold')
    if rpf:
        a = np.asarray([x for x in rpf if x is not None], dtype=float)
        if a.size > 1:
            return float(np.std(a, ddof=1))
    return float(row.get('r2_std') or 0.0)


def build_figure(rows, axis, bands, max_oc, manifest):
    C = int(bands)
    cfgs = best_unique_configs(rows, axis, bands, max_oc)
    if not cfgs:
        manifest.block('figFA param-scaling',
                       f'no rows for axis={axis} bands={bands} max_oc={max_oc}',
                       '(sweep-dir)')
        return None

    # ---- assemble plottable (real) points -----------------------------------
    pts = []                # dicts: family, x, y, yerr, d, h, L, tag
    skipped_trees = 0
    fam_plateau = {}        # family -> list of real R^2 in the 300k-620k band
    for r in cfgs:
        fam = r['family']
        x = count_params(fam, C, r.get('d_model'), r.get('num_heads'),
                          r.get('num_layers'))
        if x is None:
            if fam in _NO_PARAM:
                skipped_trees += 1
            continue
        y = float(r['r2_mean'])
        pts.append({'family': fam, 'x': x, 'y': y, 'yerr': fold_sd(r),
                    'd': r.get('d_model'), 'h': r.get('num_heads'),
                    'L': r.get('num_layers'), 'tag': r.get('tag')})
        if 300_000 <= x <= 620_000:
            fam_plateau.setdefault(fam, []).append(y)

    if not pts:
        manifest.block('figFA param-scaling',
                       'no parametric (non-tree) configs found', '(sweep-dir)')
        return None

    # overall plateau level for projecting not-run points (median of the
    # 300k-620k band across families, or global median as fallback)
    plateau_vals = [v for vs in fam_plateau.values() for v in vs]
    plateau_level = (float(np.median(plateau_vals)) if plateau_vals
                     else float(np.median([p['y'] for p in pts])))

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # plot real points grouped by family (one legend entry per family)
    fams_present = []
    for fam in fs.FAMILIES:                     # stable, flagship-first order
        fam_pts = [p for p in pts if p['family'] == fam]
        if not fam_pts:
            continue
        fams_present.append(fam)
        label, color, marker = fs.fam_style(fam)
        fam_pts.sort(key=lambda p: p['x'])
        xs = [p['x'] for p in fam_pts]
        ys = [p['y'] for p in fam_pts]
        es = [p['yerr'] for p in fam_pts]
        # connect within-family points (size sweep) with a thin guide line
        if len(fam_pts) > 1:
            ax.plot(xs, ys, color=color, lw=1.0, alpha=0.55, zorder=2)
        ax.errorbar(xs, ys, yerr=es, fmt='none', ecolor=color, elinewidth=1.0,
                    capsize=2.5, alpha=0.7, zorder=3)
        ax.scatter(xs, ys, s=46, marker=marker, facecolor=color,
                   edgecolor='white', linewidths=0.6, zorder=4)

    # ---- not-run / projected configs (HOLLOW, no fabricated R^2) ------------
    proj_handle = None
    # stagger labels (alternate above/below) so close-together projections
    # don't collide; nudge marker slightly off the plateau line per family.
    _stagger = [(0, 18, 'bottom', 0.012), (0, -18, 'top', -0.012)]
    for i, nr in enumerate(NOT_RUN):
        fam = nr['family']
        if fam not in _BUILD_NAME:
            continue
        x = count_params(fam, C, nr['d'], nr['h'], nr['L'])
        if x is None:
            manifest.block('figFA not-run projection',
                           f"could not size {fam} d{nr['d']}_h{nr['h']}_L{nr['L']}",
                           '(param_counts)')
            continue
        _, color, marker = fs.fam_style(fam)
        dx, dy, va, ynudge = _stagger[i % len(_stagger)]
        y_marker = plateau_level + ynudge
        # projected: place at the plateau level, visually distinct (hollow)
        h = ax.scatter([x], [y_marker], s=72, marker=marker,
                       facecolor='none', edgecolor=color, linewidths=1.6,
                       zorder=5)
        ax.annotate(nr['label'], xy=(x, y_marker),
                    xytext=(dx, dy), textcoords='offset points',
                    ha='center', va=va, fontsize=6.6, color=color,
                    style='italic')
        proj_handle = h
    # dashed projected band across the parameter axis at plateau level
    if proj_handle is not None:
        ax.axhline(plateau_level, color='#777777', ls='--', lw=0.8, alpha=0.6,
                   zorder=1)

    # ---- annotate the flagship (84k SGT) ------------------------------------
    flag = min((p for p in pts if p['family'] == 'sgt'),
               key=lambda p: abs(p['x'] - FLAGSHIP_PARAMS), default=None)
    if flag is not None:
        ax.annotate(f"flagship SGT\n{flag['x']/1e3:.0f}k params, "
                    f"R²={flag['y']:.3f}",
                    xy=(flag['x'], flag['y']),
                    xytext=(14, 26), textcoords='offset points',
                    fontsize=7.5, ha='left', va='bottom',
                    color=fs.fam_style('sgt')[1],
                    arrowprops=dict(arrowstyle='->', lw=0.9,
                                    color=fs.fam_style('sgt')[1]))

    # ---- axes / cosmetics ---------------------------------------------------
    ax.set_xscale('log')
    ax.set_xlabel('Trainable parameters (log scale)')
    ax.set_ylabel('Spatial-CV $R^2$ (mean $\\pm$ 1 fold SD)')
    ax.set_title('Smaller is better: spatial-CV $R^2$ vs model size\n'
                 f'(axis={axis}, {bands}-band, max OC={int(float(max_oc))}, '
                 '10-fold)', fontsize=9.5)
    ax.grid(True, which='both', alpha=0.22)
    ax.grid(True, which='minor', alpha=0.10)
    ax.axhline(0.0, color='k', lw=0.7, alpha=0.5)

    # ---- legend: families present + projected marker ------------------------
    handles = []
    for fam in fams_present:
        label, color, marker = fs.fam_style(fam)
        handles.append(Line2D([0], [0], marker=marker, color=color,
                              linestyle='-', lw=1.0, markeredgecolor='white',
                              markeredgewidth=0.6, markersize=7, label=label))
    if proj_handle is not None:
        handles.append(Line2D([0], [0], marker='o', color='#555555',
                              linestyle='None', markerfacecolor='none',
                              markeredgecolor='#555555', markeredgewidth=1.4,
                              markersize=8, label='not run (projected)'))
    ax.legend(handles=handles, loc='lower left', fontsize=7.2,
              ncol=1, handletextpad=0.5, borderaxespad=0.4)

    fig.tight_layout()

    n_proj = sum(1 for nr in NOT_RUN if nr['family'] in _BUILD_NAME)
    manifest.done(
        'figFA param-scaling',
        ['figFA_param_scaling.pdf', 'figFA_param_scaling.png'],
        f'sweep axis={axis} bands={bands} max_oc={max_oc} '
        f'(C={C}, ws={WS}, T={T}); params via param_counts.build/n_params + '
        'SimpleTransformerV2',
        f'{len(pts)} real configs across {len(fams_present)} families; '
        f'{skipped_trees} tree configs omitted (no params); '
        f'plateau~{plateau_level:.3f}; {n_proj} not-run projected')
    return fig


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                   help='sweep root (pass the JUPITER canonical dir for the '
                        'real 43-band/lon numbers).')
    p.add_argument('--axis', default='lon', choices=['lon', 'lat'],
                   help='spatial-CV split axis (canonical: lon).')
    p.add_argument('--bands', default='43',
                   help="band stack ('43'=full_extended canonical, '20', '6').")
    p.add_argument('--max-oc', type=float, default=150.0,
                   help='SOC cap of the run (canonical: 150).')
    p.add_argument('--out-dir',
                   default=str(HERE / 'out'),
                   help='output directory for the PDF/PNG.')
    return p.parse_args()


def main():
    a = parse()
    manifest = fd.Manifest()
    rows = fd.load_ranking(a.sweep_dir)
    if not rows:
        print(f'BLOCKED no ranking rows under {a.sweep_dir}')
        return 1
    fig = build_figure(rows, a.axis, str(a.bands), a.max_oc, manifest)
    if fig is None:
        manifest.write(Path(a.out_dir) / 'figFA_param_scaling_manifest.md')
        print(f'BLOCKED no plottable configs for axis={a.axis} '
              f'bands={a.bands} max_oc={a.max_oc} under {a.sweep_dir}')
        return 1
    pdf, png = fs.save(fig, a.out_dir, 'figFA_param_scaling',
                       script='figFA_param_scaling.py')
    manifest.write(Path(a.out_dir) / 'figFA_param_scaling_manifest.md')
    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
