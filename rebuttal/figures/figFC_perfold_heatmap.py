#!/usr/bin/env python3
"""
figFC_perfold_heatmap.py — per-fold hold-out R2 across the N spatial folds for
the best configuration of each model family (matched protocol; canonical:
lon-blocked, 43-band full_extended, oc150, 10-fold).

Heatmap: rows = families (ordered top-to-bottom by mean R2, best first),
columns = fold_id (0..N-1), cell value = that family's r2_per_fold[fold]
(read straight from the ranking row — same list sweep_summarize ranks on).
Diverging colormap centered at 0 so a fold where a model *fails* (negative R2,
worse than predicting the mean) is visually obvious. Every cell is annotated
with its value. A right-margin column shows mean ± per-fold SD per family
(the SD matters: spatial margins are ~0.4 fold-SD, so a single mean overstates
the certainty).

The figure does NOT assume which fold is hardest — the worst column is
detected from the data and labeled.

Numbers are READ FROM DISK via figdata; nothing is hardcoded. A family with no
config at the protocol, or with a missing/short r2_per_fold list, is logged
BLOCKED and dropped (no fabricated row).

CANONICAL run lives only on JUPITER: pass --sweep-dir there.
Smoke-test locally with the STALE bundle:
  python figFC_perfold_heatmap.py \
    --sweep-dir /home/valerian/SGTPublication/SOCrebuttal_HF/sweep \
    --axis lat --bands 20 --max-oc 150
(layout correct; NUMBERS are the stale lat/20-band values.)
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import figstyle as fs
import figdata as fd

fs.setup()

OUT_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'
NAME = 'figFC_perfold_heatmap'

# All families we try to include, in a fixed candidate order. Final row order
# is by mean R2 (best on top), so this only governs which families are sought.
CANDIDATES = ['sgt', 'vanilla', 'simpletransformer', 'lightweight',
              'rf', 'xgb', 'cnnlstm', '3dcnn']


def collect_perfold(rows, axis, bands, max_oc, manifest):
    """Return [(fam, r2_mean, r2_std, np.array per_fold)] for families that
    have a usable r2_per_fold list, plus the common fold count N."""
    best = fd.best_per_family(rows, axis=axis, bands=bands, max_oc=max_oc,
                              families=CANDIDATES)
    items = []
    lengths = []
    for fam in CANDIDATES:
        r = best.get(fam)
        if r is None:
            manifest.block(f'family {fam}',
                           f'no config at axis={axis} bands={bands} '
                           f'max_oc={max_oc}', str(fd.SWEEP_DIR_DEFAULT))
            continue
        pf = r.get('r2_per_fold')
        if not pf or not isinstance(pf, (list, tuple)) or len(pf) == 0:
            manifest.block(f'family {fam}',
                           'missing/empty r2_per_fold list',
                           str(r.get('config_dir', '')))
            continue
        arr = np.array([np.nan if v is None else float(v) for v in pf],
                       dtype=float)
        items.append(dict(fam=fam, r2=float(r['r2_mean']),
                          sd=float(r.get('r2_std') or 0.0), pf=arr, row=r))
        lengths.append(len(arr))
    if not items:
        return [], 0
    N = max(lengths)  # pad shorter lists with NaN so columns align
    for it in items:
        if len(it['pf']) < N:
            it['pf'] = np.concatenate(
                [it['pf'], np.full(N - len(it['pf']), np.nan)])
    # order rows by mean R2, best on top
    items.sort(key=lambda it: it['r2'], reverse=True)
    return items, N


def build_figure(items, N, axis, bands, max_oc):
    fams = [it['fam'] for it in items]
    M = len(items)
    data = np.vstack([it['pf'] for it in items])  # M x N

    # Diverging norm centered at 0; symmetric range from the data extent.
    finite = data[np.isfinite(data)]
    amax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    amax = max(amax, 0.1)
    norm = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax)
    cmap = plt.get_cmap('RdBu_r')   # red = positive R2 (good), blue = negative

    fig_w = max(7.4, 0.62 * N + 3.2)
    fig_h = max(3.0, 0.55 * M + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    # annotate each cell
    for i in range(M):
        for j in range(N):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=7,
                        color='#888888')
                continue
            # white text on saturated cells, dark on pale ones
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = 'white' if lum < 0.5 else '#111111'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.2,
                    color=tc)

    # worst fold (lowest mean across families) — detected, not assumed
    col_mean = np.nanmean(data, axis=0)
    worst = int(np.nanargmin(col_mean))

    ax.set_xticks(range(N))
    ax.set_xticklabels([f'F{j}' for j in range(N)], fontsize=8.2)
    ax.set_yticks(range(M))
    ax.set_yticklabels([fs.fam_label(f) for f in fams], fontsize=8.0)
    ax.set_xlabel('Spatial fold (hold-out block)')

    # highlight the worst column
    ax.add_patch(plt.Rectangle((worst - 0.5, -0.5), 1, M, fill=False,
                               edgecolor='black', linewidth=2.0, zorder=5))
    ax.text(worst, -0.62, 'worst', ha='center', va='bottom', fontsize=7.4,
            fontweight='bold')

    # gridlines between cells
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.grid(which='major', visible=False)
    ax.tick_params(which='minor', length=0)

    # flagship row marker — plotted star (font-safe, unlike a unicode glyph).
    if fs.FLAGSHIP in fams:
        fr = fams.index(fs.FLAGSHIP)
        ax.plot([-0.72], [fr], marker='*', markersize=12,
                color=fs.fam_style(fs.FLAGSHIP)[1],
                markeredgecolor='white', markeredgewidth=0.6,
                clip_on=False, zorder=6)

    # right-margin mean±SD column (as text, outside the heatmap)
    x_txt = N - 0.5 + 0.55
    ax.text(x_txt, -0.78, 'mean ± SD', ha='left', va='bottom', fontsize=8.0,
            fontweight='bold')
    for i, it in enumerate(items):
        ax.text(x_txt, i, f"{it['r2']:.3f} ± {it['sd']:.3f}",
                ha='left', va='center', fontsize=8.0,
                fontweight='bold' if it['fam'] == fs.FLAGSHIP else 'normal')

    # colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.16)
    cb.set_label('Hold-out R²  (white = 0; blue < 0 = worse than the mean)',
                 fontsize=8.0)
    cb.ax.axhline(0.0, color='black', linewidth=0.8)

    prot = (f'axis={axis} · {bands}-band · max_oc={max_oc:g} · '
            f'{N}-fold · best config per family')
    ax.set_title('Per-fold R² by family  (rows sorted by mean R²)',
                 fontsize=10.5, pad=34)
    ax.text(0.5, 1.072, prot, transform=ax.transAxes, ha='center',
            va='bottom', fontsize=7.6, color='#555555')

    fig.tight_layout()
    return fig, worst


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                   help='sweep root (JUPITER for canonical 43-band/lon runs).')
    p.add_argument('--axis', default='lon', choices=['lon', 'lat'],
                   help='spatial split axis (default lon).')
    p.add_argument('--bands', default='43',
                   help="band count tag ('43'=full_extended, default).")
    p.add_argument('--max-oc', type=float, default=150.0,
                   help='max OC cap (default 150).')
    p.add_argument('--out-dir', default=OUT_DEFAULT)
    return p.parse_args()


def main():
    a = parse()
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

    items, N = collect_perfold(rows, a.axis, a.bands, a.max_oc, manifest)
    if not items:
        print(f'BLOCKED no families with r2_per_fold at axis={a.axis} '
              f'bands={a.bands} max_oc={a.max_oc}')
        manifest.write(Path(a.out_dir) / f'{NAME}_manifest.md')
        return 1

    fig, worst = build_figure(items, N, a.axis, a.bands, a.max_oc)
    pdf, png = fs.save(fig, a.out_dir, NAME, script=NAME)

    nums = (f'worst fold=F{worst}; '
            + '; '.join(f'{it["fam"]}={it["r2"]:.3f}±{it["sd"]:.3f}'
                        for it in items))
    manifest.done(NAME, [pdf.name, png.name],
                  f'load_ranking({a.sweep_dir}) r2_per_fold best per family '
                  f'axis={a.axis} bands={a.bands} max_oc={a.max_oc}', nums)
    manifest.write(Path(a.out_dir) / f'{NAME}_manifest.md')

    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
