#!/usr/bin/env python3
"""
fig08_sgt84k_architecture.py — schematic block diagram of the flagship
SimpleSGT (CNN + GRN gate + 1-layer Transformer, ~84k params) and a small
contrast panel showing the Vanilla variant (same backbone, GRN replaced by a
plain Linear, no gate, ~115k params).

Pure-matplotlib patches/arrows — NO sweep data. Parameter counts ARE read
honestly from rebuttal/gpu_experiments/spatial_kfold/param_counts.build so the
"~84k" / "~115k" annotations cannot drift from the actual model classes.

The GRN gate is labelled prominently: it is now a live positive ablation
(SGT-with-gate > Vanilla-without-gate at matched/larger budget), so the figure
makes the gate the visual focal point of the flagship pipeline.

Smoke test (no --sweep-dir needed):
    python fig08_sgt84k_architecture.py
    -> writes out/fig08_sgt84k_architecture.{pdf,png}, prints 'OK <pdf>'.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs  # noqa: E402
import figdata as fd   # noqa: E402

import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fs.setup()

OUT_DEFAULT = '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures/out'

# canonical stack geometry (43 bands x 5x5 window x 5 years)
C, WS, T = 43, 5, 5


# ---- honest parameter counts from param_counts.build ----------------------
def param_counts():
    """(sgt_d32_params, vanilla_d64_params) or (None, None) if torch missing."""
    kf = '/home/valerian/SGTPublication/SOCmapping/rebuttal/gpu_experiments/spatial_kfold'
    sys.path.insert(0, kf)
    try:
        import param_counts as pc
        sgt = pc.n_params(pc.build('sgt', C, 32, 2, 1, WS, T))
        van = pc.n_params(pc.build('vanilla_transformer', C, 64, 2, 1, WS, T))
        return int(sgt), int(van)
    except Exception as e:  # torch unavailable / build error -> annotate "n/a"
        print(f'[WARN] param_counts.build unavailable: {type(e).__name__}: {e}',
              file=sys.stderr)
        return None, None


# ---- small drawing helpers (axes are 0..1 normalized) ---------------------
def block(ax, x, y, w, h, label, color, *, sub='', fc_alpha=0.85,
          ec='#222222', fontsize=8.0, fontweight='normal', text_color='#111111'):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle='round,pad=0.004,rounding_size=0.012',
                         linewidth=1.1, edgecolor=ec, facecolor=color,
                         alpha=fc_alpha, zorder=2)
    ax.add_patch(box)
    txt = label if not sub else f'{label}\n{sub}'
    ax.text(x, y, txt, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=3, linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, p0, p1, *, color='#333333', lw=1.4, style='-|>'):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                        linewidth=lw, color=color, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def right_of(b):
    x, y, w, h = b
    return (x + w / 2, y)


def left_of(b):
    x, y, w, h = b
    return (x - w / 2, y)


def top_of(b):
    x, y, w, h = b
    return (x, y + h / 2)


def bottom_of(b):
    x, y, w, h = b
    return (x, y - h / 2)


def build_figure(sgt_p, van_p):
    import matplotlib.pyplot as plt
    sgt_lbl, sgt_col, _ = fs.fam_style('sgt')
    van_lbl, van_col, _ = fs.fam_style('vanilla')
    neutral = '#EDEDED'
    cnn_col = '#D9E8F2'

    fig = plt.figure(figsize=(9.0, 6.2))
    # main flagship pipeline panel (top, wide) + contrast panel (bottom)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.34)
    ax = fig.add_subplot(gs[0]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axc = fig.add_subplot(gs[1]); axc.set_axis_off()
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)

    sgt_str = f'~{sgt_p/1000:.0f}k params' if sgt_p else 'params n/a'
    van_str = f'~{van_p/1000:.0f}k params' if van_p else 'params n/a'

    # ============ TOP PANEL: SimpleSGT flagship ============================
    ax.set_title(f'Flagship SimpleSGT (d32, h2, L1) — {sgt_str}',
                 fontsize=10.5, color=sgt_col, fontweight='bold', loc='left',
                 pad=2)

    yc = 0.52
    # input tensor
    b_in = block(ax, 0.075, yc, 0.115, 0.34, 'Input',
                 neutral, sub=f'{C} bands\n{WS}x{WS} window\n{T} years',
                 fontsize=7.4)
    # per-timestep CNN spatial encoder
    b_cnn = block(ax, 0.245, yc, 0.150, 0.42,
                  'Per-timestep\nCNN encoder', cnn_col,
                  sub='2x Conv2d\n+ AdaptiveAvgPool\n(shared over t)',
                  fontsize=7.3)
    # GRN gate — the focal block
    b_grn = block(ax, 0.430, yc, 0.140, 0.46, 'GRN\nGATE', sgt_col,
                  sub='Gated Residual\nNetwork', fontsize=8.6,
                  fontweight='bold', text_color='white', fc_alpha=0.95)
    # positional embedding (small) — placed clearly ABOVE the transformer
    b_pos = block(ax, 0.660, 0.85, 0.155, 0.15,
                  'Positional\nembedding', '#FBE9D0', fontsize=7.3)
    # transformer encoder over the 5 timesteps (lowered, leaving a clear gap)
    b_tr = block(ax, 0.660, 0.40, 0.155, 0.36,
                 '1-layer\nTransformer', '#F4D9B8',
                 sub=f'encoder over\n{T} timesteps\n(h2)', fontsize=7.4)
    # regression head
    b_head = block(ax, 0.890, 0.40, 0.115, 0.28,
                   'Regression\nhead', neutral, sub='SOC (g/kg)',
                   fontsize=7.4)

    arrow(ax, right_of(b_in), left_of(b_cnn))
    arrow(ax, right_of(b_cnn), left_of(b_grn))
    # GRN feeds the transformer (mid-height, clean horizontal-ish run)...
    arrow(ax, right_of(b_grn), left_of(b_tr), color=sgt_col, lw=1.5)
    # ...and the positional embedding (diagonal up-right to its left edge)
    arrow(ax, (b_grn[0] + 0.02, top_of(b_grn)[1]), left_of(b_pos),
          color=sgt_col, lw=1.5)
    # positional embedding drops straight down into the transformer top
    arrow(ax, bottom_of(b_pos), top_of(b_tr), color='#B07A2E', lw=1.2)
    arrow(ax, right_of(b_tr), left_of(b_head))

    # GRN gate callout — point to the gate's LEFT edge so it clears the arrows
    ax.annotate('learned gate selects\ninformative features\n(live + ablation)',
                xy=left_of(b_grn),
                xytext=(0.205, 0.085),
                ha='center', va='bottom', fontsize=7.2, color=sgt_col,
                fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=sgt_col, lw=1.3,
                                connectionstyle='arc3,rad=-0.2'))

    # ============ BOTTOM PANEL: Vanilla contrast ==========================
    axc.set_title(f'Vanilla contrast (d64, h2, L1) — {van_str}: '
                  'GRN gate replaced by a plain Linear (no gate)',
                  fontsize=9.0, color=van_col, fontweight='bold', loc='left',
                  pad=2)
    ycc = 0.46
    c_in = block(axc, 0.085, ycc, 0.115, 0.55, 'Input',
                 neutral, sub=f'{C}x{WS}x{WS}x{T}', fontsize=7.2)
    c_cnn = block(axc, 0.265, ycc, 0.150, 0.62, 'Per-timestep\nCNN encoder',
                  cnn_col, fontsize=7.3)
    # the swapped block: plain Linear, NO gate (use vanilla color)
    c_lin = block(axc, 0.475, ycc, 0.150, 0.66, 'Plain\nLINEAR', van_col,
                  sub='(no gate)', fontsize=8.4, fontweight='bold',
                  text_color='white', fc_alpha=0.95)
    c_tr = block(axc, 0.690, ycc, 0.150, 0.62, '1-layer\nTransformer',
                 '#CFE0EC', sub=f'over {T} timesteps', fontsize=7.3)
    c_head = block(axc, 0.890, ycc, 0.115, 0.50, 'Regression\nhead', neutral,
                   fontsize=7.3)
    for a_, b_ in [(c_in, c_cnn), (c_cnn, c_lin), (c_lin, c_tr), (c_tr, c_head)]:
        arrow(axc, right_of(a_), left_of(b_),
              color=van_col if b_ is c_lin or a_ is c_lin else '#333333',
              lw=1.4)
    axc.annotate('same backbone — only this block differs',
                 xy=(c_lin[0], c_lin[1] - c_lin[3] / 2),
                 xytext=(c_lin[0], 0.02), ha='center', va='bottom',
                 fontsize=7.0, color=van_col, fontweight='bold',
                 arrowprops=dict(arrowstyle='-|>', color=van_col, lw=1.1))

    # legend swatches tying the two focal blocks to fs family identity
    handles = [
        mpatches.Patch(facecolor=sgt_col, edgecolor='#222', label=fs.fam_label('sgt')),
        mpatches.Patch(facecolor=van_col, edgecolor='#222', label=fs.fam_label('vanilla')),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7.6,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    # accepted for interface parity with the data figures; unused (no data here)
    ap.add_argument('--sweep-dir', default=None,
                    help='ignored — this figure is pure schematic, no data')
    ap.add_argument('--bands', default='43', help='ignored (kept for parity)')
    ap.add_argument('--axis', default='lon', help='ignored (kept for parity)')
    ap.add_argument('--max-oc', type=float, default=150.0,
                    help='ignored (kept for parity)')
    a = ap.parse_args()

    man = fd.Manifest()
    sgt_p, van_p = param_counts()
    fig = build_figure(sgt_p, van_p)
    pdf, png = fs.save(fig, a.out_dir, 'fig08_sgt84k_architecture',
                       script='fig08_sgt84k_architecture.py')

    nums = (f'SGT d32_h2_L1={sgt_p:,}p; Vanilla d64_h2_L1={van_p:,}p'
            if sgt_p and van_p else 'param counts unavailable (torch?)')
    man.done('fig08 architecture schematic', [str(pdf), str(png)],
             'param_counts.build (sgt d32_h2_L1, vanilla d64_h2_L1); '
             'no sweep data', nums)
    man.write(Path(a.out_dir) / 'fig08_manifest.md')
    print(f'OK {pdf}')


if __name__ == '__main__':
    main()
