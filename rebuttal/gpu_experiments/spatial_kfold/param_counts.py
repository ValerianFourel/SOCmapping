#!/usr/bin/env python3
"""
param_counts.py — print trainable-parameter counts for the sweep model
families across a d_model grid, so a config can be matched to a target
budget (e.g. "which configs are ~500k params?").

sweep_summarize.py ranks by R² but does NOT show parameter counts; this
fills that gap. Tags here (family_d<d>_h<h>_L<L>) match the sweep run tags,
so you can cross-reference a row in the ranking with its size.

Builds the model classes directly (torch only — no accelerate / dataloaders),
mirroring run_kfold._build_model's constructor calls.

Run (project venv or any python3 with torch):
    python rebuttal/gpu_experiments/spatial_kfold/param_counts.py
    python rebuttal/gpu_experiments/spatial_kfold/param_counts.py --near 500000
    python rebuttal/gpu_experiments/spatial_kfold/param_counts.py --bands 20
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[2]   # spatial_kfold → gpu_experiments → rebuttal → SOCmapping
for sub in ('SpatiotemporalGatedTransformer', 'SimpleTransformer', '3DCNN', 'CNNLSTM'):
    sys.path.insert(0, str(SOC_ROOT / sub))

import torch  # noqa: E402


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(family, C, d, h, L, ws, T):
    """Mirror run_kfold._build_model (no two_path; eff_in_channels = C)."""
    if family == 'sgt':            # SimpleSGT — CNN + GRN gate + 1-layer Transformer
        from SimpleSGT import SimpleSGT
        return SimpleSGT(input_channels=C, height=ws, width=ws, time_steps=T,
                         d_model=d, num_heads=h, dropout=0.5)
    if family == 'vanilla_transformer':   # CNN + Transformer (no gate), scalable L
        from VanillaSpatiotemporalTransformer import VanillaSpatiotemporalTransformer
        return VanillaSpatiotemporalTransformer(
            input_channels=C, height=ws, width=ws, time_steps=T,
            d_model=d, num_heads=h, num_layers=L, dropout=0.5)
    if family == 'lightweight_transformer':   # pure Transformer (no CNN), scalable L
        from LightweightTransformer import LightweightTransformer
        return LightweightTransformer(
            input_channels=C, height=ws, width=ws, time_steps=T,
            d_model=d, num_heads=h, num_layers=L, dropout=0.5)
    if family == 'cnnlstm':        # CNN + LSTM (d = LSTM hidden width)
        from models import RefittedCovLSTM
        return RefittedCovLSTM(num_channels=C, lstm_input_size=128,
                               lstm_hidden_size=d, num_layers=L, dropout=0.5)
    if family == '3dcnn':          # 3-D CNN (ignores d/h/L)
        from modelCNNMultiYear import Small3DCNN
        return Small3DCNN(input_channels=C, input_height=ws, input_width=ws,
                          input_time=T, dropout_rate=0.5)
    raise ValueError(family)


# (family, label, [(d, h, L), ...]) — covers the CNN-front-end vs
# transformer-only families across the d_model curve.
GRID = [
    ('sgt', 'SGT gated (CNN+GRN+Transf)',
        [(64, 4, 1), (128, 4, 1), (192, 4, 1), (256, 4, 1), (320, 4, 1)]),
    ('vanilla_transformer', 'Vanilla (CNN+Transf, no gate)',
        [(64, 4, 1), (128, 4, 1), (192, 4, 1), (192, 8, 1), (256, 4, 1),
         (128, 4, 2), (192, 4, 2), (256, 4, 2)]),
    ('lightweight_transformer', 'Lightweight (Transformer, no CNN)',
        [(64, 4, 1), (128, 4, 1), (192, 4, 1), (256, 4, 1),
         (128, 4, 2), (192, 4, 2), (256, 4, 2)]),
    ('cnnlstm', 'CNN-LSTM',
        [(64, 4, 1), (128, 4, 1), (256, 4, 1)]),
    ('3dcnn', '3D-CNN',
        [(64, 4, 1)]),
]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bands', type=int, default=43,
                   help='input_channels (43 = full_extended, 20 = full_20, '
                        '6 = original_6). Default 43.')
    p.add_argument('--window-size', type=int, default=5)
    p.add_argument('--time-before', type=int, default=5)
    p.add_argument('--near', type=int, default=0,
                   help='If set, also print just the configs whose param '
                        'count is closest (±25%%) to this target, e.g. 500000.')
    return p.parse_args()


def main():
    a = parse()
    C, ws, T = a.bands, a.window_size, a.time_before
    print(f'# trainable params  (input_channels={C}, window {ws}x{ws}, {T} years)\n')
    print(f'{"family":<34}{"tag":<16}{"params":>12}')
    print('-' * 62)
    rows = []
    for family, label, configs in GRID:
        for (d, h, L) in configs:
            try:
                m = build(family, C, d, h, L, ws, T)
                np_ = n_params(m)
            except Exception as e:
                print(f'{label:<34}d{d}_h{h}_L{L:<10} ERR {type(e).__name__}: {e}')
                continue
            tag = f'd{d}_h{h}_L{L}'
            rows.append((label, family, tag, np_))
            print(f'{label:<34}{tag:<16}{np_:>12,}')
    if a.near:
        lo, hi = a.near * 0.75, a.near * 1.25
        near = sorted((r for r in rows if lo <= r[3] <= hi),
                      key=lambda r: abs(r[3] - a.near))
        print(f'\n# closest to {a.near:,} params (within ±25%):')
        for label, family, tag, np_ in near:
            print(f'  {label:<34}{family}_{tag:<22}{np_:>12,}')


if __name__ == '__main__':
    main()
