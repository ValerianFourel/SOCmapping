#!/usr/bin/env python3
"""
count_sgt_parameters.py — print parameter counts for SimpleSGT and
EnhancedSGT under both their class defaults and the rebuttal-preset
configurations used in train.py / run_kfold.py.

Run with the project venv (or any python3 with torch):

    /home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python \
        /home/valerian/SGTPublication/SOCmapping/rebuttal/count_sgt_parameters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add SGT module to path
SGT_DIR = Path('/home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer')
sys.path.insert(0, str(SGT_DIR))

from SimpleSGT import SimpleSGT
from EnhancedSGT import EnhancedSGT


# Shared input geometry (from config.py: 6 bands, 5×5 window, 5 timesteps)
COMMON = dict(input_channels=6, height=5, width=5, time_steps=5)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# Configurations to evaluate
configs = [
    # SimpleSGT
    ('SimpleSGT', 'class defaults',
        dict(d_model=128, num_heads=2, dropout=0.3)),
    ('SimpleSGT', 'rebuttal "small"',
        dict(d_model=128, num_heads=2, dropout=0.3)),

    # EnhancedSGT
    ('EnhancedSGT', 'class defaults',
        dict(d_model=128, num_heads=4, num_encoder_layers=3,
             dropout=0.3, expansion_factor=4)),
    ('EnhancedSGT', 'rebuttal "big"  (heads=8, layers=2)',
        dict(d_model=128, num_heads=8, num_encoder_layers=2,
             dropout=0.3, expansion_factor=4)),
    ('EnhancedSGT', 'historic-wandb-claimed (heads=8, layers=2)',
        dict(d_model=128, num_heads=8, num_encoder_layers=2,
             dropout=0.3, expansion_factor=4)),
    ('EnhancedSGT', 'class defaults (heads=4, layers=3)',
        dict(d_model=128, num_heads=4, num_encoder_layers=3,
             dropout=0.3, expansion_factor=4)),
]

print(f'{"Model":<14}  {"Configuration":<48}  {"d_model":>7}  {"heads":>5}  '
      f'{"layers":>6}  {"params":>11}')
print('-' * 110)
for cls_name, label, kwargs in configs:
    cls = SimpleSGT if cls_name == 'SimpleSGT' else EnhancedSGT
    m = cls(**COMMON, **kwargs)
    layers = kwargs.get('num_encoder_layers', 1)  # SimpleSGT has 1 hardcoded
    print(f'{cls_name:<14}  {label:<48}  {kwargs["d_model"]:>7}  '
          f'{kwargs["num_heads"]:>5}  {layers:>6}  {n_params(m):>11,}')

print()
print('Sanity check: Model A saved checkpoint reports model_parameters = 1,120,546')
print('  → matches EnhancedSGT class defaults (heads=4, layers=3), NOT (heads=8, layers=2).')
print('  CLI/wandb args at training time were heads=8, layers=2, but the model was')
print('  instantiated with class defaults (kwargs not forwarded at that commit).')
