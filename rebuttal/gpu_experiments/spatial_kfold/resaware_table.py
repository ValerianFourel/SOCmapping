#!/usr/bin/env python3
"""resaware_table.py — the publishable comparison table for the resolution-aware
ablation. Reads each variant's kfold_results_summary.json (the SAME across-fold
metrics every other family reports), adds the trainable-parameter count for each,
and prints/writes a markdown table comparing — on identical folds:

  (a) existing 84k SGT (small_d32_h2_L1)   [reference; pass --baseline-summary]
  (b) all_flat                              (resolution-naive concat-MLP)
  (c) full multi-branch
  (d) each single-group-ablated variant

Honest by construction: every number is read from disk; nothing hardcoded except
the band geometry. If a variant hasn't finished, it's listed as PENDING.

Run:
    python resaware_table.py --sweep-dir .../sweep/resaware --bands-list full_extended \\
        [--baseline-summary .../sweep/<sgt-d32>/kfold_results_summary.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[2]
for sub in ('', 'SpatiotemporalGatedTransformer'):
    sys.path.insert(0, str(SOC_ROOT / sub))

VARIANTS = [   # (dir, label, branches, ablate)
    ('resaware_full',       'Multi-branch (full)',       'fine_med_coarse', None),
    ('resaware_allflat',    'all_flat (naive concat)',   'all_flat',        None),
    ('resaware_abl_fine',   'multi − fine',              'fine_med_coarse', 'fine'),
    ('resaware_abl_medium', 'multi − medium',            'fine_med_coarse', 'medium'),
    ('resaware_abl_coarse', 'multi − coarse',            'fine_med_coarse', 'coarse'),
]


def _agg(summary_path: Path):
    if not summary_path.exists():
        return None
    j = json.loads(summary_path.read_text())
    ac = j.get('across_folds', {})
    fr = [f.get('r2') for f in j.get('fold_results', []) if f.get('r2') is not None]
    import statistics
    r2_sd = (statistics.pstdev(fr) if len(fr) > 1 else ac.get('r2_std'))
    return {'r2': ac.get('r2_mean'), 'r2_sd': r2_sd,
            'rmse': ac.get('rmse_mean'), 'mae': ac.get('mae_mean'),
            'rpiq': ac.get('rpiq_mean'), 'n_folds': len(fr)}


def _params_resaware(branches, bands_list, d, h, dropout=0.3):
    import torch  # noqa
    from config import bands_list_order, time_before, window_size  # noqa: E402
    import _bands as b
    from band_subsets import get_band_indices
    from ResolutionAwareNet import ResolutionAwareNet
    idx = get_band_indices(bands_list, list(bands_list_order))
    cube = [bands_list_order[i] for i in idx]
    fine, med, coarse = b.resolution_groups(cube)
    m = ResolutionAwareNet(len(idx), window_size, window_size, time_before,
                           d_model=d, num_heads=h, dropout=dropout,
                           fine_idx=fine, med_idx=med, coarse_idx=coarse,
                           branches=branches)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _params_sgt(d=32, h=2):
    from config import time_before, window_size, bands_list_order
    from band_subsets import get_band_indices
    from SimpleSGT import SimpleSGT
    n = len(get_band_indices('full_extended', list(bands_list_order)))
    m = SimpleSGT(input_channels=n, height=window_size, width=window_size,
                  time_steps=time_before, d_model=d, num_heads=h, dropout=0.5)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', type=Path, required=True,
                   help='dir holding the resaware_* variant subdirs')
    p.add_argument('--bands-list', default='full_extended')
    p.add_argument('--hidden_size', type=int, default=32)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--baseline-summary', type=Path, default=None,
                   help='kfold_results_summary.json of the 84k SGT baseline')
    p.add_argument('--out', type=Path, default=None, help='write the table here (.md)')
    a = p.parse_args()

    rows = []
    # baseline SGT row
    try:
        bp = _params_sgt()
    except Exception as e:
        bp = None; print(f'[warn] SGT param count failed: {e}', file=sys.stderr)
    base = _agg(a.baseline_summary) if a.baseline_summary else None
    rows.append(('SGT small_d32_h2_L1 (baseline)', base, bp))
    # resaware variants
    for d, label, branches, _ab in VARIANTS:
        agg = _agg(a.sweep_dir / d / 'kfold_results_summary.json')
        try:
            pc = _params_resaware(branches, a.bands_list, a.hidden_size, a.num_heads)
        except Exception as e:
            pc = None; print(f'[warn] param count {d}: {e}', file=sys.stderr)
        rows.append((label, agg, pc))

    L = ['# Resolution-aware ablation — 10-fold longitude-blocked CV', '',
         f'bands={a.bands_list}, same folds for every row, fp32, deterministic.', '',
         '| Model | R² (mean ± SD) | RMSE | MAE | RPIQ | params |',
         '|---|---|---|---|---|---|']
    for label, agg, pc in rows:
        pcs = f'{pc:,}' if pc else '—'
        if not agg or agg.get('r2') is None:
            L.append(f'| {label} | PENDING | | | | {pcs} |')
        else:
            L.append(f'| {label} | {agg["r2"]:.3f} ± {agg["r2_sd"]:.3f} | '
                     f'{agg["rmse"]:.2f} | {agg["mae"]:.2f} | {agg["rpiq"]:.3f} | {pcs} |')
    out = '\n'.join(L) + '\n'
    print(out)
    if a.out:
        a.out.write_text(out); print(f'[table] wrote {a.out}')


if __name__ == '__main__':
    main()
