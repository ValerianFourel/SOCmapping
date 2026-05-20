#!/usr/bin/env python3
"""
inspect_run.py — detailed inspector for specific spatial-CV sweep configs.

Where sweep_summarize.py shows a single ranked table across everything,
this script gives a per-config deep-dive: architecture description,
parameter count, training recipe, and per-fold R² breakdown. Designed
for writing the architecture-comparison subsection of the rebuttal.

Examples (run from SOCmapping/):
  # List all available tags grouped by sweep_group
  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py --list

  # Detailed view of one tag (use --params to load checkpoints for the
  # exact parameter count)
  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py \\
      --show small_d128_h4_L1 --group oc150 --params

  # Side-by-side comparison of the gating ablation (SGT vs Vanilla)
  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py \\
      --compare small_d128_h4_L1 vanilla_transformer_d128_h4_L1 \\
      --group oc150 --params

  # Preset: the three-way architecture comparison you asked about
  #   sgt (CNN + GRN-gated transformer)
  #   vs vanilla (CNN + transformer, NO gate)
  #   vs simpletransformer (transformer alone, NO CNN frontend)
  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py \\
      --preset cnn-frontend --group oc150

  # Preset: gating ablation
  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py \\
      --preset gating --group oc150

  # Preset: full architecture comparison (all 5 families head-to-head)
  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py \\
      --preset architecture --group oc150
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / 'sweep'

# Architecture descriptions — used to render the "what is this thing" line
# in --show and --compare. Keys match the family slug parsed from the tag.
ARCH_DESCRIPTIONS = {
    'sgt-small': (
        'SimpleSGT (CNN + GRN-gated transformer): per-band 1×1 grouped '
        'conv → spatial pool → linear projection → Gated Residual '
        'Network (GRN) → 1-layer Transformer encoder → MLP head. '
        'The GRN gate is the characteristic component vs vanilla.'
    ),
    'sgt-big': (
        'EnhancedSGT (multi-layer SGT): 3D spatial encoder → GRN gate '
        '→ multi-layer Transformer encoder → MLP head.'
    ),
    'vanilla_transformer': (
        'SimpleSGT MINUS GRN (gating ablation): per-band 1×1 grouped '
        'conv → spatial pool → linear projection → plain Linear '
        '(no GRN) → 1-layer Transformer encoder → MLP head. Same '
        'architecture as SimpleSGT minus the gate block.'
    ),
    'simpletransformer': (
        'SimpleTransformerV2 (transformer-only, NO CNN frontend): '
        'flatten (C, T, H, W) → large linear input embedding → '
        'multi-layer Transformer encoder → MLP head. Parameter count '
        'dominated by the input embedding layer (~11.2M at 20 bands, '
        '~1.7M at 6 bands, d=64 h=4 L=1).'
    ),
    '3dcnn': (
        'Small3DCNN (CNN-only, NO transformer): 3D convolutions over '
        '(C, T, H, W) → global pool → MLP head.'
    ),
    'cnnlstm': (
        'RefittedCovLSTM (CNN + LSTM, NO transformer): per-time 2D CNN '
        'spatial encoder → LSTM over time → MLP head. The recurrent '
        'counterpart to vanilla_transformer.'
    ),
    'baseline-rf': (
        'RandomForest on 80-d per-band statistics '
        '(mean/std/p10/p90 × 20 bands). No spatial structure.'
    ),
    'baseline-xgb': (
        'XGBoost on 80-d per-band statistics. GPU-accelerated training.'
    ),
}

# Family aliases used in tag patterns (must match sweep_summarize.parse_tag).
_ARCH_TAG_RE = re.compile(r'^(small_)?d(\d+)_h(\d+)_L(\d+)$')
_FAMILY_TAG_RE = re.compile(
    r'^(3dcnn|cnnlstm|simpletransformer|vanilla_transformer)'
    r'_d(\d+)_h(\d+)_L(\d+)$')
_BASELINE_TAG_RE = re.compile(r'^baseline_([a-z0-9]+)_(.+)$')


def parse_tag(tag: str) -> dict | None:
    m = _ARCH_TAG_RE.match(tag)
    if m:
        variant = 'sgt-small' if m.group(1) else 'sgt-big'
        return {
            'family': variant,
            'd_model': int(m.group(2)),
            'num_heads': int(m.group(3)),
            'num_layers': int(m.group(4)),
        }
    m = _FAMILY_TAG_RE.match(tag)
    if m:
        return {
            'family': m.group(1),
            'd_model': int(m.group(2)),
            'num_heads': int(m.group(3)),
            'num_layers': int(m.group(4)),
        }
    m = _BASELINE_TAG_RE.match(tag)
    if m:
        return {
            'family': f'baseline-{m.group(1)}',
            'd_model': None,
            'num_heads': None,
            'num_layers': None,
            'variant': m.group(2),
        }
    return None


def _cfg_dir(group: str, tag: str) -> Path:
    if group in ('(top)', 'top', ''):
        return SWEEP_DIR / tag
    return SWEEP_DIR / group / tag


def load_summary(group: str, tag: str) -> dict | None:
    p = _cfg_dir(group, tag) / 'kfold_results_summary.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def count_params(group: str, tag: str) -> tuple[int | None, str | None]:
    """Try to load one fold's best .pth and report exact parameter count.

    Counts all tensors in model_state_dict (this is the actual parameter
    count of the trained model, not buffers). Returns (n, source_file)
    or (None, None) if no checkpoint accessible.
    """
    cfg_dir = _cfg_dir(group, tag)
    for pth in sorted(cfg_dir.glob('fold_*_best.pth')):
        try:
            import torch
            ck = torch.load(str(pth), map_location='cpu', weights_only=False)
            sd = ck.get('model_state_dict', ck.get('state_dict', ck))
            if not isinstance(sd, dict):
                continue
            n = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
            return n, pth.name
        except Exception:
            continue
    return None, None


def list_all_configs() -> list[dict]:
    """Walk SWEEP_DIR recursively and group by sweep_group."""
    rows = []
    skip_dirs = {'sbatch', 'slurm_logs', 'baseline_features'}
    seen = set()
    for summary in sorted(SWEEP_DIR.rglob('kfold_results_summary.json')):
        cfg_dir = summary.parent
        if cfg_dir in seen:
            continue
        seen.add(cfg_dir)
        rel = cfg_dir.relative_to(SWEEP_DIR).parts
        if any(part in skip_dirs for part in rel):
            continue
        tag = rel[-1]
        group = '/'.join(rel[:-1]) or '(top)'
        parsed = parse_tag(tag)
        if parsed is None:
            continue
        rows.append({'group': group, 'tag': tag, **parsed})
    return rows


def _recipe_rows(recipe: dict) -> list[tuple[str, str]]:
    keys = [
        ('loss',           'loss_type'),
        ('loss_alpha',     'loss_alpha'),
        ('chi2_weight',    'chi2_weight'),
        ('lr',             'lr'),
        ('lr_schedule',    'lr_scheduler'),
        ('lr_min',         'lr_min'),
        ('num_epochs',     'num_epochs'),
        ('max_oc',         'max_oc'),
        ('sampler_mode',   'sampler_mode'),
        ('augment_train',  'augment_train'),
        ('bands_list',     'bands_list'),
        ('target_xform',   'target_transform'),
        ('seed_base',      'seed_base'),
    ]
    return [(label, str(recipe.get(k, '—'))) for label, k in keys]


def show_one(group: str, tag: str, args) -> None:
    j = load_summary(group, tag)
    if j is None:
        print(f'[error] no summary for {group}/{tag}', file=sys.stderr)
        return
    parsed = parse_tag(tag) or {}
    family = parsed.get('family', 'unknown')
    desc = ARCH_DESCRIPTIONS.get(family, '(no description registered)')

    ac = j.get('across_folds', {})
    recipe = j.get('recipe', {})
    fold_results = j.get('fold_results', [])

    print('=' * 78)
    print(f'  tag        : {tag}')
    print(f'  group      : {group}')
    print(f'  family     : {family}')
    print(f'  arch       : {desc}')
    if parsed.get('d_model') is not None:
        print(f'  hparams    : d_model={parsed["d_model"]}  '
              f'num_heads={parsed["num_heads"]}  '
              f'num_layers={parsed["num_layers"]}')
    if 'variant' in parsed:
        print(f'  variant    : {parsed["variant"]}')
    if args.params:
        n_params, src = count_params(group, tag)
        if n_params is not None:
            print(f'  params     : {n_params:,}  (from {src})')
        else:
            print(f'  params     : (no checkpoint accessible)')
    print('-' * 78)
    print('  Training recipe:')
    for label, val in _recipe_rows(recipe):
        print(f'    {label:<14}: {val}')
    print('-' * 78)
    print('  Performance (10-fold spatial CV):')
    r2m = ac.get('r2_mean'); r2s = ac.get('r2_std')
    rmse = ac.get('rmse_mean'); mae = ac.get('mae_mean'); rpiq = ac.get('rpiq_mean')
    def f(x, prec=4): return f'{x:+.{prec}f}' if isinstance(x, (int, float)) else '—'
    def g(x, prec=3): return f'{x:.{prec}f}' if isinstance(x, (int, float)) else '—'
    print(f'    r2_mean       : {f(r2m)}')
    print(f'    r2_std        : {g(r2s, 4)}')
    print(f'    rmse_mean     : {g(rmse)}')
    print(f'    mae_mean      : {g(mae)}')
    print(f'    rpiq_mean     : {g(rpiq)}')
    if isinstance(r2m, (int, float)) and isinstance(r2s, (int, float)):
        score = r2m - 0.5 * r2s
        print(f'    score (mean − 0.5σ): {score:+.4f}  (robustness-adjusted)')
    if args.show_folds and fold_results:
        print('-' * 78)
        print('  Per-fold R²:')
        r2s_arr = [f.get('r2', float('nan')) for f in fold_results]
        finite = [(i, r) for i, r in enumerate(r2s_arr) if r == r]
        if finite:
            best_i = max(finite, key=lambda t: t[1])[0]
            worst_i = min(finite, key=lambda t: t[1])[0]
        else:
            best_i = worst_i = -1
        for i, r2 in enumerate(r2s_arr):
            mark = ''
            if i == best_i: mark = '  ← best'
            elif i == worst_i: mark = '  ← worst'
            rmse_i = fold_results[i].get('rmse', float('nan'))
            n_test = fold_results[i].get('n_test', '—')
            print(f'    fold {i:>2}: R²={r2:+.4f}  RMSE={rmse_i:7.3f}  '
                  f'n_test={n_test}{mark}')
    print()


def show_compare(group: str, tags: list[str], args) -> None:
    """Side-by-side comparison table."""
    cols = []
    for tag in tags:
        j = load_summary(group, tag)
        if j is None:
            print(f'[warn] no summary for {group}/{tag}; skipping', file=sys.stderr)
            continue
        parsed = parse_tag(tag) or {}
        ac = j.get('across_folds', {})
        recipe = j.get('recipe', {})
        cols.append({
            'tag': tag,
            'group': group,
            'family': parsed.get('family', '?'),
            'd_model': parsed.get('d_model'),
            'num_heads': parsed.get('num_heads'),
            'num_layers': parsed.get('num_layers'),
            'variant': parsed.get('variant'),
            'r2_mean': ac.get('r2_mean'),
            'r2_std': ac.get('r2_std'),
            'rmse_mean': ac.get('rmse_mean'),
            'mae_mean': ac.get('mae_mean'),
            'rpiq_mean': ac.get('rpiq_mean'),
            'loss': recipe.get('loss_type'),
            'lr': recipe.get('lr'),
            'max_oc': recipe.get('max_oc'),
            'sampler': recipe.get('sampler_mode'),
            'augment': recipe.get('augment_train'),
            'epochs': recipe.get('num_epochs'),
            'bands_list': recipe.get('bands_list'),
            'fold_r2s': [f.get('r2', float('nan'))
                         for f in j.get('fold_results', [])],
        })
    if not cols:
        print('[error] no configs found to compare', file=sys.stderr)
        return
    if args.params:
        for c in cols:
            n, _ = count_params(c['group'], c['tag'])
            c['n_params'] = n

    # Print architecture descriptions up front so the table makes sense.
    print('=' * 78)
    print(f'COMPARISON ({len(cols)} configs in group "{group}")')
    print('=' * 78)
    for c in cols:
        desc = ARCH_DESCRIPTIONS.get(c['family'], '(no description)')
        print(f'  • {c["tag"]}')
        print(f'    {desc}')
        print()

    # Compute column widths
    col_w = max(22, max(len(c['tag']) for c in cols) + 2)
    label_w = 18

    def _row(label, values):
        print(f'{label:<{label_w}}' + ''.join(f'{v:<{col_w}}' for v in values))

    print('-' * (label_w + col_w * len(cols)))
    _row('Tag', [c['tag'] for c in cols])
    _row('Family', [c['family'] for c in cols])
    _row('d / h / L', [
        f"{c.get('d_model', '—')} / {c.get('num_heads', '—')} / {c.get('num_layers', '—')}"
        if c.get('d_model') is not None else (c.get('variant') or '—')
        for c in cols])
    if args.params:
        _row('Params', [f"{c.get('n_params'):,}" if c.get('n_params') else '—'
                        for c in cols])
    _row('Bands', [str(c.get('bands_list') or '—') for c in cols])
    _row('Loss', [str(c.get('loss') or '—') for c in cols])
    _row('LR', [str(c.get('lr') or '—') for c in cols])
    _row('max_oc', [str(c.get('max_oc') or '—') for c in cols])
    _row('Sampler', [str(c.get('sampler') or '—') for c in cols])
    _row('Augment', [str(c.get('augment') or '—') for c in cols])
    _row('Epochs', [str(c.get('epochs') or '—') for c in cols])
    print('-' * (label_w + col_w * len(cols)))
    _row('R² mean', [f"{c['r2_mean']:+.4f}" if c['r2_mean'] is not None else '—'
                      for c in cols])
    _row('R² std', [f"{c['r2_std']:.4f}" if c['r2_std'] is not None else '—'
                     for c in cols])
    _row('RMSE', [f"{c['rmse_mean']:.3f}" if c['rmse_mean'] is not None else '—'
                   for c in cols])
    _row('MAE', [f"{c['mae_mean']:.3f}" if c['mae_mean'] is not None else '—'
                  for c in cols])
    _row('RPIQ', [f"{c['rpiq_mean']:.3f}" if c['rpiq_mean'] is not None else '—'
                   for c in cols])
    print('-' * (label_w + col_w * len(cols)))
    # Per-fold rows
    max_folds = max(len(c['fold_r2s']) for c in cols)
    for i in range(max_folds):
        _row(f'fold {i:>2} R²', [
            f"{c['fold_r2s'][i]:+.4f}" if i < len(c['fold_r2s'])
            else '—' for c in cols])
    print()


# -----------------------------------------------------------------------
# Preset comparisons — the three head-to-head ablations the rebuttal needs.
# -----------------------------------------------------------------------
PRESETS = {
    'gating': {
        'description': 'Gating ablation: SGT (with GRN) vs Vanilla (without GRN). '
                       'Same architecture except the GRN block.',
        'tags': [
            'small_d128_h4_L1',                  # SGT, gated
            'vanilla_transformer_d128_h4_L1',    # Vanilla, no gate (matched d)
            'small_d64_h4_L1',                   # SGT, gated, smaller
            'vanilla_transformer_d64_h4_L1',     # Vanilla, no gate (matched d)
        ],
    },
    'cnn-frontend': {
        'description': 'CNN-frontend ablation: CNN + transformer (vanilla) vs '
                       'transformer alone (SimpleTransformer). Isolates the '
                       'value of the per-band 1×1 spatial encoder.',
        'tags': [
            'vanilla_transformer_d128_h4_L1',    # CNN + transformer, ~215k
            'vanilla_transformer_d64_h4_L1',     # CNN + transformer, ~95k
            'simpletransformer_d64_h4_L1',       # transformer alone, ~11.2M
        ],
    },
    'architecture': {
        'description': 'Full architecture sweep: every family head-to-head at '
                       'similar hyperparameters. Reveals which inductive '
                       'biases matter (gating, CNN frontend, recurrence, attention).',
        'tags': [
            'small_d128_h4_L1',                  # SGT
            'vanilla_transformer_d128_h4_L1',    # Vanilla
            'simpletransformer_d64_h4_L1',       # SimpleTransformer
            'cnnlstm_d64_h4_L1',                 # CNNLSTM
            '3dcnn_d64_h4_L1',                   # 3DCNN
            'baseline_xgb_shallow',              # XGBoost
            'baseline_rf_default',               # RandomForest
        ],
    },
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--list', action='store_true',
                   help='List all available tags grouped by sweep_group.')
    p.add_argument('--show', type=str, default=None,
                   help='Detailed view of ONE tag (requires --group).')
    p.add_argument('--compare', nargs='+', default=None,
                   help='Two or more tags for side-by-side comparison.')
    p.add_argument('--preset', choices=list(PRESETS.keys()), default=None,
                   help='Run a preset comparison: gating, cnn-frontend, or architecture.')
    p.add_argument('--group', type=str, default=None,
                   help='Sweep group (e.g. oc150, oc120_6band). Required for '
                        '--show / --compare / --preset.')
    p.add_argument('--family', type=str, default=None,
                   help='Filter --list to a specific architecture family.')
    p.add_argument('--show-folds', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Show per-fold R² breakdown (default: on).')
    p.add_argument('--params', action='store_true',
                   help='Load .pth files to report exact parameter counts. '
                        'Requires torch; slower on first call.')
    a = p.parse_args()

    if not SWEEP_DIR.exists():
        print(f'[error] {SWEEP_DIR} not found', file=sys.stderr)
        sys.exit(1)

    if a.show:
        if not a.group:
            print('[error] --show requires --group <sweep_group>', file=sys.stderr)
            sys.exit(1)
        show_one(a.group, a.show, a)
        return

    if a.preset:
        if not a.group:
            print('[error] --preset requires --group <sweep_group>', file=sys.stderr)
            sys.exit(1)
        preset = PRESETS[a.preset]
        print()
        print(f'PRESET: {a.preset}')
        print(preset['description'])
        print()
        show_compare(a.group, preset['tags'], a)
        return

    if a.compare:
        if not a.group:
            print('[error] --compare requires --group <sweep_group>', file=sys.stderr)
            sys.exit(1)
        show_compare(a.group, a.compare, a)
        return

    # Default: list mode
    rows = list_all_configs()
    if a.group:
        rows = [r for r in rows if r['group'] == a.group]
    if a.family:
        rows = [r for r in rows if r['family'] == a.family]

    by_group = defaultdict(list)
    for r in rows:
        by_group[r['group']].append(r)

    print(f'Found {len(rows)} configs across {len(by_group)} sweep group(s)')
    print(f'(SWEEP_DIR = {SWEEP_DIR})')
    print()
    for group in sorted(by_group):
        configs = by_group[group]
        print('━' * 78)
        print(f'  group: {group}  ({len(configs)} configs)')
        print('━' * 78)
        configs.sort(key=lambda c: (c['family'],
                                     c['d_model'] or 0,
                                     c['num_heads'] or 0))
        for c in configs:
            if c.get('d_model') is not None:
                dh = (f"d={c['d_model']:>4}  h={c['num_heads']}  "
                      f"L={c['num_layers']}")
            else:
                dh = c.get('variant', '—')
            print(f"    [{c['family']:<22}] {c['tag']:<42} {dh}")
        print()


if __name__ == '__main__':
    main()
