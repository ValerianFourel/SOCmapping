#!/usr/bin/env python3
"""
sbatch_mse200_4arch.py — final-experiment submission for the 4 transformer
families at MSE loss × 200 epochs.

Motivation
==========
The rebuttal's spatial-CV sweep used L1 loss for most configs (with
composite_l2 for some SGT/Vanilla variants at max_oc=150) and a default
30-epoch screening budget. The 200-epoch long-train Lightweight run
(commit 0ab25fb) revealed that *training budget* alone shifts Lightweight
from R² -0.06 to +0.15, but with diagnostic overfitting after the peak
suggesting the model converged at the architectural ceiling.

To close the loop, this script runs ALL FOUR transformer-family models
at the same elevated training schedule (MSE loss, 200 epochs) so the
final architecture-recommendation table compares converged ceilings,
not training-budget artefacts.

The four configs
================
1. SGT          (CNN + GRN + Transformer, 363k params)  small_d128_h4_L1
2. Vanilla      (CNN + Transformer,       215k)         vanilla_transformer_d128_h4_L1
3. Lightweight  (Transformer only,        240k)         lightweight_transformer_d128_h4_L1
4. SimpleTrans  (Transformer only,        11.2M)        simpletransformer_d64_h4_L1

Shared settings
===============
- loss_type:     mse              (was l1 or composite_l2 in earlier runs)
- num_epochs:    200              (was 30 in screening, 60 in finals)
- max_oc:        150
- target_xform:  log
- bands_list:    full_20
- sampler-mode:  qcut (default)
- augment-train: on
- d_model/h/L:   each model's canonical config
- Output namespace: sweep/oc150_mse200/<tag>/  (each tag is the model's
                    canonical tag; matches inspect_run --group oc150_mse200)

Wall time
=========
200 epochs at 10-fold × 3-folds-per-GPU on 4 GPUs ≈ 4-6 hours wall.
Total cluster time across the 4 jobs ≈ 16-24 hours; can run in parallel.

Usage
=====
    # Submit all 4
    python rebuttal/gpu_experiments/spatial_kfold/sbatch_mse200_4arch.py

    # Just generate sbatch files, don't submit
    python rebuttal/gpu_experiments/spatial_kfold/sbatch_mse200_4arch.py --dry-run

    # Submit only SGT and Vanilla
    python rebuttal/gpu_experiments/spatial_kfold/sbatch_mse200_4arch.py \\
        --only sgt,vanilla

After the runs complete
=======================
    python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py \\
        --preset transformers --group oc150_mse200 --params

    # Or to compare against the L1/composite_l2 baseline:
    python rebuttal/gpu_experiments/spatial_kfold/sweep_summarize.py | \\
        grep -E "oc150_mse200|oc150_vanilla|oc150 .*small_d128|oc150 .*simple"
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[2]
SBATCH_DIR = HERE / 'sweep' / 'sbatch'
LOG_DIR = HERE / 'sweep' / 'slurm_logs'

# ---------------------------------------------------------------------------
# The four configs. Each tuple:
#   (short_name, family, hidden_size, num_heads, num_layers, dropout)
# `short_name` is the --only key. `family` is what _build_model dispatches on.
# Tag name is auto-derived as `<family>_d{h}_h{H}_L{L}` for non-sgt families,
# and `small_d{H}_h{HEADS}_L{L}` for sgt (matches sweep_submit.tag_for).
# ---------------------------------------------------------------------------
CONFIGS = [
    ('sgt',          'sgt',                      128, 4, 1, 0.5),
    ('vanilla',      'vanilla_transformer',      128, 4, 1, 0.5),
    ('lightweight',  'lightweight_transformer',  128, 4, 1, 0.5),
    ('simpletrans',  'simpletransformer',         64, 4, 1, 0.5),
]


def tag_for(family: str, d: int, h: int, L: int) -> str:
    if family == 'sgt':
        return f'small_d{d}_h{h}_L{L}'
    return f'{family}_d{d}_h{h}_L{L}'


def build_sbatch(short: str, family: str, d: int, h: int, L: int,
                  dropout: float, args) -> str:
    tag = tag_for(family, d, h, L)
    out_dir = HERE / 'sweep' / args.sweep_name / tag
    log_path = LOG_DIR / f'{args.sweep_name}_{tag}_%j.out'
    venv_activate = (f'source {shlex.quote(str(args.venv_activate))}'
                     if args.venv_activate else 'true  # no venv activation')
    # SGT uses --model-family sgt and --model-size small.
    # All other families use --model-family <family> and --model-size small
    # (ignored by their constructors but harmless).
    model_size = 'small'
    cmd = (
        'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
        'python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py '
        f'--num-folds 10 --num-parallel 10 --folds-per-gpu 3 '
        f'--output-dir {shlex.quote(str(out_dir))} '
        '-- '
        f'--model-size {model_size} '
        f'--model-family {family} '
        f'--hidden_size {d} --num_heads {h} --num_layers {L} '
        f'--dropout_rate {dropout} '
        f'--lr {args.lr} --lr-scheduler cosine --lr-min 1e-6 '
        f'--loss_type {args.loss_type} '
        '--target_transform log '
        '--per-gpu-batch-size 256 --effective-batch-size 256 '
        f'--num-epochs {args.epochs} --seed-base {args.seed_base} '
        f'--max-oc {args.max_oc} '
        '--sampler-mode qcut --rebalance-min-ratio 0 '
        '--augment-train '
        f'--out-subdir sweep/{args.sweep_name}/{tag} '
        f'--bands-list {args.bands_list} '
        '--skip-figure'
    )
    return f'''#!/bin/bash
#SBATCH --job-name=mse200-{short}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time={args.time}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

# Generated by sbatch_mse200_4arch.py for the MSE × 200-epoch final
# transformer comparison.  Architecture: {short} (family={family},
# d={d}, h={h}, L={L}, dropout={dropout}).  All four scripts share the
# sweep namespace "{args.sweep_name}" so inspect_run --group {args.sweep_name}
# picks them up as a coherent block.

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv_activate}

mkdir -p {shlex.quote(str(LOG_DIR))}

echo "[mse200] short={short}  family={family}  d={d} h={h} L={L}  dropout={dropout}"
echo "[mse200] node=$(hostname)  job=$SLURM_JOB_ID  gpus=$(nvidia-smi -L | wc -l)"
echo "[mse200] git HEAD=$(git rev-parse --short HEAD)"
echo "---"

{cmd}

echo "---"
echo "[mse200] {short} done."
'''


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Write sbatch files but do not submit.')
    p.add_argument('--only', type=str, default=None,
                   help='Comma-separated short names to submit (sgt,vanilla,'
                        'lightweight,simpletrans). Default: all 4.')
    p.add_argument('--sweep-name', type=str, default='oc150_mse200',
                   help='Output namespace under sweep/. Default oc150_mse200.')
    p.add_argument('--loss-type', type=str, default='mse',
                   help='Training loss. Default mse.')
    p.add_argument('--epochs', type=int, default=200,
                   help='Per-config training epochs. Default 200.')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-oc', type=float, default=150.0)
    p.add_argument('--seed-base', type=int, default=42)
    p.add_argument('--bands-list', type=str, default='full_20',
                   choices=['full_20', 'original_6'])
    p.add_argument('--time', type=str, default='06:00:00',
                   help='Slurm wall-time per config. Default 06:00:00 — '
                        '200 epochs × 10 folds at 3 folds/GPU on 4 GPUs '
                        'is typically 4-5h; 6h gives headroom.')
    p.add_argument('--partition', type=str, default='booster')
    p.add_argument('--account', type=str, default='scifi')
    p.add_argument('--venv-activate', type=str,
                   default=str(SOC_ROOT.parent / 'venv' / 'bin' / 'activate'))
    return p.parse_args()


def main():
    a = parse()
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    want = set(s.strip() for s in a.only.split(',')) if a.only else None
    configs = [c for c in CONFIGS if (want is None or c[0] in want)]
    unknown = (want or set()) - {c[0] for c in CONFIGS}
    if unknown:
        print(f'[mse200] unknown --only names: {sorted(unknown)}', file=sys.stderr)
        print(f'         valid: {[c[0] for c in CONFIGS]}', file=sys.stderr)
        sys.exit(1)

    print(f'[mse200] {len(configs)} configs to submit:')
    print(f'         sweep namespace: sweep/{a.sweep_name}/')
    print(f'         loss={a.loss_type}  epochs={a.epochs}  max_oc={a.max_oc}')
    print(f'         bands={a.bands_list}  time={a.time}')
    print()

    submitted: list[tuple[str, str, str]] = []
    for short, family, d, h, L, dropout in configs:
        tag = tag_for(family, d, h, L)
        sbatch_text = build_sbatch(short, family, d, h, L, dropout, a)
        sbatch_path = SBATCH_DIR / f'{a.sweep_name}_{tag}.sbatch'
        sbatch_path.write_text(sbatch_text)
        sbatch_path.chmod(0o755)
        print(f'[mse200] {short:<14} → {sbatch_path}')

        if a.dry_run:
            print(f'  [dry-run] would submit {sbatch_path}')
            continue

        out = subprocess.run(['sbatch', str(sbatch_path)],
                              capture_output=True, text=True)
        if out.returncode != 0:
            print(f'  sbatch FAILED for {short}: {out.stderr.strip()}',
                  file=sys.stderr)
            continue
        jid = out.stdout.strip().split()[-1]
        submitted.append((short, tag, jid))
        print(f'  submitted: job_id={jid}')

    if submitted:
        print()
        print(f'[mse200] {len(submitted)} jobs queued. Monitor:')
        print(f'  squeue -u $USER | grep mse200')
        print(f'[mse200] After completion, view the scoreboard:')
        print(f'  python rebuttal/gpu_experiments/spatial_kfold/inspect_run.py '
              f'\\\n      --preset transformers --group {a.sweep_name} --params')


if __name__ == '__main__':
    main()
