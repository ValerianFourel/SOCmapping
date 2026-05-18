#!/usr/bin/env python3
"""
sweep_submit.py — architecture sweep for the SGT k-fold rebuttal experiment.

Submits one Slurm batch job per (hidden_size, num_heads, num_layers) config.
Each job runs run_folds_parallel.py at the chosen architecture for a fixed
SHORT schedule (default 80 epochs) so we can screen the architecture space
cheaply, then pick the top 1-2 configs for a full 300-epoch run.

Per-config outputs go to:
    rebuttal/gpu_experiments/spatial_kfold/sweep/<tag>/

where <tag> = "d<H>_h<HEADS>_L<LAYERS>" (e.g. d64_h2_L2).

Usage on the LOGIN node (not inside an active srun shell — sbatch jobs must
be submitted from where you have queue access):

    cd /e/project1/scifi/fourel1/SGT/SOCmapping
    source venv/bin/activate
    python rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py

Useful flags:
    --dry-run                Print sbatch commands without submitting.
    --epochs 80              Override the screening epoch count.
    --time 02:00:00          Slurm wall-time per job.
    --account scifi --partition booster
    --grid d48_h2_L1,d64_h2_L2,...   Comma-separated tags to submit only some.

After submission:
    squeue -u $USER                              # watch queue
    sweep_summarize.py                           # rank results once jobs finish
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[2]      # SOCmapping/
SWEEP_DIR = HERE / 'sweep'
SBATCH_DIR = SWEEP_DIR / 'sbatch'
LOG_DIR = SWEEP_DIR / 'slurm_logs'

# ---------------------------------------------------------------------------
# Default architecture grid. Each entry: (hidden_size, num_heads, num_layers).
# Constraint: hidden_size % num_heads == 0 and head_dim >= 16 ideally.
# All entries use --model-size big so --num_layers actually takes effect
# (SimpleSGT/'small' is hardcoded to 1 layer and silently ignores --num_layers).
# ---------------------------------------------------------------------------
DEFAULT_GRID: list[tuple[int, int, int]] = [
    # (d_model, num_heads, num_layers)
    (48,  2, 1),    # tiny floor
    (48,  2, 2),
    (64,  2, 1),
    (64,  2, 2),
    (64,  2, 3),
    (64,  4, 2),
    (96,  4, 2),
    (128, 4, 2),    # mid-size
]


def tag_for(d: int, h: int, L: int) -> str:
    return f'd{d}_h{h}_L{L}'


def build_sbatch(tag: str, d: int, h: int, L: int, args) -> str:
    out_dir_abs = SWEEP_DIR / tag
    log_path = LOG_DIR / f'{tag}_%j.out'
    cmd = (
        'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
        'python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py '
        f'--num-folds 10 --num-parallel 10 --folds-per-gpu 3 '
        f'--output-dir {shlex.quote(str(out_dir_abs))} '
        '-- '
        '--model-size big '
        f'--hidden_size {d} --num_heads {h} --num_layers {L} '
        '--dropout_rate 0.5 '
        f'--lr {args.lr} --lr-scheduler cosine --lr-min 1e-6 '
        '--loss_type l1 --target_transform log '
        '--per-gpu-batch-size 256 --effective-batch-size 256 '
        f'--num-epochs {args.epochs} --seed-base {args.seed_base} '
        f'--max-oc {args.max_oc} '
        '--sampler-mode qcut --rebalance-min-ratio 0 '
        '--augment-train '
        f'--out-subdir sweep/{tag} '
        '--skip-figure'
    )
    venv_activate = (
        f'source {shlex.quote(str(args.venv_activate))}'
        if args.venv_activate else 'true  # no venv activation requested'
    )
    return f'''#!/bin/bash
#SBATCH --job-name=sgt-{tag}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time={args.time}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv_activate}

echo "[sweep] tag={tag}  d={d} h={h} L={L}"
echo "[sweep] node=$(hostname)  job=$SLURM_JOB_ID  gpus=$(nvidia-smi -L | wc -l)"
echo "[sweep] cwd=$(pwd)"
echo "[sweep] cmd:"
echo "  {cmd}"
echo "---"

{cmd}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Write sbatch scripts but do not submit.')
    p.add_argument('--epochs', type=int, default=80,
                   help='Per-config training epochs for screening (default 80).')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-oc', type=float, default=120.0)
    p.add_argument('--seed-base', type=int, default=42)
    p.add_argument('--time', type=str, default='02:00:00',
                   help='Slurm wall-time per job. Bump if epochs > 100.')
    p.add_argument('--partition', type=str, default='booster')
    p.add_argument('--account', type=str, default='scifi')
    p.add_argument('--venv-activate', type=str,
                   default=str(SOC_ROOT.parent / 'venv' / 'bin' / 'activate'),
                   help='Path to a venv activate script to source inside each job. '
                        'Default: ../venv/bin/activate relative to SOCmapping. '
                        'Pass empty string to skip.')
    p.add_argument('--grid', type=str, default=None,
                   help='Comma-separated config tags to submit (e.g. "d64_h2_L2,d96_h4_L2"). '
                        'Default: all entries in DEFAULT_GRID.')
    a = p.parse_args()

    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if a.grid:
        wanted = set(a.grid.split(','))
        grid = [(d, h, L) for d, h, L in DEFAULT_GRID if tag_for(d, h, L) in wanted]
        missing = wanted - {tag_for(d, h, L) for d, h, L in DEFAULT_GRID}
        if missing:
            print(f'[sweep] WARNING: unknown tags ignored: {sorted(missing)}',
                  file=sys.stderr)
    else:
        grid = list(DEFAULT_GRID)

    print(f'[sweep] {len(grid)} config(s) to submit; epochs={a.epochs}  time={a.time}')
    print(f'[sweep] output root: {SWEEP_DIR}')

    submitted = []
    for d, h, L in grid:
        if d % h != 0:
            print(f'[sweep] skip d={d} h={h} (hidden_size must be divisible by num_heads)',
                  file=sys.stderr)
            continue
        tag = tag_for(d, h, L)
        script_text = build_sbatch(tag, d, h, L, a)
        script_path = SBATCH_DIR / f'{tag}.sbatch'
        script_path.write_text(script_text)
        script_path.chmod(0o755)

        if a.dry_run:
            print(f'[dry-run] would submit {script_path}')
            continue

        out = subprocess.run(['sbatch', str(script_path)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f'[sweep] sbatch FAILED for {tag}: {out.stderr.strip()}',
                  file=sys.stderr)
            continue
        # Parse "Submitted batch job 12345"
        jid = out.stdout.strip().split()[-1]
        submitted.append((tag, jid))
        print(f'[sweep] submitted {tag:>14}  job_id={jid}')

    if a.dry_run:
        print(f'\n[sweep] dry-run complete. Scripts in {SBATCH_DIR}/. '
              f'Re-run without --dry-run to submit.')
        return

    if submitted:
        print(f'\n[sweep] {len(submitted)} jobs submitted. Watch with:')
        print(f'  squeue -u $USER')
        print(f'[sweep] When done, rank with:')
        print(f'  python {HERE / "sweep_summarize.py"}')
    else:
        print('[sweep] No jobs submitted.')


if __name__ == '__main__':
    main()
