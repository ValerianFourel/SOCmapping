#!/usr/bin/env python3
"""submit_resaware_compare.py — submit the resolution-aware ablation comparison.

Runs, on the SAME 10-fold longitude-blocked splits as every other family (the
orchestrator run_folds_parallel.py owns the splits — nothing is regenerated):

  resaware_full        --branches fine_med_coarse           (the full multi-branch)
  resaware_allflat     --branches all_flat                  (resolution-NAIVE baseline)
  resaware_abl_fine    --branches fine_med_coarse --ablate-group fine
  resaware_abl_medium  ... --ablate-group medium
  resaware_abl_coarse  ... --ablate-group coarse

The existing 84k SGT (small_d32_h2_L1) is the reference — already in the sweep,
not re-run. After these finish, build the table with resaware_table.py.

Mirrors sweep_submit.build_family_sbatch exactly (run_folds_parallel orchestrator,
4-GPU, qcut sampler, D4 augmentation, log target) so results are directly
comparable; adds --branches/--ablate-group/--deterministic. hidden_size=32 →
~79k params, matched to the 84k baseline.

Run on the login node:
    python rebuttal/gpu_experiments/spatial_kfold/submit_resaware_compare.py --dry-run
    python rebuttal/gpu_experiments/spatial_kfold/submit_resaware_compare.py
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[2]                      # SOCmapping/
SBATCH_DIR = HERE / 'sbatch'
LOG_DIR = HERE / 'slurm_logs'

# (tag, branches, ablate_group)
VARIANTS = [
    ('resaware_full',       'fine_med_coarse', None),
    ('resaware_allflat',    'all_flat',        None),
    ('resaware_abl_fine',   'fine_med_coarse', 'fine'),
    ('resaware_abl_medium', 'fine_med_coarse', 'medium'),
    ('resaware_abl_coarse', 'fine_med_coarse', 'coarse'),
]


def kfold_cmd(out_dir: Path, branches: str, ablate, a) -> str:
    abl = f' --ablate-group {ablate}' if ablate else ''
    return (
        'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
        'python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py '
        f'--num-folds {a.num_folds} --num-parallel {a.num_folds} '
        f'--folds-per-gpu {a.folds_per_gpu} '
        f'--output-dir {shlex.quote(str(out_dir))} '
        '-- '
        '--model-size small --model-family resaware '
        f'--hidden_size {a.hidden_size} --num_heads {a.num_heads} --num_layers 1 '
        f'--dropout_rate {a.dropout} '
        f'--branches {branches}{abl} --deterministic '
        '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
        f'--loss_type {a.loss_type} --target_transform log '
        '--per-gpu-batch-size 256 --effective-batch-size 256 '
        f'--num-epochs {a.epochs} --seed-base {a.seed_base} '
        f'--max-oc {a.max_oc} '
        f'--split-axis {a.split_axis} --window-size {a.window_size} '
        '--sampler-mode qcut --rebalance-min-ratio 0 --augment-train '
        f'--bands-list {a.bands_list} --band-arch none --ext-reduced 8 --skip-figure'
    )


def sbatch_text(tag: str, cmd: str, a) -> str:
    log = LOG_DIR / f'resaware_{tag}_%j.out'
    venv = f'source {shlex.quote(a.venv_activate)}' if a.venv_activate else 'true'
    return f'''#!/bin/bash
#SBATCH --job-name=resaware-{tag}
#SBATCH --partition={a.partition}
#SBATCH --account={a.account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task={a.cpus_per_task}
#SBATCH --gres=gpu:4
#SBATCH --mem={a.mem}
#SBATCH --time={a.time}
#SBATCH --output={log}
#SBATCH --error={log}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "[resaware] {tag}  node=$(hostname)  job=$SLURM_JOB_ID"
{cmd}
'''


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--only', type=str, default=None,
                   help='comma-separated tags to submit (default: all 5)')
    p.add_argument('--sweep-name', default='resaware',
                   help='sub-sweep namespace under sweep/ (default resaware)')
    p.add_argument('--bands-list', default='full_extended',
                   help='full_extended (43, runs now) or full_extended_s2 (45, '
                        'needs the S2 tiles + SGT_BANDS_S2=1).')
    p.add_argument('--hidden_size', type=int, default=32)   # ~79k, ~ the 84k baseline
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--num-folds', type=int, default=10)
    p.add_argument('--folds-per-gpu', type=int, default=3)
    p.add_argument('--split-axis', default='lon')
    p.add_argument('--window-size', type=int, default=5)
    p.add_argument('--max-oc', type=float, default=150.0)
    p.add_argument('--loss-type', default='l1')
    p.add_argument('--seed-base', type=int, default=42)
    p.add_argument('--partition', default='booster')
    p.add_argument('--account', default='scifi')
    p.add_argument('--cpus-per-task', type=int, default=4)
    p.add_argument('--mem', default='0')
    p.add_argument('--time', default='02:00:00')
    p.add_argument('--venv-activate', default=str(SOC_ROOT.parent / 'venv' / 'bin' / 'activate'))
    return p.parse_args()


def main():
    a = parse()
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(',')) if a.only else None
    sweep_dir = HERE / 'sweep' / a.sweep_name
    print(f'[resaware] sweep dir: {sweep_dir}  bands={a.bands_list}  '
          f'd={a.hidden_size}  epochs={a.epochs}  axis={a.split_axis} '
          f'folds={a.num_folds}')
    submitted = []
    for tag, branches, ablate in VARIANTS:
        if only and tag not in only:
            continue
        out_dir = sweep_dir / tag
        cmd = kfold_cmd(out_dir, branches, ablate, a)
        sb = SBATCH_DIR / f'resaware_{tag}.sbatch'
        sb.write_text(sbatch_text(tag, cmd, a)); sb.chmod(0o755)
        if a.dry_run:
            print(f'[dry-run] {tag:22} branches={branches} ablate={ablate}')
            print(f'          -> {out_dir}')
            continue
        out = subprocess.run(['sbatch', str(sb)], capture_output=True, text=True)
        if out.returncode != 0:
            print(f'[submit] FAILED {tag}: {out.stderr.strip()}', file=sys.stderr)
            continue
        jid = out.stdout.strip().split()[-1]
        submitted.append((tag, jid))
        print(f'[submit] {tag:22} job={jid}  -> {out_dir}')
    if a.dry_run:
        print(f'\n[dry-run] scripts in {SBATCH_DIR}/. Re-run without --dry-run.')
        return
    if submitted:
        print(f'\n[resaware] {len(submitted)} runs queued. When done, build the table:')
        print(f'  python rebuttal/gpu_experiments/spatial_kfold/resaware_table.py '
              f'--sweep-dir {sweep_dir} --bands-list {a.bands_list}')


if __name__ == '__main__':
    main()
