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
DEFAULT_GRID: list[tuple[str, int, int, int]] = [
    # (variant, d_model, num_heads, num_layers)
    # All entries are SimpleSGT ('small'). The known-good config from
    # manual training is (--model-size small, d=64, h=4, --max-oc 90),
    # which reached R² ≈ 0.2 transiently (now actually saved after the
    # train.py best-state fix, commit b5c1cac). This grid sweeps the
    # immediate neighborhood. SimpleSGT hardcodes 1 transformer layer
    # and ignores --num_layers, so all entries have L=1.
    #
    # Constraint: d_model % num_heads == 0. With h=4 the smallest valid
    # head_dim (=d/h) is 8 at d=32, which is borderline; included as a
    # capacity floor.
    ('small',  32, 2, 1),     # tiny floor (head_dim=16)
    ('small',  32, 4, 1),     # tiny floor (head_dim=8)
    ('small',  48, 2, 1),     # head_dim=24
    ('small',  48, 4, 1),     # head_dim=12
    ('small',  64, 2, 1),     # head_dim=32
    ('small',  64, 4, 1),     # ← KNOWN-GOOD (R² ≈ 0.2 at max-oc 90)
    ('small',  96, 4, 1),     # head_dim=24
    ('small', 128, 4, 1),     # head_dim=32
]


# ---------------------------------------------------------------------------
# Baseline grid — tree ensembles on the SAME 10-fold splits.
# Each entry: (model, tag_suffix, extra_args_list_passed_to_run_baselines).
# Bundled into ONE sbatch (--gres=gpu:1) so the 80-feature per-band-stats
# extraction (~3 min) only runs once and is cached for subsequent configs.
# ---------------------------------------------------------------------------
BASELINE_GRID: list[tuple[str, str, list[str]]] = [
    # XGBoost — vary depth × n_estimators × learning rate
    ('xgb', 'default',  ['--xgb-n-estimators', '2000', '--xgb-max-depth', '6',
                          '--xgb-lr', '0.05']),
    ('xgb', 'shallow',  ['--xgb-n-estimators', '2000', '--xgb-max-depth', '4',
                          '--xgb-lr', '0.05']),
    ('xgb', 'deep',     ['--xgb-n-estimators', '1000', '--xgb-max-depth', '8',
                          '--xgb-lr', '0.05']),
    ('xgb', 'fast',     ['--xgb-n-estimators', '500',  '--xgb-max-depth', '6',
                          '--xgb-lr', '0.1']),
    # Random Forest — vary depth × n_estimators
    ('rf',  'default',  ['--rf-n-estimators', '500',   '--rf-max-depth', '0']),
    ('rf',  'shallow',  ['--rf-n-estimators', '500',   '--rf-max-depth', '8']),
    ('rf',  'deep',     ['--rf-n-estimators', '1000',  '--rf-max-depth', '0']),
]


def tag_for(variant: str, d: int, h: int, L: int) -> str:
    # 'big' tags keep the legacy d<H>_h<HEADS>_L<LAYERS> form so old summaries
    # remain parseable; 'small' tags get a 'small_' prefix.
    base = f'd{d}_h{h}_L{L}'
    return base if variant == 'big' else f'small_{base}'


def sweep_root(args) -> Path:
    """Effective output root for this sweep run. Honors --sweep-name to
    namespace max-oc sensitivity runs (sweep/oc90, sweep/oc120, ...)."""
    return SWEEP_DIR / args.sweep_name if args.sweep_name else SWEEP_DIR


def out_subdir_arg(args, tag: str) -> str:
    """The --out-subdir value to pass to run_kfold.py. Includes sweep-name."""
    if args.sweep_name:
        return f'sweep/{args.sweep_name}/{tag}'
    return f'sweep/{tag}'


def build_sbatch(tag: str, variant: str, d: int, h: int, L: int, args) -> str:
    out_dir_abs = sweep_root(args) / tag
    # Per-fold console logs stay in a flat slurm_logs/ dir; tag-prefixed
    # so namespaced sweeps don't collide on log filenames.
    name_prefix = f'{args.sweep_name}_' if args.sweep_name else ''
    log_path = LOG_DIR / f'{name_prefix}{tag}_%j.out'
    cmd = (
        'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
        'python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py '
        f'--num-folds 10 --num-parallel 10 --folds-per-gpu 3 '
        f'--output-dir {shlex.quote(str(out_dir_abs))} '
        '-- '
        f'--model-size {variant} '
        f'--hidden_size {d} --num_heads {h} --num_layers {L} '
        '--dropout_rate 0.5 '
        f'--lr {args.lr} --lr-scheduler cosine --lr-min 1e-6 '
        '--loss_type l1 --target_transform log '
        '--per-gpu-batch-size 256 --effective-batch-size 256 '
        f'--num-epochs {args.epochs} --seed-base {args.seed_base} '
        f'--max-oc {args.max_oc} '
        '--sampler-mode qcut --rebalance-min-ratio 0 '
        '--augment-train '
        f'--out-subdir {out_subdir_arg(args, tag)} '
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


def build_baseline_sbatch(args) -> str:
    """One sbatch script that runs every BASELINE_GRID config in sequence.

    All baselines share the same per-band-statistics feature extraction
    (~3 min on 14.7k samples). run_baselines.py caches that to a .npz on
    the shared filesystem, so only the first config in the bundle pays
    the I/O cost. Each subsequent config just loads the cache and fits
    its tree ensemble (~30s-2min depending on size).

    1 GPU is enough — XGBoost-GPU and cuML-RF each use a single device.
    """
    name_prefix = f'{args.sweep_name}_' if args.sweep_name else ''
    log_path = LOG_DIR / f'{name_prefix}baselines_%j.out'
    venv_activate = (
        f'source {shlex.quote(str(args.venv_activate))}'
        if args.venv_activate else 'true  # no venv activation requested'
    )
    # Each baseline writes under OUT_DIR/<output_subdir>/baseline_<model>_<suffix>/
    output_subdir = (f'sweep/{args.sweep_name}' if args.sweep_name
                     else 'sweep')

    invocations = []
    for model, suffix, extra in BASELINE_GRID:
        extra_str = ' '.join(shlex.quote(a) for a in extra)
        invocations.append(
            f'echo "[baselines] === {model}_{suffix} ==="\n'
            f'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
            f'python rebuttal/gpu_experiments/spatial_kfold/run_baselines.py '
            f'--models {model} --tag-suffix {suffix} '
            f'--max-oc {args.max_oc} --target-transform log --device cuda '
            f'--output-subdir {shlex.quote(output_subdir)} '
            f'{extra_str}'
        )
    body = '\n\n'.join(invocations)

    return f'''#!/bin/bash
#SBATCH --job-name=sgt-baselines
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output={log_path}
#SBATCH --error={log_path}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv_activate}

echo "[baselines] node=$(hostname)  job=$SLURM_JOB_ID  gpus=$(nvidia-smi -L | wc -l)"
echo "[baselines] {len(BASELINE_GRID)} configs to run (XGB + RF variants)"
echo "[baselines] feature cache: shared across configs at same --max-oc"
echo "---"

{body}

echo "---"
echo "[baselines] all configs complete."
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Write sbatch scripts but do not submit.')
    p.add_argument('--epochs', type=int, default=30,
                   help='Per-config training epochs for screening (default 30). '
                        'Earlier sweep diagnostic showed peak epochs cluster at '
                        '4-15, so 30 is plenty; 80 was waste.')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-oc', type=float, default=90.0,
                   help='Default 90 matches the known-good manual run. '
                        'Sweep this separately (try 80, 90, 100, 120) once an '
                        'architecture is locked in.')
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
                   help='Comma-separated SGT config tags to submit. '
                        'Default: all entries in DEFAULT_GRID.')
    p.add_argument('--baselines', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Also submit the bundled baseline job (RF + XGB variants). '
                        'Pass --no-baselines to skip; --baselines-only to submit just those.')
    p.add_argument('--baselines-only', action='store_true',
                   help='Submit only the baseline bundle, skip SGT configs.')
    p.add_argument('--sweep-name', type=str, default='',
                   help='Namespace outputs under sweep/<sweep-name>/ instead of '
                        'sweep/ directly. Use for max-oc sensitivity: e.g. '
                        '--max-oc 90 --sweep-name oc90, --max-oc 120 '
                        '--sweep-name oc120. Each run keeps its own results; '
                        'sweep_summarize.py walks all sub-sweeps recursively.')
    a = p.parse_args()

    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    submitted: list[tuple[str, str]] = []

    # ---- SGT configs ------------------------------------------------------
    if not a.baselines_only:
        if a.grid:
            wanted = set(a.grid.split(','))
            grid = [(v, d, h, L) for v, d, h, L in DEFAULT_GRID
                    if tag_for(v, d, h, L) in wanted]
            missing = wanted - {tag_for(v, d, h, L) for v, d, h, L in DEFAULT_GRID}
            if missing:
                print(f'[sweep] WARNING: unknown tags ignored: {sorted(missing)}',
                      file=sys.stderr)
        else:
            grid = list(DEFAULT_GRID)

        print(f'[sweep] {len(grid)} SGT config(s) to submit; '
              f'epochs={a.epochs}  time={a.time}')
        print(f'[sweep] output root: {SWEEP_DIR}')

        for variant, d, h, L in grid:
            if d % h != 0:
                print(f'[sweep] skip d={d} h={h} '
                      f'(hidden_size must be divisible by num_heads)',
                      file=sys.stderr)
                continue
            tag = tag_for(variant, d, h, L)
            script_text = build_sbatch(tag, variant, d, h, L, a)
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
            jid = out.stdout.strip().split()[-1]
            submitted.append((tag, jid))
            print(f'[sweep] submitted {tag:>18}  job_id={jid}')

    # ---- Baseline bundle (one sbatch with all RF/XGB configs) ------------
    if a.baselines or a.baselines_only:
        b_tags = [f'baseline_{m}_{s}' for m, s, _ in BASELINE_GRID]
        print(f'\n[sweep] baseline bundle: {len(BASELINE_GRID)} configs '
              f'({", ".join(b_tags)})')
        baseline_script = SBATCH_DIR / 'baselines.sbatch'
        baseline_script.write_text(build_baseline_sbatch(a))
        baseline_script.chmod(0o755)

        if a.dry_run:
            print(f'[dry-run] would submit {baseline_script}')
        else:
            out = subprocess.run(['sbatch', str(baseline_script)],
                                 capture_output=True, text=True)
            if out.returncode != 0:
                print(f'[sweep] baseline sbatch FAILED: {out.stderr.strip()}',
                      file=sys.stderr)
            else:
                jid = out.stdout.strip().split()[-1]
                submitted.append(('baselines (bundle)', jid))
                print(f'[sweep] submitted {"baselines (bundle)":>18}  job_id={jid}')

    # ---- Summary ----------------------------------------------------------
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
