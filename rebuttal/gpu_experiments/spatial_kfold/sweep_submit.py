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
    --time 00:45:00          Slurm wall-time per job (Juwels Booster is busy).
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

# Single source of truth for --bands-list → tag-suffix mapping.
sys.path.insert(0, str(HERE))
from band_subsets import band_suffix as _band_suffix  # noqa: E402

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
# Cross-architecture grid: one canonical config per sibling family, all
# trained under the same spatial-kfold pipeline as SGT. Each entry:
# (family, d_model, num_heads, num_layers). For families that don't use a
# given hyperparameter (e.g., 3DCNN ignores d_model and num_heads), we still
# pass a value so the tag is well-formed; --num_heads and --num_layers are
# silently dropped by Small3DCNN's constructor.
# ---------------------------------------------------------------------------
FAMILY_GRID: list[tuple[str, int, int, int, float]] = [
    # (family, d_model_or_hidden, num_heads, num_layers, dropout)
    # The classic sibling-architecture grid. Excludes vanilla_transformer —
    # vanilla has its own VANILLA_GRID below because it answers a different
    # question (SGT-minus-GRN ablation, not "does another architecture
    # family work?"). Controlled by --families / --no-families.
    ('3dcnn',              64, 4, 1, 0.5),
    ('cnnlstm',            64, 4, 1, 0.5),
    ('simpletransformer',  64, 4, 1, 0.5),
]


# ---------------------------------------------------------------------------
# Vanilla transformer ablation grid.
# The "vanilla" baseline is SimpleSGT with the Gated Residual Network
# (GRN) block replaced by a plain Linear projection — same spatial
# encoder, same positional embedding, same transformer encoder, same
# head. The only architectural delta is the gating, so any R² gap
# between SGT and a parameter-matched vanilla baseline is directly
# attributable to the gating mechanism. Controlled by --vanilla /
# --no-vanilla / --vanilla-only flags (parallel to --baselines /
# --families).
# ---------------------------------------------------------------------------
VANILLA_GRID: list[tuple[str, int, int, int, float]] = [
    # (family, d_model, num_heads, num_layers, dropout)
    ('vanilla_transformer', 64,  4, 1, 0.5),   # ~95k, paired with sgt small_d64_h4 (~165k)
    ('vanilla_transformer', 128, 4, 1, 0.5),   # ~215k, paired with sgt small_d128_h4 (~363k)
]


# ---------------------------------------------------------------------------
# CNN-frontend ablation: SimpleTransformer at a sweep of d_model values, so
# we get a parameter-count curve for "transformer alone, NO CNN frontend" to
# compare against the vanilla "CNN + transformer" curve at matched params.
#
# At d=64,h=4,L=1 SimpleTransformer is ~11.2M params (20-band) — totally
# dominated by the input embedding that linearly maps the flattened
# (C, T, H, W) cube to d_model. Smaller d_model shrinks that embedding
# proportionally, so this grid lets us plot R² vs param count for the
# transformer-alone family and compare it against the (CNN + transformer)
# vanilla curve at matched param count.
#
# Controlled by --simpletransformer-ablation /
# --simpletransformer-ablation-only flags (parallel to --vanilla).
# ---------------------------------------------------------------------------
SIMPLETRANSFORMER_ABLATION_GRID: list[tuple[str, int, int, int, float]] = [
    # (family, d_model, num_heads, num_layers, dropout)
    # NOTE: SimpleTransformerV2 ignores d_model and forces it to C*H*W,
    # so all of these end up at ~11M params (20-band) / ~1.7M (6-band).
    # Kept for completeness but USE LIGHTWEIGHT_TRANSFORMER_GRID for a
    # parameter-controllable transformer-alone comparison.
    ('simpletransformer',  16, 2, 1, 0.5),
    ('simpletransformer',  32, 4, 1, 0.5),
    ('simpletransformer',  64, 4, 1, 0.5),
    ('simpletransformer', 128, 4, 1, 0.5),
]


# ---------------------------------------------------------------------------
# Lightweight transformer ablation: TRUE transformer-only baseline (no CNN
# spatial encoder) at parameter counts matched to vanilla_transformer.
#
# Whereas SimpleTransformerV2 silently forces d_model = C*H*W (yielding
# ~11.2M params at 20 bands regardless of --hidden_size), the
# LightweightTransformer class respects --hidden_size and --num_layers,
# so this grid genuinely sweeps the 85k-370k parameter range — directly
# comparable to vanilla_transformer's 95k-215k. Together with VANILLA_GRID
# this gives a clean CNN-frontend ablation at matched scale:
#   SGT (CNN + GRN + Transformer)     vs vanilla (CNN + Transformer)  → gating
#   vanilla (CNN + Transformer)       vs lightweight (Transformer)    → CNN frontend
#
# Controlled by --lightweight-transformer / --lightweight-transformer-only.
# ---------------------------------------------------------------------------
LIGHTWEIGHT_TRANSFORMER_GRID: list[tuple[str, int, int, int, float]] = [
    # (family, d_model, num_heads, num_layers, dropout)
    ('lightweight_transformer',  64, 4, 1, 0.5),   # ~85k  (paired with vanilla d=64)
    ('lightweight_transformer',  96, 4, 1, 0.5),   # ~155k
    ('lightweight_transformer', 128, 4, 1, 0.5),   # ~240k (paired with vanilla d=128)
    ('lightweight_transformer',  64, 4, 2, 0.5),   # ~150k
    ('lightweight_transformer', 128, 4, 2, 0.5),   # ~370k
]


def family_tag_for(family: str, d: int, h: int, L: int) -> str:
    return f'{family}_d{d}_h{h}_L{L}'


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


def _effective_sweep_name(args) -> str:
    """sweep_name with the --bands-list suffix appended if non-default.

    Default behaviour (full_20) leaves the sweep_name unchanged so the
    existing 20-band sweep directory tree is preserved. original_6 runs
    get a "_6band" suffix automatically so they don't collide with
    20-band outputs at the same --sweep-name."""
    base = args.sweep_name
    suf = _band_suffix(getattr(args, 'bands_list', 'full_20'))
    if suf == '_20band':         # default — keep base unchanged
        return base
    return (base + suf) if base else suf.lstrip('_')


def sweep_root(args) -> Path:
    """Effective output root for this sweep run. Honors --sweep-name and
    --bands-list to namespace per-config outputs."""
    eff = _effective_sweep_name(args)
    return SWEEP_DIR / eff if eff else SWEEP_DIR


def out_subdir_arg(args, tag: str) -> str:
    """The --out-subdir value to pass to run_kfold.py. Includes sweep-name."""
    eff = _effective_sweep_name(args)
    if eff:
        return f'sweep/{eff}/{tag}'
    return f'sweep/{tag}'


def build_sbatch(tag: str, variant: str, d: int, h: int, L: int, args) -> str:
    out_dir_abs = sweep_root(args) / tag
    # Per-fold console logs stay in a flat slurm_logs/ dir; tag-prefixed
    # so namespaced sweeps don't collide on log filenames.
    name_prefix = f'{args.sweep_name.replace("/", "_")}_' if args.sweep_name else ''
    log_path = LOG_DIR / f'{name_prefix}{tag}_%j.out'
    cmd = (
        'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
        'python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py '
        f'--num-folds {args.num_folds} --num-parallel {args.num_folds} '
        f'--folds-per-gpu {args.folds_per_gpu} '
        f'--output-dir {shlex.quote(str(out_dir_abs))} '
        '-- '
        f'--model-size {variant} '
        f'--hidden_size {d} --num_heads {h} --num_layers {L} '
        '--dropout_rate 0.5 '
        f'--lr {args.lr} --lr-scheduler cosine --lr-min 1e-6 '
        f'--loss_type {args.loss_type} '
        f'--loss-alpha {args.loss_alpha} --chi2-weight {args.chi2_weight} '
        '--target_transform log '
        '--per-gpu-batch-size 256 --effective-batch-size 256 '
        f'--num-epochs {args.epochs} --seed-base {args.seed_base} '
        f'--max-oc {args.max_oc} '
        f'--split-axis {args.split_axis} --window-size {args.window_size} '
        '--sampler-mode qcut --rebalance-min-ratio 0 '
        '--augment-train '
        f'--out-subdir {out_subdir_arg(args, tag)} '
        f'--bands-list {args.bands_list} '
        f'--band-arch {args.band_arch} --ext-reduced {args.ext_reduced} '
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
#SBATCH --mem={args.mem}
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


def build_family_sbatch(tag: str, family: str, d: int, h: int, L: int,
                         dropout: float, args) -> str:
    """One sbatch per cross-architecture config (3DCNN, CNNLSTM, etc.).

    Uses the same run_folds_parallel.py orchestrator as SGT — only the
    new --model-family flag changes the model factory in run_kfold.py.
    Everything else (10 folds, 3 folds/GPU, log target, no rebalancing,
    D4 augmentation) is identical, so results are directly comparable.
    """
    out_dir_abs = sweep_root(args) / tag
    name_prefix = f'{args.sweep_name.replace("/", "_")}_' if args.sweep_name else ''
    log_path = LOG_DIR / f'{name_prefix}{tag}_%j.out'
    cmd = (
        'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
        'python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py '
        f'--num-folds {args.num_folds} --num-parallel {args.num_folds} '
        f'--folds-per-gpu {args.folds_per_gpu} '
        f'--output-dir {shlex.quote(str(out_dir_abs))} '
        '-- '
        '--model-size small '
        f'--model-family {family} '
        f'--hidden_size {d} --num_heads {h} --num_layers {L} '
        f'--dropout_rate {dropout} '
        f'--lr {args.lr} --lr-scheduler cosine --lr-min 1e-6 '
        f'--loss_type {args.loss_type} '
        f'--loss-alpha {args.loss_alpha} --chi2-weight {args.chi2_weight} '
        '--target_transform log '
        '--per-gpu-batch-size 256 --effective-batch-size 256 '
        f'--num-epochs {args.epochs} --seed-base {args.seed_base} '
        f'--max-oc {args.max_oc} '
        f'--split-axis {args.split_axis} --window-size {args.window_size} '
        '--sampler-mode qcut --rebalance-min-ratio 0 '
        '--augment-train '
        f'--out-subdir {out_subdir_arg(args, tag)} '
        f'--bands-list {args.bands_list} '
        f'--band-arch {args.band_arch} --ext-reduced {args.ext_reduced} '
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
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv_activate}

echo "[sweep] tag={tag}  family={family}  d={d} h={h} L={L} dropout={dropout}"
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
    eff_name = _effective_sweep_name(args)
    name_prefix = f'{eff_name.replace("/", "_")}_' if eff_name else ''
    log_path = LOG_DIR / f'{name_prefix}baselines_%j.out'
    venv_activate = (
        f'source {shlex.quote(str(args.venv_activate))}'
        if args.venv_activate else 'true  # no venv activation requested'
    )
    # Each baseline writes under OUT_DIR/<output_subdir>/baseline_<model>_<suffix>/
    output_subdir = (f'sweep/{eff_name}' if eff_name else 'sweep')

    invocations = []
    for model, suffix, extra in BASELINE_GRID:
        extra_str = ' '.join(shlex.quote(a) for a in extra)
        invocations.append(
            f'echo "[baselines] === {model}_{suffix} ==="\n'
            f'WANDB_MODE=disabled PYTHONUNBUFFERED=1 '
            f'python rebuttal/gpu_experiments/spatial_kfold/run_baselines.py '
            f'--models {model} --tag-suffix {suffix} --num-folds {args.num_folds} '
            f'--max-oc {args.max_oc} --target-transform log --device cuda '
            f'--split-axis {args.split_axis} --window-size {args.window_size} '
            f'--seed-base {args.seed_base} '
            f'--output-subdir {shlex.quote(output_subdir)} '
            f'--bands-list {args.bands_list} '
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
#SBATCH --time={args.time}
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
    p.add_argument('--num-folds', type=int, default=10,
                   help='Number of spatial folds per config (e.g. 5 or 10), '
                        'forwarded to run_kfold/run_baselines and the parallel '
                        'orchestrator. Default 10.')
    p.add_argument('--split-axis', type=str, default='lat',
                   choices=['lat', 'lon', 'cluster'],
                   help='Spatial-CV fold geometry forwarded to run_kfold/run_baselines: '
                        '"lat" (south↔north), "lon" (west↔east), or "cluster" '
                        '(equal-size balanced K-Means). Use a distinct --sweep-name '
                        '(e.g. oc150_cluster) so results do not collide. For cluster, '
                        '--seed-base is forwarded to both NN and baseline jobs so they '
                        'share identical folds.')
    p.add_argument('--window-size', type=int, default=5,
                   help='Spatial window edge length forwarded to every job '
                        '(model H×W and dataset crop). Default 5 (config). '
                        'Use 7 or 9 for the larger-context experiment; no data '
                        'regeneration is needed (tiles are far larger).')
    p.add_argument('--time', type=str, default='00:45:00',
                   help='Slurm wall-time per job. Default 45 min (Juwels '
                        'Booster is busy — keep jobs short). Bump only if '
                        'epochs > 100 or you switch to a heavier architecture.')
    p.add_argument('--folds-per-gpu', type=int, default=2,
                   help='Fold subprocesses packed onto each GPU, forwarded to '
                        'run_folds_parallel.py. Default 2. NOTE: 43-band runs '
                        'OOM at 3 (the runner warns about this), and 3-way GPU '
                        'time-slicing also makes each fold ~3x slower, so a '
                        'short job times out with half its folds unfinished — '
                        'which is exactly the failure this default avoids. Drop '
                        'to 1 for the heaviest models (d128) if 2 still OOMs.')
    p.add_argument('--mem', type=str, default='0',
                   help='Slurm --mem per job. Default "0" = all memory on the '
                        'node. The 43-band sweep was OOM-killed '
                        '(State=OUT_OF_MEMORY) because every concurrent fold '
                        'process stages the full raster stack into host RAM '
                        'while the template requested no --mem; "0" grabs the '
                        'whole node so the concurrent loaders fit.')
    p.add_argument('--band-arch', type=str, default='none',
                   choices=['none', 'two_path'],
                   help='Forwarded to run_kfold. "two_path" wraps the inner '
                        'model with a small Conv2d that reduces the extended '
                        'bands (idx 20..) — only active when --bands-list '
                        'full_extended is also set.')
    p.add_argument('--ext-reduced', type=int, default=8,
                   help='Forwarded to run_kfold; channels after the extended-'
                        'band reduction when --band-arch two_path is active. '
                        'Default 8 → 20 core + 8 reduced = 28 effective channels.')
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
    p.add_argument('--families', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Also submit the cross-architecture grid '
                        '(3DCNN, CNNLSTM, SimpleTransformer; FAMILY_GRID in this '
                        'file). Each family runs under the same kfold pipeline '
                        'as SGT for direct comparison. Default off; pass --families '
                        'to enable. --families-only submits just those.')
    p.add_argument('--families-only', action='store_true',
                   help='Submit only the cross-architecture grid, skip SGT and baselines.')
    p.add_argument('--vanilla', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Also submit the vanilla-transformer ablation grid '
                        '(SimpleSGT minus the GRN; VANILLA_GRID in this file). '
                        'Default off; pass --vanilla to enable, '
                        '--vanilla-only to submit just those.')
    p.add_argument('--vanilla-only', action='store_true',
                   help='Submit only the vanilla-transformer ablation grid, '
                        'skip SGT, families, and baselines.')
    p.add_argument('--simpletransformer-ablation',
                   action=argparse.BooleanOptionalAction, default=False,
                   help='Also submit the SimpleTransformer parameter-count '
                        'sweep (SIMPLETRANSFORMER_ABLATION_GRID; '
                        'd_model ∈ {16, 32, 64, 128}). Pairs with the '
                        'vanilla curve to give a "CNN+transformer vs '
                        'transformer-alone" comparison at matched param '
                        'counts. Default off.')
    p.add_argument('--simpletransformer-ablation-only', action='store_true',
                   help='Submit only the SimpleTransformer ablation grid, '
                        'skip everything else.')
    p.add_argument('--lightweight-transformer',
                   action=argparse.BooleanOptionalAction, default=False,
                   help='Also submit LightweightTransformer grid — true '
                        'transformer-only baseline at d ∈ {64,96,128} × '
                        'L ∈ {1,2}, 85k-370k params. Pairs with VANILLA_GRID '
                        'for the CNN-frontend ablation at matched scale. '
                        'Default off.')
    p.add_argument('--lightweight-transformer-only', action='store_true',
                   help='Submit ONLY the LightweightTransformer grid, '
                        'skip SGT/families/vanilla/simpletransformer/baselines.')
    p.add_argument('--loss-type', type=str, default='l1',
                   choices=['l1', 'mse', 'chi2', 'composite_l1', 'composite_l2'],
                   help='Training loss for neural-network configs (SGT + families). '
                        'composite_l1/composite_l2 add a Pearson chi-square term '
                        '(see train.py:_composite_loss). Default l1. Ignored by '
                        'the tree-baseline bundle.')
    p.add_argument('--loss-alpha', type=float, default=1.0,
                   help='Weight on the base term in composite losses (default 1.0).')
    p.add_argument('--chi2-weight', type=float, default=0.1,
                   help='Weight on the chi-square term in composite losses (default 0.1).')
    p.add_argument('--bands-list', type=str, default='full_20',
                   choices=['full_20', 'original_6', 'full_extended'],
                   help='Covariate subset (full_20 = revision expansion; '
                        'original_6 = original-paper subset). Auto-appends '
                        '"_6band" to the sweep-name namespace so 6-band and '
                        '20-band runs do not collide. Passed through to '
                        'run_kfold.py / run_baselines.py.')
    a = p.parse_args()

    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    submitted: list[tuple[str, str]] = []

    # ---- SGT configs ------------------------------------------------------
    if (not a.baselines_only and not a.families_only and not a.vanilla_only
            and not a.simpletransformer_ablation_only
            and not a.lightweight_transformer_only):
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

    # ---- Cross-architecture family grid ---------------------------------
    if ((a.families and not a.vanilla_only and not a.baselines_only
            and not a.simpletransformer_ablation_only
            and not a.lightweight_transformer_only)
            or a.families_only):
        print(f'\n[sweep] cross-architecture grid: {len(FAMILY_GRID)} configs')
        for family, d, h, L, dropout in FAMILY_GRID:
            tag = family_tag_for(family, d, h, L)
            script_text = build_family_sbatch(tag, family, d, h, L, dropout, a)
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
            print(f'[sweep] submitted {tag:>22}  job_id={jid}')

    # ---- Vanilla-transformer ablation grid ------------------------------
    if ((a.vanilla and not a.baselines_only and not a.families_only
            and not a.simpletransformer_ablation_only
            and not a.lightweight_transformer_only)
            or a.vanilla_only):
        print(f'\n[sweep] vanilla-transformer ablation grid: '
              f'{len(VANILLA_GRID)} configs (SimpleSGT minus GRN)')
        for family, d, h, L, dropout in VANILLA_GRID:
            tag = family_tag_for(family, d, h, L)
            script_text = build_family_sbatch(tag, family, d, h, L, dropout, a)
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
            print(f'[sweep] submitted {tag:>26}  job_id={jid}')

    # ---- SimpleTransformer parameter-count ablation grid ---------------
    # CNN-frontend ablation: transformer-alone curve at multiple d_model
    # values for a fair, parameter-matched comparison against the vanilla
    # (CNN+transformer) curve.
    if ((a.simpletransformer_ablation and not a.baselines_only
            and not a.families_only and not a.vanilla_only)
            or a.simpletransformer_ablation_only):
        print(f'\n[sweep] SimpleTransformer parameter ablation grid: '
              f'{len(SIMPLETRANSFORMER_ABLATION_GRID)} configs '
              f'(transformer-alone at varying d_model)')
        for family, d, h, L, dropout in SIMPLETRANSFORMER_ABLATION_GRID:
            if d % h != 0:
                print(f'[sweep] skip {family}_d{d}_h{h} '
                      f'(hidden_size must be divisible by num_heads)',
                      file=sys.stderr)
                continue
            tag = family_tag_for(family, d, h, L)
            script_text = build_family_sbatch(tag, family, d, h, L, dropout, a)
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
            print(f'[sweep] submitted {tag:>30}  job_id={jid}')

    # ---- LightweightTransformer grid (true transformer-only baseline) ---
    # Pairs with VANILLA_GRID for the CNN-frontend ablation at matched
    # parameter scale (LightweightTransformer ≈ 85k-370k, vanilla ≈
    # 95k-215k; SimpleTransformerV2 ≈ 11M which is too unfair).
    if ((a.lightweight_transformer and not a.baselines_only
            and not a.families_only and not a.vanilla_only
            and not a.simpletransformer_ablation_only)
            or a.lightweight_transformer_only):
        print(f'\n[sweep] LightweightTransformer grid: '
              f'{len(LIGHTWEIGHT_TRANSFORMER_GRID)} configs '
              f'(transformer-alone, controlled d_model)')
        for family, d, h, L, dropout in LIGHTWEIGHT_TRANSFORMER_GRID:
            if d % h != 0:
                print(f'[sweep] skip {family}_d{d}_h{h} '
                      f'(hidden_size must be divisible by num_heads)',
                      file=sys.stderr)
                continue
            tag = family_tag_for(family, d, h, L)
            script_text = build_family_sbatch(tag, family, d, h, L, dropout, a)
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
            print(f'[sweep] submitted {tag:>34}  job_id={jid}')

    # ---- Baseline bundle (one sbatch with all RF/XGB configs) ------------
    if ((a.baselines and not a.families_only and not a.vanilla_only
            and not a.simpletransformer_ablation_only
            and not a.lightweight_transformer_only)
            or a.baselines_only):
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
