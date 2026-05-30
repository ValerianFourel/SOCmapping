#!/bin/bash
#SBATCH --job-name=sgt-agg
#SBATCH --partition=booster
#SBATCH --account=scifi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/e/project1/scifi/fourel1/SGT/SOCmapping/rebuttal/gpu_experiments/spatial_kfold/sweep/slurm_logs/aggregate_%j.out
#SBATCH --error=/e/project1/scifi/fourel1/SGT/SOCmapping/rebuttal/gpu_experiments/spatial_kfold/sweep/slurm_logs/aggregate_%j.out
#
# aggregate_sweep.sbatch — rerun --aggregate-only on every NN config in a sweep
# subdir that has surviving fold_*_predictions.parquet but no
# kfold_results_summary.json (i.e. the orchestrator quit early on a fold
# SIGKILL/OOM before writing the cross-fold summary).
#
# Why a Slurm job: run_kfold.py imports torch at module top, and the login
# venv on Jupiter has no torch, so --aggregate-only can't run there. The
# aggregator itself is torch-free; we just need a compute-node Python env.
# 15-min wall-time + 1 GPU is overkill but cheap and avoids partition rules.
#
# Usage:
#   sbatch aggregate_sweep.sbatch <SWEEP_SUBDIR> [extra run_kfold args]
#
# SWEEP_SUBDIR is relative to spatial_kfold/sweep/, e.g.
#   lon_w5/oc150_extband_twopath_extband
#
# All flags after SWEEP_SUBDIR are forwarded to run_kfold --aggregate-only.
# Pass the ones required by write_results' recipe manifest:
#   --split-axis {lat|lon|cluster}
#   --window-size N
#   --bands-list {full_20|original_6|full_extended}
#   --band-arch {none|two_path}     (optional, default 'none')
#   --max-oc 150                    (or whichever was used for training)
#
# Example:
#   sbatch aggregate_sweep.sbatch lon_w5/oc150_extband_twopath_extband \
#       --split-axis lon --window-size 5 \
#       --bands-list full_extended --band-arch two_path --max-oc 150

set -euo pipefail

SOC_ROOT=/e/project1/scifi/fourel1/SGT/SOCmapping
SWEEP_ROOT="$SOC_ROOT/rebuttal/gpu_experiments/spatial_kfold/sweep"

if [ $# -lt 1 ]; then
    echo "usage: sbatch $0 <SWEEP_SUBDIR> [extra run_kfold args...]" >&2
    echo "  SWEEP_SUBDIR is relative to $SWEEP_ROOT" >&2
    exit 2
fi

SUBDIR="$1"; shift
TARGET="$SWEEP_ROOT/$SUBDIR"
if [ ! -d "$TARGET" ]; then
    echo "error: $TARGET does not exist" >&2
    exit 2
fi

source /e/project1/scifi/fourel1/SGT/venv/bin/activate
cd "$SOC_ROOT"

echo "[aggregate] sweep subdir: $SUBDIR"
echo "[aggregate] extra args:   $*"
echo "[aggregate] node=$(hostname)  job=${SLURM_JOB_ID:-no-slurm}"
echo "---"

n_skip_done=0
n_skip_empty=0
n_attempted=0
n_ok=0
n_fail=0

for d in "$TARGET"/*/; do
    name=$(basename "$d")
    case "$name" in baseline_*) continue ;; esac
    if [ -f "$d/kfold_results_summary.json" ]; then
        n_skip_done=$((n_skip_done + 1))
        continue
    fi
    if ! compgen -G "$d/fold_*_predictions.parquet" >/dev/null; then
        n_skip_empty=$((n_skip_empty + 1))
        continue
    fi
    n_attempted=$((n_attempted + 1))
    echo ">>> aggregating $name"
    if WANDB_MODE=disabled PYTHONUNBUFFERED=1 python \
            rebuttal/gpu_experiments/spatial_kfold/run_kfold.py \
            --aggregate-only \
            --num-folds 10 \
            --out-subdir "sweep/$SUBDIR/$name" \
            "$@"; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
        echo "  ^ failed (rc=$?), continuing"
    fi
done

echo "---"
echo "[aggregate] skipped (already summarized): $n_skip_done"
echo "[aggregate] skipped (no predictions):     $n_skip_empty"
echo "[aggregate] attempted:                    $n_attempted"
echo "[aggregate] ok:                           $n_ok"
echo "[aggregate] failed:                       $n_fail"
echo
echo "[aggregate] total summaries now in $SUBDIR:"
find "$TARGET" -maxdepth 2 -name kfold_results_summary.json | wc -l
