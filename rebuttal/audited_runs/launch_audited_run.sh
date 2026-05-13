#!/usr/bin/env bash
# launch_audited_run.sh — fire a rebuttal-audited SGT training run.
#
# Usage:
#     bash launch_audited_run.sh [big|small] [num_gpus] [num_runs]
#
# Defaults: big × 4 GPUs × 4 runs (= Model A v2 reproduction).
#
# All hyperparameters except --num_processes, --per-gpu-batch-size, and
# --num-runs are locked by the --rebuttal flag in train.py. Output goes
# to rebuttal/audited_runs/SGT_<size>_audited_<ts>_<git_sha>/.
set -euo pipefail

MODEL_SIZE="${1:-big}"
NUM_GPUS="${2:-4}"
NUM_RUNS="${3:-4}"

case "$MODEL_SIZE" in
    big|small) ;;
    *) echo "Usage: $0 [big|small] [num_gpus] [num_runs]"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGT_DIR="$SCRIPT_DIR/../../SpatiotemporalGatedTransformer"
cd "$SGT_DIR"

echo "=========================================================="
echo "  Audited SGT training run"
echo "  Model size:   $MODEL_SIZE"
echo "  GPUs:         $NUM_GPUS"
echo "  Runs:         $NUM_RUNS"
echo "  Working dir:  $(pwd)"
echo "  Git SHA:      $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "=========================================================="

accelerate launch --num_processes "$NUM_GPUS" train.py \
    --rebuttal \
    --model-size "$MODEL_SIZE" \
    --num-runs "$NUM_RUNS"
