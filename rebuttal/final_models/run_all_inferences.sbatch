#!/bin/bash
#SBATCH --job-name=infer-bavaria
#SBATCH --partition=booster
#SBATCH --account=scifi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=rebuttal/final_models/maps/infer_all_%j.out
#SBATCH --error=rebuttal/final_models/maps/infer_all_%j.out

# Re-run Bavaria 1mil-grid inference for every NN + tree final-model
# checkpoint, then regenerate the comparison figure. The earlier
# (pre-1393aef) maps were either all-NaN (NN, due to the wrong
# training-time dataloader) or predict-on-X=zeros constants (trees,
# same cause but silently absorbed) — both must be regenerated now
# that infer_bavaria uses MultiRasterDataset1MilMultiYears.
#
# Submit from the SOCmapping dir on the cluster:
#   sbatch rebuttal/final_models/run_all_inferences.sbatch
#
# Single GPU, sequential — ~3 min/run × ~12 runs ≈ 40 min wall.

# NOT -e: a single failing checkpoint should NOT kill the rest.
set -uo pipefail
cd /e/project1/scifi/fourel1/SGT/SOCmapping
source ../venv/bin/activate

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

mkdir -p rebuttal/final_models/maps

echo "[infer-all] node=$(hostname)  job=${SLURM_JOB_ID:-local}  gpus=$(nvidia-smi -L | wc -l)"
echo "[infer-all] cwd=$(pwd)"
echo "[infer-all] git HEAD=$(git rev-parse --short HEAD)"
echo "---"

# Auto-skip any run whose checkpoint dir is missing — lets us submit
# this once and have it fill in maps as training of slower variants
# (e.g. vanilla, sgt_6band) finishes on subsequent resubmissions.
NN_RUNS=(
    sgt_d128_h4_L1_20band
    sgt_d128_h4_L1_6band
    simpletransformer_d64_h4_L1_20band
    simpletransformer_d64_h4_L1_6band
    cnnlstm_d64_h4_L1_20band
    cnnlstm_d64_h4_L1_6band
    vanilla_transformer_d128_h4_L1_20band
    vanilla_transformer_d128_h4_L1_6band
)

TREE_RUNS=(
    xgb_shallow_20band
    xgb_shallow_6band
    rf_default_20band
    rf_default_6band
)

run_one() {
    local r="$1"
    local d="rebuttal/final_models/checkpoints/$r"
    local s="rebuttal/final_models/maps/$r/bavaria_2023_summary.json"
    if [[ ! -d "$d" ]]; then
        echo "[skip] $r — no checkpoint directory at $d (training not finished?)"
        return 0
    fi
    # Idempotency: skip runs whose summary already exists with a finite,
    # non-NaN mean. Lets a re-submission pick up where a previous time-out
    # left off, without redoing the slow ones that already finished.
    if [[ -f "$s" ]] && python -c "
import json, math, sys
try:
    m = json.load(open('$s'))['mean']
    sys.exit(0 if isinstance(m, (int, float)) and not math.isnan(m) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[skip] $r — output already exists with finite mean (idempotent)"
        return 0
    fi
    echo "=== $r ==="
    if ! python rebuttal/final_models/infer_bavaria.py --run-name "$r" --year 2023; then
        echo "[warn] $r failed (exit $?) — continuing with the next run"
    fi
    echo
}

echo "[infer-all] NN inferences (${#NN_RUNS[@]} configured)"
for r in "${NN_RUNS[@]}"; do
    run_one "$r"
done

echo "---"
echo "[infer-all] Tree inferences (${#TREE_RUNS[@]} configured)"
for r in "${TREE_RUNS[@]}"; do
    run_one "$r"
done

echo "---"
echo "[infer-all] regenerating comparison figure"
python rebuttal/final_models/compare_maps.py --year 2023 || echo "[warn] compare_maps failed"

echo "---"
echo "[infer-all] all done"
