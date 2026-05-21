#!/bin/bash
# submit_top2_100ep.sh — train + map the two top sweep configs at 100 epochs
# with a tiny random monitor holdout.
#
# Configs (from spatial_kfold sweep_ranking, 20-band stack):
#   1. simpletransformer_d64_h4_L1   (oc150, mse, was 200 epochs in sweep)
#   2. vanilla_transformer_d128_h4_L1 (oc150, l1,  was 200 epochs in sweep)
#
# Overrides vs the sweep:
#   --num-epochs   100   (cut from 200; sweep showed convergence well before)
#   --monitor-frac 0.02  (~320 random rows; best-epoch ckpt saved automatically)
#
# Run from the login node:
#   cd /e/project1/scifi/fourel1/SGT/SOCmapping
#   source ../venv/bin/activate
#   bash rebuttal/final_models/submit_top2_100ep.sh
#
# Each config queues a 4-GPU training job and a dependent 1-GPU inference job.
# Outputs:
#   checkpoints/<run-name>_20band/{final_model.pth, stats.json, config.json}
#   maps/<run-name>_20band/bavaria_2023_{predictions.parquet, summary.json, map.png}
#
# Run names are auto-suffixed with `_100ep_<loss>` so these do NOT overwrite
# any existing 60-epoch finals under the same architecture.

set -euo pipefail

SOC_ROOT="/e/project1/scifi/fourel1/SGT/SOCmapping"
VENV_ACTIVATE="${SOC_ROOT}/../venv/bin/activate"
ACCOUNT="${ACCOUNT:-scifi}"
PARTITION="${PARTITION:-booster}"
YEAR="${YEAR:-2023}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
MONITOR_FRAC="${MONITOR_FRAC:-0.02}"
BANDS_LIST="${BANDS_LIST:-full_20}"   # set to original_6 for the 6-band variant
TIME_TRAIN="${TIME_TRAIN:-02:00:00}"
TIME_INFER="${TIME_INFER:-02:00:00}"

# Derive band suffix the same way train_full.py does.
case "${BANDS_LIST}" in
    full_20)    BAND_SFX="_20band" ;;
    original_6) BAND_SFX="_6band" ;;
    *) echo "[submit] unknown BANDS_LIST=${BANDS_LIST}" >&2; exit 1 ;;
esac

if [[ "$(pwd)" != "${SOC_ROOT}" ]]; then
    echo "[submit] WARN: cwd is $(pwd) — sbatch jobs cd into ${SOC_ROOT} anyway."
fi

LOG_DIR="${SOC_ROOT}/rebuttal/final_models/slurm_logs"
mkdir -p "${LOG_DIR}"

submit_pair() {
    # $1: tag (used in job name + log name)
    # $2: base run-name (without _20band/_6band suffix — auto-appended)
    # $3: model-family
    # $4: hidden_size
    # $5: loss_type
    local tag="$1" base="$2" family="$3" d="$4" loss="$5"
    local run_name="${base}${BAND_SFX}"
    local log_train="${LOG_DIR}/final_${tag}_train_%j.out"
    local log_infer="${LOG_DIR}/final_${tag}_infer_%j.out"

    local train_jid
    train_jid=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ft-${tag}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=${TIME_TRAIN}
#SBATCH --output=${log_train}
#SBATCH --error=${log_train}
set -euo pipefail
cd ${SOC_ROOT}
source ${VENV_ACTIVATE}

echo "[final-train] tag=${tag} run_name=${base} bands=${BANDS_LIST} epochs=${NUM_EPOCHS} monitor=${MONITOR_FRAC} node=\$(hostname) job=\$SLURM_JOB_ID"

WANDB_MODE=disabled PYTHONUNBUFFERED=1 \\
accelerate launch --num_processes 4 \\
    rebuttal/final_models/train_full.py \\
    --run-name ${base} \\
    --bands-list ${BANDS_LIST} \\
    --model-family ${family} --model-size small \\
    --hidden_size ${d} --num_heads 4 --num_layers 1 --dropout_rate 0.5 \\
    --lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 \\
    --loss_type ${loss} --target_transform log --max-oc 150 \\
    --per-gpu-batch-size 256 --effective-batch-size 256 \\
    --num-epochs ${NUM_EPOCHS} --monitor-frac ${MONITOR_FRAC} \\
    --seed 42 --augment-train
EOF
)
    echo "[submit] ${tag}: train job=${train_jid}"

    local infer_jid
    infer_jid=$(sbatch --parsable --dependency=afterok:${train_jid} <<EOF
#!/bin/bash
#SBATCH --job-name=fi-${tag}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=${TIME_INFER}
#SBATCH --output=${log_infer}
#SBATCH --error=${log_infer}
set -euo pipefail
cd ${SOC_ROOT}
source ${VENV_ACTIVATE}

echo "[final-infer] tag=${tag} run_name=${run_name} year=${YEAR} node=\$(hostname) job=\$SLURM_JOB_ID"

PYTHONUNBUFFERED=1 \\
python rebuttal/final_models/infer_bavaria.py \\
    --run-name ${run_name} \\
    --year ${YEAR}
EOF
)
    echo "[submit] ${tag}: infer job=${infer_jid} (waits on ${train_jid})"
}

# Config 1 — SimpleTransformerV2 d=64, loss=MSE
submit_pair \
    "simpletransformer_100ep_mse" \
    "simpletransformer_d64_h4_L1_100ep_mse" \
    "simpletransformer" \
    64 \
    "mse"

# Config 2 — VanillaTransformer d=128, loss=L1
submit_pair \
    "vanilla_100ep_l1" \
    "vanilla_transformer_d128_h4_L1_100ep_l1" \
    "vanilla_transformer" \
    128 \
    "l1"

echo
echo "[submit] queued. Watch with:"
echo "    squeue -u \$USER"
echo "    python rebuttal/final_models/check_inference_status.py --watch"
