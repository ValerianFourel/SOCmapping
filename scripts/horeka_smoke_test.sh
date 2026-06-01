#!/bin/bash
#SBATCH --job-name=soc-smoke
#SBATCH --partition=dev_accelerated
#SBATCH --account=hk-project-CHANGEME      # ← set this to your HoreKa project
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=smoke_%j.out
#
# horeka_smoke_test.sh — one fold of small_d32_h2_L1 at 3 epochs on 1 GPU.
# Confirms that the install works end-to-end (data loads, model builds,
# training runs, predictions parquet writes). ~5 min on an A100.
#
# Usage:
#   cd $WORK/SGT/SOCmapping
#   sbatch scripts/horeka_smoke_test.sh

set -euo pipefail
source $WORK/SGT/SOCmapping/env.sh
cd $SOC_CODE_DIR

echo "[smoke] node=$(hostname)  gpus=$(nvidia-smi -L | wc -l)"
python -c 'import torch; print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'

WANDB_MODE=disabled PYTHONUNBUFFERED=1 python \
    rebuttal/gpu_experiments/spatial_kfold/run_kfold.py \
    --fold 0 --num-folds 10 \
    --model-size small --model-family sgt \
    --hidden_size 32 --num_heads 2 --num_layers 1 --dropout_rate 0.5 \
    --lr 0.0001 --lr-scheduler cosine --lr-min 1e-6 \
    --loss_type l1 --target_transform log \
    --per-gpu-batch-size 256 --effective-batch-size 256 \
    --num-epochs 3 --seed-base 42 \
    --max-oc 150 --split-axis lon --window-size 5 \
    --sampler-mode qcut --rebalance-min-ratio 0 --augment-train \
    --bands-list full_extended --band-arch two_path --ext-reduced 8 \
    --out-subdir sweep/_smoketest \
    --skip-figure

echo "[smoke] done. Check rebuttal/gpu_experiments/spatial_kfold/sweep/_smoketest/fold_0_predictions.parquet exists."
ls -l rebuttal/gpu_experiments/spatial_kfold/sweep/_smoketest/fold_0_predictions.parquet
