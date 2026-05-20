#!/bin/bash
#SBATCH --job-name=lwt-longtrain
#SBATCH --partition=booster
#SBATCH --account=scifi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=rebuttal/gpu_experiments/spatial_kfold/sweep/slurm_logs/lwt_longtrain_%j.out
#SBATCH --error=rebuttal/gpu_experiments/spatial_kfold/sweep/slurm_logs/lwt_longtrain_%j.out

# Defensive long-training run for LightweightTransformer.
# Motivation: the 30-epoch screening showed Lightweight peaking at
# median epoch 20/30 (per inspect_sweep) — i.e. still improving when
# the budget cut off. To rule out "the gap to Vanilla is just an
# undertraining artifact" we re-train ONE config (d=128, h=4, L=1)
# at 200 epochs with everything else identical.
#
# Output namespace: sweep/oc150_longtrain/lightweight_transformer_d128_h4_L1/
# So inspect_run --group oc150_longtrain picks this up alongside (or
# in place of) the 30-epoch oc150 entry without overwriting it.

set -euo pipefail
cd /e/project1/scifi/fourel1/SGT/SOCmapping
source ../venv/bin/activate

mkdir -p rebuttal/gpu_experiments/spatial_kfold/sweep/slurm_logs

echo "[lwt-longtrain] node=$(hostname)  job=$SLURM_JOB_ID  gpus=$(nvidia-smi -L | wc -l)"
echo "[lwt-longtrain] cwd=$(pwd)"
echo "[lwt-longtrain] git HEAD=$(git rev-parse --short HEAD)"
echo "---"

WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
python rebuttal/gpu_experiments/spatial_kfold/run_folds_parallel.py \
    --num-folds 10 --num-parallel 10 --folds-per-gpu 3 \
    --output-dir rebuttal/gpu_experiments/spatial_kfold/sweep/oc150_longtrain/lightweight_transformer_d128_h4_L1 \
    -- \
    --model-size small \
    --model-family lightweight_transformer \
    --hidden_size 128 --num_heads 4 --num_layers 1 \
    --dropout_rate 0.5 \
    --lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 \
    --loss_type l1 --target_transform log \
    --per-gpu-batch-size 256 --effective-batch-size 256 \
    --num-epochs 200 --seed-base 42 \
    --max-oc 150 \
    --sampler-mode qcut --rebalance-min-ratio 0 \
    --augment-train \
    --out-subdir sweep/oc150_longtrain/lightweight_transformer_d128_h4_L1 \
    --bands-list full_20 \
    --skip-figure

echo "---"
echo "[lwt-longtrain] done."
