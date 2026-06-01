#!/bin/bash
# horeka_setup.sh — first-time install of SOCmapping on KIT HoreKa.
#
# Run from $WORK (NOT $HOME — quota is tight there). Edit the ACCOUNT and
# the $WORK target if your project layout differs.
#
# After this script finishes you'll have:
#   $WORK/SOCmapping/             ← code, branch bestrun-bands
#   $WORK/SOCmapping/Data/        ← 43-band rasters + LUCAS samples (from HF)
#   $WORK/venv/                   ← Python venv with pinned deps
#
# Then run scripts/horeka_smoke_test.sh to verify the install before
# launching a sweep.
#
# Usage:
#   cd $WORK
#   bash <(curl -fsSL https://raw.githubusercontent.com/ValerianFourel/SOCmapping/bestrun-bands/scripts/horeka_setup.sh)
# or, after cloning:
#   bash SOCmapping/scripts/horeka_setup.sh

set -euo pipefail

# -------- Site-specific knobs ------------------------------------------
REPO_URL="https://github.com/ValerianFourel/SOCmapping.git"
BRANCH="bestrun-bands"
HF_DATASET="ValerianFourel/SOCmappingRastersAndSoilSamples"
WORK_ROOT="${WORK:-$HOME/work}"           # HoreKa exposes $WORK; fall back to $HOME/work
PROJECT_DIR="$WORK_ROOT/SGT"               # mirrors the Jupiter layout: SGT/{SOCmapping,Data,venv}
PYTHON_MODULE="python/3.11"                # adjust to whichever HoreKa offers (module avail python)
CUDA_MODULE="devel/cuda/12.4"              # match A100 driver — module avail devel/cuda
COMPILER_MODULE="compiler/gnu/13"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
TORCH_PINS="torch==2.5.1 torchvision==0.20.1"

# -------- 0. Module loads ----------------------------------------------
echo "[setup] Loading modules…"
module purge
module load "$COMPILER_MODULE"
module load "$CUDA_MODULE"
module load "$PYTHON_MODULE"
module list

# -------- 1. Layout ----------------------------------------------------
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# -------- 2. Clone repo -----------------------------------------------
if [ ! -d SOCmapping/.git ]; then
    echo "[setup] Cloning $REPO_URL → $PROJECT_DIR/SOCmapping"
    git clone --branch "$BRANCH" --single-branch "$REPO_URL"
else
    echo "[setup] SOCmapping/ already exists — pulling $BRANCH"
    (cd SOCmapping && git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull --ff-only)
fi

# -------- 3. Python venv -----------------------------------------------
if [ ! -d venv ]; then
    echo "[setup] Creating venv at $PROJECT_DIR/venv"
    python -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# -------- 4. Torch (CUDA-matched), then the rest -----------------------
echo "[setup] Installing torch from $TORCH_INDEX"
pip install $TORCH_PINS --index-url "$TORCH_INDEX"

echo "[setup] Installing remaining deps from SOCmapping/requirements.txt"
pip install -r SOCmapping/requirements.txt

# -------- 5. HuggingFace login + dataset pull --------------------------
mkdir -p SOCmapping/Data
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "[setup] Run 'huggingface-cli login' once with your HF token, then re-run this script."
    echo "        (You can skip this step if the dataset is public.)"
fi

echo "[setup] Pulling $HF_DATASET → SOCmapping/Data (this is the big step, can take hours)"
python - <<EOF
from huggingface_hub import snapshot_download
import os
target = os.path.join("$PROJECT_DIR", "SOCmapping", "Data")
snapshot_download(repo_id="$HF_DATASET", repo_type="dataset", local_dir=target,
                  local_dir_use_symlinks=False, max_workers=8)
print(f"Done → {target}")
EOF

# -------- 6. Environment variables you'll want in jobs -----------------
cat >"$PROJECT_DIR/SOCmapping/env.sh" <<EOF
# Source this in every Slurm job. It points the path-resolver at $PROJECT_DIR.
export SOC_PROJECT_ROOT="$PROJECT_DIR"
export SOC_CODE_DIR="$PROJECT_DIR/SOCmapping"
export SOC_DATA_DIR="$PROJECT_DIR/SOCmapping/Data"
export SOC_REBUTTAL_DIR="$PROJECT_DIR/SOCmapping/rebuttal"
export PATH="$PROJECT_DIR/venv/bin:\$PATH"
EOF

echo
echo "[setup] Done."
echo "[setup] Quick verify:"
echo "  source $PROJECT_DIR/SOCmapping/env.sh"
echo "  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "  ls $PROJECT_DIR/SOCmapping/Data/OC_LUCAS_LFU_LfL_Coordinates_v2 | head"
echo
echo "[setup] Next: edit rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py"
echo "        defaults for --partition (=accelerated) and --account (=hk-project-…),"
echo "        then run scripts/horeka_smoke_test.sh."
