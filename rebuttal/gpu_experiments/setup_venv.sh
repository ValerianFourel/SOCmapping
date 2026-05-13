#!/usr/bin/env bash
# setup_venv.sh — create .venv for the Geoderma rebuttal GPU experiments.
#
# Run this once on the host where the GPU experiments will execute
# (typically a Runpod / Lambda Labs / on-prem box). It:
#   1. Picks a CUDA-matched PyTorch wheel (or CPU if no GPU detected)
#   2. Creates .venv next to this script
#   3. Installs PyTorch + everything in requirements.txt
#   4. Sanity-checks Model B loading + a forward pass on dummy input
#
# Usage:
#   cd /path/to/SOCmapping/rebuttal/gpu_experiments
#   bash setup_venv.sh
#
# Result: ./.venv/  (≈ 3 GB; gitignored by SOCmapping/.gitignore via the
#                    "*.so" + "__pycache__/" rules and project-level .venv
#                    convention — confirm in your local .gitignore)
#
# Override CUDA detection:
#   CUDA_TAG=cu118 bash setup_venv.sh         # force cu118 wheels
#   CUDA_TAG=cpu   bash setup_venv.sh         # force CPU wheels
#
# Override Python:
#   PYTHON=/usr/bin/python3.10 bash setup_venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 1. Pick Python
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for cand in python3.10 python3.11 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
    done
fi
if [ -z "$PYTHON" ] || ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: no python3.X interpreter found. Install python3.10 or later."
    exit 1
fi
echo "Using interpreter: $($PYTHON --version) at $(command -v "$PYTHON")"

# ---------------------------------------------------------------------------
# 2. Detect CUDA → pick a torch wheel index
#    (recommendations validated against PyTorch's wheel-index table)
# ---------------------------------------------------------------------------
CUDA_TAG="${CUDA_TAG:-}"
TORCH_VERSION_DEFAULT="2.0.1"
TORCHVISION_VERSION_DEFAULT="0.15.2"

if [ -z "$CUDA_TAG" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Read driver-reported CUDA version. Example output line:
        #   | NVIDIA-SMI 535.183.01    Driver Version: 535.183.01    CUDA Version: 12.2 |
        DRIVER_CUDA=$(nvidia-smi 2>/dev/null \
            | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' \
            | head -n1 | awk '{print $3}')
        if [ -n "$DRIVER_CUDA" ]; then
            MAJ=${DRIVER_CUDA%%.*}; MIN=${DRIVER_CUDA##*.}
            echo "Detected driver CUDA: $DRIVER_CUDA"
            if [ "$MAJ" -ge 12 ]; then
                CUDA_TAG="cu121"
                TORCH_VERSION_DEFAULT="2.2.2"
                TORCHVISION_VERSION_DEFAULT="0.17.2"
            elif [ "$MAJ" -eq 11 ] && [ "$MIN" -ge 8 ]; then
                CUDA_TAG="cu118"
                TORCH_VERSION_DEFAULT="2.0.1"
                TORCHVISION_VERSION_DEFAULT="0.15.2"
            else
                CUDA_TAG="cu117"
                TORCH_VERSION_DEFAULT="2.0.1"
                TORCHVISION_VERSION_DEFAULT="0.15.2"
            fi
        else
            CUDA_TAG="cpu"
        fi
    else
        echo "No nvidia-smi → installing CPU-only torch."
        CUDA_TAG="cpu"
    fi
fi
TORCH_VERSION="${TORCH_VERSION:-$TORCH_VERSION_DEFAULT}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-$TORCHVISION_VERSION_DEFAULT}"
echo "Target: torch==$TORCH_VERSION + torchvision==$TORCHVISION_VERSION  ($CUDA_TAG)"

# ---------------------------------------------------------------------------
# 3. Create venv
# ---------------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "venv already exists at $VENV_DIR — re-using. To rebuild, remove it first."
else
    echo "Creating venv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 4. PyTorch
# ---------------------------------------------------------------------------
if [ "$CUDA_TAG" = "cpu" ]; then
    pip install --index-url "https://download.pytorch.org/whl/cpu" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
else
    pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
fi

# ---------------------------------------------------------------------------
# 5. Other deps
# ---------------------------------------------------------------------------
pip install -r "$REQ_FILE"

# ---------------------------------------------------------------------------
# 6. Smoke test: load Model B and run a forward pass
# ---------------------------------------------------------------------------
echo
echo "=== Smoke test: import torch, load EnhancedSGT, forward pass ==="
python - <<'PY'
import sys, torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  cuda_version={torch.version.cuda}")
sys.path.insert(0, "/home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer")
try:
    from EnhancedSGT import EnhancedSGT
    m = EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5, d_model=128,
                    num_heads=4, dropout=0.3, num_encoder_layers=3, expansion_factor=4)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"EnhancedSGT instantiated ({n_params:,} trainable params)")
    x = torch.randn(2, 6, 5, 5, 5)
    if torch.cuda.is_available():
        m = m.cuda(); x = x.cuda()
    y = m(x)
    print(f"forward pass OK, output shape: {tuple(y.shape)} device={y.device}")
except FileNotFoundError as e:
    # Path-not-found is acceptable on a Runpod that doesn't have the project
    # tree mounted at the same absolute path — only flag torch import failures.
    print("WARN: SGT source tree not at /home/valerian/SGTPublication/... — skipping arch smoke test.")
    print(f"      ({type(e).__name__}: {e})")
PY

echo
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Then run Experiment 2 first (~3 GPU h), Experiment 1 second (~15 GPU h):"
echo "  python rebuttal/gpu_experiments/uncertainty/mc_dropout_inference.py"
echo "  python rebuttal/gpu_experiments/uncertainty/plot_uncertainty.py"
echo "  python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py"
