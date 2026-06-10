# jupiter_env.sh — JUPITER (Jülich) activation for Slurm jobs.
#
# Use with the sweep launcher:
#   sweep_submit.py --venv-activate <repo>/scripts/jupiter_env.sh ...
#
# It is *sourced* inside the batch job, which runs under `set -euo pipefail`
# in a non-login shell where `module` is often undefined. So: bootstrap Lmod
# if needed, relax nounset (module/venv init scripts touch unbound vars), then
# load the JSC software stack and the project venv.
#
# JUPITER's venv runs on a module-provided Python, so `module load … Python …`
# is REQUIRED at runtime or the venv python fails with `libpython3.13.so …`.
#
# Override the venv location if your layout differs:
#   export SGT_VENV=/path/to/venv
set +u
if ! command -v module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash 2>/dev/null \
    || source /etc/profile.d/lmod.sh 2>/dev/null \
    || true
fi
module load Stages/2026 GCC Python CUDA
source "${SGT_VENV:-/e/project1/scifi/fourel1/SGT/venv}/bin/activate"
