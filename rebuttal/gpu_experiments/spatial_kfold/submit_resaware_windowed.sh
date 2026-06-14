#!/usr/bin/env bash
# Submit the native-resolution ResolutionAwareNet spatial-CV ablation on the
# windowed sentinel2 dataset (ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel2).
#
# Usage:
#   bash submit_resaware_windowed.sh horeka      # KIT HoreKa
#   bash submit_resaware_windowed.sh jupiter     # JSC JUPITER
#   bash submit_resaware_windowed.sh local       # no Slurm, run sequentially here
#
# Step 0 (once): pull the dataset from HF to $WINDOWS (set below).
#   hf download ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel2 \
#       --repo-type dataset --include 'windows/lucas/*' --local-dir "$DATAROOT"
set -euo pipefail
CLUSTER="${1:-local}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- where the windowed dataset lives after the HF pull -----------------------
DATAROOT="${WINDOWS:-$HOME/sgt-sentinel2}"          # override with WINDOWS=...
WINDOWS_DIR="$DATAROOT/windows/lucas"
OUTROOT="${OUTROOT:-$HERE/runs/resaware_native}"
EPOCHS="${EPOCHS:-100}"; FOLDS="${FOLDS:-10}"; WIN="${WIN:-11}"; TB="${TB:-5}"; OC="${OC:-150}"
BATCH="${BATCH:-256}"; TT="${TT:-log}"   # big batch (tiny net) + log target

# --- the ablation: 4 branch modes + 3 group ablations of the full model -------
#   tag                 branches            ablate-group
VARIANTS=(
  "fine               fine               -"
  "fine_med           fine_med           -"
  "fine_med_coarse    fine_med_coarse    -"
  "all_flat           all_flat           -"
  "abl_fine           fine_med_coarse    fine"
  "abl_medium         fine_med_coarse    medium"
  "abl_coarse         fine_med_coarse    coarse"
)

run_one() {  # tag branches ablate
  local tag="$1" br="$2" abl="$3"
  local args=(--data-root "$WINDOWS_DIR" --branches "$br" --max-oc "$OC"
              --num-folds "$FOLDS" --split-axis lon --window-size "$WIN"
              --time-before "$TB" --epochs "$EPOCHS" --batch-size "$BATCH"
              --target-transform "$TT" --seed 42 --out "$OUTROOT/$tag")
  [ "$abl" != "-" ] && args+=(--ablate-group "$abl")
  echo ">>> $tag : python train_resaware_windowed.py ${args[*]}"
  case "$CLUSTER" in
    horeka)
      # HoreKa has no single runtime env script (horeka_setup.sh is one-time);
      # load modules + venv inside the job. Override SGT_VENV if your layout differs.
      HK_VENV="${SGT_VENV:-$WORK/SGT/venv}"
      sbatch -p accelerated -A hk-project-p0026831 --gres=gpu:1 \
             --cpus-per-task 12 --time 08:00:00 -J "rsaw_$tag" \
             --wrap "module purge && module load compiler/gnu/13 devel/cuda/12.4 && source $HK_VENV/bin/activate && cd $HERE && python train_resaware_windowed.py ${args[*]}" ;;
    jupiter)
      sbatch -p booster -A scifi --gres=gpu:1 --cpus-per-task 4 --time 08:00:00 \
             -J "rsaw_$tag" \
             --wrap "cd $HERE && source $HERE/../../../scripts/jupiter_env.sh && python train_resaware_windowed.py ${args[*]}" ;;
    local)
      ( cd "$HERE" && python train_resaware_windowed.py "${args[@]}" ) ;;
    *) echo "unknown cluster: $CLUSTER"; exit 1 ;;
  esac
}

[ -d "$WINDOWS_DIR" ] || { echo "ERROR: $WINDOWS_DIR not found — run the HF pull first (see header)."; exit 1; }
for v in "${VARIANTS[@]}"; do run_one $v; done
echo "submitted ${#VARIANTS[@]} variants -> $OUTROOT/  (collect with: python resaware_table.py $OUTROOT)"
