#!/usr/bin/env bash
# reorg_sweep_lat_w5.sh — one-off: archive the existing latitude/window-5
# spatial-CV runs under sweep/lat_w5/ so the new longitude/window-9 runs
# (sweep/lon_w9/...) stay cleanly separated.
#
# Run ON THE CLUSTER, from anywhere (paths resolve from this script's location):
#     bash rebuttal/gpu_experiments/spatial_kfold/reorg_sweep_lat_w5.sh           # dry-run (default)
#     bash rebuttal/gpu_experiments/spatial_kfold/reorg_sweep_lat_w5.sh --apply    # actually move
#
# Safe: dry-run unless --apply; idempotent; never moves the infra/output items
# in KEEP[]. sweep_summarize.py picks the archived runs back up automatically
# (recursive glob; group label becomes "lat_w5/<oldgroup>").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="$HERE/sweep"
DEST="$SWEEP/lat_w5"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

# Items at the sweep root that are NOT runs — never move these.
KEEP=(lat_w5 lon_w9 sbatch slurm_logs baseline_features __pycache__
      sweep_ranking.md sweep_ranking.json)

is_kept() { local x="$1" k; for k in "${KEEP[@]}"; do [ "$x" = "$k" ] && return 0; done; return 1; }

[ -d "$SWEEP" ] || { echo "ERROR: no sweep dir at $SWEEP" >&2; exit 1; }
cd "$SWEEP"
echo "sweep dir : $SWEEP"
echo "mode      : $([ "$APPLY" = 1 ] && echo APPLY || echo DRY-RUN)"
echo

[ "$APPLY" = 1 ] && mkdir -p "$DEST"
moved=0
for x in *; do
  [ -e "$x" ] || continue
  if is_kept "$x"; then echo "  keep        $x"; continue; fi
  moved=$((moved + 1))
  if [ "$APPLY" = 1 ]; then
    mv "$x" "$DEST/" && echo "  moved       $x  ->  lat_w5/$x"
  else
    echo "  WOULD MOVE  $x  ->  lat_w5/$x"
  fi
done

echo
if [ "$APPLY" = 1 ]; then
  echo "Done — moved $moved entries into lat_w5/."
  echo "Verify:  python \"$HERE/sweep_summarize.py\""
else
  echo "[dry-run] $moved entries would move. Re-run with --apply to do it:"
  echo "    bash \"${BASH_SOURCE[0]}\" --apply"
fi
echo
echo "Then launch the new longitude/window-9 runs (nested under sweep/lon_w9/), e.g.:"
echo "    python \"$HERE/sweep_submit.py\" \\"
echo "        --split-axis lon --window-size 9 --max-oc 150 \\"
echo "        --sweep-name lon_w9/oc150 --families --dry-run"
echo "  (drop --dry-run to submit; --sweep-name lon_w9/oc150 -> sweep/lon_w9/oc150/<tag>/)"
