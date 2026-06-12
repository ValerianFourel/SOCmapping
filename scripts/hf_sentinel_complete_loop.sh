#!/bin/bash
# Resume loop for the sentinel dataset publish — upload_large_folder can
# "finish" half-committed, so re-run until the remote file count stops growing.
# Mirrors hf_large_complete_loop.sh. Run from the SOCmapping repo root after the
# 20 m export + tiling have produced the sentinel Data tree.
#
#   bash scripts/hf_sentinel_complete_loop.sh /path/to/Data_sentinel_20m
set -u
SRC="${1:-../Data_sentinel_20m}"
REPO="ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel"
LOG="hf_publish_sentinel.log"

prev=0; stall=0
for i in $(seq 1 12); do
  echo "=== PASS $i start $(date +%H:%M:%S) ===" >> "$LOG"
  python3 -u scripts/hf_publish_sentinel.py --src "$SRC" >> "$LOG" 2>&1
  n=$(python3 -c "from huggingface_hub import HfApi; print(sum(1 for s in HfApi().repo_info('$REPO',repo_type='dataset',files_metadata=True).siblings if s.rfilename!='.gitattributes'))" 2>/dev/null)
  echo "PASS $i RESULT: ${n:-ERR} remote files  ($(date +%H:%M:%S))"
  if [ "${n:-0}" -le "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev=${n:-0}
  [ "$stall" -ge 3 ] && { echo "STALLED at $n after pass $i"; break; }
  sleep 8
done
echo "=== LOOP END $(date +%H:%M:%S) ==="
