"""pull_large.py — resumable download of the unified -large dataset onto a machine.

Pulls ValerianFourel/sgt-bavaria-soc-2002-2023-large into a local dir, retrying
on transient network errors (it's a ~94-122 GB multi-GB pull). Run on HoreKa /
JUPITER (or the laptop) to sync that machine to the unified superset.

Tips:
  * On HoreKa, set HF cache on $WORK first:  export HF_HOME=$WORK/.hf_cache
  * Run under nohup/tmux so a disconnect doesn't kill it; it resumes anyway.

Usage:
    python scripts/pull_large.py <dest_dir>
    python scripts/pull_large.py <dest_dir> --workers 16
    python scripts/pull_large.py <dest_dir> --repo ValerianFourel/sgt-bavaria-soc-2002-2023
"""
from __future__ import annotations
import argparse
import sys
import time

from huggingface_hub import snapshot_download

REPO = 'ValerianFourel/sgt-bavaria-soc-2002-2023-large'
EXPECTED_FILES = 25101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dest', help='local directory to download into')
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--max-retries', type=int, default=1000)
    a = ap.parse_args()

    for attempt in range(1, a.max_retries + 1):
        try:
            p = snapshot_download(a.repo, repo_type='dataset',
                                  local_dir=a.dest, max_workers=a.workers)
            print(f'DONE -> {p}', flush=True)
            print(f'(verify: find {a.dest} -type f -not -path "*/.cache/*" | wc -l '
                  f'-> expect {EXPECTED_FILES} for the -large repo)')
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'[pull] attempt {attempt} hit {type(e).__name__}: '
                  f'{str(e)[:140]} — resuming in 15s', flush=True)
            time.sleep(15)
    sys.exit('[pull] gave up after max retries')


if __name__ == '__main__':
    main()
