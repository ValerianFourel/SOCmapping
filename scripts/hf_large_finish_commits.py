"""hf_large_finish_commits.py — land the files still missing from the -large repo.

Fallback for when upload_large_folder reports "done" but the repo is short
(its batch commits got rate-limited yet were marked committed locally). This
diffs local Data/ against the remote, then commits ONLY the missing files in a
few large, paced commits via create_commit. The LFS blobs are already on HF
storage, so each missing file is a cheap pointer add — no bulk re-upload.

Usage:
    python scripts/hf_large_finish_commits.py                 # finish the push
    python scripts/hf_large_finish_commits.py --batch 800 --sleep 5
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd

REPO_ID = 'ValerianFourel/sgt-bavaria-soc-2002-2023-large'
DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / 'Data'
IGNORE = ['.cache/**', '*.tmp', '.gitattributes', '.gitignore']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=DEFAULT_SRC)
    ap.add_argument('--repo', default=REPO_ID)
    ap.add_argument('--batch', type=int, default=800, help='files per commit')
    ap.add_argument('--sleep', type=float, default=5.0, help='seconds between commits')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    src = a.src.resolve()
    if not src.is_dir():
        sys.exit(f'no such dir: {src}')
    api = HfApi()

    remote = set(f for f in api.list_repo_files(a.repo, repo_type='dataset')
                 if f != '.gitattributes')
    local = {}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '.cache']
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), src)
            if any(fnmatch(rel, p) for p in IGNORE):
                continue
            local[rel] = os.path.join(root, fn)

    missing = sorted(set(local) - remote)
    print(f'[finish] repo has {len(remote)} files; local union {len(local)}; '
          f'missing {len(missing)}')
    if not missing:
        print('[finish] nothing to do — repo already complete.')
        return
    if a.dry_run:
        for m in missing[:20]:
            print('  ', m)
        print(f'[finish] dry-run: would commit {len(missing)} files in '
              f'{(len(missing) + a.batch - 1) // a.batch} commits.')
        return

    total = len(missing)
    for i in range(0, total, a.batch):
        chunk = missing[i:i + a.batch]
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=local[rel])
               for rel in chunk]
        for attempt in range(1, 6):
            try:
                api.create_commit(
                    repo_id=a.repo, repo_type='dataset', operations=ops,
                    commit_message=f'finish union: files {i + 1}-{i + len(chunk)}')
                break
            except Exception as e:
                wait = min(60, 5 * attempt * attempt)
                print(f'[finish]   commit {i // a.batch + 1} attempt {attempt} '
                      f'failed ({type(e).__name__}: {str(e)[:80]}); retry in {wait}s')
                time.sleep(wait)
        else:
            sys.exit(f'[finish] giving up on commit at offset {i}')
        done = min(i + len(chunk), total)
        print(f'[finish] committed {done}/{total} missing files', flush=True)
        if done < total:
            time.sleep(a.sleep)

    remaining = len(set(local) - set(
        f for f in api.list_repo_files(a.repo, repo_type='dataset')))
    print(f'[finish] done. still-missing now: {remaining}')


if __name__ == '__main__':
    main()
