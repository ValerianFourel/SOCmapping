"""hf_publish_large.py — publish the FULL union ("everything") to a new HF dataset.

Target (new): ValerianFourel/sgt-bavaria-soc-2002-2023-large  (public)

This is the unified superset used to sync all machines (laptop / HoreKa /
JUPITER). Unlike the curated canonical `sgt-bavaria-soc-2002-2023`, this one
keeps EVERYTHING in the local Data/ tree:
  * the bulk RasterTensorData (identical to canonical)
  * BOTH coordinate layouts — new SeasonalValue/StaticValue AND the old
    YearlyValue alias folders (backward-compat for old-layout code)
  * the Preprocessing/ and RasterBandsData/ intermediate trees that
    hf_publish.py deliberately drops

Only HF-cache metadata and scratch temp files are excluded.

upload_large_folder dedups identical content against the remote by sha256, so
the hundreds of redundant coordinate-grid copies cost one blob each, and the
whole job is resumable — re-run it and it picks up where it left off.

Usage (run from anywhere; HF login required: `huggingface-cli login`):
    python scripts/hf_publish_large.py                 # full push
    python scripts/hf_publish_large.py --dry-run       # list what would go
    python scripts/hf_publish_large.py --pattern 'Preprocessing/**'   # one subtree
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_large_folder


REPO_ID = 'ValerianFourel/sgt-bavaria-soc-2002-2023-large'
# The real Data/ tree (../../Data relative to this script), same source hf_publish uses.
DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / 'Data'
# Keep EVERYTHING except HF's own download/upload bookkeeping and scratch temps.
DEFAULT_IGNORE = [
    '.cache/**',
    '*.tmp',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=DEFAULT_SRC,
                   help=f'Local folder to publish (default: {DEFAULT_SRC}).')
    p.add_argument('--repo', type=str, default=REPO_ID,
                   help=f'HF dataset repo (default: {REPO_ID}).')
    p.add_argument('--private', action='store_true',
                   help='Create the repo private (default: public).')
    p.add_argument('--pattern', action='append', default=None,
                   help="Optional glob to restrict upload (relative to --src). Repeatable.")
    p.add_argument('--ignore', action='append', default=None,
                   help=f'Extra globs to skip; appended to defaults {DEFAULT_IGNORE}.')
    p.add_argument('--dry-run', action='store_true',
                   help='List what would be uploaded; do not push.')
    return p.parse_args()


def main():
    a = parse_args()
    src = a.src.resolve()
    if not src.is_dir():
        sys.exit(f'src does not exist: {src}')

    ignore = list(DEFAULT_IGNORE) + (a.ignore or [])

    print(f'[hf_publish_large] src    = {src}')
    print(f'[hf_publish_large] repo   = {a.repo}  ({"private" if a.private else "public"})')
    if a.pattern:
        print(f'[hf_publish_large] allow  = {a.pattern}')
    print(f'[hf_publish_large] ignore = {ignore}')

    if a.dry_run:
        from fnmatch import fnmatch
        matched = 0
        for root, _, files in os.walk(src, followlinks=True):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), src)
                if a.pattern and not any(fnmatch(rel, p) for p in a.pattern):
                    continue
                if any(fnmatch(rel, p) for p in ignore):
                    continue
                matched += 1
                if matched <= 20:
                    print(f'  {rel}')
        print(f'[hf_publish_large] dry-run: {matched} files would be uploaded.')
        return

    create_repo(a.repo, repo_type='dataset', exist_ok=True, private=a.private)

    upload_large_folder(
        repo_id=a.repo,
        folder_path=str(src),
        repo_type='dataset',
        allow_patterns=a.pattern,
        ignore_patterns=ignore,
        print_report=True,
    )
    print(f'[hf_publish_large] done. https://huggingface.co/datasets/{a.repo}')


if __name__ == '__main__':
    main()
