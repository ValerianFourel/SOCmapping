"""hf_publish_sentinel.py — publish the SENTINEL (20 m) dataset to a new HF repo.

Target (new): ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel  (public)

The sentinel-resolution sibling of sgt-bavaria-soc-2002-2023-large: every band
sampled on the 20 m grid (sentinel mode). Per the fine/broadcast design only the
22 FINE bands (S2 SWIR, Landsat SRC, SRTM/terrain) carry real 20 m tiles; the 23
COARSE bands (MODIS/ERA5/SoilGrids) are one nearest-pixel value broadcast across
the 9/11 window at read time — so this dataset is NOT ~156x larger than the 250 m
one (see SamplePoints/SENTINEL_MODE_RUNBOOK.md).

Mirrors scripts/hf_publish_large.py exactly (create_repo + upload_large_folder,
resumable, sha256-dedup) but points at the sentinel data tree and repo, and
uploads the dataset card. Because upload_large_folder can "finish" half-committed,
run it in the resume loop: scripts/hf_sentinel_complete_loop.sh.

Usage (HF login required: `huggingface-cli login`):
    # after the 20 m export + tiling produced the sentinel Data tree:
    python scripts/hf_publish_sentinel.py --src /path/to/Data_sentinel_20m
    python scripts/hf_publish_sentinel.py --src ... --dry-run
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_large_folder, upload_file


REPO_ID = 'ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel'
# Default sentinel data tree — a sibling of Data/ holding the 20 m regeneration.
DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / 'Data_sentinel_20m'
DEFAULT_IGNORE = ['.cache/**', '*.tmp']
# Dataset card lives next to this script's package; uploaded as README.md.
CARD = Path(__file__).resolve().parent.parent / 'SamplePoints' / 'SENTINEL_DATASET_README.md'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=DEFAULT_SRC,
                   help=f'Local sentinel (20 m) data folder (default: {DEFAULT_SRC}).')
    p.add_argument('--repo', type=str, default=REPO_ID,
                   help=f'HF dataset repo (default: {REPO_ID}).')
    p.add_argument('--private', action='store_true', help='Create the repo private.')
    p.add_argument('--pattern', action='append', default=None,
                   help='Optional glob to restrict upload (relative to --src). Repeatable.')
    p.add_argument('--ignore', action='append', default=None,
                   help=f'Extra globs to skip; appended to {DEFAULT_IGNORE}.')
    p.add_argument('--no-card', action='store_true', help='Skip uploading README.md.')
    p.add_argument('--dry-run', action='store_true',
                   help='List what would be uploaded; do not push.')
    return p.parse_args()


def main():
    a = parse_args()
    src = a.src.resolve()
    if not src.is_dir():
        sys.exit(f'src does not exist: {src}\n'
                 f'  (run the 20 m export + tiling first — see '
                 f'SamplePoints/SENTINEL_MODE_RUNBOOK.md)')

    ignore = list(DEFAULT_IGNORE) + (a.ignore or [])
    print(f'[hf_publish_sentinel] src    = {src}')
    print(f'[hf_publish_sentinel] repo   = {a.repo}  ({"private" if a.private else "public"})')
    if a.pattern:
        print(f'[hf_publish_sentinel] allow  = {a.pattern}')
    print(f'[hf_publish_sentinel] ignore = {ignore}')

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
        print(f'[hf_publish_sentinel] dry-run: {matched} files would be uploaded.')
        if CARD.exists():
            print(f'[hf_publish_sentinel] dry-run: + README.md from {CARD}')
        return

    create_repo(a.repo, repo_type='dataset', exist_ok=True, private=a.private)

    if not a.no_card and CARD.exists():
        upload_file(path_or_fileobj=str(CARD), path_in_repo='README.md',
                    repo_id=a.repo, repo_type='dataset',
                    commit_message='Add sentinel dataset card')
        print(f'[hf_publish_sentinel] uploaded README.md from {CARD}')

    upload_large_folder(
        repo_id=a.repo,
        folder_path=str(src),
        repo_type='dataset',
        allow_patterns=a.pattern,
        ignore_patterns=ignore,
        print_report=True,
    )
    print(f'[hf_publish_sentinel] done. https://huggingface.co/datasets/{a.repo}')


if __name__ == '__main__':
    main()
