"""hf_publish.py — sync the local Data tree to a canonical HF dataset.

Single source of truth: ValerianFourel/sgt-bavaria-soc-2002-2023.
Run from the SOCmapping/ project root after `huggingface-cli login`.

Designed for incremental syncs: uses upload_large_folder, which dedupes
against what's already on the remote and resumes if interrupted.

Usage:
    # full sync (everything under Data_HF/, follows symlinks)
    python scripts/hf_publish.py

    # just one tree (faster iteration)
    python scripts/hf_publish.py --pattern 'RasterTensorData/YearlyValue/**'

    # dry run — show what would be pushed without uploading
    python scripts/hf_publish.py --dry-run

The local source is `../Data_HF/` (relative to this script). That directory is
already populated with symlinks back to the canonical `../Data/`, so anything
you add there flows through. The uploader resolves symlinks before pushing.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_large_folder


REPO_ID = 'ValerianFourel/sgt-bavaria-soc-2002-2023'
DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / 'Data_HF'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=DEFAULT_SRC,
                   help=f'Local folder to sync (default: {DEFAULT_SRC}).')
    p.add_argument('--repo', type=str, default=REPO_ID,
                   help=f'HF dataset repo (default: {REPO_ID}).')
    p.add_argument('--pattern', action='append', default=None,
                   help='Optional glob to restrict upload (relative to --src). '
                        "Repeatable. e.g. --pattern 'RasterTensorData/YearlyValue/**' "
                        "--pattern 'OC_LUCAS_LFU_LfL_Coordinates_v2/**'")
    p.add_argument('--ignore', action='append', default=None,
                   help='Glob to skip (relative to --src). Repeatable.')
    p.add_argument('--dry-run', action='store_true',
                   help='List what would be uploaded; do not push.')
    return p.parse_args()


def main():
    a = parse_args()
    src = a.src.resolve()
    if not src.is_dir():
        sys.exit(f'src does not exist: {src}')

    print(f'[hf_publish] src   = {src}')
    print(f'[hf_publish] repo  = {a.repo}')
    if a.pattern: print(f'[hf_publish] allow = {a.pattern}')
    if a.ignore:  print(f'[hf_publish] ignore= {a.ignore}')

    if a.dry_run:
        # walk + apply patterns locally for preview
        from fnmatch import fnmatch
        matched = 0
        for root, _, files in os.walk(src, followlinks=True):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), src)
                if a.pattern and not any(fnmatch(rel, p) for p in a.pattern):
                    continue
                if a.ignore and any(fnmatch(rel, p) for p in a.ignore):
                    continue
                matched += 1
                if matched <= 20:
                    print(f'  {rel}')
        print(f'[hf_publish] dry-run: {matched} files would be uploaded.')
        return

    create_repo(a.repo, repo_type='dataset', exist_ok=True)

    # upload_large_folder: chunks, dedupes against remote SHAs, resumes.
    # It follows symlinks by default (uploads the resolved file content).
    upload_large_folder(
        repo_id=a.repo,
        folder_path=str(src),
        repo_type='dataset',
        allow_patterns=a.pattern,
        ignore_patterns=a.ignore,
        # commit message includes the pattern so the HF audit log is informative
        print_report=True,
    )
    print(f'[hf_publish] done. https://huggingface.co/datasets/{a.repo}')


if __name__ == '__main__':
    main()
