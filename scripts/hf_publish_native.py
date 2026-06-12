"""hf_publish_native.py — publish the NATIVE-resolution dataset (resolutions kept
SEPARATE) to a new HF dataset, structured like sgt-bavaria-soc-2002-2023-large.

Target (new): ValerianFourel/sgt-bavaria-soc-2002-2023-native  (public)

Unlike -large (one unified 250 m grid), this keeps each band at its TRUE native
resolution, grouped so the resolution-aware net loads each group directly:

    res_20m/   RasterTensorData/StaticValue/<band>/<tile>.npy    (2 bands: S2SRC_SWIR1/2)
              Coordinates1Mil/..., OC_LUCAS_LFU_LfL_Coordinates_v2/...
    res_30m/   RasterTensorData/{Yearly,Static}Value/<band>/...   (20 bands: Landsat SRC + SRTM/terrain)
              Coordinates1Mil/..., OC_.../...
    res_250m/  RasterTensorData/{Yearly,Static}Value/<band>/...   (23 coarse bands, as in -large)
              Coordinates1Mil/..., OC_.../...

Same .npy-tile + Coordinates layout as -large within each res_* root, so existing
loader code works per group. Mirrors scripts/hf_publish_large.py (create_repo +
upload_large_folder, resumable, sha256-dedup) and uploads the dataset card.

Build the res_*/ tree first (download -> pull_verify_native.py to kill all-NaN ->
tile at native TILE_PX -> coords); then:
    python scripts/hf_publish_native.py --src /path/to/Data_native
    python scripts/hf_publish_native.py --src ... --dry-run
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_large_folder, upload_file

REPO_ID = 'ValerianFourel/sgt-bavaria-soc-2002-2023-native'
DEFAULT_SRC = Path(__file__).resolve().parent.parent.parent / 'Data_native'
DEFAULT_IGNORE = ['.cache/**', '*.tmp']
CARD = Path(__file__).resolve().parent.parent / 'SamplePoints' / 'NATIVE_RES_DATASET_README.md'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=DEFAULT_SRC,
                   help=f'res_20m/ + res_30m/ + res_250m/ tree (default: {DEFAULT_SRC}).')
    p.add_argument('--repo', type=str, default=REPO_ID)
    p.add_argument('--private', action='store_true')
    p.add_argument('--pattern', action='append', default=None,
                   help='glob to restrict upload (e.g. "res_30m/**"). Repeatable.')
    p.add_argument('--ignore', action='append', default=None)
    p.add_argument('--no-card', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    src = a.src.resolve()
    if not src.is_dir():
        sys.exit(f'src does not exist: {src}\n'
                 f'  build the res_20m/res_30m/res_250m tree first (see '
                 f'NATIVE_RES_DATASET_README.md).')
    ignore = list(DEFAULT_IGNORE) + (a.ignore or [])
    print(f'[hf_publish_native] src  = {src}')
    print(f'[hf_publish_native] repo = {a.repo}  ({"private" if a.private else "public"})')
    if a.pattern:
        print(f'[hf_publish_native] allow = {a.pattern}')

    if a.dry_run:
        from fnmatch import fnmatch
        n = 0
        for root, _, files in os.walk(src, followlinks=True):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), src)
                if a.pattern and not any(fnmatch(rel, p) for p in a.pattern):
                    continue
                if any(fnmatch(rel, p) for p in ignore):
                    continue
                n += 1
                if n <= 20:
                    print(f'  {rel}')
        print(f'[hf_publish_native] dry-run: {n} files would upload'
              + (f' + README.md from {CARD}' if CARD.exists() else ''))
        return

    create_repo(a.repo, repo_type='dataset', exist_ok=True, private=a.private)
    if not a.no_card and CARD.exists():
        upload_file(path_or_fileobj=str(CARD), path_in_repo='README.md',
                    repo_id=a.repo, repo_type='dataset',
                    commit_message='Add native-resolution dataset card')
        print(f'[hf_publish_native] uploaded README.md')
    upload_large_folder(repo_id=a.repo, folder_path=str(src), repo_type='dataset',
                        allow_patterns=a.pattern, ignore_patterns=ignore,
                        print_report=True)
    print(f'[hf_publish_native] done. https://huggingface.co/datasets/{a.repo}')


if __name__ == '__main__':
    main()
