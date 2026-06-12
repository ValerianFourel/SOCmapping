#!/usr/bin/env python3
"""pull_verify_native.py — rclone-pull native-resolution GEE exports and KILL the
all-NaN ones.

EE exports of sparse/masked composites occasionally come back all-NaN (e.g. a
nearest-neighbour resample of a sparse bare-soil grid). Those are garbage and
must not reach the dataset. This script:

  1) rclone copy <drive-folder> -> <dest>   (idempotent; skips already-downloaded)
  2) opens every *.tif with rasterio and computes per-band valid-fraction
  3) deletes any file whose bands are ALL-NaN (or below --min-valid), logging it
     to <dest>/_killed_nan.log so you know which (band, year) to re-export.

Resumable: re-run it as the EE batch completes (or loop it). Verified files are
remembered in <dest>/_verified.txt so they aren't re-opened.

Run:
    python pull_verify_native.py --drive bavaria_bands_30m --dest ~/bavaria_tiffs_30m
    python pull_verify_native.py --drive bavaria_bands_2002_2023 --dest ~/bavaria_tiffs \\
        --include 's2_swir_*_20m*'
    python pull_verify_native.py --dest ~/bavaria_tiffs_30m --no-pull --dry-run   # verify only
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def rclone_pull(drive_folder: str, dest: Path, include: str | None):
    cmd = ['rclone', 'copy', f'gdrive:{drive_folder}', str(dest),
           '--drive-acknowledge-abuse', '--stats', '0']
    if include:
        cmd += ['--include', include]
    print(f'[pull] {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'[pull] rclone rc={r.returncode}', file=sys.stderr)


def valid_fractions(path: Path):
    """Per-band finite fraction; None if unreadable."""
    try:
        import rasterio
    except ImportError:
        sys.exit('rasterio not installed (pip install rasterio)')
    try:
        with rasterio.open(path) as src:
            out = []
            for i in range(1, src.count + 1):
                a = src.read(i)
                out.append(float(np.isfinite(a).mean()))
            return out, [src.descriptions[i] for i in range(src.count)]
    except Exception as e:
        print(f'[verify] unreadable {path.name}: {e}', file=sys.stderr)
        return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--drive', default=None, help='Drive folder to rclone-pull from.')
    p.add_argument('--dest', type=Path, required=True, help='local folder.')
    p.add_argument('--include', default=None, help='rclone --include glob.')
    p.add_argument('--no-pull', action='store_true', help='skip rclone, verify what is on disk.')
    p.add_argument('--min-valid', type=float, default=0.0,
                   help='delete a file if its MAX per-band valid-fraction <= this '
                        '(default 0.0 = delete only all-NaN files).')
    p.add_argument('--dry-run', action='store_true', help='report; delete nothing.')
    a = p.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)

    if a.drive and not a.no_pull:
        rclone_pull(a.drive, a.dest, a.include)

    verified_f = a.dest / '_verified.txt'
    seen = set(verified_f.read_text().split()) if verified_f.exists() else set()
    killed_log = a.dest / '_killed_nan.log'

    tifs = sorted(a.dest.glob('*.tif'))
    kept = killed = 0
    for t in tifs:
        if t.name in seen:
            continue
        fr, names = valid_fractions(t)
        if fr is None:
            continue
        maxv = max(fr) if fr else 0.0
        tag = ' '.join(f'{n or "?"}={v*100:.0f}%' for n, v in zip(names, fr))
        if maxv <= a.min_valid:
            killed += 1
            print(f'[KILL] {t.name}  all-NaN/empty ({tag})', flush=True)
            if not a.dry_run:
                t.unlink()
                with killed_log.open('a') as fh:
                    fh.write(f'{t.name}\t{tag}\n')
        else:
            kept += 1
            print(f'[ok]   {t.name}  {tag}', flush=True)
            if not a.dry_run:
                with verified_f.open('a') as fh:
                    fh.write(t.name + '\n')
    print(f'\n[verify] kept={kept}  killed={killed}'
          + ('  (dry-run, nothing deleted)' if a.dry_run else f'  (logged to {killed_log})'),
          flush=True)


if __name__ == '__main__':
    main()
