#!/usr/bin/env python3
"""Orchestrate the FULL Sentinel-grid (20 m) re-export of the SGT band stack.

The user's "sample everything a sentinel mode" decision: every band scaled to
20 m, not just the new S2 SWIR pair. This driver submits the two Earth Engine
export batches that produce the 20 m stack to Drive:

  1) gee_download_all_bands.py --category extended --scale 20
        -> all 43 covariates at 20 m (curated 20 + Tier 1/2/3). The
           `--scale 20` now also drives the source-resampling target, so fine
           bands keep native detail and coarse bands are bilinear-upsampled to
           the 20 m grid (see the EXPORT_SCALE_M wiring in that script).
  2) sgt_s2_swir_export.py --scale 20
        -> the bare-soil S2 SWIR pair (B11/B12) at 20 m (Tier 4).

This ONLY submits the EE exports. The downstream 20 m regeneration (tiling at
SGT_TILE_PX=12238, coordinate-index rebuild, dataset publish, training window)
is described in SENTINEL_MODE_RUNBOOK.md — those steps are heavy and run on the
cluster after the Drive exports finish.

Run (EE project sgtmodel):
    python SOCmapping/SamplePoints/sgt_sentinel_export.py --dry-run   # preview both batches
    python SOCmapping/SamplePoints/sgt_sentinel_export.py             # submit both
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scale', type=int, default=20, help='Sentinel pixel size (m).')
    ap.add_argument('--years', nargs=2, type=int, default=[2002, 2023])
    ap.add_argument('--drive-folder', default='bavaria_bands_20m')
    ap.add_argument('--s2-years', nargs=2, type=int, default=[2017, 2023],
                    help='S2 archive window for the bare-soil SWIR composite.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    py = sys.executable
    extended = [py, str(HERE / 'gee_download_all_bands.py'),
                '--category', 'extended', '--scale', str(args.scale),
                '--years', str(args.years[0]), str(args.years[1]),
                '--drive-folder', args.drive_folder]
    s2 = [py, str(HERE / 'sgt_s2_swir_export.py'),
          '--scale', str(args.scale),
          '--years', str(args.s2_years[0]), str(args.s2_years[1])]
    if args.dry_run:
        extended.append('--dry-run')
        s2.append('--dry-run')

    print(f'[sentinel] full-stack 20 m re-export -> Drive/{args.drive_folder}/')
    print(f'[sentinel] batch 1 (43 covariates): {" ".join(extended)}')
    rc1 = subprocess.call(extended)
    print(f'[sentinel] batch 2 (S2 SWIR pair):  {" ".join(s2)}')
    rc2 = subprocess.call(s2)
    if rc1 or rc2:
        print(f'[sentinel] WARNING non-zero exit (extended={rc1}, s2={rc2})',
              file=sys.stderr)
        return 1
    print('[sentinel] both export batches submitted. Next: tile at '
          'SGT_TILE_PX=12238 + rebuild coords. See SENTINEL_MODE_RUNBOOK.md.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
