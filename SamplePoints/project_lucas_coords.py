"""Project every LUCAS GPS point onto a (band, year) tile layout.

Produces:
  Data/OC_LUCAS_LFU_LfL_Coordinates_v2/{Static,Yearly}Value/<band>/[<year>/]coordinates.npy

where coordinates.npy has columns [latitude, longitude, tile_id, row, col]
(the format expected by NormalizedMultiRasterDatasetMultiYears).

This is the same logic as samplePoints.py but generalized: instead of
hardcoding paths to vfourel's box, it takes any tile directory and
writes the projection output to the matching coordinates directory.

Usage
-----
Single (band, year):
  python SamplePoints/project_lucas_coords.py \
      --raster-dir /home/valerian/SGTPublication/Data/RasterTensorData/YearlyValue/NDVI/2010 \
      --coords-csv /home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx

All bands & years for a tier root (auto-discovers band dirs):
  python SamplePoints/project_lucas_coords.py \
      --tier YearlyValue --bands NDVI EVI Precipitation
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.spatial import cKDTree

# Path resolution — match the project convention
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import SOC_DATA_DIR_STR

DATA_ROOT = Path(SOC_DATA_DIR_STR)
RASTER_ROOT = DATA_ROOT / 'RasterTensorData'
COORDS_ROOT = DATA_ROOT / 'OC_LUCAS_LFU_LfL_Coordinates_v2'
LUCAS_XLSX  = DATA_ROOT / 'LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx'

_FN_RE = re.compile(r'ID(?P<id>\d+)N(?P<n>-?\d+(?:_\d+)?)S(?P<s>-?\d+(?:_\d+)?)W(?P<w>-?\d+(?:_\d+)?)E(?P<e>-?\d+(?:_\d+)?)\.npy')


def parse_tile_filename(name: str) -> dict:
    m = _FN_RE.match(name)
    if not m:
        raise ValueError(f"Bad tile filename: {name}")
    return {
        'id': int(m.group('id')),
        'n': float(m.group('n').replace('_', '.')),
        's': float(m.group('s').replace('_', '.')),
        'w': float(m.group('w').replace('_', '.')),
        'e': float(m.group('e').replace('_', '.')),
    }


def load_tile_index(raster_dir: Path) -> tuple[list[dict], cKDTree]:
    """Build a list of tiles in raster_dir and a KDTree on their centres."""
    tiles = []
    for npy in sorted(raster_dir.glob('ID*.npy')):
        meta = parse_tile_filename(npy.name)
        meta['path'] = str(npy)
        meta['center_lat'] = 0.5 * (meta['n'] + meta['s'])
        meta['center_lon'] = 0.5 * (meta['w'] + meta['e'])
        tiles.append(meta)
    if len(tiles) == 0:
        raise FileNotFoundError(f"No ID*.npy tiles in {raster_dir}")
    centres = np.array([[t['center_lon'], t['center_lat']] for t in tiles])
    tree = cKDTree(centres)
    return tiles, tree


def project_point(lon: float, lat: float, tiles: list[dict], tree: cKDTree) -> tuple[int, int, int]:
    """Return (tile_id, row, col) for a single GPS point.

    Picks the tile whose centre is nearest *and* whose bbox actually
    contains the point. row/col are pixel indices into the 979 × 979
    tile array (row 0 = N edge, col 0 = W edge).
    """
    _, idx = tree.query([lon, lat], k=min(4, len(tiles)))
    candidates = np.atleast_1d(idx)
    for ci in candidates:
        t = tiles[ci]
        if t['w'] <= lon <= t['e'] and t['s'] <= lat <= t['n']:
            # convert (lon, lat) → pixel (row, col)
            n, s, w, e = t['n'], t['s'], t['w'], t['e']
            tile_px = 979
            col = int(round((lon - w) / (e - w) * (tile_px - 1)))
            row = int(round((n - lat) / (n - s) * (tile_px - 1)))
            col = max(0, min(tile_px - 1, col))
            row = max(0, min(tile_px - 1, row))
            return t['id'], row, col
    # Point is outside every tile — return sentinel
    return -1, 0, 0


def project_all(raster_dir: Path, lon_lat: np.ndarray) -> np.ndarray:
    """Project every (lon, lat) pair onto raster_dir's tile grid.

    Returns array of shape (N, 5): [lat, lon, tile_id, row, col].
    """
    tiles, tree = load_tile_index(raster_dir)
    out = np.empty((len(lon_lat), 5), dtype=np.float32)
    for i, (lon, lat) in enumerate(lon_lat):
        tid, r, c = project_point(float(lon), float(lat), tiles, tree)
        out[i] = [lat, lon, tid, r, c]
    return out


def coords_out_dir(raster_dir: Path) -> Path:
    """Mirror RasterTensorData/<tier>/<band>/[<year>] into
    OC_LUCAS_LFU_LfL_Coordinates_v2/<tier>/<band>/[<year>]/."""
    rel = raster_dir.relative_to(RASTER_ROOT)
    return COORDS_ROOT / rel


def process_single_dir(raster_dir: Path, lon_lat: np.ndarray) -> Path:
    out_dir = coords_out_dir(raster_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = project_all(raster_dir, lon_lat)
    target = out_dir / 'coordinates.npy'
    np.save(target, arr)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raster-dir', type=Path,
                    help='Single band-year folder containing ID*.npy tiles')
    ap.add_argument('--tier', choices=['StaticValue', 'YearlyValue'],
                    help='When using --bands, the tier to walk')
    ap.add_argument('--bands', nargs='+',
                    help='Band names under --tier to process (auto-discovers years for YearlyValue)')
    ap.add_argument('--coords-csv', type=Path,
                    help='Override default LUCAS_LFU xlsx path')
    args = ap.parse_args()

    # Load LUCAS GPS list
    src = args.coords_csv or LUCAS_XLSX
    if src.suffix.lower() in ('.xls', '.xlsx'):
        df = pd.read_excel(src, usecols=['GPS_LONG', 'GPS_LAT'])
    else:
        df = pd.read_csv(src, usecols=['GPS_LONG', 'GPS_LAT'])
    df = df.dropna()
    lon_lat = df[['GPS_LONG', 'GPS_LAT']].to_numpy(dtype=np.float64)
    print(f"Projecting {len(lon_lat):,} LUCAS GPS points onto tile grids")

    if args.raster_dir is not None:
        out = process_single_dir(args.raster_dir, lon_lat)
        print(f"  ✓ wrote {out}")
    elif args.tier and args.bands:
        tier_root = RASTER_ROOT / args.tier
        for band in args.bands:
            band_dir = tier_root / band
            if not band_dir.exists():
                print(f"  skip {band}: {band_dir} does not exist")
                continue
            if args.tier == 'StaticValue':
                out = process_single_dir(band_dir, lon_lat)
                print(f"  ✓ {band:25}  → {out}")
            else:
                # walk year subdirs
                years = sorted([d for d in band_dir.iterdir() if d.is_dir() and d.name.isdigit()])
                for y_dir in years:
                    out = process_single_dir(y_dir, lon_lat)
                    print(f"  ✓ {band}/{y_dir.name:5}  → {out}")
    else:
        ap.error('provide either --raster-dir, or --tier + --bands')


if __name__ == '__main__':
    main()
