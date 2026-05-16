"""Cut a downloaded Bavaria-envelope GeoTIFF into 12 .npy tiles matching the
existing layout.

Input: Bavaria-wide GeoTIFF at 250 m / EPSG:4326 (produced by
       SamplePoints/gee_download_all_bands.py). Filename pattern
       expected from the GEE exports is like:
         <category>_<band>_<year>.tif
         <category>_<band>_static.tif

Output: 12 .npy tiles per source TIFF, written under
        Data/RasterTensorData/{StaticValue,YearlyValue}/<BandName>/[<year>/]
        with filenames matching the existing IDxN..S..W..E..npy pattern.

Tile grid (parsed from existing Elevation tiles; constants mirrored here):
    3 lat-rows × 4 lon-cols = 12 tiles
    each ~1.7986° × 1.7986°, 979 × 979 px
    envelope [W 7.1864, S 46.7109, E 14.3750, N 52.1028]

Usage examples
--------------
# Cut all GeoTIFFs in a local folder (downloaded from Drive via rclone):
python SamplePoints/tiff_to_tiles.py \
    --tiff-dir ~/Downloads/bavaria_bands_2002_2023 \
    --out-root /home/valerian/SGTPublication/Data/RasterTensorData

# Single TIFF:
python SamplePoints/tiff_to_tiles.py \
    --tiff modis_NDVI_2010.tif \
    --band-name NDVI \
    --year 2010 \
    --tier YearlyValue
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Tile grid — exact reproduction of the existing Elevation layout.
# Each tile spans TILE_DEG° in both lat and lon. Edges abut without gap.
# ---------------------------------------------------------------------------
TILE_LAT_NORTH = [48.5095, 50.3062, 52.1028]
TILE_LON_WEST  = [7.1864, 8.9831, 10.7797, 12.5763]
TILE_DEG       = 1.7986
TILE_PX        = 979


def tile_bbox(tile_id: int) -> tuple[float, float, float, float]:
    """Return (N, S, W, E) for a given tile id 0-11.

    Tile ids follow the same convention as the existing layout:
      row 0 (south): ids 0, 1, 2, 3   (W → E)
      row 1 (mid):   ids 4, 5, 6, 7
      row 2 (north): ids 8, 9, 10, 11
    """
    row = tile_id // 4
    col = tile_id % 4
    n = TILE_LAT_NORTH[row]
    s = n - TILE_DEG
    w = TILE_LON_WEST[col]
    e = w + TILE_DEG
    return n, s, w, e


def encode_filename(tile_id: int) -> str:
    """Recreate the IDxN..S..W..E..npy filename for a tile."""
    n, s, w, e = tile_bbox(tile_id)
    def fmt(v: float) -> str:
        return f'{v:.15f}'.replace('.', '_').rstrip('0').rstrip('_')
    return f'ID{tile_id}N{fmt(n)}S{fmt(s)}W{fmt(w)}E{fmt(e)}.npy'


def cut_tiff(tiff_path: Path, out_dir: Path) -> int:
    """Slice one Bavaria-envelope GeoTIFF into 12 .npy tiles in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with rasterio.open(tiff_path) as src:
        for tid in range(12):
            n, s, w, e = tile_bbox(tid)
            try:
                win = from_bounds(left=w, bottom=s, right=e, top=n,
                                  transform=src.transform)
                arr = src.read(1, window=win,
                               out_shape=(TILE_PX, TILE_PX),
                               resampling=rasterio.enums.Resampling.bilinear)
            except Exception as ex:
                print(f"  tile {tid}: read failed ({ex})")
                continue
            arr = np.asarray(arr, dtype=np.float32)
            out_path = out_dir / encode_filename(tid)
            np.save(out_path, arr)
            n_written += 1
    return n_written


def make_bounds_array(out_static_root: Path, band_name: str) -> None:
    """Generate bounds_array_<band>.npy mirroring bounds_array_Elevation.npy.

    Format: float32 (12, 3) where columns are [id, center_lat, center_lon].
    Mirrors how the existing build_kdtree_from_npy_file consumes it
    (column 1 = lat-like coord, column 2 = lon-like coord).
    """
    rows = []
    for tid in range(12):
        n, s, w, e = tile_bbox(tid)
        center_lat = 0.5 * (n + s)
        center_lon = 0.5 * (w + e)
        rows.append([float(tid), center_lat, center_lon])
    bounds = np.array(rows, dtype=np.float32)
    target = out_static_root / f'bounds_array_{band_name}.npy'
    np.save(target, bounds)
    print(f"  wrote {target.name}  shape={bounds.shape}")


# Naming convention for GEE export files (set in gee_download_all_bands.py):
#   <category>_<band>_<year>.tif          for yearly
#   <category>_<band>_static.tif          for static
TIFF_PATTERN = re.compile(
    r'(?P<category>[a-z0-9]+)_(?P<band>[A-Za-z0-9_]+?)_(?P<period>static|\d{4})\.tif$'
)


def parse_tiff_filename(name: str) -> dict | None:
    """Extract (category, band, year/None) from a GEE export filename."""
    m = TIFF_PATTERN.search(name)
    if not m:
        return None
    period = m.group('period')
    return {
        'category': m.group('category'),
        'band': m.group('band'),
        'year': None if period == 'static' else int(period),
    }


def cut_all(tiff_dir: Path, out_root: Path) -> None:
    """Process every TIFF in tiff_dir matching the expected naming pattern."""
    tiffs = sorted(tiff_dir.glob('*.tif'))
    print(f"Found {len(tiffs)} GeoTIFFs in {tiff_dir}")
    bands_processed = set()
    for tp in tiffs:
        meta = parse_tiff_filename(tp.name)
        if meta is None:
            print(f"  skip (no match): {tp.name}")
            continue
        band = meta['band']
        if meta['year'] is None:
            out_dir = out_root / 'StaticValue' / band
        else:
            out_dir = out_root / 'YearlyValue' / band / str(meta['year'])
        n = cut_tiff(tp, out_dir)
        bands_processed.add((meta['year'] is None, band))
        print(f"  ✓ {tp.name:50}  → {out_dir.relative_to(out_root)}/  ({n} tiles)")

    # Emit bounds_array_<band>.npy at the appropriate tier root, once per band
    for is_static, band in bands_processed:
        tier = 'StaticValue' if is_static else 'YearlyValue'
        make_bounds_array(out_root / tier, band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiff-dir', type=Path,
                    help='Folder containing downloaded GEE GeoTIFFs')
    ap.add_argument('--tiff', type=Path,
                    help='Single GeoTIFF to cut (overrides --tiff-dir)')
    ap.add_argument('--out-root', type=Path,
                    default=Path('/home/valerian/SGTPublication/Data/RasterTensorData'),
                    help='Root of RasterTensorData/ output')
    ap.add_argument('--band-name', type=str, help='Band name (single-tiff mode)')
    ap.add_argument('--year', type=int, help='Year (single-tiff mode, yearly only)')
    ap.add_argument('--tier', choices=['StaticValue', 'YearlyValue'],
                    help='Tier (single-tiff mode)')
    args = ap.parse_args()

    if args.tiff is not None:
        if not (args.band_name and args.tier):
            ap.error('--tiff requires --band-name and --tier')
        if args.tier == 'YearlyValue' and args.year is None:
            ap.error('YearlyValue tier requires --year')
        out_dir = args.out_root / args.tier / args.band_name
        if args.year is not None:
            out_dir = out_dir / str(args.year)
        n = cut_tiff(args.tiff, out_dir)
        print(f"✓ cut {n} tiles → {out_dir}")
        make_bounds_array(args.out_root / args.tier, args.band_name)
    elif args.tiff_dir is not None:
        cut_all(args.tiff_dir, args.out_root)
    else:
        ap.error('provide --tiff-dir or --tiff')


if __name__ == '__main__':
    main()
