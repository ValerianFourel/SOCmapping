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


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf pixels with the tile's finite mean.

    `NormalizedMultiRasterDatasetMultiYears.compute_statistics` does not
    tolerate NaN — a single bad pixel pollutes per-channel mean/std and
    poisons every downstream sample. SoilGrids/OpenLandMap tiles can
    have edge or no-data NaNs; ERA5 occasionally has masked pixels.
    Falls back to 0.0 if the entire tile is NaN (shouldn't happen on
    Bavaria-land but keeps the pipeline robust).
    """
    arr = np.asarray(arr, dtype=np.float32)
    mask = ~np.isfinite(arr)
    if mask.any():
        finite = arr[~mask]
        fill = float(finite.mean()) if finite.size > 0 else 0.0
        arr = np.where(mask, fill, arr)
    return arr


def _wipe_stale_tiles(out_dir: Path) -> int:
    """Remove ID*.npy files in out_dir that don't match a canonical tile name.

    Returns the number of files removed. Used when the tile filename pattern
    has changed (e.g. when redownloading a band that was previously cut by
    an older pipeline with different bbox precision in the filenames).
    Without this, old and new tiles would coexist in the same directory and
    the dataloader's glob('ID*.npy') would pick one arbitrarily per tile id.
    """
    canonical = {encode_filename(tid) for tid in range(12)}
    removed = 0
    if out_dir.exists():
        for existing in out_dir.glob('ID*.npy'):
            if existing.name not in canonical:
                existing.unlink()
                removed += 1
    return removed


def cut_tiff(tiff_path: Path, out_dir: Path, wipe_stale: bool = True) -> int:
    """Slice one Bavaria-envelope GeoTIFF into 12 canonical-name .npy tiles.

    When wipe_stale=True (default), any pre-existing ID*.npy whose filename
    doesn't match the canonical 12-tile set is deleted first.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if wipe_stale:
        n_wiped = _wipe_stale_tiles(out_dir)
        if n_wiped:
            print(f"  wiped {n_wiped} stale tile(s) from {out_dir.name}")
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
            arr = _fill_nan(arr)
            out_path = out_dir / encode_filename(tid)
            np.save(out_path, arr)
            n_written += 1
    return n_written


def materialize_static_as_yearly(physical_dir: Path, band_name: str,
                                  out_root: Path, years: range) -> None:
    """Symlink every year directory in `years` to point at `physical_dir`.

    Layout produced:
        out_root/YearlyValue/<band>/<year>/  → symlink → physical_dir/

    The dataloader keys `coordinates.npy` by the last-two folder
    components (e.g. `'Slope/2003'`), so each year must appear as its
    own directory. Using directory-level symlinks keeps disk usage at
    1× the physical tile set per band, instead of 22×.
    """
    yearly_root = out_root / 'YearlyValue' / band_name
    yearly_root.mkdir(parents=True, exist_ok=True)
    physical_abs = physical_dir.resolve()
    for y in years:
        link = yearly_root / str(y)
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == physical_abs:
                continue  # already correct
            link.unlink()
        link.symlink_to(physical_abs, target_is_directory=True)
    print(f"  ✓ {band_name}: symlinked {len(list(years))} year-dirs "
          f"→ {physical_abs.relative_to(out_root) if physical_abs.is_relative_to(out_root) else physical_abs}")


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


def _output_complete(out_dir: Path) -> bool:
    """A (band, year) output is complete iff all 12 *canonical-name* tiles exist.

    Checking canonical names (not just any 12 ID*.npy files) means we don't
    skip a redownload of an existing band that has stale non-canonical tiles
    from an older pipeline run.
    """
    if not out_dir.exists():
        return False
    return all((out_dir / encode_filename(tid)).exists() for tid in range(12))


def cut_all(tiff_dir: Path, out_root: Path,
            materialize_yearly: set[str] | None = None,
            anchor_year: int = 2002,
            mat_years: range = range(2002, 2024),
            skip_done: bool = True,
            state=None) -> None:
    """Process every TIFF in tiff_dir matching the expected naming pattern.

    Resumable: a TIFF whose 12-tile output already exists is skipped.
    Also records progress in `state` (pipeline_state.State) if provided,
    keyed by TIFF basename under the 'cut' phase.
    """
    materialize_yearly = set(materialize_yearly or [])
    tiffs = sorted(tiff_dir.glob('*.tif'))
    print(f"Found {len(tiffs)} GeoTIFFs in {tiff_dir}")
    bands_processed = set()
    materialized = set()
    n_cut = n_skipped = 0
    for tp in tiffs:
        meta = parse_tiff_filename(tp.name)
        if meta is None:
            print(f"  skip (no match): {tp.name}")
            continue
        band = meta['band']
        is_static = meta['year'] is None
        if is_static and band in materialize_yearly:
            out_dir = out_root / 'YearlyValue' / band / str(anchor_year)
            materialized.add(band)
        elif is_static:
            out_dir = out_root / 'StaticValue' / band
        else:
            out_dir = out_root / 'YearlyValue' / band / str(meta['year'])

        already_done = skip_done and _output_complete(out_dir) and (
            state is None or state.is_done('cut', tp.name)
        )
        if already_done:
            n_skipped += 1
            print(f"  · skip (done): {tp.name:50}  → {out_dir.relative_to(out_root)}/")
        else:
            n = cut_tiff(tp, out_dir)
            n_cut += 1
            print(f"  ✓ {tp.name:50}  → {out_dir.relative_to(out_root)}/  ({n} tiles)")
            if state is not None:
                state.mark_done('cut', tp.name)
        bands_processed.add((is_static and band not in materialize_yearly, band))

    # Emit bounds_array_<band>.npy at the appropriate tier root, once per band
    for is_static, band in bands_processed:
        tier = 'StaticValue' if is_static else 'YearlyValue'
        make_bounds_array(out_root / tier, band)
    for band in materialized:
        # Materialized statics live under YearlyValue — drop bounds-array there.
        make_bounds_array(out_root / 'YearlyValue', band)

    # Symlink year directories for materialized bands.
    for band in materialized:
        physical = out_root / 'YearlyValue' / band / str(anchor_year)
        materialize_static_as_yearly(physical, band, out_root,
                                     years=[y for y in mat_years if y != anchor_year])
    print(f"\ncut summary: {n_cut} cut, {n_skipped} skipped-as-done")


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
    ap.add_argument('--materialize-yearly', nargs='+', default=[],
                    metavar='BAND',
                    help='Static-band names to write into YearlyValue/<band>/<anchor>/ '
                         'and symlink across mat-years (default 2002–2023).')
    ap.add_argument('--anchor-year', type=int, default=2002,
                    help='Physical-data year for --materialize-yearly (default 2002)')
    ap.add_argument('--mat-years', nargs=2, type=int, default=[2002, 2023],
                    metavar=('START', 'END'),
                    help='Year range to symlink for --materialize-yearly (default 2002 2023)')
    ap.add_argument('--no-skip-done', action='store_true',
                    help='Re-cut TIFFs even when their output tiles already exist on disk.')
    ap.add_argument('--no-state', action='store_true',
                    help='Do not read/write Data/pipeline_state.json.')
    args = ap.parse_args()

    # Pipeline state — opt-out via --no-state.
    state = None
    if not args.no_state:
        try:
            from pipeline_state import State
            state = State()
            state.start_phase('cut')
        except Exception as ex:
            print(f'[warn] could not load pipeline_state: {ex}; continuing without state tracking')

    mat_years = range(args.mat_years[0], args.mat_years[1] + 1)
    mat_set = set(args.materialize_yearly)

    if args.tiff is not None:
        if not args.band_name:
            ap.error('--tiff requires --band-name')
        if mat_set and args.band_name in mat_set:
            out_dir = args.out_root / 'YearlyValue' / args.band_name / str(args.anchor_year)
            n = cut_tiff(args.tiff, out_dir)
            print(f"✓ cut {n} tiles → {out_dir}")
            make_bounds_array(args.out_root / 'YearlyValue', args.band_name)
            materialize_static_as_yearly(out_dir, args.band_name, args.out_root,
                                         years=[y for y in mat_years if y != args.anchor_year])
        else:
            if not args.tier:
                ap.error('--tiff requires --tier (or --materialize-yearly including this band)')
            if args.tier == 'YearlyValue' and args.year is None:
                ap.error('YearlyValue tier requires --year')
            out_dir = args.out_root / args.tier / args.band_name
            if args.year is not None:
                out_dir = out_dir / str(args.year)
            n = cut_tiff(args.tiff, out_dir)
            print(f"✓ cut {n} tiles → {out_dir}")
            make_bounds_array(args.out_root / args.tier, args.band_name)
    elif args.tiff_dir is not None:
        cut_all(args.tiff_dir, args.out_root,
                materialize_yearly=mat_set,
                anchor_year=args.anchor_year,
                mat_years=mat_years,
                skip_done=not args.no_skip_done,
                state=state)
        if state is not None:
            state.finish_phase('cut')
    else:
        ap.error('provide --tiff-dir or --tiff')


if __name__ == '__main__':
    main()
