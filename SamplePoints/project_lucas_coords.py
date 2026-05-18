"""Project GPS points onto a (band, year) tile layout.

Two MODES (controlled by --mode):

  --mode samples (default)
      Source: LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx       (cols GPS_LONG, GPS_LAT)
              This is the COMBINED LUCAS + LFU + LfL soil-sample dataset
              (30,451 ground-truth measurements with SOC values).
      Output: Data/OC_LUCAS_LFU_LfL_Coordinates_v2/...   (training coords)

  --mode 1mil
      Source: Coordinates1Mil/coordinates_Bavaria_1mil.csv (cols longitude, latitude)
              1.3 million Bavaria reference grid points for full-map inference.
      Output: Data/Coordinates1Mil/...                     (inference coords)

Each coordinates.npy has columns [latitude, longitude, tile_id, row, col].

The projection is vectorized — 1.3M points × 20 bands typically completes
in under 5 minutes total when using --symlink-across-years.

Usage
-----
Single (band, year), samples (30k LUCAS+LFU+LfL):
  python SamplePoints/project_lucas_coords.py \
      --raster-dir Data/RasterTensorData/YearlyValue/NDVI/2010

All bands & years for a tier, 1mil:
  python SamplePoints/project_lucas_coords.py --mode 1mil \
      --tier YearlyValue --bands NDVI EVI Precipitation \
      --symlink-across-years
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

# Per-mode source + output configuration.
MODE_SAMPLES = {
    'coords_root': DATA_ROOT / 'OC_LUCAS_LFU_LfL_Coordinates_v2',
    'source':      DATA_ROOT / 'LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx',
    'lon_col':     'GPS_LONG',
    'lat_col':     'GPS_LAT',
    'label':       'combined LUCAS+LFU+LfL soil samples (30k ground-truth points)',
}
MODE_1MIL = {
    'coords_root': DATA_ROOT / 'Coordinates1Mil',
    'source':      DATA_ROOT / 'Coordinates1Mil' / 'coordinates_Bavaria_1mil.csv',
    'lon_col':     'longitude',
    'lat_col':     'latitude',
    'label':       'Bavaria 1.3M inference grid',
}
MODES = {'samples': MODE_SAMPLES, '1mil': MODE_1MIL}

# Set at runtime in main(); other functions consult COORDS_ROOT lazily via a getter.
COORDS_ROOT = MODE_SAMPLES['coords_root']   # default; rebound in main()

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
    """Return (tile_id, row, col) for a single GPS point (slow scalar path)."""
    _, idx = tree.query([lon, lat], k=min(4, len(tiles)))
    candidates = np.atleast_1d(idx)
    for ci in candidates:
        t = tiles[ci]
        if t['w'] <= lon <= t['e'] and t['s'] <= lat <= t['n']:
            n, s, w, e = t['n'], t['s'], t['w'], t['e']
            tile_px = 979
            col = int(round((lon - w) / (e - w) * (tile_px - 1)))
            row = int(round((n - lat) / (n - s) * (tile_px - 1)))
            col = max(0, min(tile_px - 1, col))
            row = max(0, min(tile_px - 1, row))
            return t['id'], row, col
    return -1, 0, 0


def project_all(raster_dir: Path, lon_lat: np.ndarray) -> np.ndarray:
    """Vectorized projection of N points onto raster_dir's 12-tile grid.

    Returns array of shape (N, 5): [lat, lon, tile_id, row, col].
    For points outside all tile bboxes, tile_id = -1 (sentinel).

    ~100× faster than per-point Python loop. Critical for the 1.3M grid.
    """
    tiles, tree = load_tile_index(raster_dir)
    n_points = len(lon_lat)
    if n_points == 0:
        return np.empty((0, 5), dtype=np.float64)

    lons = lon_lat[:, 0].astype(np.float64)
    lats = lon_lat[:, 1].astype(np.float64)

    # IMPORTANT: store as float64. The dataloader's find_coordinates_index
    # does an exact-equality lookup `coords[:, 1] == longitude`. If we save
    # as float32, the round-trip loses precision and the equality fails for
    # GPS values whose float64 representation isn't exact in float32. The
    # original samplePoints.py used float64.
    out = np.empty((n_points, 5), dtype=np.float64)
    out[:, 0] = lats
    out[:, 1] = lons
    out[:, 2] = -1.0   # tile_id sentinel
    out[:, 3] = 0.0
    out[:, 4] = 0.0

    # Tile bounds as parallel arrays: tile_n[i], tile_s[i], ... for tile index i (NOT tile_id).
    tile_n = np.array([t['n'] for t in tiles], dtype=np.float64)
    tile_s = np.array([t['s'] for t in tiles], dtype=np.float64)
    tile_w = np.array([t['w'] for t in tiles], dtype=np.float64)
    tile_e = np.array([t['e'] for t in tiles], dtype=np.float64)
    tile_id = np.array([t['id'] for t in tiles], dtype=np.float64)

    # Batch KDTree query: get k=4 nearest tile centres for each point.
    k = min(4, len(tiles))
    _, knn_idx = tree.query(np.column_stack([lons, lats]), k=k)
    knn_idx = np.atleast_2d(knn_idx)
    if knn_idx.shape[0] == 1 and knn_idx.size != n_points:
        knn_idx = knn_idx.T  # k=1 edge case

    tile_px = 979
    unassigned = np.ones(n_points, dtype=bool)

    for ki in range(k):
        if not unassigned.any():
            break
        cand = knn_idx[:, ki]                          # (N,) candidate tile-index per point
        cand_n = tile_n[cand]
        cand_s = tile_s[cand]
        cand_w = tile_w[cand]
        cand_e = tile_e[cand]
        in_bbox = (cand_w <= lons) & (lons <= cand_e) & (cand_s <= lats) & (lats <= cand_n)
        match = unassigned & in_bbox
        if not match.any():
            continue
        out[match, 2] = tile_id[cand[match]]
        col = np.round((lons[match] - cand_w[match]) / (cand_e[match] - cand_w[match]) * (tile_px - 1))
        row = np.round((cand_n[match] - lats[match]) / (cand_n[match] - cand_s[match]) * (tile_px - 1))
        out[match, 3] = np.clip(row, 0, tile_px - 1)
        out[match, 4] = np.clip(col, 0, tile_px - 1)
        unassigned &= ~match

    return out


def coords_out_dir(raster_dir: Path) -> Path:
    """Mirror RasterTensorData/<tier>/<band>/[<year>] into the configured
    COORDS_ROOT (OC_LUCAS_LFU_LfL_Coordinates_v2/ or Coordinates1Mil/)."""
    rel = raster_dir.relative_to(RASTER_ROOT)
    return COORDS_ROOT / rel


def process_single_dir(raster_dir: Path, lon_lat: np.ndarray) -> Path:
    out_dir = coords_out_dir(raster_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = project_all(raster_dir, lon_lat)
    target = out_dir / 'coordinates.npy'
    np.save(target, arr)
    return target


def symlink_coords_across_years(physical_file: Path, band: str,
                                 years: list[int]) -> None:
    """Symlink the per-band coordinates.npy across multiple year-dirs.

    Used for materialize-yearly statics AND for pure-yearly bands when
    the user opts in. Saves duplicating an identical large table
    22× per band.
    """
    physical_abs = physical_file.resolve()
    band_root = COORDS_ROOT / 'YearlyValue' / band
    for y in years:
        ydir = band_root / str(y)
        ydir.mkdir(parents=True, exist_ok=True)
        link = ydir / 'coordinates.npy'
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == physical_abs:
                continue
            link.unlink()
        link.symlink_to(physical_abs)


def main():
    global COORDS_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['samples', '1mil'], default='samples',
                    help='samples: project combined LUCAS+LFU+LfL xlsx → '
                         'OC_LUCAS_LFU_LfL_Coordinates_v2/ (training coords). '
                         '1mil:    project Bavaria 1.3M CSV → Coordinates1Mil/ (inference coords).')
    ap.add_argument('--raster-dir', type=Path,
                    help='Single band-year folder containing ID*.npy tiles')
    ap.add_argument('--tier', choices=['StaticValue', 'YearlyValue'],
                    help='When using --bands, the tier to walk')
    ap.add_argument('--bands', nargs='+',
                    help='Band names under --tier to process (auto-discovers years for YearlyValue)')
    ap.add_argument('--coords-csv', type=Path,
                    help='Override the per-mode default coords source path')
    ap.add_argument('--materialize-yearly', nargs='+', default=[],
                    metavar='BAND',
                    help='For each named band: project once against '
                         'YearlyValue/<band>/<anchor>/ then symlink '
                         'coordinates.npy across all mat-years. Requires '
                         'tiff_to_tiles.py to have been run with the '
                         'matching --materialize-yearly first.')
    ap.add_argument('--anchor-year', type=int, default=2002,
                    help='Anchor year for --materialize-yearly (default 2002)')
    ap.add_argument('--mat-years', nargs=2, type=int, default=[2002, 2023],
                    metavar=('START', 'END'),
                    help='Year range to symlink for --materialize-yearly (default 2002 2023)')
    ap.add_argument('--symlink-across-years', action='store_true',
                    help='For pure-yearly --bands (not in --materialize-yearly): '
                         'project once at the first available year and symlink '
                         'coordinates.npy across the rest. Optional disk saver.')
    ap.add_argument('--no-skip-done', action='store_true',
                    help='Re-project even when coordinates.npy already exists at the output path.')
    ap.add_argument('--no-state', action='store_true',
                    help='Do not read/write Data/pipeline_state.json.')
    args = ap.parse_args()

    # Pipeline state — opt-out via --no-state.
    state = None
    if not args.no_state:
        try:
            from pipeline_state import State
            state = State()
            state.start_phase('project')
        except Exception as ex:
            print(f'[warn] could not load pipeline_state: {ex}; continuing without state tracking')

    def _key(band, year):
        return f'{band}/{year}' if year is not None else band

    def _output_exists(raster_dir: Path) -> bool:
        """coordinates.npy exists AND is at least as new as any tile file
        in raster_dir. If tiles were recut (e.g. after a redownload), the
        coordinates.npy is stale and must be regenerated.
        """
        target = coords_out_dir(raster_dir) / 'coordinates.npy'
        if not target.exists():
            return False
        coords_mtime = target.stat().st_mtime
        for tile in raster_dir.glob('ID*.npy'):
            if tile.stat().st_mtime > coords_mtime:
                return False     # tile newer than coords ⇒ stale
        return True

    # Configure source + output root from --mode.
    mode_cfg = MODES[args.mode]
    COORDS_ROOT = mode_cfg['coords_root']
    src = args.coords_csv or mode_cfg['source']
    lon_col, lat_col = mode_cfg['lon_col'], mode_cfg['lat_col']

    if src.suffix.lower() in ('.xls', '.xlsx'):
        df = pd.read_excel(src, usecols=[lon_col, lat_col])
    else:
        df = pd.read_csv(src, usecols=[lon_col, lat_col])
    df = df.dropna()
    lon_lat = df[[lon_col, lat_col]].to_numpy(dtype=np.float64)
    print(f"[{args.mode}] Projecting {len(lon_lat):,} points ({mode_cfg['label']})")
    print(f"  source: {src}")
    print(f"  output: {COORDS_ROOT}")

    mat_set = set(args.materialize_yearly)
    mat_years = list(range(args.mat_years[0], args.mat_years[1] + 1))

    skip_done = not args.no_skip_done
    n_proj = n_skipped = 0

    if args.raster_dir is not None:
        if skip_done and _output_exists(args.raster_dir):
            print(f"  · skip (done): {coords_out_dir(args.raster_dir) / 'coordinates.npy'}")
            n_skipped += 1
        else:
            out = process_single_dir(args.raster_dir, lon_lat)
            print(f"  ✓ wrote {out}")
            n_proj += 1
            if state is not None:
                state.mark_done('project', str(args.raster_dir))
        if state is not None:
            state.finish_phase('project')
        print(f"\nproject summary: {n_proj} projected, {n_skipped} skipped-as-done")
        return

    if not (args.tier and args.bands):
        ap.error('provide either --raster-dir, or --tier + --bands')

    tier_root = RASTER_ROOT / args.tier
    for band in args.bands:
        band_dir = tier_root / band
        if not band_dir.exists():
            print(f"  skip {band}: {band_dir} does not exist")
            continue

        if band in mat_set:
            # Materialize-yearly: project once against anchor year, symlink the rest.
            anchor_dir = band_dir / str(args.anchor_year)
            if not anchor_dir.exists():
                print(f"  skip {band}: anchor {anchor_dir} does not exist "
                      f"(run tiff_to_tiles.py --materialize-yearly {band} first)")
                continue
            if skip_done and _output_exists(anchor_dir):
                print(f"  · skip (done): {band:25} anchor coords already at {coords_out_dir(anchor_dir)}")
                n_skipped += 1
            else:
                physical = process_single_dir(anchor_dir, lon_lat)
                print(f"  ✓ {band:25} anchor → {physical}")
                n_proj += 1
                if state is not None:
                    state.mark_done('project', _key(band, args.anchor_year))
            symlink_coords_across_years(coords_out_dir(anchor_dir) / 'coordinates.npy', band,
                                        [y for y in mat_years if y != args.anchor_year])
            print(f"    └─ symlinked {len(mat_years) - 1} year-dirs")
            continue

        if args.tier == 'StaticValue':
            if skip_done and _output_exists(band_dir):
                print(f"  · skip (done): {band:25} {coords_out_dir(band_dir)}")
                n_skipped += 1
            else:
                out = process_single_dir(band_dir, lon_lat)
                print(f"  ✓ {band:25}  → {out}")
                n_proj += 1
                if state is not None:
                    state.mark_done('project', _key(band, None))
            continue

        # Pure yearly band
        years = sorted([d for d in band_dir.iterdir()
                        if d.is_dir() and d.name.isdigit()])
        if args.symlink_across_years and years:
            anchor = years[0]
            if skip_done and _output_exists(anchor):
                print(f"  · skip (done): {band}/{anchor.name:5} anchor coords exist")
                n_skipped += 1
            else:
                physical = process_single_dir(anchor, lon_lat)
                print(f"  ✓ {band}/{anchor.name:5} anchor → {physical}")
                n_proj += 1
                if state is not None:
                    state.mark_done('project', _key(band, int(anchor.name)))
            other_years = [int(d.name) for d in years[1:]]
            symlink_coords_across_years(coords_out_dir(anchor) / 'coordinates.npy', band, other_years)
            print(f"    └─ symlinked {len(other_years)} year-dirs")
        else:
            for y_dir in years:
                if skip_done and _output_exists(y_dir):
                    n_skipped += 1
                    print(f"  · skip (done): {band}/{y_dir.name}")
                    continue
                out = process_single_dir(y_dir, lon_lat)
                print(f"  ✓ {band}/{y_dir.name:5}  → {out}")
                n_proj += 1
                if state is not None:
                    state.mark_done('project', _key(band, int(y_dir.name)))

    if state is not None:
        state.finish_phase('project')
    print(f"\nproject summary: {n_proj} projected, {n_skipped} skipped-as-done")


if __name__ == '__main__':
    main()
