"""Compute slope, aspect, and TWI from existing Elevation tiles.

Inputs:  Data/RasterTensorData/StaticValue/Elevation/ID*.npy   (979x979)
Outputs: Data/RasterTensorData/YearlyValue/{Slope,Aspect,TWI}/<year>/
         (same tile filenames, float32)

The output is materialized under YearlyValue (not StaticValue) so the
dataloader's `int(year)` parse on the leaf folder name works without
edits. Year 2002 is the physical anchor; 2003..2023 are symlinks.

Each tile's geo-extent is encoded in the filename:
    IDxxN<top_lat>S<bot_lat>W<left_lon>E<right_lon>.npy

Pixel spacing is derived from that extent so slope is in real-world
units (degrees from horizontal). At Bavaria latitudes (~48-50N):
  - 1° latitude  ≈ 111.0 km
  - 1° longitude ≈  71.0 km  (varies with lat: 111 * cos(lat))

After this script runs, append to each model's config.py:
    bands_list_order = [
        'Elevation', 'LAI', 'LST', 'MODIS_NPP',
        'SoilEvaporation', 'TotalEvapotranspiration',   # existing 6
        ..., 'Slope', 'Aspect', 'TWI',                  # appended
    ]

PlanCurvature (originally listed in the BAND_EXPANSION_PLAN) is dropped
here — TWI is the stronger SOC predictor (wet soils → high SOC) and
plan curvature is highly correlated with slope.
"""
import os
import re
import sys
import numpy as np
from pathlib import Path

# Path resolution — match how the rest of the project does it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import SOC_DATA_DIR_STR

ELEV_DIR = Path(SOC_DATA_DIR_STR) / 'RasterTensorData' / 'StaticValue' / 'Elevation'
OUT_ROOT = Path(SOC_DATA_DIR_STR) / 'RasterTensorData'
ANCHOR_YEAR = 2002
MAT_YEARS = range(2002, 2024)
DERIVED_BANDS = ('Slope', 'Aspect', 'TWI')

R_EARTH_M = 6371000.0  # mean Earth radius


def parse_bbox_from_filename(fname: str) -> tuple[float, float, float, float]:
    """Extract (N, S, W, E) bounds in degrees from the IDxNxxSxxWxxExx pattern.

    Filename pattern: IDnnN<lat>S<lat>W<lon>E<lon>.npy where '_' in numbers
    is a decimal point (e.g. '48_5095' = 48.5095°).
    """
    name = fname.replace('_', '.')
    m = re.search(r'N(-?\d+(?:\.\d+)?)S(-?\d+(?:\.\d+)?)W(-?\d+(?:\.\d+)?)E(-?\d+(?:\.\d+)?)', name)
    if not m:
        raise ValueError(f"Could not parse bbox from {fname}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))


def pixel_size_meters(north_deg: float, south_deg: float,
                      west_deg: float,  east_deg: float,
                      n_rows: int, n_cols: int) -> tuple[float, float]:
    """Return (dy_m, dx_m) — pixel size in meters at the tile's centre latitude."""
    centre_lat = 0.5 * (north_deg + south_deg)
    lat_range_deg = abs(north_deg - south_deg)
    lon_range_deg = abs(east_deg - west_deg)
    m_per_deg_lat = (np.pi / 180.0) * R_EARTH_M
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(centre_lat))
    dy_m = (lat_range_deg * m_per_deg_lat) / n_rows
    dx_m = (lon_range_deg * m_per_deg_lon) / n_cols
    return dy_m, dx_m


def compute_slope_aspect(z: np.ndarray, dy_m: float, dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope in degrees, aspect in degrees [0, 360).

    Uses numpy.gradient (central differences). Slope = atan(|grad|),
    Aspect = atan2(dz/dy, dz/dx) measured clockwise from north (standard
    GIS convention). NaNs are propagated.
    """
    z = z.astype(np.float32)
    dzdy, dzdx = np.gradient(z, dy_m, dx_m)  # rows=y (south→north), cols=x (west→east)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    aspect_rad = np.arctan2(-dzdx, dzdy)        # 0 = north, +ve = east
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    return slope_deg, aspect_deg.astype(np.float32)


def compute_twi(z: np.ndarray, slope_deg: np.ndarray, dy_m: float, dx_m: float) -> np.ndarray:
    """Topographic Wetness Index = ln(specific catchment area / tan(slope)).

    Specific catchment area approximated by upslope-area heuristic:
    we estimate it as 1 / (number of pixels above us in flow path).
    Cheap proxy: use the elevation rank within a local neighbourhood
    as a quick proxy. A proper D8 flow-accumulation is more accurate
    but requires pysheds/richdem; this proxy is correlated and cheap.

    For a real D8 TWI, install `richdem` and replace the body of this
    function with:
        import richdem as rd
        rda = rd.rdarray(z, no_data=-9999)
        accum = rd.FlowAccumulation(rda, method='D8')
        twi = np.log(accum / np.tan(np.radians(np.maximum(slope_deg, 0.1))))
    """
    # Proxy specific catchment area: pixels lower than z in a 9x9 window.
    # Higher count → more area drains here → wetter.
    from scipy.ndimage import generic_filter
    def lower_count(window):
        c = window[len(window) // 2]
        return np.sum(window < c)
    proxy_sca = generic_filter(z.astype(np.float32), lower_count, size=9, mode='nearest')
    proxy_sca = proxy_sca + 1.0  # avoid log(0)
    slope_rad = np.radians(np.maximum(slope_deg, 0.1))  # floor slope to avoid /0
    twi = np.log(proxy_sca / np.tan(slope_rad)).astype(np.float32)
    return twi


def process_tile(tile_path: Path, anchor_dirs: dict[str, Path]) -> None:
    """Process one Elevation tile → write Slope/Aspect/TWI into anchor year dirs."""
    z = np.load(tile_path)
    n_rows, n_cols = z.shape
    n, s, w, e = parse_bbox_from_filename(tile_path.name)
    dy_m, dx_m = pixel_size_meters(n, s, w, e, n_rows, n_cols)

    slope, aspect = compute_slope_aspect(z, dy_m, dx_m)
    twi = compute_twi(z, slope, dy_m, dx_m)

    np.save(anchor_dirs['Slope']  / tile_path.name, slope)
    np.save(anchor_dirs['Aspect'] / tile_path.name, aspect)
    np.save(anchor_dirs['TWI']    / tile_path.name, twi)


def symlink_years(band: str) -> None:
    """Symlink YearlyValue/<band>/<year>/ → <band>/<ANCHOR_YEAR>/ for non-anchor years."""
    band_root = OUT_ROOT / 'YearlyValue' / band
    anchor_dir = band_root / str(ANCHOR_YEAR)
    physical_abs = anchor_dir.resolve()
    for y in MAT_YEARS:
        if y == ANCHOR_YEAR:
            continue
        link = band_root / str(y)
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == physical_abs:
                continue
            link.unlink()
        link.symlink_to(physical_abs, target_is_directory=True)


def main():
    # Pipeline state — opt-out via PIPELINE_STATE=off env var.
    state = None
    if os.environ.get('PIPELINE_STATE', '').lower() not in ('off', '0', 'false'):
        try:
            from pipeline_state import State
            state = State()
            state.start_phase('derive')
        except Exception as ex:
            print(f'[warn] could not load pipeline_state: {ex}; continuing without state tracking')

    anchor_dirs = {b: OUT_ROOT / 'YearlyValue' / b / str(ANCHOR_YEAR) for b in DERIVED_BANDS}
    for d in anchor_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tiles = sorted(ELEV_DIR.glob('ID*.npy'))
    print(f"Processing {len(tiles)} Elevation tiles → {', '.join(DERIVED_BANDS)} "
          f"(anchor year {ANCHOR_YEAR}, symlinked across {MAT_YEARS.start}-{MAT_YEARS.stop-1})")
    n_done = n_skip = 0
    for tp in tiles:
        # Skip-if-done: all three output tiles exist for this Elevation tile.
        skip = state is not None and state.is_done('derive', tp.name) and all(
            (anchor_dirs[b] / tp.name).exists() for b in DERIVED_BANDS
        )
        if skip:
            print(f"  · skip (done): {tp.name}")
            n_skip += 1
            continue
        try:
            process_tile(tp, anchor_dirs)
            print(f"  ✓ {tp.name}")
            n_done += 1
            if state is not None:
                state.mark_done('derive', tp.name)
        except Exception as ex:
            print(f"  ✗ {tp.name}: {ex}")
    print(f"derive summary: {n_done} computed, {n_skip} skipped-as-done")

    # Bounds arrays for YearlyValue (one per band, mirrors anchor-year tiles).
    static_bounds = OUT_ROOT / 'StaticValue' / 'bounds_array_Elevation.npy'
    if static_bounds.exists():
        import shutil
        yearly_tier = OUT_ROOT / 'YearlyValue'
        yearly_tier.mkdir(parents=True, exist_ok=True)
        for b in DERIVED_BANDS:
            dst = yearly_tier / f'bounds_array_{b}.npy'
            if not dst.exists():
                shutil.copy(static_bounds, dst)
                print(f"cloned bounds_array → YearlyValue/{dst.name}")

    # Symlink 2003..2023 to 2002 for each derived band.
    for b in DERIVED_BANDS:
        symlink_years(b)
        print(f"  symlinked {len(MAT_YEARS) - 1} year-dirs for {b}")

    if state is not None:
        state.finish_phase('derive')

    print("\nNext steps:")
    print(f"  1. python SamplePoints/project_lucas_coords.py --tier YearlyValue \\")
    print(f"         --bands {' '.join(DERIVED_BANDS)} --materialize-yearly {' '.join(DERIVED_BANDS)}")
    print(f"  2. Append {DERIVED_BANDS} to bands_list_order in each model's config.py")
    print(f"     and add the corresponding path variables.")
    print(f"  3. Retrain with strict=False on any pre-existing 6-channel checkpoint.")


if __name__ == '__main__':
    main()
