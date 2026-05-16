"""Compute slope, aspect, plan curvature, and TWI from existing Elevation tiles.

Inputs:  Data/RasterTensorData/StaticValue/Elevation/ID*.npy   (979x979 int16)
Outputs: Data/RasterTensorData/StaticValue/{Slope,Aspect,PlanCurvature,TWI}/
         (same tile filenames, float32)

Each tile's geo-extent is encoded in the filename:
    IDxxN<top_lat>S<bot_lat>W<left_lon>E<right_lon>.npy

Pixel spacing is derived from that extent so slope is in real-world
units (degrees from horizontal). At Bavaria latitudes (~48-50N):
  - 1° latitude  ≈ 111.0 km
  - 1° longitude ≈  71.0 km  (varies with lat: 111 * cos(lat))

After this script runs, append to config.py:
    bands_list_order = ['Elevation', 'Slope', 'Aspect', 'PlanCurvature', 'TWI',
                        'LAI', 'LST', 'MODIS_NPP', 'SoilEvaporation',
                        'TotalEvapotranspiration']

(Append, don't prepend — preserves checkpoint compatibility for the
existing channels; the 4 new channels need first-conv re-init via
load_state_dict(..., strict=False).)
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
OUT_ROOT = Path(SOC_DATA_DIR_STR) / 'RasterTensorData' / 'StaticValue'

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


def compute_plan_curvature(z: np.ndarray, dy_m: float, dx_m: float) -> np.ndarray:
    """Plan curvature: curvature of contour lines (perpendicular to slope).

    Positive = divergent (ridge-like), negative = convergent (valley-like).
    Computed as second-derivative combination per Zevenbergen-Thorne 1987.
    """
    z = z.astype(np.float32)
    dzdy, dzdx = np.gradient(z, dy_m, dx_m)
    d2zdy2, d2zdyx = np.gradient(dzdy, dy_m, dx_m)
    _,      d2zdx2 = np.gradient(dzdx, dy_m, dx_m)
    p = dzdx; q = dzdy
    p2_q2 = p * p + q * q
    # Avoid /0 on perfectly flat pixels
    safe = np.where(p2_q2 > 1e-12, p2_q2, 1.0)
    plan = (d2zdx2 * q * q - 2 * d2zdyx * p * q + d2zdy2 * p * p) / (safe * np.sqrt(safe))
    plan = np.where(p2_q2 > 1e-12, plan, 0.0).astype(np.float32)
    return plan


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


def process_tile(tile_path: Path, out_dirs: dict[str, Path]) -> None:
    z = np.load(tile_path)
    n_rows, n_cols = z.shape
    n, s, w, e = parse_bbox_from_filename(tile_path.name)
    dy_m, dx_m = pixel_size_meters(n, s, w, e, n_rows, n_cols)

    slope, aspect = compute_slope_aspect(z, dy_m, dx_m)
    plan = compute_plan_curvature(z, dy_m, dx_m)
    twi = compute_twi(z, slope, dy_m, dx_m)

    np.save(out_dirs['Slope']          / tile_path.name, slope)
    np.save(out_dirs['Aspect']         / tile_path.name, aspect)
    np.save(out_dirs['PlanCurvature']  / tile_path.name, plan)
    np.save(out_dirs['TWI']            / tile_path.name, twi)


def main():
    out_dirs = {b: OUT_ROOT / b for b in ('Slope', 'Aspect', 'PlanCurvature', 'TWI')}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tiles = sorted(ELEV_DIR.glob('ID*.npy'))
    print(f"Processing {len(tiles)} Elevation tiles → 4 new bands")
    for tp in tiles:
        try:
            process_tile(tp, out_dirs)
            print(f"  ✓ {tp.name}")
        except Exception as ex:
            print(f"  ✗ {tp.name}: {ex}")

    # Also clone the bounds_array_Elevation.npy for the new bands — the
    # samplePoints.py pipeline expects one bounds-array per band, but
    # since the four new bands share the Elevation tile layout exactly,
    # they share the same bounds array.
    bounds_src = OUT_ROOT / 'bounds_array_Elevation.npy'
    if bounds_src.exists():
        for b in out_dirs:
            dst = OUT_ROOT / f'bounds_array_{b}.npy'
            if not dst.exists():
                import shutil
                shutil.copy(bounds_src, dst)
                print(f"cloned bounds_array → {dst.name}")

    print("\nNext steps:")
    print("  1. Re-run SamplePoints/samplePoints.py to project LUCAS points onto the 4 new bands")
    print("     (it will skip already-processed bands and only add the new ones).")
    print("  2. Append the 4 new band names to bands_list_order in config.py:")
    print('     bands_list_order = ["Elevation", "Slope", "Aspect", "PlanCurvature", "TWI",')
    print('                         "LAI", "LST", "MODIS_NPP", "SoilEvaporation",')
    print('                         "TotalEvapotranspiration"]')
    print("  3. Retrain with strict=False on any pre-existing checkpoint loading")
    print("     (input_channels goes from 6 to 10).")


if __name__ == '__main__':
    main()
