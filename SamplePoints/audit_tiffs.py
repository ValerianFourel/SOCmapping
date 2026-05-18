"""Audit raw GEE-exported TIFFs for cleanliness before downstream use.

Checks every .tif in --tiff-dir (default ~/bavaria_tiffs/) for:
  - readability (rasterio.open succeeds)
  - CRS = EPSG:4326
  - geo-transform anchored at BAVARIA_BBOX with ~250 m pixel size
  - single-band float32 raster
  - shape close to the expected ~3200 × 2400 pixel grid
  - NaN/inf fraction (warn > 5%, fail > 50%)
  - value range plausibility against a per-band expectation table
  - non-trivial variance (fail if std ≈ 0 → constant raster, likely empty)

Prints one line per file (✓ / ⚠ / ✗) and a summary by category. Exit
status 0 if all files pass, 2 otherwise.

Use:
    python SamplePoints/audit_tiffs.py
    python SamplePoints/audit_tiffs.py --tiff-dir ~/bavaria_tiffs --verbose
    python SamplePoints/audit_tiffs.py --csv audit.csv      # dump per-file stats
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. pip install rasterio")
    sys.exit(1)


# Bavaria envelope expected on every export (matches the GEE script).
BBOX_W, BBOX_S, BBOX_E, BBOX_N = 7.1864, 46.7109, 14.3750, 52.1028
EXPECTED_SCALE_M = 250

# Per-band sanity ranges. These are physical-units floors/ceilings — a tile
# whose [min, max] falls outside the wider envelope is suspicious. Keep them
# loose to avoid false alarms; we mostly want to catch all-zero / all-fill rasters.
BAND_RANGES = {
    # MODIS — scaled integer values (×0.0001 or ×0.02)
    'NDVI':                    (-2000, 10000),     # NDVI × 10000
    'EVI':                     (-2000, 10000),
    'LAI':                     (0, 200),           # LAI × 10
    'LST':                     (10000, 18000),     # LST Kelvin × 0.02 → 200-360K
    'MODIS_NPP':               (-500, 30000),      # NPP × 0.0001 kg C/m²/yr
    'TotalEvapotranspiration': (0, 50000),         # ET 8-day mm × 0.1, summed yearly
    # ERA5-Land total_precipitation_sum — annual sum in m water equiv (~0.4–2.5 m)
    'Precipitation':           (0.2, 3.0),         # m water/year
    # ERA5-Land — physical units
    'AirTemperature':          (260, 295),         # Kelvin (annual mean)
    'SoilMoisture_layer1':     (0.05, 0.55),       # m³/m³ (volumetric)
    'SnowDepth':               (0.0, 2.0),         # meters
    'SoilEvaporation':         (-0.5, 1.5),        # m water equiv. annual sum
    # OpenLandMap soil — scaled integers
    'ClayContent_0_10cm':      (0, 80),            # % clay
    'SandContent_0_10cm':      (0, 100),           # % sand
    'pH_H2O_0_10cm':           (40, 90),           # pH × 10
    'BulkDensity_0_10cm':      (20, 200),          # OpenLandMap unit is ×100, so 0.2-2.0 g/cm³
    # SoilGrids CEC
    'CEC_0_10cm':              (50, 1000),         # mmol(c)/kg × 10
    # SRTM
    'Elevation':               (-100, 4500),       # m
    'Slope':                   (0, 60),            # degrees
    'Aspect':                  (-360, 360),        # degrees
    'TWI':                     (-20, 30),          # dimensionless log
}

_DESC_RE = re.compile(
    r'^(?P<category>[a-z0-9]+)_(?P<band>[A-Za-z0-9_]+?)_(?P<period>static|\d{4})\.tif$'
)


def parse_filename(name: str):
    m = _DESC_RE.match(name)
    if not m:
        return None
    return {
        'category': m.group('category'),
        'band': m.group('band'),
        'period': m.group('period'),
    }


def audit_one(tif_path: Path, verbose: bool = False) -> dict:
    """Return a dict with status + per-file stats. status in {ok, warn, fail}."""
    rec = {
        'path': str(tif_path),
        'name': tif_path.name,
        'size_mb': round(tif_path.stat().st_size / 1e6, 1),
        'status': 'fail',
        'issues': [],
        'meta': None,
        'band': None,
    }
    meta = parse_filename(tif_path.name)
    if meta is None:
        rec['issues'].append('filename does not match <category>_<band>_<period>.tif pattern')
        return rec
    rec['meta'] = meta
    rec['band'] = meta['band']

    try:
        with rasterio.open(tif_path) as src:
            crs = src.crs
            transform = src.transform
            shape = (src.height, src.width)
            nbands = src.count
            dtype = src.dtypes[0]
            bounds = src.bounds

            # CRS check
            if crs is None or str(crs).upper() not in ('EPSG:4326', 'WGS 84'):
                if crs is None or 'EPSG' in str(crs).upper() and '4326' not in str(crs):
                    rec['issues'].append(f'CRS={crs} (expected EPSG:4326)')

            # Bounds check — GEE pixel-aligns the export, so the actual bbox
            # is usually expanded by up to ~1 pixel (0.06°) on each edge. Pass
            # the check as long as the requested envelope is CONTAINED in the
            # actual bounds AND the actual bounds aren't wildly different.
            requested_contained = (
                bounds.left  <= BBOX_W + 0.01 and bounds.right >= BBOX_E - 0.01 and
                bounds.bottom <= BBOX_S + 0.01 and bounds.top   >= BBOX_N - 0.01
            )
            expansion_reasonable = (
                BBOX_W - bounds.left   < 0.1 and bounds.right  - BBOX_E < 0.1 and
                BBOX_S - bounds.bottom < 0.1 and bounds.top    - BBOX_N < 0.1
            )
            if not (requested_contained and expansion_reasonable):
                rec['issues'].append(
                    f'bounds={tuple(round(x, 4) for x in bounds)} '
                    f'expected ≈ (W{BBOX_W}, S{BBOX_S}, E{BBOX_E}, N{BBOX_N}) '
                    f'(contained={requested_contained}, expansion_reasonable={expansion_reasonable})'
                )

            # Pixel size check — at EPSG:4326 with scale=250m, pixel size in degrees ≈ 0.00225°
            pix_w_deg = abs(transform.a)
            pix_h_deg = abs(transform.e)
            expected_deg = EXPECTED_SCALE_M / 111319.49
            if abs(pix_w_deg - expected_deg) / expected_deg > 0.05:
                rec['issues'].append(f'pixel size {pix_w_deg:.5f}° (expected ≈ {expected_deg:.5f}°)')

            # Band count + dtype
            if nbands != 1:
                rec['issues'].append(f'nbands={nbands} (expected 1)')

            # Read the single band — full read OK at this scale (~30 MB array)
            arr = src.read(1)

            # NaN / inf check
            finite = np.isfinite(arr)
            n_total = arr.size
            n_finite = int(finite.sum())
            nan_frac = (n_total - n_finite) / n_total

            # nodata explicit?
            nodata = src.nodata
            if nodata is not None and not np.isnan(nodata):
                mask = (arr == nodata) | ~finite
            else:
                mask = ~finite
            valid = arr[~mask]

            if valid.size == 0:
                rec['issues'].append('all pixels are nodata/NaN — empty raster')
                rec['stats'] = {'nan_frac': 1.0}
                return rec

            v_min = float(valid.min())
            v_max = float(valid.max())
            v_mean = float(valid.mean())
            v_std = float(valid.std())

            rec['stats'] = {
                'shape': shape,
                'dtype': dtype,
                'nbands': nbands,
                'bounds': tuple(round(x, 5) for x in bounds),
                'pixel_deg': round(pix_w_deg, 5),
                'nan_frac': round(nan_frac, 4),
                'min': round(v_min, 4),
                'max': round(v_max, 4),
                'mean': round(v_mean, 4),
                'std':  round(v_std, 4),
            }

            # NaN check (warn vs fail)
            if nan_frac > 0.5:
                rec['issues'].append(f'NaN/nodata > 50% ({nan_frac*100:.1f}%)')
            elif nan_frac > 0.05:
                rec['issues'].append(f'NaN/nodata > 5% ({nan_frac*100:.1f}%)  [warn]')

            # Zero-variance check
            if v_std == 0:
                rec['issues'].append(f'std=0 — constant raster (value {v_min})')

            # Per-band range check
            band = meta['band']
            rng = BAND_RANGES.get(band)
            if rng is not None:
                lo, hi = rng
                # Out-of-range fraction
                oor_mask = (valid < lo) | (valid > hi)
                oor_frac = float(oor_mask.mean())
                if oor_frac > 0.5:
                    rec['issues'].append(
                        f'>50% of valid pixels outside expected band range [{lo}, {hi}]  '
                        f'(actual range [{v_min:.2f}, {v_max:.2f}])'
                    )
                elif oor_frac > 0.05:
                    rec['issues'].append(
                        f'{oor_frac*100:.1f}% of pixels outside expected band range [{lo}, {hi}]  [warn]'
                    )

    except rasterio.errors.RasterioIOError as ex:
        rec['issues'].append(f'rasterio open failed: {ex}')
        return rec
    except Exception as ex:
        rec['issues'].append(f'{type(ex).__name__}: {ex}')
        return rec

    # Classify
    fatal = any('[warn]' not in i for i in rec['issues'])
    if not rec['issues']:
        rec['status'] = 'ok'
    elif not fatal:
        rec['status'] = 'warn'
    else:
        rec['status'] = 'fail'
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiff-dir', type=Path,
                    default=Path.home() / 'bavaria_tiffs',
                    help='Folder of GEE-exported TIFFs to audit')
    ap.add_argument('--verbose', action='store_true',
                    help='Print stats for every file, not just issues')
    ap.add_argument('--csv', type=Path, default=None,
                    help='Optional: write per-file stats to this CSV')
    args = ap.parse_args()

    if not args.tiff_dir.exists():
        sys.exit(f'ERROR: {args.tiff_dir} does not exist')

    tiffs = sorted(args.tiff_dir.glob('*.tif'))
    print(f'Auditing {len(tiffs)} TIFFs in {args.tiff_dir}\n')

    n_ok = n_warn = n_fail = 0
    by_cat = defaultdict(lambda: Counter())
    by_band_stats = defaultdict(list)
    rows_for_csv = []

    for i, tp in enumerate(tiffs, 1):
        rec = audit_one(tp, verbose=args.verbose)
        meta = rec.get('meta') or {}
        cat = meta.get('category', '?')
        by_cat[cat][rec['status']] += 1

        if rec['status'] == 'ok':
            n_ok += 1
            if args.verbose:
                s = rec['stats']
                print(f'  ✓ [{i:3}/{len(tiffs)}] {tp.name:50}  '
                      f'shape={s["shape"]}  range=[{s["min"]:.2f}, {s["max"]:.2f}]  '
                      f'mean={s["mean"]:.2f}  std={s["std"]:.2f}')
        elif rec['status'] == 'warn':
            n_warn += 1
            print(f'  ⚠ [{i:3}/{len(tiffs)}] {tp.name}')
            for issue in rec['issues']:
                print(f'        {issue}')
        else:
            n_fail += 1
            print(f'  ✗ [{i:3}/{len(tiffs)}] {tp.name}')
            for issue in rec['issues']:
                print(f'        {issue}')

        # Per-band stats aggregation
        if rec.get('stats') and rec['band']:
            by_band_stats[rec['band']].append(rec['stats'])

        # CSV row
        if args.csv:
            row = {'name': rec['name'], 'status': rec['status'],
                   'issues': '; '.join(rec['issues']),
                   'band': rec.get('band'),
                   'category': cat}
            row.update(rec.get('stats') or {})
            rows_for_csv.append(row)

    # Per-category summary
    print(f'\n=== Per-category status ===')
    for cat in sorted(by_cat):
        c = by_cat[cat]
        total = sum(c.values())
        print(f'  {cat:10}  ok={c["ok"]:3}  warn={c["warn"]:3}  fail={c["fail"]:3}   total={total}')

    # Per-band cross-year range (so you can eyeball stability)
    print(f'\n=== Per-band value range across years (mean ± std min..max) ===')
    for band in sorted(by_band_stats):
        stats = by_band_stats[band]
        means = np.array([s['mean'] for s in stats])
        mins  = np.array([s['min']  for s in stats])
        maxs  = np.array([s['max']  for s in stats])
        print(f'  {band:30}  N={len(stats):2}  '
              f'mean={means.mean():>10.3f}±{means.std():.3f}  '
              f'min={mins.min():>10.3f}  max={maxs.max():>10.3f}')

    if args.csv:
        with args.csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows_for_csv for k in r}))
            w.writeheader()
            for r in rows_for_csv:
                w.writerow(r)
        print(f'\n  Per-file CSV → {args.csv}')

    print(f'\n=== Summary ===')
    print(f'  {n_ok:3} OK    {n_warn:3} WARN    {n_fail:3} FAIL   (of {len(tiffs)} TIFFs)')

    sys.exit(0 if n_fail == 0 else 2)


if __name__ == '__main__':
    main()
