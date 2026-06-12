#!/usr/bin/env python3
"""Sentinel-2 SWIR bare-soil export for the SGT band stack (Tier 4).

Exports a MULTI-YEAR STATIC bare-soil composite of the two Sentinel-2 SWIR
bands from COPERNICUS/S2_SR_HARMONIZED:

    S2SRC_SWIR1  = B11, ~1.61 um (20 m native)
    S2SRC_SWIR2  = B12, ~2.19 um (20 m native)
    S2SRC_ExposureCount = count of valid bare-soil observations (validity mask)

WHY STATIC (not per-year, unlike the Landsat SRC):
    Sentinel-2 surface reflectance (S2_SR_HARMONIZED) only starts ~2017, but the
    SGT dataset spans 2002-2023 with 5-year windows. A per-year S2 stack would be
    NoData for 2002-2016 — breaking the input cube for most of the record. So we
    take ONE bare-soil composite over the whole S2 archive (default 2017-2023)
    and materialize it as a StaticValue band (same value for every year window),
    exactly like Elevation. This is the Zepp/Broeg/Tziolas exposed-soil SWIR
    approach: a single multi-year bare-soil mosaic of the highest-SOC-signal bands.

SENTINEL MODE (--scale):
    Default export pixel size is 20 m ("sentinel size" — the native B11/B12
    resolution). For the existing 250 m project grid, pass --scale 250 (the
    standardize/tile step then aligns it to the 979x979 tiles). See the README
    S2_SWIR_BANDS_README.md for the two integration paths (20 m sentinel-mode
    variant vs 250 m drop-in for the current 43-band stack).

Pipeline:
    1) this script        -> Drive GeoTIFF  s2_swir_baresoil_<y0>_<y1>.tif (3 bands)
    2) sgt_landsat_mb_cut.py (analog)  -> RasterTensorData/StaticValue/S2SRC_*/  tiles
    3) standardize + publish to HF  ValerianFourel/sgt-bavaria-soc-2002-2023-large
    4) train with  SGT_BANDS_S2=1  --bands-list full_extended_s2  (45 bands)

Run:
    python sgt_s2_swir_export.py                 # 2017-2023, 20 m, bare-soil
    python sgt_s2_swir_export.py --scale 250     # 250 m, drops into the 43-band grid
    python sgt_s2_swir_export.py --years 2017 2023 --cloud-max 40
"""
import argparse
import sys

sys.path.insert(0, "/home/valerian/SGTPublication/SOCmapping/SamplePoints")
import ee  # noqa: E402

ee.Initialize(project="sgtmodel")
import gee_download_all_bands as g  # noqa: E402

DRIVE_FOLDER = "bavaria_bands_2002_2023"
_S2_SR = "COPERNICUS/S2_SR_HARMONIZED"

# Reuse the Landsat SRC bare-soil thresholds for cross-sensor consistency.
_NDVI_MIN, _NDVI_MAX, _NBR2_MAX = g._SRC_NDVI_MIN, g._SRC_NDVI_MAX, g._SRC_NBR2_MAX


def _s2_mask_scl(img):
    """Keep only vegetation(4)/bare-soil(5)/unclassified(7); drop cloud/shadow/snow/water."""
    scl = img.select("SCL")
    keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(7))
    return img.updateMask(keep)


def s2_bare_soil_swir(aoi, y0, y1, cloud_max):
    """Multi-year bare-soil composite of B11/B12 (+ ExposureCount), 3 bands.

    SR scaled to reflectance (x1e-4); bare soil via NDVI in [min,max] AND NBR2 <
    max (same thresholds as the Landsat SRC); median over the masked stack.
    """
    def prep(img):
        sr = img.select(["B4", "B8", "B11", "B12"]).multiply(1e-4)
        ndvi = sr.normalizedDifference(["B8", "B4"])
        nbr2 = sr.normalizedDifference(["B11", "B12"])
        bare = (ndvi.gt(_NDVI_MIN).And(ndvi.lt(_NDVI_MAX)).And(nbr2.lt(_NBR2_MAX)))
        return (sr.select(["B11", "B12"])
                  .rename(["S2SRC_SWIR1", "S2SRC_SWIR2"])
                  .updateMask(bare))

    coll = (ee.ImageCollection(_S2_SR)
            .filterDate(f"{y0}-01-01", f"{y1 + 1}-01-01")
            .filterBounds(aoi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            .map(_s2_mask_scl)
            .map(prep))

    ref_proj = ee.Image(coll.first()).select("S2SRC_SWIR1").projection()
    comp = (coll.reduce(ee.Reducer.median(), parallelScale=8)
                .rename(["S2SRC_SWIR1", "S2SRC_SWIR2"]))
    count = (coll.select("S2SRC_SWIR1").reduce(ee.Reducer.count(), parallelScale=8)
                 .rename("S2SRC_ExposureCount"))
    out = comp.addBands(count).toFloat().clip(aoi)
    return out.setDefaultProjection(ref_proj)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2017, 2023],
                    help="S2 archive window for the bare-soil composite (default 2017-2023).")
    ap.add_argument("--scale", type=int, default=20,
                    help="Export pixel size in metres. 20 = sentinel-mode (native B11/B12); "
                         "250 = drop into the existing 43-band project grid.")
    ap.add_argument("--cloud-max", type=int, default=40,
                    help="Drop S2 scenes with CLOUDY_PIXEL_PERCENTAGE >= this (default 40).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the composite + print band names; submit no Export task.")
    args = ap.parse_args()

    aoi = ee.Geometry.Rectangle(g.BAVARIA_BBOX)
    y0, y1 = args.years
    img = s2_bare_soil_swir(aoi, y0, y1, args.cloud_max)
    desc = f"s2_swir_baresoil_{y0}_{y1}"
    bands = img.bandNames().getInfo()
    print(f"[s2-swir] composite {desc}  bands={bands}  scale={args.scale} m  "
          f"cloud<{args.cloud_max}%")
    if args.dry_run:
        print("[s2-swir] dry-run — no export submitted.")
        return
    g.submit_export(img, desc, DRIVE_FOLDER, args.scale, aoi)
    print(f"[s2-swir] submitted Export.image.toDrive -> {DRIVE_FOLDER}/{desc}.tif")
    print("[s2-swir] next: tile with the sgt_landsat_mb_cut.py pattern into "
          "RasterTensorData/StaticValue/S2SRC_SWIR1|S2SRC_SWIR2/, then publish to "
          "HF sgt-bavaria-soc-2002-2023-large. See S2_SWIR_BANDS_README.md.")


if __name__ == "__main__":
    main()
