# SamplePoints — Bavaria 2002–2023 data rig

End-to-end pipeline that produces a 20-band on-disk layout the existing
SOCmapping dataloader consumes without any architecture or dataloader
changes. The 12-tile Bavaria layout parsed from
`Data/RasterTensorData/StaticValue/Elevation/` is the authority — every
new band is sliced to match it bit-for-bit.

## What changed vs. the original pipeline

- **Curated band set** — 6 existing + 14 new = 20 channels. Excludes
  target-leakage bands, categoricals, and Sentinel-1/2 (gap years).
- **Statics as yearly via symlinks** — soil chemistry (Clay/Sand/pH/
  BulkDensity/CEC) and DEM derivatives (Slope/Aspect/TWI) are stored
  under `YearlyValue/<band>/<year>/`, with 2003..2023 directories
  symlinking to the physical 2002 directory. This lets the dataloader's
  `int(year)` parse work with zero code changes.
- **Bilinear resampling** on coarse GEE sources (ERA5 ~11 km, CHIRPS
  5.5 km) instead of GEE's default nearest-neighbor at export.
- **NaN handling** in `tiff_to_tiles.py` — bad pixels filled with the
  tile's finite mean to keep `NormalizedMultiRasterDatasetMultiYears.
  compute_statistics` clean.

## The 12-tile Bavaria layout (frozen here)

```
   N edge (lat)            W edge (lon)
   52.10  ┌─ ID  8 ─┬─ ID  9 ─┬─ ID 10 ─┬─ ID 11 ─┐  ← row 2 (north)
   50.30  ├─ ID  4 ─┼─ ID  5 ─┼─ ID  6 ─┼─ ID  7 ─┤  ← row 1
   48.51  ├─ ID  0 ─┼─ ID  1 ─┼─ ID  2 ─┼─ ID  3 ─┤  ← row 0 (south)
   46.71  └─────────┴─────────┴─────────┴─────────┘
          7.19      8.98     10.78     12.58     14.37
```

Each tile is **1.7986° × 1.7986°, 979 × 979 px** (~204 m pixel size).
Encoded in filenames as `IDxN<top>S<bot>W<left>E<right>.npy`
(underscores = decimal points).

Bavaria's real GPS extent (from `Coordinates1Mil/coordinates_Bavaria_1mil.csv`)
is **lon [8.0, 13.9], lat [47.2, 50.6]** — comfortably inside the buffered
envelope above.

## Curated band set (14 new + 6 existing = 20 total)

| Tier | Band | Source | Why for SOC |
|---|---|---|---|
| existing | Elevation | SRTM/EU-DEM static | topography baseline |
| existing | LAI | MODIS MCD15A3H yearly | canopy biomass input |
| existing | LST | MODIS MOD11A2 yearly | surface T, decomposition |
| existing | MODIS_NPP | MOD17A3HGF yearly | net carbon flux |
| existing | SoilEvaporation | yearly | surface moisture flux |
| existing | TotalEvapotranspiration | MOD16A2 yearly | water-balance |
| new yearly | NDVI | MOD13Q1, 250 m | vegetation greenness |
| new yearly | EVI | MOD13Q1, 250 m | better in dense canopy |
| new yearly | Precipitation | CHIRPS, 5.5 km→bilinear | wetter → more SOC |
| new yearly | AirTemperature | ERA5-Land monthly | air-T (vs LST surface-T) |
| new yearly | SoilMoisture_layer1 | ERA5-Land monthly | direct decomposition driver |
| new yearly | SnowDepth | ERA5-Land monthly | alpine insulation |
| new static→yearly | ClayContent_0_10cm | OpenLandMap 250 m | clay protects organic carbon |
| new static→yearly | SandContent_0_10cm | OpenLandMap 250 m | sandy soils lose C fast |
| new static→yearly | pH_H2O_0_10cm | OpenLandMap 250 m | controls microbial activity |
| new static→yearly | BulkDensity_0_10cm | OpenLandMap 250 m | stocks denominator + predictor |
| new static→yearly | CEC_0_10cm | OpenLandMap 250 m | cation exchange |
| new static→yearly | Slope | GEE: ee.Terrain.slope(SRTM 30 m) | erosion vs deposition |
| new static→yearly | Aspect | GEE: ee.Terrain.aspect(SRTM 30 m) | cooler N-faces hold more SOC |
| new static→yearly | TWI | GEE: ln(MERIT Hydro upa / tan(slope)) | wet-soil / peatland indicator |

**Excluded** (with reason): `OrganicCarbon_Reference` (target leakage),
`TextureUSDA` (categorical), Sentinel-1/2 (gap years for 2002–2023),
NPP/ET/GPP/PET/FPAR/LST_Day/LST_Night/BurnedArea (redundant with
existing 6 or low SOC signal in Bavaria).

## End-to-end run

For a fresh end-to-end run, use the orchestrator:

```bash
bash SamplePoints/run_full_pipeline.sh
```

The orchestrator runs 6 phases in order: `plan → gee → pull → cut →
project → verify`. Each phase can be skipped with `SKIP_<PHASE>=1`
env var, or you can run a single phase:

```bash
bash SamplePoints/run_full_pipeline.sh gee       # submit GEE batch only
bash SamplePoints/run_full_pipeline.sh verify    # sanity tests only
bash SamplePoints/run_full_pipeline.sh derive    # FALLBACK: local Slope/Aspect/TWI
                                                 #   if GEE access is unavailable
```

Slope, Aspect, and TWI are computed server-side by GEE
(`ee.Terrain.slope/aspect` on SRTM 30 m, and TWI from MERIT Hydro's
upstream-area band), so there's no local DEM derivation step in the
default sequence.

Env vars:
- `DRIVE_FOLDER` (default `bavaria_bands_2002_2023`) — Drive output dir.
- `DRIVE_FOLDER_ID` — required for the `pull` phase. Get it from the
  Drive URL after the GEE tasks finish: `drive.google.com/drive/folders/<ID>`.
- `TIFF_LOCAL_DIR` (default `~/bavaria_tiffs`) — local stash for pulled TIFFs.
- `YEAR_START`/`YEAR_END` (default `2002`/`2023`).

## Manual / single-step usage

### Phase 1 — `gee_download_all_bands.py`
Submits Earth Engine `Export.image.toDrive()` tasks for the curated
band set at 250 m / EPSG:4326. ~140 tasks for 2002–2023 (6 yearly × 22
+ 5 soil + 3 terrain = 140). Slope/Aspect are built server-side via
`ee.Terrain.slope/aspect` on SRTM; TWI is built server-side from
MERIT Hydro `upa` (upstream drainage area) and SRTM-derived slope.

```bash
pip install earthengine-api
earthengine authenticate
python SamplePoints/gee_download_all_bands.py --dry-run
python SamplePoints/gee_download_all_bands.py
# or call a single category:
python SamplePoints/gee_download_all_bands.py --category modis
python SamplePoints/gee_download_all_bands.py --category soil
```

Monitor at https://code.earthengine.google.com/tasks. The script auto-
applies `.resample('bilinear')` on coarse sources before export.

### Phase 2 — `compute_dem_derivatives.py` (FALLBACK only)
**Not run by default.** Slope/Aspect/TWI now come from GEE
(`gee_download_all_bands.py --category curated` includes them in the
`topo` group). Use this script only as a fallback when GEE access is
unavailable — it computes the three from the existing local Elevation
tiles and writes them directly to `YearlyValue/<band>/2002/` plus
2003..2023 symlinks.

```bash
python SamplePoints/compute_dem_derivatives.py
```

### Phase 3 — `pull_from_drive.py`
After GEE completes, pull the TIFFs locally:

```bash
pip install gdown
python SamplePoints/pull_from_drive.py \
    --folder-id <DRIVE_FOLDER_ID> --out ~/bavaria_tiffs
```

Find the folder ID in the Drive URL after the export finishes:
`drive.google.com/drive/folders/<ID>`.

### Phase 4 — `tiff_to_tiles.py`
Cuts each downloaded GeoTIFF into 12 `.npy` tiles. For static soil
bands, the `--materialize-yearly` flag stores the physical tiles under
`YearlyValue/<band>/2002/` and creates symlinks for 2003..2023.

```bash
python SamplePoints/tiff_to_tiles.py \
    --tiff-dir ~/bavaria_tiffs \
    --materialize-yearly ClayContent_0_10cm SandContent_0_10cm \
                         pH_H2O_0_10cm BulkDensity_0_10cm CEC_0_10cm \
                         Slope Aspect TWI
```

Pure-yearly bands (NDVI/EVI/Precipitation/AirTemperature/SoilMoisture_layer1/
SnowDepth) are auto-routed to `YearlyValue/<band>/<year>/` without
needing `--materialize-yearly`.

### Phase 5 — `project_lucas_coords.py`
Projects LUCAS GPS points onto each (band, year) tile grid → writes
`coordinates.npy` (cols: `[lat, lon, tile_id, row, col]`).

```bash
# Pure-yearly bands — use --symlink-across-years to write one
# coordinates.npy per band and symlink across all 22 year-dirs:
python SamplePoints/project_lucas_coords.py \
    --tier YearlyValue \
    --bands NDVI EVI Precipitation AirTemperature SoilMoisture_layer1 SnowDepth \
    --symlink-across-years

# Materialized statics — anchor + symlink (soil chemistry + terrain):
python SamplePoints/project_lucas_coords.py \
    --tier YearlyValue \
    --bands ClayContent_0_10cm SandContent_0_10cm pH_H2O_0_10cm \
            BulkDensity_0_10cm CEC_0_10cm \
            Slope Aspect TWI \
    --materialize-yearly ClayContent_0_10cm SandContent_0_10cm pH_H2O_0_10cm \
                         BulkDensity_0_10cm CEC_0_10cm \
                         Slope Aspect TWI
```

## On-disk layout produced

```
Data/RasterTensorData/
  StaticValue/
    Elevation/  ID0..npy ... ID11..npy            (existing 12 tiles)
  YearlyValue/
    LAI/, LST/, MODIS_NPP/, SoilEvaporation/,
    TotalEvapotranspiration/                       (existing yearly bands)
    NDVI/, EVI/, Precipitation/, AirTemperature/,
    SoilMoisture_layer1/, SnowDepth/               (new yearly bands)
      2002/ ... 2023/   ID0..npy ... ID11..npy
    ClayContent_0_10cm/, SandContent_0_10cm/,
    pH_H2O_0_10cm/, BulkDensity_0_10cm/,
    CEC_0_10cm/, Slope/, Aspect/, TWI/             (new statics-as-yearly)
      2002/   ID0..npy ... ID11..npy               (physical)
      2003/ → 2002/  2004/ → 2002/  ...            (symlinks)

Data/OC_LUCAS_LFU_LfL_Coordinates_v2/
  YearlyValue/<band>/<year>/coordinates.npy        (per (band, year))
```

## Config updates (already applied)

All 9 model configs (`TemporalFusionTransformer`, `SimpleTransformer`,
`2DCNN`, `3DCNN`, `CNNLSTM`, `VAETransformer`, `FoundationalModels`,
`balancedDataset`, `BaselinesXGBoostAndRF`) have been updated:

1. `bands_list_order` extended from 6 → 20 entries (existing 6
   preserved in the same positions; 14 new bands appended).
2. Per-band path variables added for the 14 new bands.
3. `SamplesCoordinates_Yearly` and `DataYearly` extended in
   `bands_list_order` order.
4. Seasonal blocks left untouched.

After the pipeline runs, the model will pick up the new bands
automatically — `input_channels = len(bands_list_order)`. **Pretrained
6-channel checkpoints will need `strict=False`** + reinitialization
of the first conv layer's 14 new channels.

## Validation

The orchestrator's `verify` phase checks:
- Each band has 12 tiles per sampled year (2002, 2010, 2023).
- All `coordinates.npy` files load with shape `(30227, 5)`.
- Each model's config imports cleanly with aligned 20-entry lists.

Manual checks:

```bash
# Tile count
for b in NDVI EVI Precipitation AirTemperature SoilMoisture_layer1 SnowDepth \
         ClayContent_0_10cm SandContent_0_10cm pH_H2O_0_10cm BulkDensity_0_10cm CEC_0_10cm \
         Slope Aspect TWI; do
  echo "=== $b ==="
  for y in 2002 2010 2023; do
    n=$(ls Data/RasterTensorData/YearlyValue/$b/$y/ID*.npy 2>/dev/null | wc -l)
    echo "  $y: $n tiles (expect 12)"
  done
done
```

## Notes on HuggingFace upload

`huggingface_hub.upload_folder` resolves symlinks and uploads the
underlying file content. For the static-as-yearly bands, this means
22× duplicate uploads per band. If HF disk cost matters, you have two
options:

1. Upload only the `2002/` anchor for each materialized band, plus a
   small `materialize_static_as_yearly.py` helper that the consumer
   runs after `git clone` / `huggingface-cli download` to recreate the
   symlinks. (Not implemented here — separate change.)
2. Accept the duplication and move on. At ~46 MB per band-year, 8
   materialized bands × 21 duplicate years = ~7.7 GB extra on HF.
