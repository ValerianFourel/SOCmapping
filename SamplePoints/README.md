# SamplePoints — full data rig

End-to-end pipeline from Earth Engine to a trainable channel in the SGT/TFT
dataloader. Three scripts cover the three phases. The exact 12-tile layout
parsed from `Data/RasterTensorData/StaticValue/Elevation/` is the
authority — every new band is sliced to match it bit-for-bit.

## The 12-tile Bavaria layout (frozen here)

```
   N edge (lat)            W edge (lon)
   52.10  ┌─ ID  8 ─┬─ ID  9 ─┬─ ID 10 ─┬─ ID 11 ─┐  ← row 2 (north)
   50.30  ├─ ID  4 ─┼─ ID  5 ─┼─ ID  6 ─┼─ ID  7 ─┤  ← row 1
   48.51  ├─ ID  0 ─┼─ ID  1 ─┼─ ID  2 ─┼─ ID  3 ─┤  ← row 0 (south)
   46.71  └─────────┴─────────┴─────────┴─────────┘
          7.19      8.98     10.78     12.58     14.37
```

Each tile is **1.7986° × 1.7986°, 979 × 979 px** (~200 m pixel size).
Encoded in filenames as `IDxN<top>S<bot>W<left>E<right>.npy`
(underscores = decimal points).

Bavaria's real GPS extent (from `Coordinates1Mil/coordinates_Bavaria_1mil.csv`)
is **lon [8.0, 13.9], lat [47.2, 50.6]** — comfortably inside the buffered
envelope above.

## Three-phase workflow

### Phase 1 — `gee_download_all_bands.py`
Submits Earth Engine `Export.image.toDrive()` tasks for the buffered
Bavaria envelope at 250 m / EPSG:4326. One GeoTIFF per (band, year)
plus static layers. 30 bands × up to 22 years ≈ **416 tasks**.

```bash
pip install earthengine-api
earthengine authenticate
python SamplePoints/gee_download_all_bands.py --dry-run     # plan
python SamplePoints/gee_download_all_bands.py               # submit all
# or category-by-category:
python SamplePoints/gee_download_all_bands.py --category modis
python SamplePoints/gee_download_all_bands.py --category chirps
python SamplePoints/gee_download_all_bands.py --category era5
python SamplePoints/gee_download_all_bands.py --category topo
python SamplePoints/gee_download_all_bands.py --category soil
python SamplePoints/gee_download_all_bands.py --category s1
python SamplePoints/gee_download_all_bands.py --category s2
```

Monitor at https://code.earthengine.google.com/tasks. Pull from Drive
with `rclone` or `gdown` into a local folder, e.g.
`~/bavaria_bands_2002_2023/`.

### Phase 2 — `tiff_to_tiles.py`
Cuts each downloaded GeoTIFF into the 12 `.npy` tiles, with the exact
filename pattern the dataloader expects. Also generates the
`bounds_array_<band>.npy` file.

```bash
python SamplePoints/tiff_to_tiles.py \
    --tiff-dir ~/bavaria_bands_2002_2023 \
    --out-root /home/valerian/SGTPublication/Data/RasterTensorData
```

Output structure created:
```
Data/RasterTensorData/
  StaticValue/
    Elevation_SRTM/  ID0..npy  ID1..npy  ...  ID11..npy
    ClayContent_0_10cm/  ...
    bounds_array_Elevation_SRTM.npy
    bounds_array_ClayContent_0_10cm.npy
  YearlyValue/
    NDVI/
      2002/  ID0..npy  ...  ID11..npy
      2003/  ...
      2023/  ...
    Precipitation/
      ...
    bounds_array_NDVI.npy
    bounds_array_Precipitation.npy
```

### Phase 3 — `project_lucas_coords.py`
For each (band, year) folder, projects all LUCAS GPS points into
tile-local pixel coordinates and writes `coordinates.npy`. Mirrors the
RasterTensorData directory structure under
`OC_LUCAS_LFU_LfL_Coordinates_v2/`.

```bash
# All new yearly bands at once:
python SamplePoints/project_lucas_coords.py \
    --tier YearlyValue \
    --bands NDVI EVI Precipitation AirTemperature SoilMoisture_layer1 \
            SoilMoisture_layer2 SnowDepth FPAR LST_Night NPP_v2 GPP \
            ET_v2 PET BurnedArea

# All new static bands:
python SamplePoints/project_lucas_coords.py \
    --tier StaticValue \
    --bands Elevation_SRTM ClayContent_0_10cm SandContent_0_10cm \
            pH_H2O_0_10cm BulkDensity_0_10cm CEC_0_10cm \
            TextureUSDA
```

After this completes, `Data/OC_LUCAS_LFU_LfL_Coordinates_v2/` contains
a `coordinates.npy` for every (tier, band, year), matching the format
the dataloader's `MultiRasterDatasetMultiYears.find_coordinates_index`
expects: `[lat, lon, tile_id, row, col]`.

### Phase 4 (manual) — update `config.py`
Append new band names to `bands_list_order`. Order matters — appending
preserves checkpoint compatibility for the first 6 channels:

```python
bands_list_order = [
    # original 6 (DO NOT REORDER — saved checkpoints rely on this)
    'Elevation', 'LAI', 'LST', 'MODIS_NPP',
    'SoilEvaporation', 'TotalEvapotranspiration',
    # appended via this rig:
    'NDVI', 'EVI', 'Precipitation', 'AirTemperature',
    'SoilMoisture_layer1', 'ClayContent_0_10cm', 'SandContent_0_10cm',
    'pH_H2O_0_10cm', 'BulkDensity_0_10cm', 'CEC_0_10cm',
    'Slope', 'Aspect', 'PlanCurvature', 'TWI',
]
```

Then retrain. The model's input-conv channel count goes from 6 to N;
saved checkpoints will need `load_state_dict(..., strict=False)` and
reinit of the first conv layer.

## Naming conventions to keep aligned

| Phase | Filename pattern |
|---|---|
| Phase 1 → Drive | `<category>_<band>_<year>.tif` or `<category>_<band>_static.tif` |
| Phase 2 → npy | `IDxN<lat>S<lat>W<lon>E<lon>.npy` (underscores = decimals) |
| Phase 3 → coords | `coordinates.npy` shape `(N, 5)` columns `[lat, lon, tile_id, row, col]` |

`tiff_to_tiles.py` parses Phase-1 filenames with the regex
`(?P<category>[a-z0-9]+)_(?P<band>[A-Za-z0-9_]+?)_(?P<period>static|\d{4})\.tif$`
so renaming the GEE exports breaks the auto-routing — keep the
underscore-delimited `<cat>_<band>_<period>` format.

## Validation sanity tests

After Phase 2, check each new band has 12 tiles per (year) and bounds-array
exists:
```bash
ls Data/RasterTensorData/YearlyValue/NDVI/2010/ | wc -l   # expect 12
ls Data/RasterTensorData/StaticValue/bounds_array_NDVI.npy
```

After Phase 3, check coordinates.npy has 30,227 rows (= LUCAS count) and
all 5 columns:
```bash
python -c "import numpy as np; a = np.load('Data/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/NDVI/2010/coordinates.npy'); print(a.shape, a.dtype)"
```

The projector has been validated against the existing Elevation tile
layout: 100 % tile_id agreement on all 30,227 LUCAS points (row/col
agree to within ~50 px, comfortably inside the 5×5 sampling window).
