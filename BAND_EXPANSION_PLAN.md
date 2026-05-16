# Band expansion plan for SOC mapping

## Status: existing 6 bands, 2007–2023

All complete in `Data/RasterTensorData/`. No redownload needed:

| Band | Yearly coverage | Source | Native res |
|---|---|---|---|
| Elevation | static, 12 tiles | likely SRTM / EU-DEM | ~250 m |
| LAI | 2000–2023 (24 yrs) | MODIS MCD15A2H/A3H | 500 m → resampled |
| LST | 2000–2023 (24 yrs) | MODIS MOD11A2 | 1 km → resampled |
| MODIS_NPP | 2001–2023 (23 yrs) | MODIS MOD17A3HGF | 500 m |
| SoilEvaporation | 2000–2023 (24 yrs) | likely GLDAS / ERA5 | ~10 km → resampled |
| TotalEvapotranspiration | 2001–2023 (23 yrs) | MODIS MOD16A2 | 500 m |

Tile layout: 12 tiles, ~979×979 px each, ~200 m pixel size. The
`samplePoints.py` pipeline projects each LUCAS GPS_LONG/GPS_LAT into the
tile-local (row, col) and saves to `OC_LUCAS_LFU_LfL_Coordinates_v2/`.

---

## Expansion in three phases

### Phase 1 — DEM derivatives (free, derive from existing Elevation)

**Add 4 new bands at zero cost** by computing topographic derivatives
from the Elevation tiles you already have:

| New band | What it captures | SOC relevance |
|---|---|---|
| `Slope` | gradient magnitude, degrees | erosion vs deposition; flats accumulate SOC |
| `Aspect` | gradient direction, 0–360° | N-facing slopes (N. hemisphere) = cooler/wetter → more SOC |
| `PlanCurvature` | curvature along contour | flow convergence/divergence |
| `TWI` | log(flow_accum / tan(slope)) | predicts wet soils → high SOC |

Script: `SOCmapping/SamplePoints/compute_dem_derivatives.py` (see below).
Runs per-tile, writes to `Data/RasterTensorData/StaticValue/{Slope,Aspect,PlanCurvature,TWI}/`
with identical filename pattern to `Elevation/`. Then add the four new
names to `bands_list_order` in `config.py` — dataloader picks them up
automatically.

Effort: ~1 hour one-time. **Expected R² gain: 0.02–0.06** on SOC tasks
in studies using similar covariates.

### Phase 2 — Climate (ERA5-Land via CDS API)

| New band | ERA5-Land variable | Why |
|---|---|---|
| `Precipitation` | `total_precipitation` (annual sum) | wetter = more SOC |
| `AirTemperature` | `t2m` (annual mean) | colder = more SOC (vs LST = surface) |
| `SoilMoisture` | `swvl1` or `swvl2` (annual mean) | direct decomposition driver |

ERA5-Land has 1981–present coverage at 9 km native, daily/hourly.
Download recipe (template — needs your CDS API key in `~/.cdsapirc`):

```python
# SOCmapping/SamplePoints/download_era5_yearly.py
import cdsapi
c = cdsapi.Client()
for year in range(2007, 2024):
    for var in ['2m_temperature', 'total_precipitation', 'volumetric_soil_water_layer_1']:
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': var,
                'year': str(year),
                'month': [f'{m:02d}' for m in range(1, 13)],
                'day':   [f'{d:02d}' for d in range(1, 32)],
                'time':  '12:00',
                'area': [50.6, 8.9, 47.3, 13.9],  # Bavaria N, W, S, E
                'format': 'netcdf',
            },
            f'era5_{var}_{year}.nc',
        )
```

Then aggregate daily → annual (sum for precipitation, mean for the
others) and resample to your 979×979 tile grid via `rasterio.warp`.

Effort: ~half a day (CDS queue can be slow; submit all 17 years × 3
variables in one batch). **Expected R² gain: 0.05–0.10** combined.

### Phase 3 — Static soil properties (SoilGrids 250 m, ISRIC)

| New band | SoilGrids variable | Why |
|---|---|---|
| `ClayContent` | `clay/clay_0-5cm_mean` | clay protects organic carbon |
| `SandContent` | `sand/sand_0-5cm_mean` | inverse — sandy soils lose C |
| `pH_H2O` | `phh2o/phh2o_0-5cm_mean` | controls microbial activity |
| `BulkDensity` | `bdod/bdod_0-5cm_mean` | needed for stocks; also a SOC predictor |
| `CEC` | `cec/cec_0-5cm_mean` | cation exchange capacity |

These are static (single snapshot), 250 m globally. Direct WMS download
or VRT mosaic, no auth needed:

```python
# SOCmapping/SamplePoints/download_soilgrids.py
import rioxarray as rxr
url_root = 'https://files.isric.org/soilgrids/latest/data/'
bbox = (8.9, 47.3, 13.9, 50.6)  # W, S, E, N
for var, depth in [('clay', '0-5cm'), ('sand', '0-5cm'), ('phh2o', '0-5cm'),
                   ('bdod', '0-5cm'), ('cec', '0-5cm')]:
    url = f'{url_root}{var}/{var}_{depth}_mean.vrt'
    da = rxr.open_rasterio(url).rio.clip_box(*bbox)
    da.rio.to_raster(f'soilgrids_{var}_{depth}.tif')
```

Effort: ~1 hour. **Expected R² gain: 0.08–0.15** (clay+pH are
historically the strongest non-vegetation SOC predictors in EU soils).

---

## Total potential R² gain (additive, may not stack linearly)

| Phase | Cumulative bands | Expected R² uplift |
|---|---|---|
| 0 (current) | 6 | baseline |
| 1 (+ DEM derivatives) | 10 | +0.02 – 0.06 |
| 2 (+ ERA5 climate) | 13 | +0.05 – 0.10 |
| 3 (+ SoilGrids) | 18 | +0.08 – 0.15 |

Total realistic envelope: **+0.15 – 0.30 R²** going from 6 to 18 bands,
assuming the model isn't already at the covariate ceiling. Phase 1 is
free and should be done first to test that the gain materializes
before investing in Phase 2/3 downloads.

---

## Integration steps (per new band)

1. **Add raster TIFs / npy tiles** under `Data/RasterTensorData/{YearlyValue, StaticValue}/{BandName}/`
   matching the 979×979 tile layout (same `IDxN..S..W..E..` filename pattern).
2. **Resample bounds-array** `Data/RasterTensorData/{tier}/bounds_array_{BandName}.npy`
   if the new band's spatial extent differs from existing ones.
3. **Run `SamplePoints/samplePoints.py`** to project LUCAS points onto
   the new tiles → writes `OC_LUCAS_LFU_LfL_Coordinates_v2/{tier}/{BandName}/`.
4. **Add band name to `config.py`** `bands_list_order` list. Order
   matters — appending preserves checkpoint compatibility; prepending
   breaks it.
5. **Update `MAX_OC`-style normalization stats** if the new band has a
   very different scale. The `NormalizedMultiRasterDatasetMultiYears`
   class auto-computes per-channel mean/std, so no manual step needed.
6. **Retrain.** Model auto-adapts to the new `input_channels`. Saved
   checkpoints won't load `strict=True` because Conv1 input channel
   dim changed; use `strict=False` and reinit the first conv layer.

---

## Recommended sequencing

If you want maximum R²-gain-per-hour-spent:

1. **This week**: Phase 1 (DEM derivatives) — purely local compute, no
   downloads. Test the model gain. If +0.05 materializes, move on.
2. **Next**: Phase 3 (SoilGrids) — static, one-time download, classic
   strong predictors.
3. **Later**: Phase 2 (ERA5) — requires CDS queue patience but is the
   most direct decomposition driver.
