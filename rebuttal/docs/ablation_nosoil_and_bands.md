# No-Soil (Circularity) Ablation & 43-Band Dictionary

This document records (1) the results of the **circularity ablation** — re-running the
top models on the canonical 43-band stack with the co-measured soil properties removed —
and (2) a **band dictionary** giving the Google Earth Engine origin of every band in the
`full_extended` stack.

---

## 1. Circularity ablation — drop the co-measured soil covariates

**Why.** Reviewers flagged a circularity concern: five of the covariates
(`ClayContent_0_10cm`, `SandContent_0_10cm`, `pH_H2O_0_10cm`, `BulkDensity_0_10cm`,
`CEC_0_10cm`) are lab/SoilGrids soil properties co-determined with the SOC target, so using
them to predict SOC risks leaking the answer. We re-ran the top models with exactly those
five bands removed.

**What changed.** New band subset `full_extended_nosoil` = canonical 43-band stack minus the
5 soil properties → **38 bands** (added in commit `d438967`; selected via
`--bands-list full_extended_nosoil`). Everything else is held fixed at the canonical recipe,
so any change is attributable to removing the soil covariates:

| setting | value |
|---|---|
| spatial CV | longitude 10-fold (deciles), 1.2 km train/test buffer |
| max OC | 150 g/kg | 
| window | 5×5 | seed | 342 |
| band encoder | `two_path` (core kept raw, Tier 1/2/3 reduced to 8 ch); split now 15 core + 23 ext |
| sweep group | `crisp_lon_f10_oc150_topdogs_extband_nosoil` (vs with-soil `…_extband`) |

**Result — same config, with soil → without soil (43-band lon 10-fold, R² mean):**

| model (config) | R² with soil | R² no-soil | Δ |
|---|---|---|---|
| **simpletransformer_d64_h4_L1** (flagship) | 0.401 | 0.410 † | +0.009 † |
| sgt small_d32_h2_L1 (seed 342) | 0.375 | 0.368 | −0.007 |
| vanilla_transformer_d64_h4_L1 | 0.381 | 0.357 | −0.024 |
| vanilla_transformer_d128_h4_L1 | 0.373 | 0.363 | −0.010 |
| vanilla_transformer_d192_h8_L1 | 0.367 | 0.357 | −0.010 |
| cnnlstm_d64_h4_L1 | 0.352 | 0.334 | −0.019 |
| 3dcnn_d64_h4_L1 (degenerate) | −0.481 | −0.511 | −0.030 |

† simpletransformer no-soil is currently a **7/10-fold** number; the 3 missing folds are being
filled by rerun job `941969` (resume skips the 7 done folds). Update to the 10-fold value once
it lands. The larger/unstable vanilla configs (d128_L2, d192_L2, d256_*) move by ±0.03 in
either direction (run-to-run noise) and are not the headline.

**Takeaway.** Removing the five co-measured soil properties barely moves the needle — the top
transformer is essentially unchanged (slightly up), and the other strong models lose only
~0.01–0.02 R². The reported skill is therefore **not an artifact of circularity**: the models
are not relying on the lab-measured soil covariates.

**Reproduce / refresh.**
```bash
# submit (JUPITER):
python rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py \
  --partition booster --account scifi --venv-activate scripts/jupiter_env.sh --cpus-per-task 4 \
  --small --families --vanilla --lightweight-transformer --no-baselines \
  --bands-list full_extended_nosoil --split-axis lon --max-oc 150 --epochs 100 \
  --seed-base 342 --time 04:00:00 --sweep-name crisp_lon_f10_oc150_topdogs
# rank:
python rebuttal/gpu_experiments/spatial_kfold/sweep_summarize.py   # group crisp_lon_f10_oc150_topdogs_extband_nosoil
```
Note: the lightweight_transformer arm of this run produced no output (jobs failed — check
`sacct`); resubmit `sweep/sbatch/lightweight_transformer_*.sbatch` if those rows are needed.

---

## 2. Band dictionary — GEE origin of every `full_extended` band

Order and grouping follow `_bands.py` (`FULL_EXTENDED_BANDS`). Authoritative downloader:
`SamplePoints/gee_download_all_bands.py` (`--category extended`). All bands are resampled /
aggregated to **250 m, EPSG:4326** at export; temporal coverage is **2002–2023** except the
static soil/terrain layers. Derived bands are computed in-pipeline, not pre-existing assets.

| # | Band | GEE asset ID | Source band(s) | Cadence / reduction | Native res | Units / scaling |
|---|------|--------------|----------------|---------------------|------------|-----------------|
| | **Original 6** | | | | | |
| 1 | Elevation | `USGS/SRTMGL1_003` | `elevation` | static | 30 m | metres |
| 2 | LAI | `MODIS/061/MCD15A3H` | `Lai` | yearly mean (4-day) | 500 m | ×0.1 |
| 3 | LST | `MODIS/061/MOD11A2` | `LST_Day_1km` | yearly mean (8-day) | 1 km | K ×0.02 |
| 4 | MODIS_NPP | `MODIS/061/MOD17A3HGF` | `Npp` | yearly (annual) | 500 m | kgC/m²/yr ×0.0001 |
| 5 | SoilEvaporation | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `evaporation_from_bare_soil_sum` | yearly sum | ~9 km | m w.e. |
| 6 | TotalEvapotranspiration | `MODIS/061/MOD16A2GF` | `ET` | yearly sum (8-day) | 500 m | mm ×0.1 |
| | **Revision +14** | | | | | |
| 7 | NDVI | `MODIS/061/MOD13Q1` | `NDVI` | yearly mean (16-day) | 250 m | ×0.0001 |
| 8 | EVI | `MODIS/061/MOD13Q1` | `EVI` | yearly mean (16-day) | 250 m | ×0.0001 |
| 9 | Precipitation | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `total_precipitation_sum` | yearly sum | ~9 km | m w.e. |
| 10 | AirTemperature | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `temperature_2m` | yearly mean | ~9 km | K |
| 11 | SoilMoisture_layer1 | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `volumetric_soil_water_layer_1` | yearly mean | ~9 km | m³/m³ |
| 12 | SnowDepth | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `snow_depth` | yearly mean | ~9 km | m |
| 13 | **ClayContent_0_10cm** ⚠ | `OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02` | `b10` | static | 250 m | % (×100 conv.) |
| 14 | **SandContent_0_10cm** ⚠ | `OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02` | `b10` | static | 250 m | % (×100 conv.) |
| 15 | **pH_H2O_0_10cm** ⚠ | `OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02` | `b10` | static | 250 m | pH ×10 |
| 16 | **BulkDensity_0_10cm** ⚠ | `OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02` | `b10` | static | 250 m | kg/dm³ ×10 |
| 17 | **CEC_0_10cm** ⚠ | `projects/soilgrids-isric/cec_mean` | `cec_0-5cm_mean` | static | 250 m | mmol(c)/kg ×10 |
| 18 | Slope | `USGS/SRTMGL1_003` (`ee.Terrain.slope`) | derived | static | 30 m | degrees |
| 19 | Aspect | `USGS/SRTMGL1_003` (`ee.Terrain.aspect`) | derived | static | 30 m | degrees 0–360 |
| 20 | TWI | `USGS/SRTMGL1_003` + `MERIT/Hydro/v1_0_1` | derived | static | 30 m / 90 m | ln(SCA/tanβ) |
| | **Tier 1 — Landsat bare-soil composite (SRC)** | | | | | |
| 21 | SRC_Blue | Landsat C2 L2 (LT05/LE07/LC08/LC09) | Blue | yearly median, bare-soil filtered | 30 m | SR [0–1] |
| 22 | SRC_Green | Landsat C2 L2 | Green | yearly median, bare-soil | 30 m | SR [0–1] |
| 23 | SRC_Red | Landsat C2 L2 | Red | yearly median, bare-soil | 30 m | SR [0–1] |
| 24 | SRC_NIR | Landsat C2 L2 | NIR | yearly median, bare-soil | 30 m | SR [0–1] |
| 25 | SRC_SWIR1 | Landsat C2 L2 | SWIR1 | yearly median, bare-soil | 30 m | SR [0–1] |
| 26 | SRC_SWIR2 | Landsat C2 L2 | SWIR2 | yearly median, bare-soil | 30 m | SR [0–1] |
| 27 | SRC_RCC | Landsat C2 L2 (derived) | R/(R+G+B) | yearly median, bare-soil | 30 m | chromatic |
| 28 | SRC_BCC | Landsat C2 L2 (derived) | B/(R+G+B) | yearly median, bare-soil | 30 m | chromatic |
| 29 | SRC_NBR2 | Landsat C2 L2 (derived) | (SWIR1−SWIR2)/(SWIR1+SWIR2) | yearly median | 30 m | index |
| 30 | SRC_BSI | Landsat C2 L2 (derived) | (SWIR1+R−NIR−B)/(SWIR1+R+NIR+B) | yearly median | 30 m | index |
| 31 | SRC_ExposureCount | Landsat C2 L2 | count of valid bare-soil obs | yearly | 30 m | integer mask |
| | **Tier 2 — multi-scale terrain (SRTM-derived, static)** | | | | | |
| 32 | TPI_90 | `USGS/SRTMGL1_003` (derived) | elev − focal_mean(90 m) | static | 30 m | metres |
| 33 | TPI_300 | `USGS/SRTMGL1_003` (derived) | elev − focal_mean(300 m) | static | 30 m | metres |
| 34 | TPI_1000 | `USGS/SRTMGL1_003` (derived) | elev − focal_mean(1000 m) | static | 30 m | metres |
| 35 | TRI | `USGS/SRTMGL1_003` (derived) | stddev(elev, 90 m) | static | 30 m | metres |
| 36 | Roughness | `USGS/SRTMGL1_003` (derived) | max−min(elev, 90 m) | static | 30 m | metres |
| | **Tier 3 — climate / phenology (yearly, derived)** | | | | | |
| 37 | ClimaticWaterBalance | `ECMWF/ERA5_LAND/MONTHLY_AGGR` (derived) | precip_sum − pot_evap_sum | yearly | ~9 km | m/yr (P−PET) |
| 38 | SoilTemperature_layer1 | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `soil_temperature_level_1` | yearly mean | ~9 km | K |
| 39 | FrostDays | `ECMWF/ERA5_LAND/DAILY_AGGR` (derived) | count(`temperature_2m_min` < 273.15 K) | yearly count | ~9 km | days/yr |
| 40 | GrowingDegreeDays | `ECMWF/ERA5_LAND/DAILY_AGGR` (derived) | Σ max(T2m−5°C, 0) | yearly sum | ~9 km | K·day/yr (GDD5) |
| 41 | NDVI_Amplitude | `MODIS/061/MOD13Q1` (derived) | yearly max−min NDVI | yearly | 250 m | ×0.0001 |
| 42 | NDVI_Integral | `MODIS/061/MOD13Q1` (derived) | yearly Σ NDVI | yearly | 250 m | ×0.0001 |
| 43 | EVI_Amplitude | `MODIS/061/MOD13Q1` (derived) | yearly max−min EVI | yearly | 250 m | ×0.0001 |

⚠ = the five co-measured soil properties dropped in the `full_extended_nosoil` ablation.

**Distinct GEE datasets used (9 families):**
- MODIS — `MCD15A3H` (LAI), `MOD11A2` (LST), `MOD17A3HGF` (NPP), `MOD16A2GF` (ET), `MOD13Q1` (NDVI/EVI + phenology)
- ERA5-Land — `ECMWF/ERA5_LAND/MONTHLY_AGGR` (climate), `ECMWF/ERA5_LAND/DAILY_AGGR` (FrostDays/GDD)
- SRTM — `USGS/SRTMGL1_003` (elevation + 7 derived terrain bands)
- OpenLandMap — clay / sand / pH / bulk-density soil assets
- ISRIC SoilGrids 2.0 — `projects/soilgrids-isric/cec_mean` (CEC; replaces the deprecated OpenLandMap CEC asset)
- MERIT Hydro — `MERIT/Hydro/v1_0_1` (upstream area, supports TWI)
- Landsat Collection 2 Level 2 — `LT05/LE07/LC08/LC09 C02 T1_L2` (bare-soil composite, SRC_*)

**Bare-soil composite (SRC_*) detail.** A per-year `_bare_soil_composite(year, aoi)` merges the
four Landsat sensors, masks cloud/shadow/snow via `QA_PIXEL`, keeps bare-soil pixels
(NDVI ∈ [0.15, 0.25], NBR2 < 0.075), takes the per-sensor median of ≤20 least-cloudy scenes,
derives RCC/BCC/NBR2/BSI from the reflectance composite, and records `ExposureCount` as a
validity mask before aggregating 30 m → 250 m. (See `project_landsat_src_capped` — the capped
20-scene median is a methodology deviation worth noting in the paper.)

*Source of truth for exact reducers/thresholds: `SamplePoints/gee_download_all_bands.py`.*
