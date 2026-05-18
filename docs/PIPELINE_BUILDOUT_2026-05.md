# Bavaria SOC pipeline buildout — May 2026 session log

**Dates:** 2026-05-16 → 2026-05-17
**Goal:** extend the SGT/TFT input from 6 bands to a curated 20-band set covering Bavaria 2002–2023 at 250 m, with pixel-perfect alignment across all channels and parallel coordinate trees for both training (LUCAS+LFU+LfL 30 k samples) and inference (1.3 M Bavaria grid), without modifying the model or dataloader.

---

## Executive summary

| Component | Status |
|---|---|
| Curated band set (20 channels) | ✅ Validated on GEE, redownloaded all 6 existing bands for consistency |
| GEE export | ✅ 251 tasks submitted, 251 COMPLETED, 0 FAILED (after fixes) |
| Local pull (rclone) | ✅ 251 TIFFs in `~/bavaria_tiffs/` (~2.9 GB) |
| TIFF cut → 12 npy tiles | ✅ 251 × 12 = 3 012 canonical-name tiles in `Data/RasterTensorData/` |
| LUCAS+LFU+LfL coords projection (samples) | ✅ 30 227 points × 20 bands |
| 1.3 M Bavaria grid projection (1mil) | ✅ 1 300 000 points × 20 bands (vectorized — 0.4 s per band) |
| TIFF audit | ✅ 250 OK, 1 minor WARN, 0 FAIL (final) |
| HF mirror (`Data_HF/`) | ✅ Symlink-based, ~43 GB resolved |
| **HF dataset push** | ⏳ **Not yet run — last step** |

---

## What still needs to be done

### 1. Push to HuggingFace (last step before the data is shareable)

```bash
# One-time setup
pip install huggingface_hub
huggingface-cli login    # paste a write-scoped token from huggingface.co/settings/tokens

# Push (use the script we built)
cd /home/valerian/SGTPublication
python3 SOCmapping/SamplePoints/push_to_hf.py \
    --repo-id <your-username>/sgt-bavaria-soc-2002-2023 \
    --create-repo \
    --private          # or omit for a public dataset
```

Expected: ~30 min – 2 h upload for 43 GB at typical home bandwidth. The script uses `multi_commits=True`, so a network drop is recoverable — re-running resumes from the last committed chunk.

After completion, the dataset card lives at `https://huggingface.co/datasets/<username>/sgt-bavaria-soc-2002-2023`.

### 2. Optional follow-ups (not blocking the publication)

- **Retrain the model** with the 20-channel input (`bands_list_order` already updated in every model's `config.py`; load checkpoints with `strict=False` and reinit the first conv).
- **Drive cleanup**: the failed-task records on GEE accumulate forever (e.g. 19 historical FAILED `MODIS/061/MOD16A2` records). They're harmless audit trail but show up via `earthengine task list`. Optionally delete via the GEE web UI.
- **Dataloader memory**: full 20-band instantiation needs ~20 GB RAM because `RasterTensorDataset.__init__` eagerly pre-loads all 12 tiles per (band, year) into memory. On a 16 GB laptop this OOM-kills. Not a data problem — a dataloader design choice. To work around: lazy-load tiles (a small dataloader change), or train on machines with ≥32 GB RAM, or use a subset year range.

---

## Decisions log (chronological)

| When | Decision | Why |
|---|---|---|
| 2026-05-16 day 1 | Curate the band set down from "everything on GEE" to a focused 20-channel list | Avoid SOC-target leakage (`OrganicCarbon_Reference`), categorical bands (`TextureUSDA`), and Sentinel‑1/2 (which only start 2014/2015 → coverage gap for 2002–2023). |
| day 1 | Default pixel grid: EPSG:4326, 250 m, region `[W 7.1864, S 46.7109, E 14.3750, N 52.1028]`, 12 tiles × 979×979 px | Matches the existing 12-tile layout already on disk — minimal disruption to downstream code. |
| day 1 | Static-as-yearly via symlinks | Dataloader special-cases only `'Elevation'` as static. To add new static bands (soil chemistry, terrain) without touching `__getitem__`, materialize them at `YearlyValue/<band>/2002/` and symlink 2003..2023 to the anchor. Disk cost stays at 1× physical copy per band. |
| day 1 | Drop `OrganicCarbon_Reference` and `TextureUSDA` from curated default | First leaks the model target. Second is a categorical class index — bilinear resampling produces meaningless floats. |
| day 2 | Switch Slope/Aspect/TWI from local DEM derivation to GEE server-side | User preference: get everything from GEE for consistency. Server-side: `ee.Terrain.slope(SRTM)`, `ee.Terrain.aspect(SRTM)`, and `ln(MERIT_Hydro.upa × 10⁶ / tan(slope_floored))` for TWI. |
| day 2 | Add 6 existing bands (LAI, LST, MODIS_NPP, SoilEvaporation, TotalEvapotranspiration, Elevation) to the curated redownload list | The pre-existing on-disk tiles had bbox-filename drift (~5 m, 3 % of a pixel). Redownloading onto the same canonical grid gives pixel-perfect alignment across all 20 channels. |
| day 2 | Wipe stale non-canonical tiles before cut | After redownload, the new canonical tile filenames differ from the old. If both coexist, `glob('ID*.npy')` returns ambiguous results. `tiff_to_tiles._wipe_stale_tiles` removes any `ID*.npy` whose name isn't in the canonical 12-tile set before writing. |
| day 2 | Switch projection script to support both `--mode samples` (LUCAS+LFU+LfL 30 k points) and `--mode 1mil` (1.3 M Bavaria grid) | Inference needs the full Bavaria grid mapping; training needs the ground-truth sample mapping. Same projection logic, two trees: `OC_LUCAS_LFU_LfL_Coordinates_v2/` and `Coordinates1Mil/`. |
| day 2 | Vectorize `project_all` | Per-point Python loop was too slow for 1.3 M points. Vectorized version: 0.4 s for the whole batch, 3.4 M pts/sec. |
| day 2 | `OpenLandMap/SOL/SOL_CEC_USDA-4B1A_M/v02` deprecated | GEE returned HTTP 400 "Image asset not found". Probed alternatives — `projects/soilgrids-isric/cec_mean` works (band `cec_0-5cm_mean`, mmol(c)/kg × 10). Substituted. |
| day 2 | `MODIS/061/MOD16A2` (TotalEvapotranspiration) only covers 2021+ | 19 of 22 ET tasks failed for 2002–2020. Switched to `MODIS/061/MOD16A2GF` (gap-filled v6.1) which covers the full 2001-present archive. |
| day 2 | `modis_NDVI_2003` SUCCEEDED but no file on Drive | GEE reported task COMPLETED but Drive had no output (silent write failure, probably quota or sync hiccup). Cleared from state, resubmitted, second attempt produced the file. Confirms the per-task resilient retry path. |
| day 2 | `MODIS/061/MOD16A2` → `MOD16A2GF` swap | Same product family, gap-filled variant, full archive coverage. Documented in `BANDS['modis']['TotalEvapotranspiration']` notes. |
| day 2 | `chirps` Precipitation switched to ERA5-Land | CHIRPS doesn't cover >50°N. ~13.5 % of LUCAS+LFU+LfL samples and ~17.7 % of the 1.3 M grid fall in that strip. ERA5-Land has global coverage. Asset: `ECMWF/ERA5_LAND/MONTHLY_AGGR`, band `total_precipitation_sum`. |
| day 2 | Pipeline state tracker (`Data/pipeline_state.json`) | Resumability across phases (gee/pull/cut/project/verify) and per-item granularity (per task, per TIFF, per (band, year)). Atomic write via tempfile+rename. |
| day 2 | `pull_from_drive.py` only marks phase done when all expected files are present | Earlier version marked done even on partial completion. Now compares local `.tif` count against `gee.done` set; phase stays `in_progress` until 100 % match. |
| day 2 | TIFF audit reveals all band ranges plausible after fixes | 250 OK, 1 minor WARN (SoilGrids CEC 8 % NaN over water/urban — inherent to the dataset, accepted). |

---

## Scripts created / modified

All paths relative to `/home/valerian/SGTPublication/`.

### New scripts (under `SOCmapping/SamplePoints/`)

| Script | Purpose |
|---|---|
| `pipeline_state.py` | JSON-backed resumable state for the whole pipeline. `State()` class with `start_phase`, `finish_phase`, `is_done(phase, item)`, `mark_done`, `summary()`. CLI: `python pipeline_state.py {summary,show,reset-phase,init}`. |
| `gee_download_all_bands.py` | Submits 251 `Export.image.toDrive()` tasks via the `--category curated` default. Includes `--validate-assets`, `--dry-run`, per-task try/except so one bad asset doesn't abort the batch, state-tracker integration. |
| `pull_from_drive.py` | Pulls TIFFs from Drive. Default backend: **rclone** (proper OAuth, resumable, handles private folders). Fallback: gdown (public folders only). Marks pull phase done only after local count matches `gee.done`. |
| `tiff_to_tiles.py` | Cuts each TIFF into 12 canonical-name npy tiles (979×979 each). `--materialize-yearly BAND ...` writes statics to `YearlyValue/<band>/2002/` + 21 year-symlinks. `_wipe_stale_tiles` removes non-canonical leftovers before writing. NaN-fill with tile mean. |
| `project_lucas_coords.py` | Projects GPS → (tile_id, row, col) per (band, year). `--mode samples` (LUCAS+LFU+LfL xlsx, 30k pts → `OC_LUCAS_LFU_LfL_Coordinates_v2/`). `--mode 1mil` (Bavaria CSV, 1.3M pts → `Coordinates1Mil/`). Vectorized. `--symlink-across-years` avoids 21× duplication per band. mtime-based staleness check forces regen when tiles are newer. |
| `compute_dem_derivatives.py` | (Legacy fallback) Local Slope/Aspect/TWI from the existing Elevation tiles. Not used in default pipeline — these come from GEE now. Kept for environments without GEE access. |
| `monitor_gee.py` | Live status of the GEE batch: `--watch`, `--failed`, `--running`. Deduplicates by description (latest/best state wins) so stale FAILED records from earlier retries don't show. |
| `retry_failed_gee.py` | After a server-side FAILED, removes the task's desc from `pipeline_state.gee.done` so the next `gee` run resubmits. |
| `probe_cec_alternatives.py` | One-shot CEC asset probe used during the OpenLandMap-CEC outage. Tested 7 candidates, picked SoilGrids 2.0. |
| `audit_tiffs.py` | Per-TIFF cleanliness check: CRS, bounds, shape, NaN fraction, per-band value-range plausibility. Output: OK/WARN/FAIL per file + per-category + per-band cross-year stats. |
| `scaffold_synthetic_bands.py` | Symlink the 14 new bands to LAI's directories for offline pipeline testing before the real GEE data arrives. `--undo` cleans up. |
| `test_pipeline.py` | End-to-end dataloader smoke test (20 channels in → tensor shape `(20, 5, 5, 5)` + forward + backward through `EnhancedTFT`). |
| `prepare_hf_export.py` | Builds `Data_HF/` as a symlink-based mirror of the publishable subset. Writes `manifest.json` + `README.md`. |
| `push_to_hf.py` | Uploads `Data_HF/` to HF as a dataset repo with `multi_commits=True` for resumable upload. |
| `pull_from_hf.py` | Downloads the dataset back from HF into a target folder. |
| `run_full_pipeline.sh` | Orchestrator: 6 phases (plan → gee → pull → cut → project → verify). State-aware skip; `FORCE=1` overrides; single-phase invocation supported (`bash run_full_pipeline.sh cut`). |

### Modified scripts

| File | Change |
|---|---|
| `SOCmapping/*/config.py` (9 model configs) | `bands_list_order` extended from 6 → 20 entries. Added per-band path variables for the 14 new bands. Extended `SamplesCoordinates_Yearly` and `DataYearly` lists. All Yearly path entries aligned. Seasonal blocks unchanged (out of scope). |

### Documentation

| File | Contents |
|---|---|
| `SOCmapping/SamplePoints/README.md` | End-to-end pipeline guide; updated when bands or sources changed. |
| `SOCmapping/docs/PIPELINE_BUILDOUT_2026-05.md` | This file. |

---

## Data inventory

### 20 input channels (final `bands_list_order`)

| # | Band | Tier on disk | Source asset | Reducer | Notes |
|---|---|---|---|---|---|
| 1 | Elevation | StaticValue | `USGS/SRTMGL1_003` | static | 30 m → 250 m |
| 2 | LAI | YearlyValue | `MODIS/061/MCD15A3H` band `Lai` | mean | LAI × 10 |
| 3 | LST | YearlyValue | `MODIS/061/MOD11A2` band `LST_Day_1km` | mean | Kelvin × 0.02 |
| 4 | MODIS_NPP | YearlyValue | `MODIS/061/MOD17A3HGF` band `Npp` | mean | kg C/m²/yr × 0.0001 |
| 5 | SoilEvaporation | YearlyValue | `ECMWF/ERA5_LAND/MONTHLY_AGGR` band `evaporation_from_bare_soil_sum` | sum | m water/year |
| 6 | TotalEvapotranspiration | YearlyValue | `MODIS/061/MOD16A2GF` band `ET` | sum | mm × 0.1 |
| 7 | NDVI | YearlyValue | `MODIS/061/MOD13Q1` band `NDVI` | mean | NDVI × 10000 |
| 8 | EVI | YearlyValue | `MODIS/061/MOD13Q1` band `EVI` | mean | EVI × 10000 |
| 9 | Precipitation | YearlyValue | `ECMWF/ERA5_LAND/MONTHLY_AGGR` band `total_precipitation_sum` | sum | m water/year (switched from CHIRPS for full-Bavaria coverage) |
| 10 | AirTemperature | YearlyValue | `ECMWF/ERA5_LAND/MONTHLY_AGGR` band `temperature_2m` | mean | Kelvin |
| 11 | SoilMoisture_layer1 | YearlyValue | `ECMWF/ERA5_LAND/MONTHLY_AGGR` band `volumetric_soil_water_layer_1` | mean | m³/m³ |
| 12 | SnowDepth | YearlyValue | `ECMWF/ERA5_LAND/MONTHLY_AGGR` band `snow_depth` | mean | m |
| 13 | ClayContent_0_10cm | static→yearly symlinks | `OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02` | static | % clay |
| 14 | SandContent_0_10cm | static→yearly | `OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02` | static | % sand |
| 15 | pH_H2O_0_10cm | static→yearly | `OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02` | static | pH × 10 |
| 16 | BulkDensity_0_10cm | static→yearly | `OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02` | static | g/cm³ × 100 |
| 17 | CEC_0_10cm | static→yearly | `projects/soilgrids-isric/cec_mean` band `cec_0-5cm_mean` | static | mmol(c)/kg × 10 (SoilGrids 2.0; substituted after OpenLandMap CEC outage) |
| 18 | Slope | static→yearly | server-side: `ee.Terrain.slope(USGS/SRTMGL1_003)` | static | degrees |
| 19 | Aspect | static→yearly | server-side: `ee.Terrain.aspect(USGS/SRTMGL1_003)` | static | degrees clockwise from N |
| 20 | TWI | static→yearly | server-side: `ln(MERIT/Hydro/v1_0_1.upa × 10⁶ / tan(slope_floored))` | static | dimensionless |

### Coordinate trees

| Tree | Source CSV/XLSX | Rows | Use |
|---|---|---|---|
| `OC_LUCAS_LFU_LfL_Coordinates_v2/` | `LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx` | **30 227** (combined LUCAS + LFU + LfL after dropna) | Training (ground-truth + SOC values) |
| `Coordinates1Mil/` | `coordinates_Bavaria_1mil.csv` | **1 300 000** | Inference (full-Bavaria SOC mapping) |

Both trees have the same structure: `{Static,Yearly}Value/<band>/[<year>/]coordinates.npy`, where each `coordinates.npy` has columns `[lat, lon, tile_id, row, col]`. Yearly bands store one physical anchor + 21 year-symlinks per band (since GPS doesn't change year-to-year).

### On-disk layout summary

```
Data/
├── pipeline_state.json                                 # state tracker (all phases done)
├── LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx               # 30k samples + OC
├── LUCAS_LFU_Bavaria_OC_joint_data_modified.xlsx       # seasonal-resolution joint
├── Coordinates1Mil/
│   ├── coordinates_Bavaria_1mil.csv                    # 1.3M reference grid
│   ├── StaticValue/Elevation/coordinates.npy
│   └── YearlyValue/<band>/<year>/coordinates.npy       # 20 bands × 22 years
├── OC_LUCAS_LFU_LfL_Coordinates_v2/
│   ├── StaticValue/Elevation/coordinates.npy
│   └── YearlyValue/<band>/<year>/coordinates.npy
├── RasterTensorData/
│   ├── StaticValue/Elevation/IDxN…S…W…E….npy           # 12 canonical tiles
│   └── YearlyValue/<band>/<year>/IDx…npy               # 20 bands × 22 years × 12 tiles
└── RasterBandsData/                                    # legacy v1, kept untouched
```

Original `Data/` directory unmodified except for the new ERA5-Precipitation tiles and the recut versions of the 6 pre-existing bands (now aligned canonical filenames).

---

## Operations reference

### Pipeline state inspection / control

```bash
python3 SOCmapping/SamplePoints/pipeline_state.py summary
python3 SOCmapping/SamplePoints/pipeline_state.py show               # raw JSON
python3 SOCmapping/SamplePoints/pipeline_state.py reset-phase --phase pull
python3 SOCmapping/SamplePoints/pipeline_state.py init               # reinit empty
```

### Re-running phases

```bash
bash SOCmapping/SamplePoints/run_full_pipeline.sh                    # full pipeline; state-aware skips
bash SOCmapping/SamplePoints/run_full_pipeline.sh gee                # single phase
FORCE=1 bash …/run_full_pipeline.sh cut                              # force a phase even if 'done'

DRIVE_FOLDER_ID=12X21qgpt1tqmDr-nfNyZe-7m7DBW2wlP \
  bash …/run_full_pipeline.sh pull                                   # provide Drive folder ID for the pull
```

### GEE monitoring

```bash
python3 SOCmapping/SamplePoints/monitor_gee.py                        # snapshot summary
python3 SOCmapping/SamplePoints/monitor_gee.py --watch                # live refresh every 30s
python3 SOCmapping/SamplePoints/monitor_gee.py --failed               # with real error messages
python3 SOCmapping/SamplePoints/monitor_gee.py --no-dedupe            # show every historical task record
```

### Asset validation (before submitting)

```bash
python3 SOCmapping/SamplePoints/gee_download_all_bands.py --validate-assets
python3 SOCmapping/SamplePoints/gee_download_all_bands.py --dry-run --years 2002 2023
```

### TIFF audit

```bash
python3 SOCmapping/SamplePoints/audit_tiffs.py
python3 SOCmapping/SamplePoints/audit_tiffs.py --verbose
python3 SOCmapping/SamplePoints/audit_tiffs.py --csv /tmp/audit.csv
```

### HF push

```bash
python3 SOCmapping/SamplePoints/prepare_hf_export.py
python3 SOCmapping/SamplePoints/push_to_hf.py \
    --repo-id <username>/<dataset-name> --create-repo [--private]
```

### HF pull (on another machine)

```bash
pip install huggingface_hub
huggingface-cli login          # only for private repos
python3 SOCmapping/SamplePoints/pull_from_hf.py \
    --repo-id <username>/<dataset-name> --out ./Data_HF_pulled
export SOC_DATA_DIR=$(realpath ./Data_HF_pulled)
```

---

## Caveats & known limitations

1. **CEC has 8.2 % NaN** — SoilGrids 2.0 has scattered no-data pixels over water bodies and unsoiled urban patches. `tiff_to_tiles._fill_nan()` replaces them with the per-tile mean. Acceptable for SOC modeling at 250 m.

2. **Dataloader RAM usage** — `RasterTensorDataset.__init__` eagerly pre-loads all 12 tiles per (band, year) into `self.data_cache`. For 20 bands × 22 years × ~46 MB each, that's ~20 GB. On 16 GB laptops, instantiating the full `NormalizedMultiRasterDatasetMultiYears` will OOM-kill. Workarounds: train on machines with ≥32 GB RAM; or change the dataloader to lazy-load (single-line edit in `_create_id_mapping` / `get_tensor_by_location` — keep `self.data_cache` empty, always `np.load(filepath)` on demand).

3. **Existing checkpoints break** — the 6→20 channel expansion means pretrained checkpoints have a mismatched first conv (6 input channels vs 20 now). Load with `strict=False` and reinit the first conv layer; the rest of the weights transfer. Cleaner: train from scratch on the 20-channel setup.

4. **MOD16A2 vs MOD16A2GF**: TotalEvapotranspiration data values may differ slightly from the pre-existing on-disk data, because the gap-filled variant interpolates missing 8-day composites. For annual sums (which is what we use), the difference is small.

5. **GEE failed-task records persist**: 19 historical `MOD16A2` FAILED records and 1 `OpenLandMap_CEC` FAILED record remain in the GEE task list as audit trail. They're harmless but visible via `earthengine task list`. The `monitor_gee.py` `--no-dedupe` flag shows them; the default deduped view hides them.

6. **HF symlinks**: `Data_HF/` uses local symlinks for space efficiency (24 KB local pointing at 43 GB real). `huggingface_hub.upload_folder` follows symlinks and uploads the resolved file content. Pulling from HF on another machine produces real files, not symlinks.

7. **Bbox drift between v1 and v2 tile filenames**: prior to this session, the pre-existing 6 bands had 15-decimal-precision bboxes in their tile filenames (e.g. `N48_509545174475655…`); the new pipeline uses 4-decimal precision (`N48_5095…`). Ground drift is ~5 m, well below 3 % of a 250 m pixel. Redownload + re-cut resolved this — all 20 bands now share canonical-name tiles with identical bboxes. `_wipe_stale_tiles` was added to `tiff_to_tiles.py` to delete pre-v2 leftovers before writing canonical names.

---

## Resumability guarantees

Every phase is idempotent + state-aware:

| Phase | Per-item resume key | What happens on re-run after partial completion |
|---|---|---|
| gee | task description (`modis_NDVI_2003`) | Already-submitted descriptions skipped; missing/failed get resubmitted |
| pull | TIFF basename | rclone `--ignore-existing` skips local files; missing ones downloaded |
| cut | TIFF basename + 12 canonical tile names | If 12 canonical tiles exist, the TIFF is skipped; non-canonical leftovers wiped |
| project | per (band, year) `coordinates.npy` existence + mtime | Regenerated if any tile is newer than `coordinates.npy` |
| verify | (just runs `test_pipeline.py`) | safe to re-run |

If the machine reboots mid-pipeline, re-run `bash run_full_pipeline.sh` and it picks up exactly where it stopped — no double work, no double GEE submissions, no double-uploaded HF chunks.

---

## Acknowledgements / sources

- **MODIS** products: NASA LP DAAC.
- **ERA5-Land**: Copernicus Climate Change Service / ECMWF.
- **CHIRPS** (not used in final): UCSB CHG.
- **OpenLandMap** soil rasters: EnvirometriX / Mendeley Data.
- **SoilGrids 2.0**: ISRIC – World Soil Information.
- **SRTM**: NASA / USGS.
- **MERIT Hydro**: Yamazaki Lab, Univ. of Tokyo.
- **Google Earth Engine**: data hosting and server-side computation.

LUCAS soil samples: Eurostat (EU Land Use/Cover Area frame Survey).
LFU samples: Bayerisches Landesamt für Umwelt.
LfL samples: Bayerische Landesanstalt für Landwirtschaft.

---

*Document generated: 2026-05-17. Pipeline state: all 6 phases done. Next action: HuggingFace push.*
