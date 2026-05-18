# Full session log — 2026-05-17

End-to-end build of a 20-channel, 2002–2023, 250 m Bavaria SOC dataset,
plus dataloader verification and HuggingFace publication.

**Final state at end of session: ✅ Pipeline complete, dataset live at**
`https://huggingface.co/datasets/ValerianFourel/sgt-bavaria-soc-2002-2023`
(private, 9 366 files, ~44 GB).

This document supersedes the earlier `PIPELINE_BUILDOUT_2026-05.md` and
`NEXT_STEPS.md` — they cover the same work but in progress; this one
is the post-completion summary.

---

## 1. Final state

### Pipeline phases (all done)

| Phase | Items completed | Notes |
|---|---|---|
| `plan` | dry-run printed 251 tasks | curated set |
| `gee` | 251 tasks SUCCEEDED (after substitutions) | server-side, 1–6 h passive |
| `pull` | 251 TIFFs in `~/bavaria_tiffs/` | via rclone, ~2.9 GB |
| `cut` | 251 × 12 = 3 012 canonical-name tiles | with stale-tile wipe |
| `project` | 20 bands × 2 modes (samples + 1mil) | float64 coords, mtime-aware |
| `verify` | TIFF audit + dataloader test | 250 OK / 1 minor WARN / 0 FAIL |
| **HF push** | **9 366 files at `ValerianFourel/sgt-bavaria-soc-2002-2023`** | **upload_large_folder, 8 workers** |

### Verified working

- **20-channel tensor** `(C=20, H=5, W=5, T=5)` produced by
  `MultiRasterDatasetMultiYears.__getitem__` on a random 2007 LUCAS sample
  (POINTID `gzf9IWQa`, lat 49.82°N, lon 10.92°E, OC 7.9 g/kg).
- **All 9 statics** identical across the T axis (Elevation + 5 soil + 3 terrain).
- **All 11 yearly bands** vary across the T axis (LAI/LST/MODIS_NPP/SoilEvap/TotalEvap/NDVI/EVI/Precipitation/AirTemperature/SoilMoisture_layer1/SnowDepth).
- **100 % finite values** in the 2 500-element output tensor.
- **Per-band ranges all physically plausible** (NDVI 0.4–0.7, LST ~285 K,
  Precipitation 0.6–1.1 m/yr, BulkDensity 1.1–1.4 g/cm³, etc.)
- **TIFF audit**: 250 OK + 1 minor WARN (SoilGrids CEC 8 % NaN over water/urban — accepted).
- **HF dataset**: 9 366 files, including 1.3 M Bavaria grid coordinates,
  the combined LUCAS+LFU+LfL 30 k samples xlsx, all 20-band tiles +
  bounds arrays, both coord trees (training + inference).

### What we did NOT change

- `SOCmapping/TemporalFusionTransformer/dataloader/dataloaderMultiYears.py`
- Any model file (`model.py`, `train.py`, `running.py`, etc.)
- The training loop or loss code
- The seasonal-mode paths (legacy SeasonalValue trees retained for old tests, not extended to new bands)
- Original `Data/` directory (only added new tiles, never deleted existing ones except recutting the 6 pre-existing bands onto the canonical grid)
- The 9 model `config.py` files — only `bands_list_order` and per-band path entries were appended; everything else (window_size, MAX_OC, paths to legacy data, …) untouched

---

## 2. Chronological log

### Phase A — Buildout (initial scripts + data prep, earlier in session)

| Decision | Why |
|---|---|
| Curate band list to 20 channels | The repo had `gee_download_all_bands.py` aiming for ~18 bands; needed to drop SOC-leakage (`OrganicCarbon_Reference`), categorical (`TextureUSDA`), and Sentinel-1/2 (2014+ gap years) before submitting. |
| Static-as-yearly via symlinks | Dataloader special-cases only the string `'Elevation'` as static. To add 8 more statics without touching the dataloader, materialize them at `YearlyValue/<band>/2002/` and symlink years 2003..2023 to that anchor. |
| Pixel grid: EPSG:4326 / 250 m / fixed 12-tile bbox | Matches existing 12-tile Elevation layout already on disk. |
| Slope/Aspect/TWI from GEE server-side | User preference (everything from GEE for consistency). `ee.Terrain.slope/aspect(SRTM)` + `ln(MERIT/Hydro.upa × 10⁶ / tan(slope_floored))`. |
| Redownload the 6 existing bands too | Pre-existing on-disk tile filenames had 15-decimal bbox precision; new pipeline uses 4-decimal canonical names. Redownload + recut onto the canonical grid for pixel-perfect alignment across all 20 channels. |
| Pipeline state tracker (`Data/pipeline_state.json`) | Resumability across all 7 phases + per-item granularity (per task, per TIFF, per (band, year)). Atomic write via tempfile + rename. |

### Phase B — GEE asset substitutions (mid-session, after initial submission)

| Issue | Fix |
|---|---|
| `OpenLandMap/SOL/SOL_CEC_USDA-4B1A_M/v02` returns HTTP 400 | Probed 7 candidates with `probe_cec_alternatives.py`. Switched to `projects/soilgrids-isric/cec_mean` band `cec_0-5cm_mean`. |
| 19 `modis_TotalEvapotranspiration_2002..2020` tasks FAILED | `MODIS/061/MOD16A2` only covers 2021+ on GEE. Switched to gap-filled `MODIS/061/MOD16A2GF` which covers 2001-present. Cleared the 19 FAILED entries from state with `retry_failed_gee.py` so they get resubmitted. |
| `modis_NDVI_2003` reported COMPLETED but file missing on Drive | Silent Drive write failure (quota or sync hiccup). Cleared from state, resubmitted, second attempt produced the file. |

### Phase C — Local pipeline run (pull → cut → project → verify)

| Step | Outcome |
|---|---|
| `gdown` failed with public-link rate limit at 32 / 251 TIFFs | Switched `pull_from_drive.py` default backend to **rclone** with proper OAuth. Resumed pull cleanly, fetched the remaining 219. |
| 1 missing TIFF after pull (`modis_NDVI_2003.tif`) | Handled via resubmit-and-pull-again. |
| `tiff_to_tiles.py` cut all 251 TIFFs into canonical 12-tile sets | `_wipe_stale_tiles()` removed pre-existing non-canonical-named tiles for the 6 redownloaded bands before writing the new canonical names. Otherwise dataloader's `glob('ID*.npy')` would pick ambiguously. |
| `project_lucas_coords.py` ran for samples + 1mil modes | Vectorized projection — 0.4 s per 1.3 M points. Wrote per-(band, year) `coordinates.npy` with 1 physical anchor + 21 file-symlinks across years. |
| `verify` phase: TIFF audit + dataloader smoke test | Initial audit FAILED 251/251 due to bbox-tolerance script bug and wrong BulkDensity expected range. Fixed both, re-ran: 228 OK / 23 WARN. The 22 WARNs were CHIRPS Precipitation NaN (±50°N coverage gap); 1 was SoilGrids CEC 8 % NaN. |

### Phase D — Precipitation: CHIRPS → ERA5-Land

| Discovery | Action |
|---|---|
| CHIRPS only covers ±50°N latitude; ~13.5 % of LUCAS+LFU+LfL samples and 17.7 % of the 1.3 M grid fall above 50°N. tile-mean NaN-fill would propagate southern-Bavaria mean precipitation to northern-Bavaria samples. | User chose to switch to ERA5-Land. |
| Switched `BANDS['era5']['Precipitation']` to use `ECMWF/ERA5_LAND/MONTHLY_AGGR` band `total_precipitation_sum` with annual sum reducer. Removed CHIRPS Precipitation from `_CURATED_BANDS`. | |
| Cleared 22 `chirps_Precipitation_*` entries from `pipeline_state.gee.done`, `pull.done`, `cut.done`. Deleted the 22 local CHIRPS TIFFs and the 22 × 12 = 264 cut tiles under `YearlyValue/Precipitation/`. Removed orphan CHIRPS files from Drive (Drive Trash). | |
| Resubmitted 22 era5_Precipitation tasks. State tracker correctly skipped the 229 already-submitted entries. | |
| Waited for 22 tasks to COMPLETE on GEE (~10 min). | |
| Re-pulled (rclone fetched the 22 new TIFFs, also re-fetched 22 stale CHIRPS files that needed manual cleanup). Cleaned up the local + Drive CHIRPS files again. | |
| Re-cut the 22 era5_Precipitation TIFFs. | |
| Re-projected (mtime-aware check detected the new tiles, regenerated Precipitation `coordinates.npy` for both modes). | |
| Re-audit: **250 OK / 1 WARN / 0 FAIL.** Precipitation now 0.44–2.25 m/year (440–2 250 mm/yr, normal Bavaria). | |

### Phase E — Dataloader full-band test (random 2007 sample)

- Wrote `tmp_dataloader_check.py`. Picks a random 2007 LUCAS+LFU+LfL
  sample, sets up 96 (band, year) paths (Elevation + 19 yearly bands × 5 years 2003..2007),
  instantiates `MultiRasterDatasetMultiYears` with a 1-row dataframe,
  pulls `__getitem__(0)`.
- **First attempt failed**: `ValueError: Coordinates (...) not found in StaticValue/Elevation`.
- Diagnosed: a precision regression I'd introduced in
  `project_lucas_coords.py` — saving `coordinates.npy` as `float32`
  instead of the original `float64`. Round-trip lost precision; the
  dataloader's exact-equality lookup `coords[:, 1] == longitude` then
  failed for samples with many-decimal GPS values.
- **Fixed**:
  - `project_lucas_coords.py`: changed `out = np.empty((n_points, 5), dtype=np.float32)` → `float64`. Same for `tile_id` array. Removed `astype(np.float32)` on row/col.
  - Deleted all `coordinates.npy` files in both trees, reset `project` phase, re-ran.
  - Discovered orchestrator's `YEARLY_BANDS` list was only the **6 new** bands, not all 11. Fixed to include LAI/LST/MODIS_NPP/SoilEvaporation/TotalEvapotranspiration too. Manually projected those 5 missing bands.
- **Re-test passed**:
  - tensor shape `(20, 5, 5, 5)`, dtype `torch.float32`, 100 % finite
  - all 9 statics constant across T (proves symlink trick works end-to-end)
  - all 11 yearly bands vary across T (proves real per-year data is loaded)
  - per-band ranges all physically plausible for that location (central Bavaria, near Nuremberg)

### Phase F — HuggingFace push

1. Installed `huggingface_hub` v1.15. Logged in with user token. `whoami` confirmed `ValerianFourel`, role `write`.
2. **First push attempt** (`upload_folder` from `Data_HF/`) uploaded only **5 files** — the directory-level symlinks in `Data_HF/` (Coordinates1Mil, OC_LUCAS_LFU_LfL_Coordinates_v2, RasterTensorData) were not followed by `upload_folder`. Only the file-level symlinks (xlsx) got resolved.
3. **Patched `push_to_hf.py`** to upload directly from `/home/valerian/SGTPublication/Data/` with `allow_patterns` (Coordinates1Mil, OC_LUCAS_LFU_LfL_Coordinates_v2, RasterTensorData, xlsx) + `ignore_patterns` (RasterBandsData v1, pipeline_state, __pycache__, .bak).
4. **Second push attempt** (`upload_folder`) was slow: 4 minutes for 0.5 MB. HF library suggested `upload_large_folder` for big uploads.
5. Killed second attempt. **Third push** with `HfApi().upload_large_folder(num_workers=8)` — parallel uploads, chunked, content-hash deduplication.
6. Mid-upload, found and fixed the float32 → float64 precision bug (above). Stopped the upload, regenerated coords, restarted upload.
7. **Upload completed** in ~22 minutes total: **9 366 files** at ~250 files/min average.
8. HF deduplicated identical content automatically — the 8 static-as-yearly bands have only 12 raster tiles each on HF (not 22 × 12), saving ~8 GB. The 22 per-year `coordinates.npy` files for each band still appear as 22 entries but share storage via HF's content-hash dedup.

---

## 3. Scripts created today

All under `SOCmapping/SamplePoints/`. New scripts (not modifying anything in the model or dataloader code):

| Script | Purpose |
|---|---|
| `pipeline_state.py` | JSON-backed resumable state for the whole pipeline. `State()` class with `start_phase`, `finish_phase`, `is_done(phase, item)`, `mark_done`, `summary()`. CLI: `python pipeline_state.py {summary,show,reset-phase,init}`. |
| `gee_download_all_bands.py` | Submits 251 `Export.image.toDrive()` tasks via the `--category curated` default. Includes `--validate-assets`, `--dry-run`, per-task try/except, state-tracker integration. |
| `pull_from_drive.py` | Pulls TIFFs from Drive. Default backend: rclone (proper OAuth, resumable). Fallback: gdown. Marks pull phase done only after local count matches `gee.done`. |
| `tiff_to_tiles.py` | Cuts each TIFF into 12 canonical-name `.npy` tiles (979×979 each). `--materialize-yearly BAND ...` flag for static-as-yearly. `_wipe_stale_tiles` cleans non-canonical leftovers. NaN-fill with tile mean. |
| `project_lucas_coords.py` | Vectorized projection of GPS → `(tile_id, row, col)` per (band, year). Two modes: `samples` (30k LUCAS+LFU+LfL → `OC_LUCAS_LFU_LfL_Coordinates_v2/`) and `1mil` (1.3 M Bavaria grid → `Coordinates1Mil/`). mtime-aware skip-done. **Stores float64 for dataloader precision** (post-fix). |
| `compute_dem_derivatives.py` | Legacy fallback for Slope/Aspect/TWI from local Elevation (not used — GEE provides these now). |
| `monitor_gee.py` | Live status of the GEE batch with `--watch`, `--failed`, `--running`, `--no-dedupe`. Deduplicates by description (latest/best state). |
| `retry_failed_gee.py` | After a server-side FAILED, removes the task's description from `pipeline_state.gee.done` so a subsequent `gee` run resubmits. |
| `probe_cec_alternatives.py` | One-shot CEC asset probe used during the OpenLandMap-CEC outage. Tested 7 candidates → picked SoilGrids 2.0. |
| `audit_tiffs.py` | Per-TIFF cleanliness check: CRS, bounds, shape, NaN fraction, per-band value-range plausibility. CSV export, per-category + per-band cross-year stats. |
| `scaffold_synthetic_bands.py` | Symlinks the 14 new bands to LAI's directories for offline pipeline testing before real GEE data arrives. `--undo` cleans up. |
| `test_pipeline.py` | End-to-end dataloader smoke test (20 channels in → tensor shape `(20, 5, 5, 5)` + forward + backward through `EnhancedTFT`). |
| `tmp_dataloader_check.py` | Today's focused test: random 2007 LUCAS+LFU+LfL sample × full 20-band 5-year tensor. Uses a 96-path subset to fit in 5 GB RAM. Reports cross-T behavior verification. |
| `prepare_hf_export.py` | Builds `Data_HF/` as a symlink-based mirror with `manifest.json` + `README.md`. |
| `push_to_hf.py` | Uploads to HF via `upload_large_folder(num_workers=8)` with allow/ignore patterns. Uploads directly from `Data/` because `upload_folder` doesn't traverse directory symlinks. |
| `pull_from_hf.py` | Downloads the dataset from HF on another machine. |
| `run_full_pipeline.sh` | Orchestrator: 6 phases (plan → gee → pull → cut → project → verify). State-aware skip; `FORCE=1` override; single-phase invocation supported. |

## 4. Files modified today

| File | Change |
|---|---|
| 9 × `SOCmapping/*/config.py` | `bands_list_order` extended 6 → 20 entries. Added per-band path variables for the 14 new bands. Extended `SamplesCoordinates_Yearly` and `DataYearly` lists. Seasonal blocks unchanged. |

---

## 5. Caveats discovered today (worth knowing)

1. **CEC has 8 % NaN** — SoilGrids 2.0 has scattered no-data over water/urban patches. NaN-filled with per-tile mean. Acceptable for SOC at 250 m.

2. **Dataloader RAM usage** — `RasterTensorDataset.__init__` eagerly pre-loads all 12 tiles per (band, year) into `self.data_cache`. For 20 × 22 = 440 datasets at ~46 MB each, that's ~20 GB. On a 16 GB laptop, instantiating the full `NormalizedMultiRasterDatasetMultiYears` will OOM-kill. Workarounds:
   - Train on ≥32 GB RAM machines
   - Use a year-range subset
   - Make `RasterTensorDataset` lazy (single-line edit in `_create_id_mapping` / `get_tensor_by_location`)

3. **Existing checkpoints break** — 6 → 20 channel expansion means pretrained checkpoints have a mismatched first conv. Load with `strict=False` and reinit first conv; rest of the weights transfer.

4. **CHIRPS replaced by ERA5-Land for Precipitation** — CHIRPS doesn't cover >50°N. ~14 % of Bavaria samples would have had tile-mean fill. ERA5-Land provides full latitude coverage. Unit changed from mm/year (CHIRPS) → m water/year (ERA5). Dataloader auto-normalizes per channel, so the magnitude change is transparent to the model.

5. **MOD16A2 → MOD16A2GF for TotalEvapotranspiration** — the non-gap-filled v6.1 asset only covers 2021+ on GEE. Gap-filled variant has the full 2001-present archive with same band names + units.

6. **GEE failed-task records persist forever** — historical FAILED tasks from CHIRPS, MOD16A2, OpenLandMap-CEC, and `modis_NDVI_2003` (the silent Drive write failure) remain in the GEE task list as audit trail. Harmless. `monitor_gee.py` deduplicates them out of the default summary; `--no-dedupe` shows all.

7. **Float32 vs Float64 coords** — original samplePoints.py used float64; my initial `project_lucas_coords.py` saved float32. Round-trip lost precision, causing the dataloader's exact-equality GPS lookup to fail. **Fixed**: coords are now float64 on disk for both modes.

8. **HF directory symlinks not traversed by `upload_folder`** — fixed by uploading directly from `Data/` with `allow_patterns`. `upload_large_folder` was needed because `upload_folder` was too slow on 44 GB. Both can be re-run after a network drop and will resume from the last committed chunk.

9. **SeasonalValue subtree got swept up** — the existing `Data/RasterTensorData/SeasonalValue/` (legacy seasonal-aggregated tiles for LAI/LST/SoilEvap/TotalEvap from an earlier pipeline) was uploaded because the allow pattern `RasterTensorData/**` matches it. It doesn't include the 14 new bands, so it's only useful for seasonal-mode training on the 5 original bands. Can be deleted from HF with `HfApi().delete_folder(...)` if not wanted.

---

## 6. Operations reference (cheat sheet)

```bash
# State inspection / control
python3 SOCmapping/SamplePoints/pipeline_state.py summary
python3 SOCmapping/SamplePoints/pipeline_state.py reset-phase --phase pull

# Re-run phases (state-aware; FORCE=1 overrides skip)
bash SOCmapping/SamplePoints/run_full_pipeline.sh
bash SOCmapping/SamplePoints/run_full_pipeline.sh cut       # single phase
FORCE=1 bash SOCmapping/SamplePoints/run_full_pipeline.sh project

# Provide Drive folder ID for the pull (only needed once)
DRIVE_FOLDER_ID=12X21qgpt1tqmDr-nfNyZe-7m7DBW2wlP \
  bash SOCmapping/SamplePoints/run_full_pipeline.sh pull

# GEE monitoring
python3 SOCmapping/SamplePoints/monitor_gee.py
python3 SOCmapping/SamplePoints/monitor_gee.py --watch
python3 SOCmapping/SamplePoints/monitor_gee.py --failed

# Audit
python3 SOCmapping/SamplePoints/audit_tiffs.py
python3 SOCmapping/SamplePoints/audit_tiffs.py --csv /tmp/audit.csv

# HF push / pull
python3 SOCmapping/SamplePoints/push_to_hf.py --repo-id <user>/<name> --create-repo --private
python3 SOCmapping/SamplePoints/pull_from_hf.py --repo-id <user>/<name> --out ./Data_HF_pulled
```

---

## 7. What's next (none required — pipeline complete)

The data is published. Optional follow-ups:

- **Retrain** with the 20-channel input (`bands_list_order` is 20 in every model config). Pretrained 6-channel checkpoints will need `strict=False` + first-conv reinit.
- **Slim the HF repo** by deleting the SeasonalValue subtree (~5 GB) if you don't use seasonal-mode training.
- **Make the HF repo public** if intended for the SGT publication. Currently private.
- **Dataloader memory fix** — convert `RasterTensorDataset._create_id_mapping` to lazy-load instead of eagerly pre-loading every tile. Brings 20-channel instantiation from 20 GB to negligible RAM.
- **Drive cleanup** — purge Drive Trash to free space (orphan CHIRPS files + failed-task aborted writes).

---

## 8. Reference URLs

| Resource | URL |
|---|---|
| HuggingFace dataset (private) | <https://huggingface.co/datasets/ValerianFourel/sgt-bavaria-soc-2002-2023> |
| GEE task list | <https://code.earthengine.google.com/tasks> |
| Drive folder of source TIFFs | drive.google.com/drive/folders/12X21qgpt1tqmDr-nfNyZe-7m7DBW2wlP |
| MODIS LP DAAC | <https://lpdaac.usgs.gov/> |
| ERA5-Land | <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means> |
| OpenLandMap | <https://openlandmap.org/> |
| SoilGrids 2.0 | <https://soilgrids.org/> |
| MERIT Hydro | <https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/> |

---

*Generated 2026-05-17 at end of session. All 7 pipeline phases done. Dataset live on HF.*
