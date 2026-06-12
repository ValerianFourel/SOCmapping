# Sentinel mode — regenerate the WHOLE stack at 20 m

Decision (confirmed): **all bands scaled to 20 m**, not just the new S2 SWIR
pair. This is a full dataset regeneration. Branch: `bestrun-bands-s2swir`.

The current dataset is a **250 m** grid (979×979 tiles, `TILE_DEG=1.7986°`,
12 tiles). Sentinel mode rebuilds the same tile *layout* at **20 m** pixels.

## Design: fine bands get a real window, coarse bands are broadcast

The chosen sampling (avoids the storage blow-up):

- **Window**: `9×9` or `11×11` (default 11; `SGT_WINDOW_SIZE=9`) at the 20 m grid.
- **Fine bands** (`_bands.is_sentinel_fine`, native ≤ 30 m — the **22** bands:
  S2 SWIR ×2, Landsat SRC ×11, SRTM/terrain ×9): read the **real** 9×9/11×11
  spatial patch from a 20 m tile.
- **Coarse bands** (≥ 250 m — the **23** bands: MODIS, ERA5, SoilGrids): there is
  no sub-window detail at 20 m, so read the **single nearest-pixel value at the
  point** and **broadcast** it across the whole window (one value → 9×9/11×11).

**Storage consequence (the win):** only the 22 fine bands need 20 m tiles; the 23
coarse bands stay **one value per point** (no 20 m tiles at all). So instead of
~156× the whole dataset, the cost is ~(22 fine bands × 11×11 windows) +
(23 coarse scalars) — close to the current windowed footprint, not multi-TB.

The per-band classification lives in `_bands.py`:
`BAND_NATIVE_M`, `is_sentinel_fine(band)`, `SENTINEL_FINE_BANDS` (22),
`BROADCAST_COARSE_BANDS` (23), and `sentinel_mode()` (reads `SGT_SENTINEL_MODE`).

## What's wired on this branch (the enablers)

- **`gee_download_all_bands.py`** — `--scale` now also sets `EXPORT_SCALE_M`, so
  the source-resampling target follows the export grid. `--category extended
  --scale 20` exports all 43 covariates at a true 20 m (fine bands keep detail,
  coarse bands bilinear-upsample to 20 m). The S2 SWIR pair is `sgt_s2_swir_export.py
  --scale 20`.
- **`tiff_to_tiles.py`** — `TILE_PX` is now env-configurable:
  `SGT_TILE_PX=12238` (= `round(979 * 250/20)`) rebuilds each `TILE_DEG°` tile at
  20 m. Same abutting (lat,lon) layout, only pixel density changes.
- **`sgt_sentinel_export.py`** — one entry point that submits both EE export
  batches (43 covariates + S2 SWIR) at 20 m.
- **Bands**: `full_extended_s2` (45) + `SGT_BANDS_S2=1` opt-in (Tier 4 S2 SWIR),
  see `S2_SWIR_BANDS_README.md`.

## Window: 9×9 / 11×11 (set by the env flag)

The broadcast design fixes the window at the sentinel size — default **11×11**
(220 m of fine-band detail), or `SGT_WINDOW_SIZE=9` for 9×9. No huge window is
needed: fine bands carry the local 20 m detail, coarse bands are a single
broadcast value, so a 9/11 window is the full design.

```bash
export SGT_SENTINEL_MODE=1          # fine=real patch, coarse=broadcast
export SGT_WINDOW_SIZE=11           # or 9
export SGT_TILE_PX=12238            # tiling density for the FINE bands' 20 m tiles
```
`config.py` reads these: `window_size` becomes 11 (or 9) when sentinel mode is on
and stays **5** otherwise, so non-sentinel runs are unchanged.

## Full pipeline (cluster / EE-authenticated)

```bash
# 1) Export ALL bands at 20 m to Drive (two EE batches).
python SOCmapping/SamplePoints/sgt_sentinel_export.py --dry-run     # preview
python SOCmapping/SamplePoints/sgt_sentinel_export.py               # submit
#   -> Drive/bavaria_bands_20m/  (43 covariates + s2_swir_baresoil_2017_2023.tif)

# 2) Tile every exported GeoTIFF at 20 m into the RasterTensorData layout.
export SGT_TILE_PX=12238
#   reuse the existing cut scripts (they read TILE_PX from tiff_to_tiles):
#     sgt_landsat_mb_cut.py  (multiband SRC / extended)
#     + the per-band cut path for the static/yearly bands
#   writes RasterTensorData/{YearlyValue,StaticValue}/<band>/<tile>.npy at 20 m.
#   Standardize: float32, NoData -> NaN, same 12-tile abutting grid.

# 3) Rebuild coordinate indices at 20 m (sample points + 1.3M grid):
#   Coordinates1Mil/<tier>/<band>/  and
#   OC_LUCAS_LFU_LfL_Coordinates_v2/<tier>/<band>/
#   must index the 20 m pixel grid (pixel i,j of each point changes with TILE_PX).
#   Regenerate with the existing coordinate-builder against the 20 m tiles.

# 4) Publish the 20 m variant to a DISTINCT HF path so the 250 m -large is intact:
#   e.g. ValerianFourel/sgt-bavaria-soc-2002-2023-large  under a  20m/  prefix
#   (or a sibling dataset -large-20m). Same dtype/grid/NoData convention.

# 5) Train / map at sentinel resolution:
export SGT_BANDS_S2=1
#   set window_size per the decision table above (config.py + --window-size)
python rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py \
    --bands-list full_extended_s2 --window-size <5|25|63> \
    --max-oc 150 --loss-type l1 --split-axis lon --num-folds 10 \
    --sweep-name sentinel20m_exts2band ...
```

## Still TODO (cluster-side, not codeable/testable from here)

- **Tiling at 20 m**: the cut scripts read `TILE_PX` (now env-driven) but were
  exercised at 979; verify they stream 12238×12238 tiles without OOM (window the
  reads, as `sgt_landsat_mb_cut.py` already does).
- **Coordinate rebuild**: the pixel (i,j) index of every sample/grid point is
  resolution-dependent — must be recomputed against the 20 m tiles.
- **Storage/throughput**: size the Drive + workspace for ~156× tiles (or window
  on extraction to avoid materializing full 20 m tiles).
- **Memory note**: update `project_hf_large_superset` / `project_band_*` once the
  20 m variant path on HF is fixed.

This branch makes the export + tiling *capable* of 20 m and documents the rest;
the heavy regeneration runs where the data and EE auth live.
