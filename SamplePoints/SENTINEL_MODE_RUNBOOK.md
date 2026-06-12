# Sentinel mode — regenerate the WHOLE stack at 20 m

Decision (confirmed): **all bands scaled to 20 m**, not just the new S2 SWIR
pair. This is a full dataset regeneration. Branch: `bestrun-bands-s2swir`.

The current dataset is a **250 m** grid (979×979 tiles, `TILE_DEG=1.7986°`,
12 tiles). Sentinel mode rebuilds the same tile *layout* at **20 m** pixels.

## ⚠️ Scale reality (read first)

20 m is **12.5× finer per axis ⇒ ~156× more pixels per band per tile** than
250 m. Two things this hits:

- **Stored tiles** (`tiff_to_tiles.py` writes full 979×979 tiles today): at
  `TILE_PX=12238` each tile is ~156× larger. If the dataset stores full tiles,
  the ~94 GB `-large` becomes **multi-TB**.
- **Per-point windows** (what the model actually consumes): a `window_size=5`
  window is `5×5` pixels = **100 m** at 20 m (vs **1.25 km** at 250 m). Same
  storage per window, but a much smaller physical footprint. To keep the 1.25 km
  context you'd raise `window_size` to ~63 (`63×20 m ≈ 1.26 km`) — which grows
  the windowed data ~156× too.

**So the single biggest decision is `window_size` / physical extent** (next
section). Confirm it before committing storage.

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

## window_size decision (do this first)

| Goal | window_size @ 20 m | physical window | windowed storage vs now |
|---|---|---|---|
| Same pixel count, finer detail | 5 | 100 m | ~1× (cheapest) |
| Same physical context as 250 m | 63 | 1.26 km | ~156× |
| Compromise | 25 | 500 m | ~25× |

Set it in the model `config.py` (`window_size`) and the spatial-CV
`--window-size`. The tiles must be cut at `SGT_TILE_PX=12238` regardless; only
the *window* extent differs.

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
