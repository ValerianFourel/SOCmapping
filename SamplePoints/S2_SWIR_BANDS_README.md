# Tier 4 — Sentinel-2 SWIR bare-soil bands (B11/B12)

Adds the two Sentinel-2 SWIR bands — the Zepp/Broeg/Tziolas exposed-soil SOC
signal — to the SGT covariate stack, taking the extended stack from **43 → 45
bands** (`--bands-list full_extended_s2`). Branch: `bestrun-bands-s2swir`.

| Band | Sentinel-2 | wavelength | native res |
|---|---|---|---|
| `S2SRC_SWIR1` | B11 | ~1.61 µm | 20 m |
| `S2SRC_SWIR2` | B12 | ~2.19 µm | 20 m |
| `S2SRC_ExposureCount` | — | count of valid bare-soil obs (validity mask) | — |

## Why a STATIC multi-year composite (the key decision)

Sentinel-2 surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`) only begins
**~2017**, but the SGT dataset spans **2002–2023** with 5-year windows. A
per-year S2 stack would be NoData for 2002–2016, breaking the input cube for most
of the record. So the S2 SWIR is built as **one bare-soil composite over the
whole S2 archive (default 2017–2023)** and stored as a `StaticValue` band — the
same value for every year window, exactly like `Elevation`. This matches the
exposed-soil-mosaic methodology of the cited papers (a single multi-year
bare-soil composite of the highest-SOC-signal bands), and it gives full spatial
coverage with no temporal gap. The bare-soil mask reuses the Landsat SRC
thresholds (NDVI ∈ [0.15, 0.25], NBR2 < 0.075) for cross-sensor consistency.

## "Sentinel mode" — resolution (the second decision)

`sgt_s2_swir_export.py --scale` controls the export pixel size:

- **`--scale 20` (default, sentinel mode):** export at B11/B12 native **20 m**.
  Use this for a high-resolution Sentinel variant where the input bands are at
  sentinel size. NOTE: the existing stack is on a **250 m** grid (979×979 tiles),
  so a 20 m S2 band does **not** align with it — a full sentinel-resolution
  dataset means re-gridding **every** band to 20 m (a large EE re-export +
  retile, see "Open decision" below).
- **`--scale 250` (drop-in):** area-mean the 20 m composite onto the existing
  250 m project grid so the two S2 bands stack pixel-for-pixel with the current
  43 bands. This is the fast path to `full_extended_s2` on today's data.

## Safety: opt-in, 43-band runs unchanged

The S2 bands are **off by default**. The per-model `config.py` only appends them
to `bands_list_order` (43 → 45) when **`SGT_BANDS_S2=1`** AND the S2 rasters
exist on disk. `band_subsets.get_band_indices('full_extended')` is pinned to its
43 named bands (no longer "all channels"), so existing 43-band runs/checkpoints
are byte-identical. `full_extended_s2` (45) raises a clear error if the S2 bands
aren't present, telling you to set the flag.

## Pipeline (run on the cluster / EE-authenticated env)

```bash
# 1) Export the bare-soil S2 SWIR composite to Drive (EE project sgtmodel).
#    Default = 2017-2023, 20 m, bare-soil. Use --scale 250 for the drop-in grid.
python SOCmapping/SamplePoints/sgt_s2_swir_export.py --scale 250        # drop-in
#   or:  --scale 20   for the sentinel-resolution variant
python SOCmapping/SamplePoints/sgt_s2_swir_export.py --dry-run          # preview bands

# 2) Tile the Drive GeoTIFF into the project RasterTensorData layout, mirroring
#    the Landsat multiband cut (one band per StaticValue dir):
#      RasterTensorData/StaticValue/S2SRC_SWIR1/<tile>.npy
#      RasterTensorData/StaticValue/S2SRC_SWIR2/<tile>.npy
#      + matching OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/<band>/ and
#        Coordinates1Mil/StaticValue/<band>/ index files.
#    Use sgt_landsat_mb_cut.py as the template (it already windows a multiband
#    GeoTIFF into the 12 tiles); point it at s2_swir_baresoil_*.tif and the
#    StaticValue tier. (Clean: NoData -> NaN, float32, same 979x979 tile grid.)

# 3) Standardize + publish to the unified -large dataset:
#      ValerianFourel/sgt-bavaria-soc-2002-2023-large
#    Add the StaticValue/S2SRC_* dirs (rasters + coord indices) alongside the
#    existing RasterTensorData/Coordinates so a fresh pull on any machine gets
#    the S2 bands. Same dtype/grid/NoData convention as the other StaticValue
#    bands (Elevation) so the dataloader treats them identically.

# 4) Train / map with the 45-band stack:
export SGT_BANDS_S2=1
python rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py \
    --bands-list full_extended_s2 --max-oc 150 --loss-type l1 \
    --split-axis lon --num-folds 10 --sweep-name exts2band ...
# run-name auto-suffix: _exts2band
```

## Standardization for `sgt-bavaria-soc-2002-2023-large`

The S2 SWIR bands must match the dataset's existing StaticValue convention so a
pull is drop-in:
- **Grid:** the 250 m, 979×979, 12-tile Bavaria layout (use `--scale 250`, or
  area-mean during tiling). For the 20 m sentinel variant, a parallel
  `*_20m` tile set is required — keep it under a separate path so the 250 m
  stack is untouched.
- **dtype / NoData:** `float32`, NoData → `NaN` (the dataloader's normalizer
  clamps; `S2SRC_ExposureCount == 0` marks pixels with no bare-soil observation).
- **Layout:** `RasterTensorData/StaticValue/S2SRC_SWIR1|S2SRC_SWIR2/` +
  `Coordinates1Mil/StaticValue/<band>/` + `OC_.../StaticValue/<band>/`, indexed
  so the appended channels line up with `bands_list_order` index-for-index
  (`_bands.build_yearly_paths` already produces these paths for static bands).

## Open decision for the maintainer

The drop-in (`--scale 250`) path is implemented end-to-end and integrates into
the current 43-band stack as `full_extended_s2` immediately. A **full
sentinel-resolution dataset** ("sample everything at sentinel size") is a much
larger job — re-exporting and re-tiling all 43+ bands at 20 m, ~12× more pixels,
new tile geometry, and a separate `-large` variant. If that's the goal rather
than adding the 2 S2 SWIR bands at the current resolution, say so and the export
scale + tiling can be generalized to a sentinel-grid mode for the whole stack.

## Files changed on this branch

- `_bands.py` — `TIER4_S2SWIR_BANDS`, `FULL_EXTENDED_S2_BANDS`, S2 bands in `STATIC_BANDS`.
- `rebuttal/gpu_experiments/spatial_kfold/band_subsets.py` — pin `full_extended`
  to 43; add `full_extended_s2` (45) + `_exts2band` suffix.
- `SpatiotemporalGatedTransformer/config.py`, `balancedDataset/config.py` —
  env-gated (`SGT_BANDS_S2=1`) append of the S2 bands (43 → 45).
- `run_kfold.py`, `train_full.py`, `train_full_baselines.py`, `sweep_submit.py`
  — add `full_extended_s2` to `--bands-list`.
- `SamplePoints/sgt_s2_swir_export.py` — the EE export driver.
