---
license: cc-by-4.0
pretty_name: "Bavaria SOC 2002–2023 — Sentinel 20 m (sentinel mode)"
tags:
  - soil-organic-carbon
  - digital-soil-mapping
  - sentinel-2
  - remote-sensing
  - bavaria
---

# sgt-bavaria-soc-2002-2023-large-sentinel

The **20 m sentinel-resolution** sibling of
[`ValerianFourel/sgt-bavaria-soc-2002-2023-large`](https://huggingface.co/datasets/ValerianFourel/sgt-bavaria-soc-2002-2023-large).
Every band is sampled on the Sentinel-2 **20 m** grid ("sentinel mode") for SGT
soil-organic-carbon mapping over Bavaria, 2002–2023.

## What "sentinel mode" means (and why it isn't 156× bigger)

Each sample is a `time_before`-year stack of `11×11` (or `9×9`) windows at 20 m,
but bands are handled by their native resolution:

- **Fine bands (22)** — Sentinel-2 SWIR (B11/B12, 20 m), Landsat SRC bare-soil
  reflectance/indices (30 m), and SRTM/terrain (Elevation, Slope, Aspect, TWI,
  TPI×3, TRI, Roughness, 30 m): stored as **real 20 m tiles**, so the window
  carries genuine sub-pixel-of-the-old-grid detail.
- **Coarse bands (23)** — MODIS (250 m–1 km), ERA5-Land (~11 km), SoilGrids
  (250 m): there is no sub-window detail at 20 m, so each is the **single nearest
  value at the point, broadcast** across the window. They cost one value per
  point — **no 20 m tiles** — so the dataset is close to the windowed footprint
  of the 250 m version, not multi-TB.

The per-band split is `_bands.is_sentinel_fine()` (22 fine / 23 broadcast = 45
when the S2 SWIR pair is included via `full_extended_s2`).

## Sentinel-2 SWIR (the new bands)

`S2SRC_SWIR1` (B11, ~1.61 µm) and `S2SRC_SWIR2` (B12, ~2.19 µm) — a multi-year
**bare-soil** composite (2017–2023, the exposed-soil SOC signal of
Zepp/Broeg/Tziolas). Because S2 SR only starts ~2017 while the record is
2002–2023, these are stored as a **static** band (same value per year window),
like Elevation. `S2SRC_ExposureCount` is the bare-soil-observation validity mask.

## Layout

Mirrors the `-large` tree (`RasterTensorData/{YearlyValue,StaticValue}/<band>/`,
`Coordinates1Mil/...`, `OC_LUCAS_LFU_LfL_Coordinates_v2/...`) but the FINE bands'
rasters and the coordinate indices are on the 20 m grid (`TILE_PX=12238` per
`TILE_DEG°` tile). Coarse bands keep their native rasters (read as a broadcast
point value). `float32`, NoData → `NaN`.

## Use it

```bash
huggingface-cli download ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel \
    --repo-type dataset --local-dir Data_sentinel_20m
export HF_HOME=$WORK/.cache/huggingface   # keep cache off $HOME on clusters
# train at sentinel resolution:
export SGT_SENTINEL_MODE=1 SGT_WINDOW_SIZE=11 SGT_BANDS_S2=1
python rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py \
    --bands-list full_extended_s2 --window-size 11 --sweep-name sentinel20m ...
```

## Regenerate from source

EE export → tile → standardize → publish, all documented in
`SOCmapping/SamplePoints/SENTINEL_MODE_RUNBOOK.md`. Published with
`SOCmapping/scripts/hf_publish_sentinel.py` (resumable `upload_large_folder`).

Branch: `bestrun-bands-s2swir`. Model code:
`github.com/ValerianFourel/SOCmapping`.
