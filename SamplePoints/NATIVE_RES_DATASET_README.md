---
license: cc-by-4.0
pretty_name: "Bavaria SOC 2002–2023 — native multi-resolution (20 m / 30 m / 250 m)"
tags:
  - soil-organic-carbon
  - digital-soil-mapping
  - sentinel-2
  - multi-resolution
  - bavaria
---

# sgt-bavaria-soc-2002-2023-native

The **native multi-resolution** companion to
[`ValerianFourel/sgt-bavaria-soc-2002-2023-large`](https://huggingface.co/datasets/ValerianFourel/sgt-bavaria-soc-2002-2023-large).
Same Bavaria SOC stack, 2002–2023, but each band is kept at its **true native
resolution** instead of all being resampled to 250 m — for the resolution-aware
multi-branch network (`--model-family resaware`).

## Layout — like `-large`, but split by resolution (not unified)

Each `res_*/` root uses the **same `.npy`-tile + Coordinates layout as `-large`**,
so existing loader code works per group:

```
res_20m/    2 bands  — S2SRC_SWIR1, S2SRC_SWIR2 (Sentinel-2 B11/B12, bare-soil, 20 m)
res_30m/    20 bands — 11 Landsat SRC + 9 SRTM/terrain (Elevation, Slope, Aspect,
                       TWI, TPI_90/300/1000, TRI, Roughness), 30 m
res_250m/   23 bands — MODIS (LAI/NPP/ET/LST/NDVI/EVI/phenology), ERA5 climate,
                       SoilGrids — kept at 250 m (identical to -large)

  each res_*/:
    RasterTensorData/{YearlyValue,StaticValue}/<band>/<tile>.npy
    Coordinates1Mil/<tier>/<band>/...
    OC_LUCAS_LFU_LfL_Coordinates_v2/<tier>/<band>/...
```

The three groups map 1:1 onto the `resaware` net's branches: `res_20m` + `res_30m`
→ **fine** (real spatial patch), `res_250m` → **medium/coarse** (centre value).
Resolutions are NOT unified, by design — the network processes each at its own
scale and fuses late.

## How it was built (and cleaned)

1. **Export** at native scale (EE): S2 SWIR @20 m, Landsat SRC + terrain @30 m
   (`sgt_s2_swir_export.py --scale 20`, `gee_download_all_bands.py --category
   landsat --scale 30` / `--category topo --scale 30`). Coarse bands reuse the
   existing 250 m exports.
2. **Pull + kill all-NaN** with `SamplePoints/pull_verify_native.py` — every TIFF
   is opened and any **all-NaN** file is deleted (logged to `_killed_nan.log`);
   valid sparse bands (e.g. bare-soil composites are ~40–60% valid) are kept.
3. **Tile** at native `TILE_PX` (`SGT_TILE_PX≈8158` for 30 m, `12238` for 20 m)
   into the `res_*/RasterTensorData/...` layout; rebuild coordinate indices.
4. **Publish** with `scripts/hf_publish_native.py` (resumable `upload_large_folder`).

## Use it (for the new branch)

```bash
huggingface-cli download ValerianFourel/sgt-bavaria-soc-2002-2023-native \
    --repo-type dataset --local-dir Data_native
# resolution-aware net (branch bestrun-bands-s2swir):
python rebuttal/gpu_experiments/spatial_kfold/submit_resaware_compare.py \
    --bands-list full_extended_s2
```

Model code: `github.com/ValerianFourel/SOCmapping` (branch `bestrun-bands-s2swir`).
