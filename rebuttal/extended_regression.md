# Extended regression with land-use — T2.9

## Source of the land-use label

No ESA WorldCover or `land_use`/`landcover`/`lulc` column is attached to the 16,514-row model-ready dataset itself. Per-sample land-cover labels are available, however, in the LUCAS Topsoil 2015 and LUCAS Soil 2018 surveys, both of which sit under `Data/Preprocessing/SoilSamples_From_Raw_to_MLready/LucasBodenDaten/`. Each LUCAS record carries an `LC1` (2nd-level) code plus an `LC0_Desc` 8-class group (Artificial land / Cropland / Woodland / Shrubland / Grassland / Bareland / Water / Wetlands). The 2018 file gives `LC0_Desc` directly; for the 2015 file the LC0 class is derived from the first letter of `LC1` via the standard LUCAS hierarchy mapping.

- LUCAS 2018 DE rows: 779
- LUCAS 2015 DE rows: 1,687
- Combined unique POINTIDs: 1,805

After joining on `POINTID` against the 16,514-row model-ready dataset, **1,239 samples (7.5%)** carry a land-cover label. The remaining 15,275 samples come from the LfL/LfU agricultural soil surveys whose POINTID scheme does not match LUCAS; no separate WorldCover raster is bundled with the repo so they cannot be labelled without external data.

### LC class counts in the matched subset

| LC0_Desc | n |
|----------|---|
| Artificial land | 7 |
| Bareland | 16 |
| Cropland | 678 |
| Grassland | 494 |
| Woodland | 42 |

## Baseline (same LC-labeled subset, no land-use term)

- n = 1237, R² = 0.1131, adj R² = 0.1117

| term | coef | SE | p | 95% CI |
|------|------|----|---|--------|
| `const` | -803.3121 | 346.3246 | 0.0205 | [-1482.7623, -123.8619] |
| `year` | +0.4024 | 0.1721 | 0.0195 | [+0.0647, +0.7401] |
| `elevation` | +0.0457 | 0.0038 | 3.80e-32 | [+0.0383, +0.0530] |

## Extended:  SOC ~ year + altitude + C(LC0_Desc)

- n = 1237, R² = 0.3091, adj R² = 0.3057, AIC = 10408.2

**Year coefficient with land-use controlled:** β_year = +0.3019 (SE 0.1523, 95% CI [+0.0031, +0.6007], p = 0.0476)

| term | coef | SE | t | p | 95% CI |
|------|------|----|---|---|--------|
| `Intercept` | -583.5351 | 306.5616 | -1.903 | 0.0572 | [-1184.9765, +17.9063] |
| `C(LC0_Desc)[T.Bareland]` | -17.2938 | 7.3438 | -2.355 | 0.0187 | [-31.7015, -2.8860] |
| `C(LC0_Desc)[T.Cropland]` | -15.0694 | 6.1584 | -2.447 | 0.0145 | [-27.1515, -2.9874] |
| `C(LC0_Desc)[T.Grassland]` | +2.8829 | 6.1867 | +0.466 | 0.6413 | [-9.2548, +15.0206] |
| `C(LC0_Desc)[T.Woodland]` | +7.6312 | 6.6425 | +1.149 | 0.2508 | [-5.4007, +20.6631] |
| `year` | +0.3019 | 0.1523 | +1.983 | 0.0476 | [+0.0031, +0.6007] |
| `elevation` | +0.0237 | 0.0035 | +6.694 | 3.29e-11 | [+0.0167, +0.0306] |

## Baseline → extended comparison

| metric | baseline (no LU) | extended (+ C(LC0_Desc)) | Δ |
|--------|------------------|--------------------------|---|
| β_year | +0.4024 | +0.3019 | -0.1005 |
| R² | 0.1131 | 0.3091 | +0.1960 |
| adj R² | 0.1117 | 0.3057 | +0.1940 |
