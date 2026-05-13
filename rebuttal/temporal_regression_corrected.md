# Temporal regression — CORRECTED (Task A)

## Diagnosis

The earlier `temporal_regression.py` regressed SOC on year + altitude using the full LUCAS xlsx (30,451 rows; 2000–2023). After the paper’s `OC ≤ 150` cap that left 26,344 observations spanning 2000–2023 — a wider time window than the model was trained on. The model-ready dataset that the paper actually reports on is the post-`filter_dataframe` **2007–2023, OC ≤ 150, non-null GPS** subset: **16,514 rows**, which is also exactly what is stored in every `train_val_data_*.parquet`.

| step | rows |
|------|------|
| LUCAS xlsx raw | 30,451 |
| after filter (year∈[2007,2023], OC≤150, non-null GPS+OC) | 16514 |
| after altitude merge (no losses) | 16514 |

## Result on the corrected base

**OLS SOC ~ year + altitude** on the 16,514-row 2007–2023 sample:

- β_year = **+0.7519** (SE 0.0397, 95% CI [+0.6741, +0.8296], p = 2.47e-79)
- R² = **0.2463**, adj R² = 0.2462, n = 16514

_Paper-reported values for cross-check:_ β_year = +0.751, R² = 0.246.

**β_year matches the paper to within 0.05 g/kg/yr** (corrected = +0.7519, paper = +0.751). The earlier discrepancy was solely due to the time window: the regression in `temporal_regression.py` had been run on 2000–2023 (n = 26,344) instead of the model-ready 2007–2023 subset.

## All four T2.2 / T2.3 models on the corrected base

| model | n | R² | β_year | SE | p_year | 95% CI on β_year | β_altitude | p_altitude |
|-------|---|----|--------|----|--------|------------------|------------|------------|
| OLS  SOC ~ year + altitude  (all 16,514 samples, 2007–2023) | 16514 | 0.2463 | +0.7519 | 0.0397 | 2.47e-79 | [+0.6741, +0.8296] | +0.05840 | 0.00e+00 |
| OLS  SOC ~ year + altitude  (year < 2022) | 16335 | 0.2232 | +0.6086 | 0.0404 | 5.48e-51 | [+0.5295, +0.6878] | +0.05694 | 0.00e+00 |
| WLS  SOC ~ year + altitude  (weights = 1 / n_year) | 16514 | 0.3411 | +1.3969 | 0.0422 | 9.01e-233 | [+1.3142, +1.4796] | +0.06463 | 0.00e+00 |
| OLS  SOC ~ year + altitude  (years with n ≥ 100) | 16459 | 0.2352 | +0.6823 | 0.0398 | 3.57e-65 | [+0.6042, +0.7604] | +0.05760 | 0.00e+00 |

## Per-year sample count in the 16,514-row model-ready set

| year | n | in n≥100 OLS? |
|------|---|---------------|
| 2007 | 1425 | ✔ |
| 2008 | 755 | ✔ |
| 2009 | 632 | ✔ |
| 2010 | 125 | ✔ |
| 2011 | 2230 | ✔ |
| 2012 | 3034 | ✔ |
| 2013 | 2697 | ✔ |
| 2014 | 873 | ✔ |
| 2015 | 1564 | ✔ |
| 2016 | 136 | ✔ |
| 2017 | 1286 | ✔ |
| 2018 | 1043 | ✔ |
| 2019 | 171 | ✔ |
| 2020 | 151 | ✔ |
| 2021 | 213 | ✔ |
| 2022 | 124 | ✔ |
| 2023 | 55 | — |

### `OLS  SOC ~ year + altitude  (all 16,514 samples, 2007–2023)`

- n = 16514, R² = 0.2463, adj. R² = 0.2462, F p = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -1518.276232 | 79.745260 | -19.039 | 5.81e-80 | -1674.585527 | -1361.966936 |
| `year` | +0.751852 | 0.039651 | +18.962 | 2.47e-79 | +0.674131 | +0.829573 |
| `elevation` | +0.058398 | 0.000874 | +66.834 | 0.00e+00 | +0.056686 | +0.060111 |

### `OLS  SOC ~ year + altitude  (year < 2022)`

- n = 16335, R² = 0.2232, adj. R² = 0.2231, F p = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -1229.457545 | 81.233407 | -15.135 | 2.13e-51 | -1388.683897 | -1070.231193 |
| `year` | +0.608612 | 0.040381 | +15.072 | 5.48e-51 | +0.529460 | +0.687764 |
| `elevation` | +0.056943 | 0.000879 | +64.768 | 0.00e+00 | +0.055220 | +0.058667 |

### `WLS  SOC ~ year + altitude  (weights = 1 / n_year)`

- n = 16514, R² = 0.3411, adj. R² = 0.3410, F p = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -2814.544824 | 84.850321 | -33.171 | 1.24e-233 | -2980.860589 | -2648.229060 |
| `year` | +1.396939 | 0.042195 | +33.107 | 9.01e-233 | +1.314233 | +1.479645 |
| `elevation` | +0.064628 | 0.000934 | +69.183 | 0.00e+00 | +0.062797 | +0.066459 |

### `OLS  SOC ~ year + altitude  (years with n ≥ 100)`

- n = 16459, R² = 0.2352, adj. R² = 0.2351, F p = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -1377.981053 | 80.137559 | -17.195 | 1.08e-65 | -1535.059336 | -1220.902771 |
| `year` | +0.682295 | 0.039843 | +17.125 | 3.57e-65 | +0.604199 | +0.760392 |
| `elevation` | +0.057601 | 0.000875 | +65.795 | 0.00e+00 | +0.055885 | +0.059316 |
