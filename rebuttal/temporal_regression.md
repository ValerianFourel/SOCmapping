# Temporal regression — SOC ~ year + altitude (T2.2 / T2.3)

_Master sample file:_ `/home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx`

_Altitude join:_ `/home/valerian/SGTPublication/Data/OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation/coordinates.npy` × tiles in `/home/valerian/SGTPublication/Data/RasterTensorData/StaticValue/Elevation`

After dropping rows with NaN coordinates/year/OC and applying the paper’s `MAX_OC ≤ 150` cap (and dropping any rows still missing elevation), the regression dataframe has **26,344** observations spanning **2000–2023**.

## Per-year sample counts

| year | n | included in n≥100 OLS? |
|------|---|------------------------|
| 2000 | 161 | ✔ |
| 2001 | 2264 | ✔ |
| 2002 | 2763 | ✔ |
| 2003 | 2936 | ✔ |
| 2004 | 940 | ✔ |
| 2005 | 685 | ✔ |
| 2006 | 81 | — |
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

## Coefficient table — four models

| model | n | R² | adj. R² | β_year | SE_year | p_year | 95% CI on β_year | β_altitude | p_altitude |
|-------|---|----|---------|--------|---------|--------|------------------|------------|------------|
| OLS  SOC ~ year + altitude  (all years) | 26344 | 0.2141 | 0.2141 | +0.2820 | 0.0175 | 2.11e-58 | [+0.2478, +0.3162] | +0.05659 | 0.00e+00 |
| OLS  SOC ~ year + altitude  (year < 2022) | 26165 | 0.1981 | 0.1980 | +0.2281 | 0.0175 | 8.29e-39 | [+0.1938, +0.2623] | +0.05491 | 0.00e+00 |
| WLS  SOC ~ year + altitude  (weights = 1 / n_year) | 26344 | 0.3074 | 0.3073 | +0.8232 | 0.0220 | 1.28e-298 | [+0.7801, +0.8663] | +0.06516 | 0.00e+00 |
| OLS  SOC ~ year + altitude  (years with n ≥ 100) | 26208 | 0.2076 | 0.2075 | +0.2631 | 0.0173 | 8.54e-52 | [+0.2291, +0.2970] | +0.05547 | 0.00e+00 |

Interpretation:

- β_year is the change in SOC (g/kg) per additional calendar year, holding altitude constant.
- β_altitude is the change in SOC (g/kg) per 1 m of elevation.

### `OLS  SOC ~ year + altitude  (all years)`

- n = 26344, R² = 0.2141, adj. R² = 0.2141, F p-value = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -571.204259 | 35.044097 | -16.300 | 1.94e-59 | -639.892583 | -502.515934 |
| `year_int` | +0.282004 | 0.017460 | +16.152 | 2.11e-58 | +0.247782 | +0.316226 |
| `elevation` | +0.056588 | 0.000698 | +81.050 | 0.00e+00 | +0.055219 | +0.057956 |

### `OLS  SOC ~ year + altitude  (year < 2022)`

- n = 26165, R² = 0.1981, adj. R² = 0.1980, F p-value = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -462.181434 | 35.082271 | -13.174 | 1.65e-39 | -530.944603 | -393.418264 |
| `year_int` | +0.228054 | 0.017474 | +13.051 | 8.29e-39 | +0.193804 | +0.262304 |
| `elevation` | +0.054911 | 0.000702 | +78.215 | 0.00e+00 | +0.053535 | +0.056287 |

### `WLS  SOC ~ year + altitude  (weights = 1 / n_year)`

- n = 26344, R² = 0.3074, adj. R² = 0.3073, F p-value = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -1657.698372 | 44.160427 | -37.538 | 1.81e-300 | -1744.255195 | -1571.141548 |
| `year_int` | +0.823222 | 0.022000 | +37.418 | 1.28e-298 | +0.780100 | +0.866344 |
| `elevation` | +0.065157 | 0.000733 | +88.865 | 0.00e+00 | +0.063720 | +0.066594 |

### `OLS  SOC ~ year + altitude  (years with n ≥ 100)`

- n = 26208, R² = 0.2076, adj. R² = 0.2075, F p-value = 0.00e+00

| term | coef | std err | t | p | 95% CI lo | 95% CI hi |
|------|------|---------|---|---|-----------|-----------|
| `const` | -532.732636 | 34.794950 | -15.311 | 1.10e-52 | -600.932635 | -464.532636 |
| `year_int` | +0.263053 | 0.017334 | +15.175 | 8.54e-52 | +0.229078 | +0.297029 |
| `elevation` | +0.055470 | 0.000696 | +79.681 | 0.00e+00 | +0.054105 | +0.056834 |
