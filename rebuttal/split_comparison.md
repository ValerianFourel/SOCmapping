# Spatial vs random validation split — descriptive comparison

_Spatial split source:_ `/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/train_val_data_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_normalize_LOSS_composite_l2.parquet`

_Random split:_ synthetic, drawn here with seed = `20260513`. Same size (1359 rows) sampled without replacement from the full pool of 16514 rows in the same parquet.

> **Caveat.** Every train/val parquet I found in the repository (`residualModels1mil_normalize_*`, `_log_*`, `_normalize_mse`) uses the same `create_validation_train_sets(distance_threshold=1.2)` spatial-split logic. There is no surviving parquet from a purely random split. The "random" column below is therefore a synthetic reference draw — it shows what a same-size random validation set looks like on this dataset, which is the closest you can get to an apples-to-apples random-vs-spatial comparison without retraining.

## Side-by-side descriptive statistics

| metric | spatial val | random val | Δ (random − spatial) |
|--------|-------------|------------|----------------------|
| n | 1359 | 1359 | — |
| mean (g/kg) | 16.337 | 22.903 | +6.567 |
| sd (g/kg) | 7.474 | 20.290 | +12.815 |
| median (g/kg) | 14.400 | 16.600 | +2.200 |
| q25 (g/kg) | 12.400 | 12.625 | +0.225 |
| q75 (g/kg) | 17.400 | 24.050 | +6.650 |
| IQR (g/kg) | 5.000 | 11.425 | +6.425 |
| min (g/kg) | 4.460 | 3.700 | -0.760 |
| max (g/kg) | 85.000 | 147.000 | +62.000 |
| n > 50 g/kg | 8 | 107 | — |
| % > 50 g/kg | 0.59% | 7.87% | +7.285 |
| n > 120 g/kg | 0 | 12 | — |
| % > 120 g/kg | 0.00% | 0.88% | +0.883 |

## Geographic extent

| split | lat range | lon range | unique locations |
|-------|-----------|-----------|------------------|
| spatial_val | [47.3412, 50.4876] | [9.6002, 13.8337] | 513 |
| random_val | [47.2409, 50.5842] | [9.5407, 13.8768] | 1088 |

## Per-year sample count in each validation set

| year | spatial val n | random val n |
|------|---------------|--------------|
| 2007 | 32 | 116 |
| 2008 | 11 | 58 |
| 2009 | 94 | 63 |
| 2010 | 9 | 15 |
| 2011 | 246 | 176 |
| 2012 | 285 | 250 |
| 2013 | 192 | 220 |
| 2014 | 113 | 67 |
| 2015 | 203 | 113 |
| 2016 | 12 | 9 |
| 2017 | 19 | 111 |
| 2018 | 34 | 93 |
| 2019 | 12 | 16 |
| 2020 | 47 | 15 |
| 2021 | 40 | 19 |
| 2022 | 9 | 13 |
| 2023 | 1 | 5 |
