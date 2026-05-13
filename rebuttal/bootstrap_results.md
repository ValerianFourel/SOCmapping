# Bootstrap CIs — TFT 1 mil, composite_l2_v2, run 1

_Source parquet:_ `/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/train_val_data_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_normalize_LOSS_composite_l2.parquet`

_Predictions source:_ `/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/analysis_results.pkl`

## Column layout

| column | dtype | notes |
|--------|-------|-------|
| `GPS_LONG` | float64 | geographic coordinate |
| `GPS_LAT` | float64 | geographic coordinate |
| `year` | int64 | sample year |
| `OC` | float64 | actual SOC concentration (g/kg) — the regression target |
| `survey_date` | datetime64[us] |  |
| `season` | object |  |
| `bin` | int64 |  |
| `dataset_type` | object | split indicator (`train` / `val`) |

The parquet has **no** `predicted` column — it is the train/val split snapshot stored at training time. Predicted vs. actual pairs for the same model live in `analysis_results.pkl` (residualsStudy output), and are loaded from there for the bootstrap.

## Descriptive statistics by split

| metric | train | val |
|--------|-------|-----|
| `n` | 15155 | 1359 |
| `mean` | 23.082 | 16.337 |
| `sd` | 20.317 | 7.474 |
| `median` | 16.800 | 14.400 |
| `q25` | 12.600 | 12.400 |
| `q75` | 24.300 | 17.400 |
| `iqr` | 11.700 | 5.000 |
| `min` | 0.839 | 4.460 |
| `max` | 150.000 | 85.000 |
| `n_gt_50` | 1166 | 8 |
| `pct_gt_50` | 7.69% | 0.59% |
| `n_gt_120` | 154 | 0 |
| `pct_gt_120` | 1.02% | 0.00% |

### Cross-check — same stats on the pickled targets (predictions file)

| metric | train (pickle) | val (pickle) |
|--------|----------------|--------------|
| `n` | 15155 | 1359 |
| `mean` | 23.082 | 16.337 |
| `sd` | 20.317 | 7.474 |
| `median` | 16.800 | 14.400 |
| `q25` | 12.600 | 12.400 |
| `q75` | 24.300 | 17.400 |
| `iqr` | 11.700 | 5.000 |
| `min` | 0.839 | 4.460 |
| `max` | 150.000 | 85.000 |
| `n_gt_50` | 1166 | 8 |
| `pct_gt_50` | 7.69% | 0.59% |
| `n_gt_120` | 154 | 0 |
| `pct_gt_120` | 1.02% | 0.00% |

(Pickle counts can differ from parquet counts by a handful: the parquet is the post-split snapshot at training time, the pickle records the samples the residualsStudy could actually re-encode at inference.)

## Point estimates

| metric | train | val |
|--------|-------|-----|
| `n` | 15155 | 1359 |
| `r2` | 0.8659 | 0.6258 |
| `rmse` | 7.5378 | 4.7583 |
| `mae` | 3.3695 | 2.7912 |
| `rpiq` | 1.5522 | 1.0508 |
| `bias_pred_minus_actual` | -0.3137 | 1.1287 |

## Bootstrap 95% CIs (n_boot=1000, seed=20260513)

Resampling with replacement on the **validation set** (1359 samples).

| metric | mean | SD | median | 95% CI lo | 95% CI hi |
|--------|------|----|--------|-----------|-----------|
| `r2` | 0.6245 | 0.0471 | 0.6269 | 0.5257 | 0.7115 |
| `rmse` | 4.7497 | 0.2999 | 4.7388 | 4.1767 | 5.3348 |
| `mae` | 2.7913 | 0.1052 | 2.7874 | 2.5996 | 2.9989 |
| `rpiq` | 1.0414 | 0.0760 | 1.0381 | 0.9019 | 1.1884 |

### Training-set bootstrap (for completeness, 15155 samples)

| metric | mean | SD | median | 95% CI lo | 95% CI hi |
|--------|------|----|--------|-----------|-----------|
| `r2` | 0.8659 | 0.0066 | 0.8660 | 0.8525 | 0.8780 |
| `rmse` | 7.5420 | 0.1868 | 7.5362 | 7.1866 | 7.9281 |
| `mae` | 3.3713 | 0.0540 | 3.3691 | 3.2679 | 3.4783 |
| `rpiq` | 1.5515 | 0.0395 | 1.5515 | 1.4732 | 1.6247 |
