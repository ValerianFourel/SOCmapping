# Full proper-R² leaderboard — every model in `ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping`

_Source roots:_ 6 local directories incl. fresh HF snapshot at `_hf_full_weights_repo/`. Scan found 30 `analysis_results.pkl` files; 60 (split × file) entries; **26 unique (split × identity)** rows after deduping mirror copies.

## Metric glossary

| metric | formula | interpretation |
|---|---|---|
| **`r2` (proper R²)** | `1 − Σ(a−p)² / Σ(a−ā)²` | 0 = mean-predictor; **negative = worse than predicting the mean** |
| `pearson_r2` | `corr(p,a)²` | What v1 paper called "R²"; bias-invariant |
| `rmse` | `√mean((p−a)²)` | absolute error in OC units (g/kg) |
| `mae` | `mean(|p−a|)` | robust absolute error (g/kg) |
| `bias` | `mean(p−a)` | systematic offset (g/kg) |
| `sd_y` / `iqr_y` | `SD / IQR of actual y` | spread of the target on this split |
| **`rmse_over_sd_y`** | `RMSE / SD(y)` | `< 1` = useful; `= 1` = mean-predictor; `> 1` = worse |
| **`rpd`** | `SD(y) / RMSE` | calibrated; `1` = mean-predictor; **`≥ 2` = good (DSM convention)** |
| `rpiq` | `IQR(y) / RMSE` | v1 metric; has non-zero floor at `rpiq_mean_predictor` |
| `rpiq_mean_predictor` | `IQR(y) / SD(y)` | what `rpiq` would be for a constant-mean predictor — the actual "zero-signal" floor |
| `is_useful` | `r2 > 0` | quick boolean — is this model better than predicting the mean? |
| `is_leakage` | model trained on full data, "val" is in-pool | `_R2_1.0000` in saved checkpoint name |

## Val-set leaderboard (deduplicated, ranked by proper R²)

| # | Architecture | Size | Transform | Loss | n_val | Pearson r² | **R² (proper)** | RMSE | bias | SD(y) | RMSE/SD | RPD | RPIQ | useful? | leak? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TFT/SGT | 1mil | normalize | composite_l2 (≡MSE) | 1359 | 0.7799 | **+0.7571** | 3.68 | -1.13 | 7.47 | 0.49 | 2.03 | 1.36 | ✅ | ⚠️ leak |
| 2 | TFT/SGT | 1mil | normalize | composite_l2 (≡MSE) | 1376 | 0.6833 | **+0.6728** | 4.80 | +0.36 | 8.39 | 0.57 | 1.75 | 1.06 | ✅ |  |
| 3 | TFT/SGT | 1mil | log | L1 | 1378 | 0.6836 | **+0.6441** | 4.35 | -1.12 | 7.29 | 0.60 | 1.68 | 1.13 | ✅ | ⚠️ leak |
| 4 | TFT/SGT | 1mil | normalize | composite_l2 (≡MSE) | 1359 | 0.6258 | **+0.5944** | 4.76 | +1.13 | 7.47 | 0.64 | 1.57 | 1.05 | ✅ |  |
| 5 | TFT/SGT | 1mil | normalize | L1 | 1626 | 0.6145 | **+0.5196** | 5.27 | +1.33 | 7.60 | 0.69 | 1.44 | 1.03 | ✅ |  |
| 6 | TFT/SGT | 360k | normalize | L1 | 1361 | 0.5681 | **+0.4143** | 6.09 | +1.24 | 7.96 | 0.77 | 1.31 | 0.90 | ✅ |  |
| 7 | TFT/SGT | 360k | none | composite_l2 (≡MSE) | 1473 | 0.5057 | **+0.2013** | 6.92 | +2.32 | 7.75 | 0.89 | 1.12 | 0.72 | ✅ |  |
| 8 | TFT/SGT | 1mil | log | L1 | 1378 | 0.4620 | **+0.0628** | 7.05 | +0.48 | 7.29 | 0.97 | 1.03 | 0.69 | ✅ |  |
| 9 | 3DCNN |  | none | MSE | 1661 | 0.1897 | **-0.2584** | 8.73 | +1.86 | 7.79 | 1.12 | 0.89 | 0.58 | ❌ |  |
| 10 | SimpleTransformer | 2mil | none | composite_l2 (≡MSE) | 1361 | 0.4101 | **-0.4296** | 9.55 | +4.05 | 7.99 | 1.20 | 0.84 | 0.53 | ❌ |  |
| 11 | SimpleTransformer | 2mil | none | composite_l2 (≡MSE) | 1566 | 0.4534 | **-0.4475** | 9.21 | +4.67 | 7.66 | 1.20 | 0.83 | 0.50 | ❌ |  |
| 12 | CNN-LSTM |  | none | composite_l2 (≡MSE) | 1377 | 0.3062 | **-1.6215** | 11.71 | +4.62 | 7.24 | 1.62 | 0.62 | 0.47 | ❌ |  |
| 13 | SimpleTransformer | 20k | none | composite_l2 (≡MSE) | 1325 | 0.2509 | **-1.7939** | 12.58 | +4.51 | 7.53 | 1.67 | 0.60 | 0.42 | ❌ |  |

## Train-set rows (for completeness)

| # | Architecture | Size | Transform | Loss | n_train | Pearson r² | R² (proper) | RMSE | bias | RPD | RPIQ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TFT/SGT | 1mil | normalize | composite_l2 (≡MSE) | 15155 | 0.8659 | +0.8623 | 7.54 | -0.31 | 2.70 | 1.55 |
| 2 | TFT/SGT | 1mil | normalize | composite_l2 (≡MSE) | 15155 | 0.8622 | +0.8560 | 7.71 | -1.52 | 2.64 | 1.52 |
| 3 | TFT/SGT | 1mil | normalize | composite_l2 (≡MSE) | 15138 | 0.8524 | +0.8474 | 7.93 | -1.21 | 2.56 | 1.46 |
| 4 | TFT/SGT | 360k | none | composite_l2 (≡MSE) | 15041 | 0.8411 | +0.8399 | 8.15 | +0.41 | 2.50 | 1.44 |
| 5 | TFT/SGT | 1mil | normalize | L1 | 14888 | 0.8244 | +0.8181 | 8.72 | -0.69 | 2.34 | 1.34 |
| 6 | TFT/SGT | 360k | normalize | L1 | 15153 | 0.8092 | +0.8075 | 8.91 | -0.82 | 2.28 | 1.31 |
| 7 | SimpleTransformer | 2mil | none | composite_l2 (≡MSE) | 15153 | 0.8266 | +0.8006 | 9.07 | +1.59 | 2.24 | 1.28 |
| 8 | SimpleTransformer | 2mil | none | composite_l2 (≡MSE) | 14948 | 0.8267 | +0.7919 | 9.31 | +2.43 | 2.19 | 1.27 |
| 9 | TFT/SGT | 1mil | log | L1 | 15136 | 0.7605 | +0.7475 | 10.22 | -1.73 | 1.99 | 1.15 |
| 10 | TFT/SGT | 1mil | log | L1 | 15136 | 0.7777 | +0.7318 | 10.53 | -3.08 | 1.93 | 1.11 |
| 11 | SimpleTransformer | 20k | none | composite_l2 (≡MSE) | 15189 | 0.6128 | +0.6121 | 12.65 | -0.29 | 1.61 | 0.93 |
| 12 | CNN-LSTM |  | none | composite_l2 (≡MSE) | 15137 | 0.5828 | +0.5824 | 13.14 | -0.41 | 1.55 | 0.89 |
| 13 | 3DCNN |  | none | MSE | 14853 | 0.2513 | +0.2355 | 17.89 | -2.53 | 1.14 | 0.66 |

## Notes

- **`leak`** rows are re-evaluations of an "AllDataFit" / "fullDatasetRun" model (trained with `use_validation=False`) on rows that were inside its training pool. They look better than the corresponding parent file but **must not be cited as held-out R²**.
- **`useful`** is just `r2 > 0`. A model worse than predicting the mean has negative R² and `useful = ❌`. Such models also have `RPD < 1` and `RMSE/SD > 1`.
- **`rpiq_mean_predictor`** typically falls in 0.6–0.8 on Bavarian SOC val sets; **RPIQ values ≤ ~0.8 are at or below the mean-predictor floor.** RPIQ ≥ 2 = "good" by DSM convention.
