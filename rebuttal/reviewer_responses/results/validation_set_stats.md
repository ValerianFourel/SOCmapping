# Validation-set descriptive statistics — random vs spatial-CV

Addresses reviewer comments R1.3 (single split weaker than CV) and R3.3 (R²/RMSE paradox).

## Random 91/9 split (fixed seed = 42)

| stat | training (91%) | validation (9%) |
|---|---|---|
| n samples | 15028 | 1486 |
| mean SOC (g/kg) | 22.54 | 22.34 |
| std SOC (g/kg) | 19.79 | 18.42 |
| median SOC | 16.40 | 16.60 |
| IQR | 11.18 | 10.68 |
| max SOC | 150.00 | 148.10 |
| % > 50 g/kg | 7.16 | 6.59 |
| % > 120 g/kg | 0.96 | 0.67 |

## Spatial-CV (10 latitude-decile folds) — aggregated across folds

| stat | training (across folds) | validation (across folds) |
|---|---|---|
| mean n samples per fold | 14863 | 1651 |
| fold-averaged mean SOC | 22.53 | 22.53 |
| fold-averaged SD SOC | 19.59 | 16.49 |
| fold-avg median SOC | 16.47 | 18.17 |
| fold-avg % > 50 g/kg | 7.11 | 7.11 |
| fold-avg % > 120 g/kg | 0.93 | 0.93 |

## Per-fold validation breakdown

| fold | lat range | n val | mean | SD | max | % > 50 | % > 120 |
|---|---|---|---|---|---|---|---|
| 0 | [47.22, 47.97] | 1651 | 48.29 | 30.81 | 150.00 | 35.37 | 4.97 |
| 1 | [47.97, 48.24] | 1651 | 26.34 | 18.79 | 149.60 | 7.21 | 0.79 |
| 2 | [48.24, 48.42] | 1652 | 19.14 | 15.30 | 147.00 | 4.78 | 0.36 |
| 3 | [48.42, 48.65] | 1651 | 16.98 | 15.07 | 146.00 | 3.33 | 0.73 |
| 4 | [48.65, 48.83] | 1652 | 16.97 | 15.12 | 148.10 | 3.69 | 0.30 |
| 5 | [48.83, 49.12] | 1651 | 19.85 | 21.62 | 149.00 | 5.81 | 1.57 |
| 6 | [49.12, 49.44] | 1651 | 16.86 | 9.85 | 120.60 | 1.70 | 0.06 |
| 7 | [49.44, 49.76] | 1652 | 18.52 | 11.35 | 138.30 | 1.76 | 0.12 |
| 8 | [49.76, 50.08] | 1651 | 18.90 | 13.33 | 149.00 | 3.45 | 0.12 |
| 9 | [50.08, 50.58] | 1652 | 23.42 | 13.67 | 145.00 | 4.00 | 0.30 |

## The R²/RMSE paradox, explained

The mean-only baseline (predict `mean(y_train)` for every validation sample) yields:

| Split | val SS_tot | val variance | mean-only RMSE | mean-only R² |
|---|---|---|---|---|
| **Random 91/9** | 503812.5 | 339.27 | 18.41 | -0.0001 |
| Spatial fold 0 | 1566043.5 | 949.12 | 42.05 | -0.8639 |
| Spatial fold 1 | 582509.8 | 353.04 | 19.26 | -0.0509 |
| Spatial fold 2 | 386439.2 | 234.06 | 15.75 | -0.0605 |
| Spatial fold 3 | 374664.7 | 227.07 | 16.27 | -0.1672 |
| Spatial fold 4 | 377486.6 | 228.64 | 16.33 | -0.1670 |
| Spatial fold 5 | 771038.9 | 467.30 | 21.81 | -0.0189 |
| Spatial fold 6 | 160149.0 | 97.06 | 11.69 | -0.4093 |
| Spatial fold 7 | 212597.5 | 128.77 | 12.19 | -0.1542 |
| Spatial fold 8 | 293081.3 | 177.63 | 13.92 | -0.0913 |
| Spatial fold 9 | 308729.8 | 187.00 | 13.71 | -0.0053 |

**Reading**: a validation set with higher variance (more heavy-tail samples) inflates SS_tot, which makes R² = 1 − SS_res/SS_tot larger for the *same* residual sum of squares. The random split typically has higher variance (it samples the long SOC tail uniformly), so its R² appears higher even when its residuals are larger. The R²/RMSE paradox is therefore not a sign of better model behavior under random splitting — it is a sign that R² is *not invariant* to the validation distribution. Spatial CV reports the model behavior on distribution-shifted hold-outs (specific geographies), which is the more relevant operational claim.