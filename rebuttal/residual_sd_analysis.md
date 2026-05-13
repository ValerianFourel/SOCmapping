# Training residual SD analysis — T2.7

_Predictions source:_ `/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/analysis_results.pkl`

Residual = prediction − actual (g/kg). Comparison between training and validation residuals; within each split, broken down by the OC ≤ 50 vs. OC > 50 g/kg threshold where peatlands dominate.

## Tail counts (actual OC concentration)

| split | n | n > 50 g/kg | % > 50 | n > 120 g/kg | % > 120 |
|-------|---|-------------|--------|--------------|---------|
| train | 15155 | 1166 | 7.69% | 154 | 1.02% |
| val | 1359 | 8 | 0.59% | 0 | 0.00% |

## Residual statistics by SOC stratum

| split | stratum | n | mean residual | **SD residual** | RMSE | MAE | min | max |
|-------|---------|---|---------------|-----------------|------|-----|-----|-----|
| train | all | 15155 | -0.3137 | **7.5315** | 7.5378 | 3.3695 | -104.55 | +101.08 |
| train | OC ≤ 50 g/kg | 13989 | +0.5480 | **5.2894** | 5.3175 | 2.5750 | -36.76 | +101.08 |
| train | OC > 50 g/kg | 1166 | -10.6519 | **16.9124** | 19.9812 | 12.9012 | -104.55 | +30.84 |
| train | OC > 120 g/kg | 154 | -21.3352 | **25.4967** | 33.1821 | 22.2384 | -104.55 | +5.11 |

| val | all | 1359 | +1.1287 | **4.6242** | 4.7583 | 2.7912 | -34.30 | +47.37 |
| val | OC ≤ 50 g/kg | 1351 | +1.1938 | **4.4933** | 4.6476 | 2.7493 | -27.24 | +47.37 |
| val | OC > 50 g/kg | 8 | -9.8549 | **10.7655** | 14.0900 | 9.8549 | -34.30 | -0.96 |
| val | OC > 120 g/kg | 0 | — | — | — | — | — | — |

## Headline

- Training residual SD across **all** samples: **7.5315** (n=15155).
- Training residual SD restricted to **OC ≤ 50 g/kg**: **5.2894** (n=13989, 92.3% of train).
- Training residual SD restricted to **OC > 50 g/kg**: **16.9124** (n=1166, 7.7% of train).
- Validation residual SD (all): 4.6242 (n=1359).

**Reading.** The OC > 50 g/kg tail comprises only 7.7% of the training samples but its residual SD (16.91) is **3.2× the SD on the OC ≤ 50 bulk (5.29)** — confirming that the high training residual SD is driven by peatland-class outliers, not by mis-fit on the soil-mineral majority. The validation split, which contains essentially no high-SOC samples (see Task 5), is on a fundamentally easier stratum than the training set the model was scored on.
