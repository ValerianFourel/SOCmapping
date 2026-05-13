# Models inventory log — every `analysis_results.pkl` we have

**Date written:** 2026-05-13
**Source:** every `analysis_results.pkl` found under
  `/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/`,
  the three `residual_Maps_Bavaria_*` folders, and
  `SOCmapping/SpatiotemporalGatedTransformer/residual_analysis/`.

**Count:** 17 files, listed individually (no dedup). Three of the 17
are mirror copies under `residual_Maps_Bavaria_*` of files in the
`Archive/` directory; both copies are listed so the audit trail is
explicit.

**Metrics convention (final, after commit `e414a47`):**
- `r2 (Pearson²)` = `corr(pred, actual)²` — what v1 manuscript called "R²"
- **`r2 (proper)`** = `1 − Σ(actual − pred)² / Σ(actual − mean(actual))²`
  — coefficient of determination; this is the scientifically meaningful
  number for held-out data
- `bias` = `mean(pred − actual)` g/kg
- **`useful`** = `r2 (proper) > 0` (model beats predicting the mean)
- **`leak`** = the model was re-evaluated on rows that were in its
  training pool (`_R2_1.0000` filename suffix, or path contains
  `AllDataFit` / `fullDatasetRun`)

---

## All 17 entries (ranked by proper R² descending)

| # | n_val | Pearson r² | **proper R²** | RMSE | bias | useful | leak | path |
|---|---|---|---|---|---|---|---|---|
| 3 | 1,359 | 0.7799 | **+0.7571** | 3.68 | −1.13 | ✅ | **⚠️** | `…/Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/residual_analysis_AllDataFit/` |
| 4 | 1,376 | 0.6833 | **+0.6728** | 4.80 | +0.36 | ✅ | | `…/Archive/residual_analysis1mil_normalize_compositel2_TemporalFusionTransformer/` |
| 10 | 1,378 | 0.6836 | **+0.6441** | 4.35 | −1.12 | ✅ | **⚠️** | `…/Archive/residual_analysis_1mil_TRANSFORM_log_LOSS_l1_TemporalFusionTransformer/residual_analysis_fullDatasetRun/` |
| 2 | 1,359 | 0.6258 | **+0.5944** | 4.76 | +1.13 | ✅ | | `…/Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/` ← **MODEL A (canonical)** |
| 15 | 1,359 | 0.6258 | +0.5944 | 4.76 | +1.13 | ✅ | | `…/residual_Maps_Bavaria_1milTFT/residual_Maps_Bavaria_1milTFT/` ← mirror of #2 |
| 8 | 1,626 | 0.6145 | **+0.5196** | 5.27 | +1.33 | ✅ | | `…/Archive/residual_analysis_1mil_OC150_2007to2023_transform_normalize_loss_l1_TemporalFusionTransformer/` |
| 13 | 1,361 | 0.5680 | **+0.4141** | 6.09 | +1.24 | ✅ | | `…/Archive/residual_analysis_TFT360kparams_OC150_2007to2023_transform_normalize_loss_l1/` |
| 16 | 1,361 | 0.5681 | +0.4143 | 6.09 | +1.24 | ✅ | | `…/residual_Maps_Bavaria_360kTFT/residual_Maps_Bavaria_360kTFT/` ← mirror of #13 |
| 1 | 1,473 | 0.5057 | **+0.2013** | 6.92 | +2.32 | ✅ | | `SOCmapping/SpatiotemporalGatedTransformer/residual_analysis/` |
| 6 | 1,473 | 0.5057 | +0.2013 | 6.92 | +2.32 | ✅ | | `…/Archive/residual_analysis360k_…_TRANSFORM_none_LOSS_composite_l2_TemporalFusionTransformer/` ← mirror of #1 |
| 9 | 1,378 | 0.4620 | **+0.0628** | 7.05 | +0.48 | ✅ | | `…/Archive/residual_analysis_1mil_TRANSFORM_log_LOSS_l1_TemporalFusionTransformer/` ← **MODEL B (deployment / Fig 14)** |
| 12 | 1,661 | 0.1897 | **−0.2584** | 8.73 | +1.86 | ❌ | | `…/Archive/residual_analysis_…_TRANSFORM_none_LOSS_mse_R2_3DCNN/` |
| 11 | 1,361 | 0.4101 | **−0.4296** | 9.55 | +4.05 | ❌ | | `…/Archive/residual_analysis_2mil_SimpleTransformer_…_LOSS_composite_l2/` |
| 7 | 1,566 | 0.4534 | **−0.4475** | 9.21 | +4.67 | ❌ | | `…/Archive/residual_analysisFInalfinalResults…_2mil_SimpleTransformer/` |
| 17 | 1,566 | 0.4534 | −0.4475 | 9.21 | +4.67 | ❌ | | `…/residual_Maps_Bavaria_v3_2milSimpleTransformer/residual_Maps_Bavaria_v3_2milSimpleTransformer/` ← mirror of #7 |
| 14 | 1,377 | 0.3062 | **−1.6215** | 11.71 | +4.62 | ❌ | | `…/Archive/residual_analysis_cnnlstm_model_BEST_OVERALL_…_R2_0.2937/` |
| 5 | 1,325 | 0.2509 | **−1.7939** | 12.58 | +4.51 | ❌ | | `…/Archive/residual_analysis20k_SimpleTransformer_…_LOSS_composite_l2/` |

Indices (#1–#17) are arbitrary insertion order from the file walk; **kept here for cross-reference with the JSON dump**. The ranked column above is sorted by proper R².

---

## Mirror-copy groups (same predictions in multiple folders)

| canonical (`Archive/`) | mirror (`residual_Maps_Bavaria_*` / local) |
|---|---|
| #2 — Model A 1mil normalize+L2 | #15 |
| #6 — TFT 360k none+L2 | #1 (local in SGT/residual_analysis) |
| #13 — TFT 360k normalize+L1 | #16 |
| #7 — 2mil SimpleTransformer | #17 |

When citing, cite the `Archive/` version — it's the source of record. The `residual_Maps_Bavaria_*` copies appear to be staging artifacts from when the Bavaria-wide residual maps were generated.

---

## Notes per row (where there's something non-obvious)

### #2 / #15 — Model A (canonical, Table 2 baseline)
- `model_path`: `…/residualModels1mil_normalize_composite_l2_v2/TFT_model_BEST_OVERALL_..._R2_0.6909.pth`
- **Auditable**: .pth checkpoint, 4 train/val parquets, normalization-stats .pkl all present in the parent folder. pkl val (513 unique points) matches `train_val_data_run_1.parquet` exactly (513/513). Citing this row in Table 2 is reproducible end-to-end.

### #3 — Model A v2 *AllDataFit* (data leakage)
- `model_path`: ends in `_R2_1.0000.pth` (no-validation placeholder)
- Same val rows as #2 but the model was trained on all 16,514 rows including those val rows. **Do not cite** as held-out R².

### #4 — Model A "v1" (no `_v2` suffix in source folder)
- `model_path`: `…/residualModels1mil/TFT_model_BEST_OVERALL_..._R2_0.6256.pth`
- **Provenance broken**: the source folder `residualModels1mil/` (no suffix) is gone from disk, only the `.pkl` survived. No saved train_val parquet matches the val rows (only 41 % overlap with closest candidate). Has the best proper R² of the non-leaked rows (0.673) but cannot be reproduced — **do not cite as headline**.

### #9 — Model B (deployment / Fig 14)
- `model_path`: `…/residualModels1mil_TRANSFORM_log_LOSS_l1/TFT_model_..._R2_0.6287.pth`
- Used to generate the Bavaria-wide prediction map (paper Fig 14) and as the base model for MC-dropout uncertainty experiment.
- Trained with `use_validation=False`, so its "val" is a sanity-check holdout rather than a tuning target — proper R² 0.063 reflects that, **not** the map quality.

### #10 — Model B *fullDatasetRun* (data leakage)
- `model_path`: ends in `_R2_1.0000.pth`
- Same situation as #3 (trained on all data, re-eval on rows it saw). Do not cite.

### #5 / #7 / #11 / #12 / #14 / #17 — Negative-R² baselines
- Bias > +4 g/kg in every case → predictions clustered near training mean → coefficient of determination is **worse than the mean predictor** (RPD < 1, RMSE > SD(y))
- These were architecture-exploration runs, not benchmark-quality comparisons against published DSM methods.
- See `proper_r2_all_projects_FULL.md` for full metric set (RPD, RMSE/SD, IQR(y)).

---

## Where the source data lives

```
Local:
  /home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/
    Archive/                                         ← canonical residual analyses
    TemporalFusionTransformer/                       ← .pth checkpoints + parquets
    3DCNN/                                            ← 3DCNN residuals
    CNNLSTM/                                          ← CNN-LSTM residuals
    SimpleTransformer/                                ← SimpleTransformer residuals

  /home/valerian/SGTPublication/residual_Maps_Bavaria_*/     ← Bavaria-map mirrors
  /home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer/residual_analysis/
                                                              ← local SGT residuals

HF mirror:
  https://huggingface.co/ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping
```

## Machine-readable companion

`rebuttal/models_inventory_log.json` (gitignored per `*.json` repo policy)
has the same 17 records with **all** metrics (Pearson r², proper R², RMSE,
MAE, bias, SD(y), IQR(y), RPD, RPIQ, RMSE/SD(y), `is_useful`,
`leakage_flag`, full pkl path and model_path) — regenerate any time with:

```bash
VPY=/home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python
$VPY /home/valerian/SGTPublication/SOCmapping/rebuttal/compute_all_proper_metrics.py
# (writes proper_r2_all_projects_FULL.md/json which contains the same 17 entries
#  deduplicated; for the non-deduplicated raw listing this models_inventory_log
#  is the source of record)
```

---

## What to actually cite in the rebuttal

| paper section | row to cite | why |
|---|---|---|
| Table 2 — Model A baseline | **#2** | auditable; published checkpoint; matches parquet |
| Figure 14 caption — deployment model | **#9** | the actual map-generating model |
| Architecture comparison (if kept in main text) | #2, #13, #11, #14, #12 | one per architecture family |
| Architecture comparison (full, supplementary) | All 17 | for completeness, with the mirrors annotated |
| **Do NOT cite** | #3, #4, #10 | leakage or broken provenance |
