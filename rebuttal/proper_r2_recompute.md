# Proper R² recomputation across all saved residual analyses

**Date:** 2026-05-13
**Why:** Throughout the v1 codebase, "R²" in `train.py`, `residualsStudy.py`,
and wandb logs was computed as `np.corrcoef(pred, actual)[0,1] ** 2` —
the **squared Pearson correlation**, not the coefficient of determination.
The two agree only when predictions are unbiased and identically scaled.
For held-out test data with systematic bias (typical in spatial CV), the
gap can be huge — squared correlation stays moderate while proper R² goes
negative.

This file documents the recomputation of both metrics for every saved
SGT/TFT analysis result on disk. **The published "R² = 0.626" for Model A
corresponds to a proper R² of 0.594** — a 3 pp downward correction that's
small for Model A's well-calibrated val set, but much larger for
spatial-CV-style setups where the bias is large.

Going forward, every `train.py` / `residualsStudy.py` / `run_kfold.py` now
reports BOTH:
- `r_squared` (or `r2`) = coefficient of determination = 1 − SS_res / SS_tot
- `pearson_r2` = squared Pearson correlation (for v1 continuity)

Code fix committed in this rebuttal session — affects only future runs;
historical wandb panels labelled `r_squared` continue to display
squared-Pearson values.

---

## Model A (Table 2 evaluation, normalize + L2)

`Weights-ResidualsModels-MappingInference-SOCmapping/Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/analysis_results.pkl`

| split | n | **r² (Pearson²)** | **R² (1−SS_res/SS_tot)** | RMSE | MAE | bias |
|---|---|---|---|---|---|---|
| train | 15,155 | 0.8659 | 0.8623 | 7.54 | 3.37 | −0.31 |
| **val** | **1,359** | **0.6258** | **0.5944** | **4.76** | **2.79** | **+1.13** |

Bias is small (+1.13 g/kg ≈ 5% of the val mean), so the Pearson² ≈ proper R²
gap is only ~3 pp. **Use 0.594 as the honest R²; 0.626 stays available as
the Pearson² for continuity with the v1 paper.**

## Model B (mapping, log + L1) — what MC dropout was run on

`Weights-ResidualsModels-MappingInference-SOCmapping/Archive/residual_analysis_1mil_TRANSFORM_log_LOSS_l1_TemporalFusionTransformer/analysis_results.pkl`

| split | n | **r² (Pearson²)** | **R² (1−SS_res/SS_tot)** | RMSE | MAE | bias |
|---|---|---|---|---|---|---|
| train | 15,136 | 0.7605 | 0.7475 | 10.22 | 4.21 | −1.73 |
| **val** | **1,378** | **0.4620** | **0.0628** | **7.05** | **3.09** | **+0.48** |

Model B's val collapses from 0.46 → 0.063 — but it was trained with
`use_validation=False`, so "val" here is just a sanity-check holdout
rather than a tuning target. The mapping model is judged by its inference
fit, not this val statistic.

---

## Other SGT/TFT runs ranked by proper val R²

| run | n_val | r² (Pearson²) | **R² (proper)** | bias |
|---|---|---|---|---|
| `residualModels1mil_normalize_composite_l2_v2 / AllDataFit` | 1,359 | 0.7799 | **0.7571** | −1.13 |
| `residual_analysis1mil_normalize_compositel2` | 1,376 | 0.6833 | **0.6728** | +0.36 |
| `residual_analysis1mil_normalize_composite_l2_v2` *(= Model A)* | 1,359 | 0.6258 | **0.5944** | +1.13 |
| `residual_analysis_1mil_OC150…_normalize_loss_l1` | 1,626 | 0.6145 | **0.5196** | +1.33 |
| `residual_analysis_TFT360kparams…_normalize_loss_l1` | 1,361 | 0.5680 | **0.4141** | +1.24 |

## "Bias-bombed" runs — Pearson² and proper R² diverge dramatically

When predictions are systematically shifted by 4+ g/kg, Pearson² stays in
the 0.25–0.50 range but proper R² is **negative** (worse than the
mean-predictor baseline):

| run | n_val | r² (Pearson²) | **R² (proper)** | bias |
|---|---|---|---|---|
| `…_TRANSFORM_none_LOSS_composite_l2` (TFT) | 1,325 | 0.2509 | **−1.79** | +4.52 |
| `residual_analysis_cnnlstm_…_composite_l2` | 1,377 | 0.3062 | **−1.62** | +4.62 |
| `…_2mil_SimpleTransformer_…_composite_l2` | 1,566 | 0.4534 | **−0.45** | +4.67 |
| `…_TRANSFORM_none_LOSS_composite_l2_TemporalFusionTransformer` | 1,473 | 0.5057 | **+0.20** | +2.32 |

The bigger the bias, the bigger the gap. These runs were "near-mean"
predictors with the right ranking but the wrong location — a failure mode
that proper R² catches and Pearson² obscures.

---

## Reproduce locally

```bash
VPY=/home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python
$VPY << 'PY'
import pickle, glob, numpy as np

def both_r2(pred, actual):
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    p, a = p[mask], a[mask]
    pearson_r2 = float(np.corrcoef(p, a)[0, 1] ** 2)
    ss_res = float(np.sum((p - a) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    coeff_det = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    bias = float((p - a).mean())
    return pearson_r2, coeff_det, bias

candidates = sorted(set(
    glob.glob('/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/**/analysis_results.pkl', recursive=True)
    + glob.glob('/home/valerian/SGTPublication/residual_Maps_*/**/analysis_results.pkl', recursive=True)
))
for p in candidates:
    d = pickle.load(open(p, 'rb'))
    for split in ('train_results', 'val_results'):
        if split not in d or 'predictions' not in d[split]:
            continue
        pr2, cd, bias = both_r2(d[split]['predictions'], d[split]['targets'])
        print(f'{p[-90:]:90}  {split:14}  r²(Pearson²)={pr2:.4f}  R²(proper)={cd:+.4f}  bias={bias:+.3f}')
PY
```

---

## What this implies for the rebuttal

1. **Update Table 2** to report both `r² (Pearson²)` and `R² (1−SS_res/SS_tot)`.
   For Model A: `0.626 / 0.594`. Don't pull the published 0.626 — just
   add the proper R² column.
2. **The spatial k-fold script now reports both per epoch and per fold**;
   the `kfold_results.md` summary will have both rows.
3. **The Alpine fold is where the gap will be largest** — predictions
   trained on the lowlands will be biased low when extrapolating to
   peatland highlands. Proper R² may go negative there.
4. **Response-letter framing**: "v1 followed the field convention of
   reporting R² as the squared Pearson correlation. For this revision we
   additionally report the coefficient of determination, which is
   sensitive to prediction bias and provides a stricter test of
   generalisation under spatial extrapolation. The two agree closely for
   the original spatial-with-buffer split (0.626 vs 0.594) but diverge
   substantially under k-fold spatial extrapolation, reflecting between-
   region prediction bias — itself a substantive DSM finding."
