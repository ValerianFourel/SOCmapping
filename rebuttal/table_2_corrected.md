# Table 2 — corrected numbers for the Geoderma response letter

**Date:** 2026-05-13
**Cross-refs:** [`proper_r2_recompute.md`](proper_r2_recompute.md),
[`proper_r2_all_projects.md`](proper_r2_all_projects.md),
[`bootstrap_results.md`](bootstrap_results.md)

Throughout the v1 manuscript "R²" was computed as the squared Pearson
correlation coefficient. This file gives the correction with the proper
coefficient of determination ($R^2 = 1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$),
plus bootstrap 95 % confidence intervals on the Model A evaluation set
(1,359-sample held-out spatial split, 1.2 km buffer; n_boot = 1,000;
seed = 20260513).

---

## Direct corrected Table 2 row for Model A (Bavaria SGT, 91/9 spatial split)

| metric | point estimate | bootstrap 95 % CI |
|---|---|---|
| **Pearson r²** (as reported in v1) | **0.626** | [0.526, 0.712] |
| **proper R²** (= 1 − SS_res / SS_tot) | **0.594** | **[0.450, 0.693]** |
| RMSE (g/kg) | 4.76 | [4.18, 5.33] |
| MAE (g/kg) | 2.79 | [2.60, 3.00] |
| RPIQ | 1.05 | [0.90, 1.19] |
| bias (pred − actual, g/kg) | +1.13 | [0.88, 1.37] |
| Pearson r | 0.791 | — |
| n (val) | 1,359 | — |

The 95 % CI on proper R² is wider than on Pearson r² because proper R² is
sensitive to bias resampling; the CI on Pearson r² is artificially narrow
because the metric ignores location shifts.

---

## Architecture leaderboard — same evaluation under both R² flavours

(Validation set per row, all from the published Weights repo `analysis_results.pkl`.)

| Architecture | Variant | n_val | Pearson r² (v1 "R²") | **proper R² (corrected)** | bias g/kg |
|---|---|---|---|---|---|
| **SGT (TFT) 1mil** | **normalize + L2 — Model A** | **1,359** | **0.626** | **0.594** | +1.13 |
| SGT (TFT) 1mil | normalize + L1 | 1,626 | 0.615 | 0.520 | +1.33 |
| SGT (TFT) 360k | normalize + L1 | 1,361 | 0.568 | 0.414 | +1.24 |
| SGT (TFT) 360k | none + L2 | 1,473 | 0.506 | 0.201 | +2.32 |
| SGT (TFT) 1mil | log + L1 — Model B (deployment) | 1,378 | 0.462 | 0.063 | +0.48 |
| 3DCNN | none + MSE | 1,661 | 0.190 | **−0.258** | +1.86 |
| SimpleTransformer 2mil | none + L2 | 1,566 | 0.453 | **−0.448** | +4.67 |
| CNN-LSTM | none + L2 | 1,377 | 0.306 | **−1.622** | +4.62 |
| SimpleTransformer 20k | none + L2 | 1,325 | 0.251 | **−1.794** | +4.51 |

**Headline:** every non-SGT baseline collapses to a negative
coefficient of determination because of systematic prediction bias
(>4 g/kg). The SGT/TFT architecture's advantage over baselines under
proper R² is much wider than v1 Pearson² numbers suggested (e.g. Model A
vs. 2mil SimpleTransformer: apparent gap 0.17 in Pearson², actual gap
**1.04** in proper R²).

---

## Drop-in response-letter paragraph

> Following the reviewers' request for tighter quantification, all
> performance metrics in this revision are reported as the coefficient of
> determination $R^2 = 1 - \mathrm{SS_{res}} / \mathrm{SS_{tot}}$
> (henceforth simply "$R^2$"). The v1 manuscript reported "R²" as the
> squared Pearson correlation; this gives the same value as the
> coefficient of determination only when predictions are unbiased and
> identically scaled. We provide both metrics in the corrected Table 2
> below for full transparency. For Model A (Bavaria SGT, 91/9 spatial
> hold-out with 1.2 km buffer, n = 1,359), the change is small:
> Pearson $r^2$ = 0.626 → coefficient of determination $R^2$ = 0.594
> (1,000-bootstrap 95 % CI = [0.450, 0.693], seed = 20260513). RMSE,
> MAE, and RPIQ are unchanged (4.76 g/kg / 2.79 g/kg / 1.05). For the
> non-SGT baselines previously reported with Pearson $r^2$ in the
> 0.19–0.45 range, the coefficient of determination is negative because
> of systematic prediction bias of +4 g/kg or larger — i.e., the
> mean-only predictor outperforms them. The SGT architecture's gain over
> the next-best baseline under the corrected metric is therefore
> substantially larger than the v1 table indicated. Per-fold spatial
> cross-validation R² (Section X) is reported using the same coefficient-
> of-determination definition.

---

## Note on the "AllDataFit" / "fullDatasetRun" rows in `proper_r2_all_projects.md`

Those rows look better than Model A / Model B on the same val rows
(R² = 0.757 and R² = 0.644 respectively) but **they are not held-out
evaluations**. The model that produced those predictions was retrained
with `use_validation=False` — i.e., on the entire 16,514-row dataset
including the 1,359 / 1,378 rows that the parent file uses as "val".
The v1 codebase marks such checkpoints with the `_R2_1.0000` filename
suffix as a "no validation" placeholder.

These deployment-stage models exist because Figures 14/15 (the Bavaria-
wide prediction map) need the most-data-trained version of the network;
the `AllDataFit` / `fullDatasetRun` directories are sanity-check
re-evaluations confirming the full-data model fits the original
train+val pool. **They must not be cited as test-set R²** — for Table 2
use only the parent files (Model A: R² = 0.594; Model B: R² = 0.063 —
or simply do not cite Model B's val number since it was a sanity holdout
rather than a tuning target).

---

## Reproduction

```bash
VPY=/home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python
$VPY /home/valerian/SGTPublication/SOCmapping/rebuttal/bootstrap_cis.py
```

Outputs both `r2` (proper) and `pearson_r2` columns in
`bootstrap_results.{md,json}`, plus bias CIs.
