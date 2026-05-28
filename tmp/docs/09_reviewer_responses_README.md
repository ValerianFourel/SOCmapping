# Reviewer-Response Analyses (Geoderma GEODER-D-26-01032)

This directory holds **focused analysis scripts that re-target the previously-written
rebuttal analyses** at the rebuttal's new flagship architecture:
**Vanilla (CNN + Transformer, 215k params)**, retained at the same hyperparameters
as the original SGT (d=128, h=4, L=1).

Where the earlier `rebuttal/*.py` scripts (bootstrap_cis, temporal_regression,
multi_run_cv, etc.) operated on the *original paper's* TFT composite_l2_v2
run, the scripts here operate on the **spatial-CV sweep** (`sweep/oc150*/*/`)
and the **production-mapping** final-models (`final_models/`). They produce
artifacts ready to drop into the revised manuscript and the response letter.

## Reviewer-comment crosswalk

Every script maps to one or more reviewer comments from the action plan:

| Script | Tier-2 item | Reviewers | Purpose |
|---|---|---|---|
| `flagship_summary.py` | (framing) | R1.4, R2.M1, R4.2 | Headline table establishing Vanilla as the recommended architecture; reports R², σ, params, peak-epoch, production-map mean across the 8-family comparison. Drop directly into Discussion §X. |
| `bootstrap_ci.py` | **T2.1** | R1.3, R3.6, R4 implied | 95 % bootstrap CIs on R²μ, RMSE, MAE across the 10 spatial-CV folds. Computed for Vanilla flagship + 7 comparison architectures. |
| `validation_set_stats.py` | **T2.4** | R1.3, R3.3 | Descriptive stats for spatial-CV vs random validation splits on the model-ready dataset (n, mean, SD, median, IQR, % > 50, % > 120). Explains the R²/RMSE paradox. |
| `temporal_sensitivity.py` | **T2.2 + T2.3 + T2.9** | R1.2, R2.M4, R3.5, R3-mod10 | β_year regression with three sensitivities: (1) exclude 2022-2023, (2) weighted 1/n_year, (3) restrict to n_year ≥ 100; multivariate version including altitude + (optionally) land-use class. |
| `nn_distances.py` | **T2.6** | R2.M3 | Nearest-neighbour distance distribution; reports quantiles & % below 300 m. |
| `residual_audit.py` | **T2.7** | R3-mod11 | Stratifies the cross-fold residuals by SOC bin to demonstrate that the training-SD anomaly is dominated by carbon-rich samples (peat / fen). |
| `run_all.py` | — | — | Orchestrator: runs every script above, in order, with shared defaults. |

## Defaults — every script targets these

- **Flagship config**: `vanilla_transformer_d128_h4_L1` at `oc150_vanilla` sweep group
- **Comparison configs**:
  - `small_d128_h4_L1` (SGT) at `oc150`
  - `lightweight_transformer_d128_h4_L1` at `oc150`
  - `simpletransformer_d64_h4_L1` at `oc150`
  - `cnnlstm_d64_h4_L1` at `oc150`
  - `3dcnn_d64_h4_L1` at `oc150`
  - `baseline_rf_default` at `oc150`
  - `baseline_xgb_shallow` at `oc150`

Override via `--flagship-tag`, `--flagship-group`, `--compare TAG@GROUP[,…]` on each script.

## Output

Each script writes both a `.json` (machine-readable) and a `.md` (paste-ready)
to `rebuttal/reviewer_responses/results/`. Re-running a script overwrites that
script's outputs only — others are preserved.

## Run on the cluster

```bash
cd /e/project1/scifi/fourel1/SGT/SOCmapping
source ../venv/bin/activate

# Full reviewer-response package, ~5–10 min wall:
python rebuttal/reviewer_responses/run_all.py

# Individual analyses:
python rebuttal/reviewer_responses/flagship_summary.py
python rebuttal/reviewer_responses/bootstrap_ci.py --n-boot 10000
python rebuttal/reviewer_responses/validation_set_stats.py
python rebuttal/reviewer_responses/temporal_sensitivity.py
python rebuttal/reviewer_responses/nn_distances.py
python rebuttal/reviewer_responses/residual_audit.py
```

## Why a new directory instead of editing the old scripts

The old `rebuttal/bootstrap_cis.py` (and friends) are pinned to a single
TFT 1mil composite_l2_v2 run from the original Weights/ResidualsModels
tree. They will remain valid as the "original-paper" analyses if anyone
asks. The scripts here use the rebuttal's expanded multi-architecture
sweep — the answer set has gone from "1 model × 1 split" to "8 families ×
10 spatial folds" — so re-pointing the old scripts would lose the
architecture comparison. Keeping both lets the response letter cite both
when useful: *"The 95 % CI on the original TFT run is [a, b] (Table X);
the same metric across the 10-fold spatial CV on the new flagship is
[c, d] (Table Y)."*
