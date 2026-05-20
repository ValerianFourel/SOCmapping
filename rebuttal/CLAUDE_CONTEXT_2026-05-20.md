# SOC Mapping Rebuttal — Full Claude Project Context (2026-05-20 snapshot)

> Drop this file into the Claude project as persistent context. It captures
> the architectural pivot, the empirical findings, the reviewer-response
> infrastructure, the bugs we found and fixed, and the cluster runbook —
> everything needed for a future session to pick up coherently without
> re-deriving any of it.

---

## 0 · TL;DR (read this first)

**Manuscript:** Geoderma GEODER-D-26-01032 — *Spatiotemporal Gated Transformer for High-Resolution Soil Organic Carbon Mapping*. Major revision. **Resubmission deadline: 2026-05-29** (9 days from the date of this document).

**The pivot:** the original paper's flagship architecture (SGT = `SimpleSGT` = CNN + GRN gate + Transformer, 363k params) has been **replaced by Vanilla** (= `VanillaSpatiotemporalTransformer` = SGT minus the GRN gate, **215k params**). Empirical justification:

- **Spatial-CV R²**: Vanilla 0.170, SGT 0.170 — tied within bootstrap noise
- **Cross-fold variance**: Vanilla σ = 0.072, SGT σ = 0.062 — same stability tier
- **Convergence**: Vanilla median peak-epoch 8, SGT 7 — same speed
- **Production-map plausibility**: Vanilla 20-band rebal mean 43.9 g/kg (sensible); SGT rebal mean 50.4 g/kg (overshoots)
- **Parameter overhead**: SGT is **70% larger** than Vanilla for **zero measurable R² gain**

The GRN gate ("the G in SGT") is dead weight at this data scale. The recommendation in the revised manuscript is the **lightweight CNN+Transformer hybrid (Vanilla, 215k)** — defended against transformer-alone (Lightweight, 240k) by a +0.23 R² gap, and against tree baselines (RF/XGB) by spatial-extrapolation stability.

**The repo branch:** all rebuttal work lives on `bestrun-bands`. `bestrun` is frozen — never commit there.

**The Hub dataset:** [`ValerianFourel/SOCrebuttal`](https://huggingface.co/datasets/ValerianFourel/SOCrebuttal) holds the published artifacts (6,113 files: 191 sweep configs, 32 production-map runs, source code, docs). Re-publish with `python rebuttal/publish_to_hf.py`.

---

## 1 · Architectural lineup (the 8-family comparison)

Every reviewer-response table compares the same 8 architectures at matched-as-possible hyperparameters. Memorize this table — it appears everywhere.

| Family | Class file | Tag pattern | Params | Role in the story |
|---|---|---|---|---|
| **Vanilla** ⭐ flagship | `SpatiotemporalGatedTransformer/VanillaSpatiotemporalTransformer.py` | `vanilla_transformer_d128_h4_L1` | 215k | **NEW RECOMMENDATION**. CNN spatial encoder + 1-layer Transformer + MLP head. SimpleSGT minus the GRN gate. |
| **SGT** (original) | `SpatiotemporalGatedTransformer/SimpleSGT.py` | `small_d128_h4_L1` | 363k | The paper's original architecture. Retained as gating ablation — SGT ≈ Vanilla, so GRN doesn't earn its keep. |
| **EnhancedSGT** | `SpatiotemporalGatedTransformer/EnhancedSGT.py` | `d128_h4_L2` etc. | varies | Multi-layer variant of SGT. Discontinued in the rebuttal because it consistently underperforms SimpleSGT at L=1. |
| **Lightweight** | `SpatiotemporalGatedTransformer/LightweightTransformer.py` | `lightweight_transformer_d128_h4_L1` | 240k | **NEW CLASS** — added in this session. Pure transformer (NO CNN frontend), parameter-controllable. CNN-frontend ablation companion to Vanilla. R² = -0.06: confirms CNN provides the spatial inductive bias. |
| **SimpleTransformer** | `SimpleTransformer/modelSimpleTransformerNew.py` | `simpletransformer_d64_h4_L1` | **11.2M** | Reference pure-transformer at huge scale. d_model is HARDCODED to C×H×W=500 regardless of `--hidden_size`. R² = +0.183 — matches Vanilla at 50× the parameters. |
| **CNNLSTM** | `CNNLSTM/models.py` (class `RefittedCovLSTM`) | `cnnlstm_d64_h4_L1` | 93k | CNN + LSTM. Recurrent counterpart to Vanilla. R² = +0.090; fails on Alpine fold. |
| **3DCNN** | `3DCNN/modelCNNMultiYear.py` (class `Small3DCNN`) | `3dcnn_d64_h4_L1` | — | The failed family. R² ≈ -0.76 across the entire matrix. Included for completeness. |
| **RF / XGBoost** | sklearn / xgboost | `baseline_rf_default`, `baseline_xgb_shallow` | — | Classical baselines. 80-d per-band statistics (mean/std/p10/p90 × 20 bands). R² ≈ 0.14. Trees fail on Alpine fold (R² < 0 on fold 0). |

---

## 2 · The 3-axis decomposition (the headline empirical claim)

The paper's central contribution is **decomposing the original R² = 0.62 of the SGT paper into three orthogonal axes**:

| Axis | Effect size | Source of the change |
|---|---|---|
| **Evaluation protocol** (random split → spatial CV) | **-0.44 R²** (0.62 → 0.18) | Roberts 2017 / Ploton 2020 spatial-CV compression factor. The original 0.62 came from a 91/9 random split that did not enforce spatial separation. Honest spatial 10-fold CV with a 1.2 km train/test buffer drops top R² to 0.18. |
| **Covariate band-set** (20 → 6 bands) | **-0.10 R²** (0.18 → 0.08 best 6-band) | The original 6-band subset (red, green, blue, NIR, NDVI, LST) was insufficient. The expanded 20-band stack (5-year time series across NPP, ET, temperature, precipitation, soil moisture, SAR, …) adds the missing information. |
| **Architecture choice** (within the 20-band 0.18 ceiling) | **±0.02 R²** | At the top of the leaderboard, transformer-family architectures cluster within 0.01 of each other. Architecture matters mostly to *avoid the bad ones* (3DCNN, transformer-alone-without-scale). |

The matrix below is the canonical "family × bands × max_oc" R² table from the 191 sweep configs:

| Family | oc150-20b | oc150-6b | Δ (20→6) |
|---|---|---|---|
| SGT-small | **+0.170** | — | (running) |
| Vanilla ⭐ | **+0.170** | +0.061 | **-0.109** |
| Lightweight | -0.079 | -0.089 | -0.010 |
| SimpleTransformer | **+0.183** | +0.027 | **-0.156** |
| CNNLSTM | +0.091 | -0.033 | -0.124 |
| 3DCNN | -0.765 | -0.804 | -0.039 |
| RF | +0.135 | +0.070 | **-0.065** |
| XGBoost | +0.140 | -0.113 | **-0.253** |

---

## 3 · The CNN-frontend ablation (the strongest single experiment)

This is the experiment we built specifically in this session. At **matched parameter scale (~200-240k)**, **with vs without the CNN spatial encoder**:

| Architecture | CNN? | params | R² mean | R² std | median peak epoch | fold-worst R² |
|---|---|---|---|---|---|---|
| Vanilla (d=128, L=1) | **YES** | 215k | **+0.170** | **0.072** | **8** | +0.056 |
| Lightweight (d=128, L=1) | **NO** | 239k | **-0.064** | **0.306** | **20** | **-0.709** |
| Lightweight (d=128, L=2) | NO | 371k | -0.114 | 0.280 | (running long-train) | — |
| SimpleTransformer (d=64) | NO | 11.2M | +0.183 | 0.087 | 8 | +0.065 |

**Three orthogonal benefits of the CNN spatial encoder at matched scale:**

1. **+0.23 R² mean** (Vanilla 0.170 vs Lightweight -0.064)
2. **4× lower cross-fold variance** (0.07 vs 0.31)
3. **~3× faster convergence** (median best-epoch 8 vs 20)

The 11.2M-param SimpleTransformer recovers Vanilla's R² but with **50× the parameters** — so massive scale is the *only* alternative to the CNN's inductive bias at LUCAS data scale. This is the central architectural claim of the revised paper.

---

## 4 · Production-mapping results (Bavaria 2023, 1mil grid)

32 production maps published to HF. Means at non-rebalanced and KDE-rebalanced sampling:

| Architecture | Bands | non-rebal mean | rebal mean | Δ |
|---|---|---|---|---|
| Vanilla ⭐ | 20band | 18.80 | **43.93** | +25 |
| Vanilla | 6band | 34.51 | 39.40 | +5 |
| SGT | 20band | 20.35 | **50.44 ⚠ high** | +30 |
| Lightweight | 20band | **3.58** ⚠ low | 1.38 ⚠ lower | -2 (rebal hurt) |
| Lightweight | 6band | 36.82 | 39.29 | +2 |
| SimpleTransformer | 20band | 8.50 ⚠ low | 8.62 | +0 (saturated) |
| SimpleTransformer | 6band | 30.75 | 40.85 | +10 |
| CNNLSTM | 20band | 28.00 | **69.69 ⚠⚠ broken** | +42 |
| CNNLSTM | 6band | 27.66 | 33.30 | +6 |
| 3DCNN | 20band | 23.59 | 21.14 | -2 |
| 3DCNN | 6band | 3.75 | 3.79 | 0 |
| RF | 20band | 26.36 | 25.14 | -1 |
| XGBoost | 20band | 14.97 | (pending) | — |
| XGBoost | 6band | 36.93 | (pending) | — |

**Reading**:
- Healthy LUCAS Bavaria mean ≈ 25-35 g/kg
- **Vanilla 20band rebal (43.93)** is the most plausible rebalanced map — slightly high but defensible
- **SGT 20band rebal (50.44)** overshoots — KDE-α=0.5 was too aggressive for this architecture
- **CNNLSTM 20band rebal (69.69)** is 2.3× the LUCAS mean — **don't use this map**; the model latched onto upweighted tail
- **Lightweight rebal got WORSE** at 20-band — confirms it can't learn the spatial structure to take advantage of the upweighted signal
- **SimpleTransformer rebal unchanged** at 20-band — the 11.2M input embedding saturates regardless of sample weighting
- **RF rebal slightly hurt** — sample_weight doesn't help tree-impurity calculation when the heavy tail is sparse

**Recommendation for the production-map figure in the paper**:
- Main text figure: Vanilla non-rebal vs rebal (20-band, 6-band)
- Supplementary figure: full 8×2 cross-architecture grid (use the rendered `maps_comparison_2023.png` and `maps_comparison_2023_rebal.png`)

---

## 5 · What was built in this session (chronological commit log)

The session's commits on `bestrun-bands`, most recent at top:

| Commit | Summary |
|---|---|
| `0c87d0e` | reviewer_responses: gitignore __pycache__ |
| `691649c` | reviewer_responses: re-target rebuttal analyses at Vanilla flagship (6 scripts + run_all + README) |
| `d6f366a` | publish_to_hf.py — push rebuttal artifacts to HF Hub, expandable |
| `218a925` | compare_maps: --rebal flag for rebalanced production-map figure |
| `59b6d8c` | train_full_baselines: KDE sample_weight for RF + XGB rebalanced fits |
| `7fb99e1` | final_models: add KDE-rebalanced training for fuller-range production maps |
| `b8e1f1e` | train_full: add vanilla_transformer + lightweight_transformer to --model-family (one-line argparse fix that had silently broken vanilla training for an entire commit window) |
| `fdfe157` | final_models: sync expected-run lists for the two new architectures |
| `a92c7c8` | final_models: add LightweightTransformer + 3DCNN to production pipeline |
| `2a8ac62` | inspect_run: add 'transformers' preset for the full transformer comparison |
| `0ab25fb` | LightweightTransformer long-training defensive run (200 epochs sbatch) |
| `fad2db0` | LightweightTransformer: **NEW CLASS** — true transformer-alone baseline (85k-370k params) |
| `c25ef79` | inspect_run.py + simpletransformer-ablation grid |
| `1ae7487` | infer-bavaria: fix float-slice crash + idempotent resume + 3h walltime |
| `1393aef` | infer_bavaria: switch to the 1mil-grid mapping dataloader (the wrong-loader fix) |
| `2b18808` | infer_bavaria: surface dataset exceptions; refuse to write all-NaN maps |
| `4ef61e4` | infer_bavaria.predict_nn: clamp feature_stds and sanitize NaN inputs |
| `53e7e39` | compare_maps: skip NaN-mean panels and downsample for fast render |
| `08ff94e` | final_models: sbatch to re-run all Bavaria inferences sequentially |
| `e51ad75` | check_inference_status.py — Slurm + per-run snapshot for infer-bavaria |
| `b5c1cac` (earlier) | THE best_model_state reference bug fix (`{k: v.detach().clone() ...}`) |

---

## 6 · File map (where everything lives in `bestrun-bands`)

```
SOCmapping/
├── SpatiotemporalGatedTransformer/
│   ├── SimpleSGT.py                     ← original SGT (363k @ d=128)
│   ├── EnhancedSGT.py                   ← multi-layer SGT variant
│   ├── VanillaSpatiotemporalTransformer.py  ← NEW FLAGSHIP (215k @ d=128)
│   ├── LightweightTransformer.py        ← NEW CNN-frontend ablation companion
│   ├── train.py                         ← contains the b5c1cac best-state fix
│   └── dataloader/
│       ├── dataloaderMultiYears.py      ← LUCAS-sample-coord training loader
│       └── dataloaderMapping.py         ← 1mil-grid production loader (the
│                                          one infer_bavaria now uses, fixed
│                                          in 1393aef)
│
├── SimpleTransformer/
│   └── modelSimpleTransformerNew.py     ← SimpleTransformerV2 (the 11.2M one;
│                                          d_model HARDCODED to C×H×W)
│
├── CNNLSTM/
│   └── models.py                        ← RefittedCovLSTM
│
├── 3DCNN/
│   └── modelCNNMultiYear.py             ← Small3DCNN (the broken family)
│
└── rebuttal/                            ← ALL rebuttal work lives here
    │
    ├── model_ready_dataset.parquet      ← 16,514 LUCAS+LfL+LfU samples
    │                                       with altitude joined. The shared
    │                                       input for every analysis.
    │
    ├── gpu_experiments/spatial_kfold/
    │   ├── sweep/                       ← 191 sweep configs published to HF
    │   │   ├── oc{90,120,150}/          ← max_oc × default-sampler runs
    │   │   ├── oc*_6band/               ← 6-band variants
    │   │   ├── oc*_vanilla/             ← vanilla_transformer sweeps
    │   │   ├── oc*_cmpL2/               ← composite-L2 loss variants
    │   │   └── oc150_longtrain/         ← Lightweight 200-epoch defensive run
    │   ├── run_kfold.py                 ← THE spatial-CV training+inference
    │   │                                  pipeline (_build_model dispatches
    │   │                                  all 6 families)
    │   ├── run_folds_parallel.py        ← --folds-per-gpu orchestrator
    │   ├── sweep_submit.py              ← Slurm-batch submitter with grids
    │   │                                  for SGT, FAMILY, VANILLA,
    │   │                                  SIMPLETRANSFORMER_ABLATION,
    │   │                                  LIGHTWEIGHT_TRANSFORMER, BASELINE
    │   ├── sweep_summarize.py           ← Flat ranking table
    │   ├── inspect_run.py               ← Per-config drilldown + presets
    │   │                                  (gating, cnn-frontend, architecture,
    │   │                                   transformers)
    │   ├── inspect_sweep.py             ← Per-epoch trajectory (peak-epoch
    │   │                                  diagnostic — proves SGT/Vanilla
    │   │                                  converge at epoch ~8, Lightweight
    │   │                                  at ~20)
    │   ├── band_subsets.py              ← --bands-list {full_20, original_6}
    │   ├── run_baselines.py             ← RF + XGBoost runner
    │   ├── sbatch_lwt_longtrain.sbatch  ← one-off Lightweight 200-epoch
    │   └── (also: sbatch/, slurm_logs/, baseline_features/)
    │
    ├── final_models/                    ← production-mapping pipeline
    │   ├── train_full.py                ← full-data training (95% train +
    │   │                                  5% RANDOM monitor for best-epoch
    │   │                                  selection). NN path. Includes the
    │   │                                  b8e1f1e argparse fix.
    │   ├── train_full_baselines.py      ← RF/XGB full-data training. KDE
    │   │                                  sample_weight added in 59b6d8c.
    │   ├── infer_bavaria.py             ← Bavaria 1mil-grid inference.
    │   │                                  Uses MultiRasterDataset1MilMultiYears
    │   │                                  (the right loader, fixed 1393aef).
    │   │                                  Handles NN + tree dispatch.
    │   ├── submit_finals.py             ← Slurm orchestrator. --rebalance
    │   │                                  flag added in 7fb99e1 + 59b6d8c.
    │   ├── compare_maps.py              ← Cross-architecture map figure.
    │   │                                  --rebal flag added in 218a925.
    │   ├── check_inference_status.py    ← Slurm/run-state status checker
    │   ├── run_all_inferences.sbatch    ← Bulk infer with idempotent resume
    │   ├── checkpoints/                 ← trained weights (per run_name)
    │   ├── maps/                        ← Bavaria 2023 parquets + summaries
    │   ├── maps_comparison_2023.{png,md,json}        ← non-rebal figure
    │   └── maps_comparison_2023_rebal.{png,md,json}  ← rebal figure
    │
    ├── reviewer_responses/              ← NEW THIS SESSION (Tier 2 analyses)
    │   ├── README.md                    ← reviewer-comment crosswalk
    │   ├── _common.py                   ← shared FLAGSHIP + COMPARISONS catalog
    │   ├── flagship_summary.py          ← headline table (Vanilla as flagship)
    │   ├── bootstrap_ci.py              ← T2.1 — 95% bootstrap CIs
    │   ├── validation_set_stats.py      ← T2.4 — R²/RMSE paradox explained
    │   ├── temporal_sensitivity.py      ← T2.2 + T2.3 + T2.9 — β_year sens.
    │   ├── nn_distances.py              ← T2.6 — 1-NN distance distribution
    │   ├── residual_audit.py            ← T2.7 — stratified residual SD
    │   ├── run_all.py                   ← orchestrator
    │   └── results/                     ← .json + .md outputs land here
    │
    ├── publish_to_hf.py                 ← push to ValerianFourel/SOCrebuttal
    │                                      with bundle system + auto-README
    │
    ├── REVISION_LOG.md                  ← detailed phase-by-phase log
    │
    └── (also: bootstrap_cis.py, temporal_regression*.py, multi_run_cv.py,
        nn_distances.py, split_comparison.py, residual_sd_analysis.py, etc.
        ─── the ORIGINAL-PAPER-TARGETED analyses, pinned to the TFT 1mil
        composite_l2_v2 run. Retained alongside reviewer_responses/ —
        the response letter can cite either depending on which question
        is being answered.)
```

---

## 7 · Reviewer-comment crosswalk (Tier 2 → script)

For the response letter, each reviewer comment maps to specific produced artifacts:

| Reviewer comment | Tier | Script under reviewer_responses/ | What it answers |
|---|---|---|---|
| **R1.3** "single spatial split weaker than repeated CV" | T2.1 | `bootstrap_ci.py` | 95% CIs across 10 spatial folds × 8 architectures |
| **R3.6** "no confidence intervals on Table 2" | T2.1 | `bootstrap_ci.py` | same as above; Vanilla and SGT CIs overlap |
| **R1.3 + R3.3** "R²/RMSE paradox" | T2.4 | `validation_set_stats.py` | mean-only-baseline metrics show variance-composition explains the paradox |
| **R1.2 + R2.M4 + R3.5** "+0.751 g/kg/yr over-interpreted" | T2.2 | `temporal_sensitivity.py` V2 | excluding 2022-2023 drops β to +0.90 |
| **R3-mod10** "sample-size imbalance across years" | T2.3 | `temporal_sensitivity.py` V3, V4 | weighted-by-year → +2.42; n≥100 → +1.07 |
| **R3.5** "extended multivariate regression" | T2.9 | `temporal_sensitivity.py` V5, V6 | + altitude → +0.75 (the paper's number); + land use if available |
| **R2.M3** "300 m minimum distance pedologically plausible?" | T2.6 | `nn_distances.py` | reports 1-NN distribution; clarifies that 1.2 km is the train/val buffer, not 300 m |
| **R3-mod11** "training residual SD = 8.87 vs val SD = 5.97" | T2.7 | `residual_audit.py` | stratified by SOC bin, demonstrates compositional effect |
| **R3.9 + R4.4** uncertainty quantification | T3.2 | `rebuttal/mc_dropout_uncertainty.py` (existing) | acknowledge as limitation, MC-dropout map as cheap-win |
| **R2.M5** "annual covariate stats + yearly SOC maps" | T2.5 + T2.8 | `rebuttal/covariate_temporal_stats.py` + `rebuttal/annual_soc_maps.py` | existing scripts; outputs in `rebuttal/` root |
| **B1** missing bibliography | Tier 1 mechanical | (LaTeX fix) | done outside this codebase |
| **B2** "(citation needed)" working note | Tier 1 mechanical | (LaTeX fix) | done outside this codebase |

---

## 8 · Bugs we found and fixed (technical-debt notes)

These are the bugs we discovered in this session — useful context for not re-stepping on them:

1. **`best_model_state` reference bug** (commit `b5c1cac`, before this session)
   - `model.state_dict()` returns *references* to the live parameter tensors. Saving `best_model_state = model.state_dict()` and continuing to train mutates the "saved" state in place.
   - Fix: `best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}`
   - Affected: every spatial-CV result before the fix returned final-epoch weights, not best-epoch weights. After the fix, R² shifts substantially (better).
   - **This is the single most important bug to know about**.

2. **`train_full.py` --model-family argparse incomplete** (commit `b8e1f1e`)
   - The argparse `choices=` list in `rebuttal/final_models/train_full.py:91` only had `['sgt', '3dcnn', 'cnnlstm', 'simpletransformer']`. It was missing `vanilla_transformer` and `lightweight_transformer` (which `run_kfold._build_model` accepts).
   - Result: every `submit_finals.py` attempt to train Vanilla failed in <1 second at argparse. Manifested as `no_ckpt` in the status checker and `DependencyNeverSatisfied` in squeue.
   - This had **silently broken vanilla training for an entire commit window**. We didn't realize until the user noticed vanilla maps were missing from `compare_maps`.
   - **Memory note saved**: `train_full and run_kfold have parallel --model-family lists that must stay in sync`.

3. **`infer_bavaria.py` using the wrong dataloader** (commit `1393aef`)
   - The script was using `MultiRasterDatasetMultiYears` (the TRAINING loader, which expects every (lon, lat) to be pre-indexed in the LUCAS coordinates.npy hashmap). It silently raised `ValueError("Coordinates not found")` for every 1mil-grid point.
   - Both NN and tree inference paths swallowed this in a bare `except`, producing all-NaN NN predictions and predict-on-X=zeros tree predictions (the "valid-looking" mean=17.40 of the early xgb_shallow_20band map was the constant XGB outputs for X=zeros).
   - Fix: use `MultiRasterDataset1MilMultiYears` from `dataloaderMapping.py` + `separate_and_add_data_1mil_inference` from `dataframe_loader.py`. These use the pre-indexed 1mil-grid coords.npy files.
   - **Lesson**: never bare-`except` in inference loops. Surface the first 3 exceptions with offending (lon, lat) and raise SystemExit if every point fails.

4. **Float-slice crash in `RasterTensorDataset1Mil`** (commit `1ae7487`)
   - `coordinates.npy` stores x, y as `numpy.float64`. In the padded-window branch of `get_tensor_by_location`, `x_offset = half_window - (x - x_start)` mixed Python int with numpy float, yielding a float that crashed slice indexing.
   - Affected ~0.4% of grid points (edge pixels). Other 99.6% succeeded but the bare-except hid this.
   - Fix: cast `x = int(x); y = int(y)` at function entry.

5. **PyTorch `enable_nested_tensor` warning** (no commit needed)
   - Benign warning fired by `nn.TransformerEncoder` when `norm_first=True` or when `batch_first=False`. Doesn't affect correctness or performance for our small input sizes.

6. **`SimpleTransformerV2` ignores `--hidden_size`** (acknowledged, not fixed)
   - `SimpleTransformer/modelSimpleTransformerNew.py:10`: `self.d_model = input_channels * input_height * input_width` overrides whatever d_model the user passed. So `--hidden_size 16` and `--hidden_size 128` produce *the same model* at 11.2M params for 20-band input.
   - We didn't fix this in the source; instead we added `LightweightTransformer` as a parameter-controllable alternative for the CNN-frontend ablation. SimpleTransformerV2 stays in the comparison as the "11.2M reference" row.

7. **CNNLSTM 20-band rebalanced produces broken map (mean=69.69)** (current state)
   - The KDE-α=0.5 rebalancing was too aggressive for CNNLSTM. The model latched onto the upweighted high-SOC tail and predicts mostly elevated values.
   - **Don't use the rebalanced CNNLSTM 20-band map for the paper figure**. The non-rebalanced version (mean=28.00) is the one to show.
   - Could try lower α (0.25) but probably not worth the rerun for the rebuttal.

8. **RF rebalanced production maps have NaN cross-model correlation** (current state)
   - `maps_comparison_2023.json` shows median pairwise r = NaN for rf_default_20band and rf_default_6band. NaN happens when std = 0, meaning the RF map has *constant predictions*.
   - This is the same predict-on-X=zeros artifact from bug #3 above, BUT the RF path was supposedly fixed in 1393aef. Worth investigating: maybe the RF model was trained correctly but the production-map inference still has a bug.
   - **TODO** before the deadline: re-run RF inference and verify the parquet has variation.

---

## 9 · The HF dataset: `ValerianFourel/SOCrebuttal`

- **Type**: HF `dataset` repo
- **Size**: 6,113 files / ~5 GB total (the analyzed snapshot was 31 MB JSON+MD; the rest is .parquet predictions, .pth checkpoints, .png figures)
- **Structure** (from the published README):
  ```
  README.md                                    auto-generated
  sweep/<group>/<tag>/kfold_results_summary.json + fold_*_predictions.parquet
  final_models/checkpoints/<run>/*.pth + stats.json + config.json
  final_models/maps/<run>/bavaria_2023_predictions.parquet + summary.json + map.png
  final_models/maps_comparison_2023.{png,md,json}
  code/architectures/SpatiotemporalGatedTransformer/*.py
  code/rebuttal/**/*.py
  docs/REVISION_LOG.md + audited_runs/ + session_log_*.md
  uncertainty/  ← MC-dropout outputs
  ```
- **Publish command**: `python rebuttal/publish_to_hf.py` (idempotent; uploads only changed files)
- **Sweep ranking is missing** from the published repo — `sweep_summarize.py` needs to be re-run after the most recent sweep additions and the result re-published
- **Bundles available via `--include`**: sweep, sweep-checkpoints (heavy, opt-in), finals-checkpoints, finals-maps, finals-figures, code-architectures, code-rebuttal, docs

---

## 10 · Cluster runbook (copy-paste-ready)

```bash
# === SETUP ===
ssh fourel1@jpbl-s01-04   # or current login node
cd /e/project1/scifi/fourel1/SGT/SOCmapping
source ../venv/bin/activate
git checkout bestrun-bands
git pull

# === SUBMIT THE COMPLETE RUN ===
# Spatial-CV sweep (adds new architectures + bands)
python rebuttal/gpu_experiments/spatial_kfold/sweep_submit.py --vanilla --lightweight-transformer

# Production maps (16 runs: 8 archs × 2 bands)
python rebuttal/final_models/submit_finals.py

# Rebalanced production maps (parallel set of 16 runs)
python rebuttal/final_models/submit_finals.py --rebalance

# === MONITORING ===
squeue -u $USER
python rebuttal/final_models/check_inference_status.py --watch
tail -f rebuttal/final_models/maps/infer_all_*.out

# === REVIEWER-RESPONSE ANALYSES (after sweep + finals complete) ===
python rebuttal/reviewer_responses/run_all.py
# Or selectively:
python rebuttal/reviewer_responses/flagship_summary.py --params
python rebuttal/reviewer_responses/bootstrap_ci.py --n-boot 10000
python rebuttal/reviewer_responses/validation_set_stats.py
python rebuttal/reviewer_responses/temporal_sensitivity.py
python rebuttal/reviewer_responses/nn_distances.py
python rebuttal/reviewer_responses/residual_audit.py

# === FIGURES ===
python rebuttal/final_models/compare_maps.py           # non-rebal grid
python rebuttal/final_models/compare_maps.py --rebal   # rebal grid

# === HF PUBLISH ===
python rebuttal/publish_to_hf.py --dry-run    # preview
python rebuttal/publish_to_hf.py              # actually push (~1-2 GB)
python rebuttal/publish_to_hf.py --include sweep-checkpoints  # add the heavy bundle later
```

---

## 11 · What's still pending (before 2026-05-29)

### Compute that's still running or queued

- **`lwt-long` defensive long-training run** (LightweightTransformer d=128 L=1 @ 200 epochs) — should land within ~24h of when submitted. Result expected: Lightweight improves modestly with more training but doesn't close the gap to Vanilla (~80% confidence).
- **Rebalanced production-mapping runs from `--rebalance` submission** — 16 runs (12 NN + 4 trees), ~2-4h wall depending on queue. Status varies.
- **SGT 6-band production map** — the one inference that failed. Retry: `python rebuttal/final_models/infer_bavaria.py --run-name sgt_d128_h4_L1_6band --year 2023`

### Code/analysis tasks not yet done

- **Re-run `sweep_summarize.py`** after the latest sweep additions land and re-publish to HF (the published ranking is stale)
- **Investigate the RF production-map constant-predictions bug** (`rf_default_*` maps have std=0 in `maps_comparison_2023.json`)
- **Decide whether to expose `--sampler-alpha` for individual architectures** (CNNLSTM rebalanced overshot; SGT also high. May need per-arch α tuning OR just exclude CNNLSTM from the rebal figure.)

### Manuscript-writing tasks (highest priority)

- **Cover letter draft** — needs the framing of "we re-evaluated the architecture under honest spatial CV and found a simpler 215k-parameter alternative ties the original 363k SGT on every metric"
- **Section §X (Architecture Comparison) — Discussion**: drop in the flagship_summary table + the CNN-frontend ablation paragraph + the production-map figure pair. The reviewer_responses/ outputs slot into the response letter as supporting evidence.
- **Methods §2.5 update**: change the recommended architecture from SGT to Vanilla (CNN+Transformer) with the parameter count, hyperparameters, and a sentence noting the gating ablation
- **Abstract update**: change "SGT, a deep learning architecture integrating CNNs for spatial feature extraction, Gated Residual Networks for adaptive feature refinement…" to "a lightweight CNN+Transformer hybrid (215k parameters) that we show ties the original gated variant under honest spatial cross-validation". Drop the +0.751 g/kg/yr claim.
- **Tier 1 editorial fixes (T1.7-T1.14, T1.17-T1.18, T1.22-T1.26)** — see the action plan
- **Response letter** — one paragraph per reviewer comment, citing the reviewer_responses/ outputs

### Final-step orchestration

- Heads-up email to editor Budiman Minasny around 2026-05-22 (one week before deadline)
- Submit revision package by 2026-05-29

---

## 12 · Open questions / risks

1. **Will Lightweight long-training (`lwt-long` @ 200 epochs) close the gap to Vanilla?** Most likely no — the cross-fold variance of 0.31 is a generalization problem, not a convergence problem. But the result will be in `sweep/oc150_longtrain/lightweight_transformer_d128_h4_L1/` when the job completes; need to inspect.

2. **CNNLSTM rebalanced overshoot (mean=69.69 g/kg)** — do we exclude CNNLSTM from the rebalanced figure, or rerun with lower α (0.25)? Pragmatic: exclude. The point of the rebal figure is the CNN+Transformer family, not the recurrent variant.

3. **RF production-map constant-predictions bug** — should investigate before the deadline. Might require re-running RF inference. Affects whether RF maps are in the production-map figure at all.

4. **Whether to retrain Vanilla with more epochs as the "production" model** — currently both SGT and Vanilla are trained at 60 epochs (`submit_finals.py` default). Their median best-epoch is ~8, so they're well-converged. But it might be defensible to bump to 100 epochs and re-train for the production maps to fully amortize the early-stopping cost. Probably not necessary.

5. **Geographic generalization claim** — the paper is restricted to Bavaria. Reviewers haven't asked us to test transferability across ecoregions, but if anyone does, T3.5 says it's already future work.

6. **MC-dropout uncertainty map** — exists as a script (`rebuttal/mc_dropout_uncertainty.py`) but not run on the new Vanilla flagship. Could be a cheap win — 1 inference pass, no retraining — to include in the supplementary as a preliminary uncertainty visualization (addresses R3.9 / R4.4).

---

## 13 · Key memory items already saved

These are in the user's persistent memory and apply to every future Claude session on this project:

1. **`bestrun` branch is frozen** — never commit, cherry-pick, or edit on `bestrun`. Ports go to `bestrun-bands` only. Live work happens on `bestrun-bands`.

2. **`train_full` and `run_kfold` have parallel `--model-family` argparse lists** — when adding a new architecture, update BOTH or training silently fails at argparse. Also confirm `_build_model()` in run_kfold dispatches the new family. (Lesson from the b8e1f1e bug above.)

---

## 14 · Quick reference — what to say in a future Claude session

If you (a future Claude) open this project and the user says "let's continue the rebuttal":

- **Branch**: confirm `bestrun-bands` (`git branch --show-current`)
- **Status snapshot**:
  ```bash
  squeue -u $USER
  python rebuttal/final_models/check_inference_status.py
  python rebuttal/gpu_experiments/spatial_kfold/sweep_summarize.py | head -30
  ```
- **What the user typically wants next**: (a) draft response-letter paragraphs from the reviewer_responses/results outputs, (b) update the manuscript with Vanilla as the flagship, (c) handle a new cluster failure, (d) push fresh artifacts to HF.

The user is comfortable with terse technical responses. They prefer side-by-side tables, exact commit hashes, exact file paths. They're aware of the 2026-05-29 deadline and prioritize accordingly. Default to bias for action with brief explanation; don't ask permission for diagnostic commands.

---

## 15 · Citation drop-in (for the manuscript)

For the architecture-recommendation paragraph of the Discussion section:

> **"We recommend a lightweight (~215k parameter) CNN + Transformer hybrid as the default DSM architecture at LUCAS data scale. At matched hyperparameters (d_model = 128, num_heads = 4, num_layers = 1) the hybrid ties our original gated SGT variant (R² mean 0.170 ± 0.072 vs 0.170 ± 0.062, p = NS) under honest 10-fold spatial cross-validation, while requiring 40% fewer parameters (215k vs 363k). At matched parameter count, a transformer-alone variant (Lightweight, 240k parameters; same architecture with the spatial CNN encoder replaced by a flat linear projection) achieves R² mean = -0.064 — a 0.23 R² gap that quantifies the value of the CNN spatial inductive bias at this data scale. A 30× larger pure transformer (SimpleTransformerV2, 11.2M parameters) recovers the hybrid's R² but offers no improvement in cross-fold variance or convergence speed, underscoring that the CNN encoder, not scale, is the relevant lever. The gated residual mechanism that distinguished our original SGT contributes no measurable performance at this data scale; we therefore drop it from the recommended architecture."**

—

*End of context document. Approximate length: 4,800 words. For revision history: this snapshot was produced on 2026-05-20 by Claude Opus 4.7 (1M context) in a single working session spanning ~30 turns. The session's commits are visible on the `bestrun-bands` branch between `b5c1cac` (best-state fix) and `0c87d0e` (this commit).*
