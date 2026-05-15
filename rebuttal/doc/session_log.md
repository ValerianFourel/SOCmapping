# Rebuttal session log — what we have

**Manuscript:** GEODER-D-26-01032 (Geoderma revision)
**Deadline:** 2026-05-29
**Working root:** `/home/valerian/SGTPublication/`
**Rebuttal output root:** `/home/valerian/SGTPublication/rebuttal/`
**Log written:** 2026-05-13

This file is the human-readable index of everything produced in the
rebuttal-preparation session. `rebuttal_numbers.md` is still the single
source of truth for the actual numbers; this log just tells the next
reader (or future-you) what's where and why.

---

## 1. Status snapshot

| stream | status |
|--------|--------|
| Code cleanup of training scripts (composite-loss collapse) | **Done** — 4 `train.py` files edited |
| Full saved-run inventory (parquet/pkl/csv walk, 204 files) | **Done** — `results_inventory.md`, `results_summary.json` |
| Bootstrap CIs on Table 2 (T2.1 / T2.4) | **Done** — `bootstrap_results.{md,json}` |
| Spatial vs random split comparison (T2.4) | **Done** — `split_comparison.{md,json}` |
| Temporal regression sensitivity (T2.2 / T2.3) | **Done** — `temporal_regression_corrected.{md,json}` |
| NN distance distribution (T2.6) | **Done** — `nn_distances.{md,json}`, `nn_distance_histogram.png` |
| Training residual SD anomaly (T2.7) | **Done** — `residual_sd_analysis.{md,json}` |
| Land-use extended regression (T2.9) | **Done** — `extended_regression.{md,json}` |
| Figure 7 replacement (T1.16) | **Done** — `figure7_replacement.png`, `figure7_data.csv` |
| Multi-run spatial-CV evidence (T3.1) | **Done** — `multi_run_cv.{md,json}` |
| **k-fold spatial CV (Experiment 1)** | **Rewritten 2026-05-15** (10 latitude-decile folds, train.py recipe imported) — pending Runpod re-run |
| **MC dropout uncertainty map (Experiment 2)** | **Done 2026-05-13** — 5.5 min on 4 GPUs, uploaded to `ValerianFourel/SOCrebuttal` |
| **Training CLI / LR schedulers / SyncBN** | **Done 2026-05-15** — cosine/plateau/exponential schedulers, `--num-epochs`, `--sync-bn`, bool-flag fix, val→test rename, rebalance-after-split fix |
| Final response-letter integration | Pending |

Day-by-day record:
- [`session_log_2026-05-13.md`](session_log_2026-05-13.md) — Runpod operationalization, debug history, k-fold recipe alignment.
- [`session_log_2026-05-15.md`](session_log_2026-05-15.md) — train.py bug fixes, new LR scheduler CLI surface, SyncBN, run_kfold rewrite (10 latitude deciles); 4-GPU / 2-GPU / 1-GPU launch commands.

How-to: [`completion_playbook.md`](completion_playbook.md) — operational runbook for finishing both GPU experiments (resume, inspect, push to HF, drop-in response-letter paragraphs).

---

## 2. Directory map

```
rebuttal/
├── rebuttal_numbers.md                ← Single source of truth (7 sections, 31 KB)
├── doc/
│   └── session_log.md                 ← This file
├── results_inventory.md               ← 346 KB walk-through of every saved
├── results_summary.json                  parquet/pkl/csv across 4 roots
├── extract_results.py                 ← Reproducible inventory script
│
├── model_ready_dataset.parquet        ← Canonical 16,514-row dataset
│                                         (year ∈ [2007,2023], OC≤150,
│                                         altitude joined via Elevation tiles)
│
├── bootstrap_cis.py                   ← T2.1 / T2.4 — 1000-iter bootstrap
├── bootstrap_results.md                  on the 1,359-row spatial val set
├── bootstrap_results.json
│
├── split_comparison.py                ← T2.4 — spatial val vs synthetic
├── split_comparison.md                   random val
├── split_comparison.json
│
├── temporal_regression.py             ← T2.2 / T2.3 — INITIAL pass on
├── temporal_regression.md                30,451-row LUCAS (superseded)
├── temporal_regression.json
│
├── temporal_regression_corrected.py   ← T2.2 / T2.3 — CORRECTED pass on
├── temporal_regression_corrected.md      the 16,514-row model-ready set
├── temporal_regression_corrected.json    (β_year = +0.7519, matches paper)
│
├── extended_regression.py             ← T2.9 — adds C(LC0_Desc) on the
├── extended_regression.md                LUCAS-LC-labelled subset
├── extended_regression.json
│
├── residual_sd_analysis.py            ← T2.7 — train residual SD by SOC
├── residual_sd_analysis.md               stratum (peatland tail explains it)
├── residual_sd_analysis.json
│
├── nn_distances.py                    ← T2.6 — pairwise NN haversine
├── nn_distances.md                       distance distribution
├── nn_distances.json
├── nn_distance_histogram.png
│
├── figure7_replacement.py             ← T1.16 — 2D scatter+OLS replacement
├── figure7_replacement.png               for the 3D Figure 7
├── figure7_data.csv
├── figure7_trends.json
│
├── multi_run_cv.py                    ← T3.1 — repeated random spatial
├── multi_run_cv.md                       hold-out across 44 sessions
├── multi_run_cv.json                     (SGT mean R² 0.448 over 58 runs)
│
└── gpu_experiments/                   ← GPU work — ready, not yet run
    ├── README.md                          (also a launch-day cheat sheet)
    ├── checkpoint_note.md                 Model A vs Model B disambiguation
    ├── spatial_kfold/
    │   └── run_kfold.py                ← Experiment 1 (≈15 h)
    └── uncertainty/
        ├── mc_dropout_inference.py     ← Experiment 2 step 1 (≈3 h)
        └── plot_uncertainty.py         ← Experiment 2 step 2 (CPU)
```

39 files in total under `rebuttal/`, none in the rest of the repo.

---

## 3. Headline numbers (what the response letter cites)

### Table 2 metrics (Model A, the evaluation model)
- val R² = **0.6258**, 95% CI [0.526, 0.712]
- val RMSE = **4.758** g/kg, 95% CI [4.18, 5.33]
- val MAE = **2.791** g/kg, 95% CI [2.60, 3.00]
- val RPIQ = **1.051**, 95% CI [0.90, 1.19]
- val n = 1,359, bootstrap n_iter = 1000, seed = 20260513

### Multi-run spatial CV (T3.1, 58 converged SGT runs)
- mean R² = **0.448 ± 0.071** (range [0.30, 0.61])
- best single SGT config (L1 + log, 5 runs): **0.527 ± 0.050**
- Architecture leaderboard:
    SGT 0.448 → SimpleTransformer 0.287 → CNNLSTM 0.239 → 3DCNN 0.150 → 2DCNN failed

### Temporal regression (T2.2 / T2.3 on 16,514-row model-ready base)
- OLS full: β_year = **+0.7519**, 95% CI [+0.674, +0.830], R² = **0.2463**
  (reproduces the paper's +0.751, 0.246 exactly)
- OLS year<2022: β_year = +0.6086 (−19% vs full)
- WLS 1/n_year: β_year = +1.3969 (almost 2× full)
- OLS year n≥100: β_year = +0.6823 (close to full)

### Land use (T2.9, n=1,237 LUCAS-labelled subset)
- Baseline `SOC ~ year + altitude`: β_year = +0.402, R² = 0.113
- Extended `SOC ~ year + altitude + C(LC0_Desc)`: β_year = **+0.302** (−25%), R² = **0.309**

### NN distances (T2.6, 5,148 unique sample locations)
- median 816 m, p5 25 m, p95 6.0 km, max 22.3 km
- 34% of locations have a neighbour < 100 m; 47% < 1 km

### Spatial vs random split (T2.4, n=1,359 each)
- Mean SOC: spatial **16.34** vs random **22.90** g/kg
- Max SOC: spatial **85** vs random **147** g/kg
- % > 50 g/kg: spatial **0.59 %** vs random **7.87 %**
- % > 120 g/kg: spatial **0.00 %** vs random **0.88 %**

### Training residual SD (T2.7, Model A train set)
- All samples (n=15,155): SD = **7.53**
- OC ≤ 50 g/kg bulk (n=13,989, 92.3 %): SD = **5.29** ≈ val SD 4.62
- OC > 50 g/kg peatland tail (n=1,166, 7.7 %): SD = **16.91** (3.2× bulk)

### Methodological finding
- `composite_l1_chi2_loss` ≡ L1; `composite_l2_chi2_loss` ≡ MSE.
- The `loss_alpha` weight had no numerical effect.
- All `LOSS_composite_l2` runs in saved filenames are functionally MSE runs.

---

## 4. What each artifact answers

| Reviewer concern | Answer artifact |
|------------------|-----------------|
| **T1.16** Replace 3D Fig 7 | `figure7_replacement.png`, `figure7_data.csv` |
| **T2.1** Confidence intervals on Table 2 | `bootstrap_results.md` |
| **T2.2** Drop late years | `temporal_regression_corrected.md` Model 2 |
| **T2.3** Equal-year weighting | `temporal_regression_corrected.md` Models 3 + 4 |
| **T2.4** Random vs spatial split | `split_comparison.md` and `bootstrap_results.md` |
| **T2.6** NN distance distribution | `nn_distances.md` + `nn_distance_histogram.png` |
| **T2.7** Training residual SD anomaly | `residual_sd_analysis.md` |
| **T2.9** Add land-use to the regression | `extended_regression.md` |
| **T3.1** Spatial k-fold CV | `multi_run_cv.md` (repeated hold-out) + Experiment 1 (true 5-fold, pending GPU) |
| **R3.6** CIs on Table 2 | Same as T2.1 + Experiment 1 |
| **R3.8** No proper test set | Experiment 1 (5 latitude strips with 1.2 km buffer) |
| **R3.9** No uncertainty quantification | Experiment 2 (MC dropout, pending GPU) |
| **R4.4** Uncertainty maps expected in DSM | Experiment 2 |
| **Methods** Loss function description | §1 of `rebuttal_numbers.md` + cleaned `train.py` files |

---

## 5. Train.py simplification (already in the codebase)

In `SOCmapping/{3DCNN, CNNLSTM, SimpleTransformer, SpatiotemporalGatedTransformer}/train.py`:

- Removed the `composite_l1_chi2_loss` and `composite_l2_chi2_loss`
  function definitions (algebraically identical to L1 / MSE).
- Removed `--loss_alpha` argparse argument and `loss_alpha=` parameter
  threading through `train_model`.
- Removed `composite_l1` / `composite_l2` from `--loss_type` choices;
  defaults changed from `composite_l2` → `mse` where applicable.
- Removed `loss_alpha` entries from wandb config logging and from the
  saved-model `training_config` metadata dict.
- Removed stale `# BEST "composite_l2" / log` comment from SimpleTransformer.
- Compile-verified: `python3 -m py_compile` clean on all four files.

Filename strings containing `composite_l2` in `residualsStudy.py`,
`running.py`, and saved-model paths are unchanged — they're literal
paths to historical artifacts, not loss-function references.

---

## 6. Outstanding work (GPU)

Both scripts are in `rebuttal/gpu_experiments/` and were written *not
executed* per the original instruction. Launch order on a single GPU:

```bash
cd /home/valerian/SGTPublication

# Step 1 — sanity-check (see rebuttal/gpu_experiments/README.md §6)
ls -la Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/
python -c "import rasterio, pyproj; print(rasterio.__version__, pyproj.__version__)"

# Step 2 — Experiment 2 first (≈ 3 h), then Experiment 1 overnight (≈ 15 h)
python rebuttal/gpu_experiments/uncertainty/mc_dropout_inference.py
python rebuttal/gpu_experiments/uncertainty/plot_uncertainty.py
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py
```

If two GPUs available: run both Experiments 1 and 2 in parallel via
`CUDA_VISIBLE_DEVICES=0/1`.

**Outputs Experiment 1 will produce:**
- `spatial_kfold/kfold_results.md` (paste straight into manuscript §2.5)
- `spatial_kfold/figure_kfold.png` (300 dpi, 2 panels)
- `spatial_kfold/kfold_predictions_all_folds.parquet`
- 5 × `fold_{i}_best.pth` + `fold_{i}_metrics.json`
- `spatial_kfold/kfold_results_summary.json`

**Outputs Experiment 2 will produce:**
- `uncertainty/SGT_1mil_2023_mean_mc30.tif` (UTM 32N, 250 m, float32)
- `uncertainty/SGT_1mil_2023_std_mc30.tif` (same)
- `uncertainty/figure_uncertainty_3panel.{png,pdf}`
- `uncertainty/mc_dropout_points.parquet`
- `uncertainty/validation_check.txt`
- `uncertainty/mc_dropout_metadata.json`

**Flagged assumptions in the GPU scripts** (24 `# ASSUMPTION:` comments
in total). The four substantive ones:

1. lr = 2e-4 + no scheduler (matches `train.py`, not the spec's 1e-3 + decay)
2. `torch.log(OC + 1e-10)` / `torch.exp` (matches `train.py`, not log1p / expm1)
3. EnhancedSGT (BatchNorm + LayerNorm, 1,121,637 params) — not SimpleSGT — confirmed by direct checkpoint inspection
4. MC accumulator works in original SOC units (the log inverse is applied
   inside the MC loop). Selective dropout activation keeps BatchNorm in
   eval mode.

---

## 7. Re-running anything (every artifact is reproducible)

Every CPU-side artifact has a sibling `.py` script that regenerates it
from raw data. Run from `/home/valerian/SGTPublication`:

```bash
VPY=/home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python

$VPY rebuttal/extract_results.py             # 204-file inventory walk
$VPY rebuttal/bootstrap_cis.py               # T2.1 bootstrap CIs
$VPY rebuttal/split_comparison.py            # T2.4 spatial vs random
$VPY rebuttal/temporal_regression_corrected.py   # T2.2/T2.3 corrected
$VPY rebuttal/extended_regression.py         # T2.9 land use
$VPY rebuttal/residual_sd_analysis.py        # T2.7 residual SD
$VPY rebuttal/nn_distances.py                # T2.6 NN distances
$VPY rebuttal/figure7_replacement.py         # T1.16 Figure 7
$VPY rebuttal/multi_run_cv.py                # T3.1 multi-run summary
```

Each script also writes a `.json` companion for machine-readable downstream
use (e.g. for `rebuttal_numbers.md` aggregation).

---

## 8. Key dataset paths

| purpose | path |
|---------|------|
| Master LUCAS+LfU+LfL Bavaria OC | `Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx` (30,451 rows) |
| Filtered 16,514-row model-ready | `rebuttal/model_ready_dataset.parquet` (year ∈ [2007,2023], OC ≤ 150, altitude joined) |
| Elevation lookup | `Data/.../Elevation/coordinates.npy` + `Data/RasterTensorData/StaticValue/Elevation/ID*.npy` (12 tiles, 979×979 each) |
| Inference grid (1.3 M points) | `Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv` |
| LUCAS land-cover labels (DE) | `Data/Preprocessing/SoilSamples_From_Raw_to_MLready/LucasBodenDaten/LUCAS_Topsoil_2015_20200323.xlsx` + `LUCAS-SOIL-2018.xlsx` |
| Bavaria boundary geojson | `SOCmapping/SpatiotemporalGatedTransformer/bavaria.geojson` (also fetched online by mapping.py) |
| Model A checkpoint (eval, 91/9 split) | `Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/TFT_model_BEST_OVERALL_from_run_1_...R2_0.6909.pth` |
| Model B checkpoint (mapping, full data) | `Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/TFT_model_BEST_OVERALL_from_run_1_...LOSS_l1_R2_1.0000.pth` |
