# GPU experiments for the Geoderma rebuttal (GEODER-D-26-01032)

Deadline: **2026-05-29**.

## Context

These two experiments address the only remaining GPU-dependent reviewer
requests for this revision. Every CPU-based analysis is already complete
in `rebuttal/`; the headline numbers (Table 2 bootstrap CIs, β_year
sensitivities, NN-distance distribution, residual-SD breakdown, land-use
regression, spatial-vs-random split comparison, T3.1 multi-run spatial-CV
evidence) live in `rebuttal/rebuttal_numbers.md`.

| Reviewer concern | Experiment |
|------------------|------------|
| **R1.3** single spatial split is weak | Experiment 1 |
| **R3.6** no confidence intervals on Table 2 | Experiment 1 |
| **R3.8** no proper test set | Experiment 1 |
| **R3.9** no uncertainty quantification | Experiment 2 |
| **R4.4** uncertainty maps expected in DSM | Experiment 2 |

The two existing checkpoints (Model A, Model B) and which experiment
uses which are documented in `checkpoint_note.md`. **No retraining of the
existing checkpoints is needed.** Experiment 1 trains 5 fresh models
from scratch (one per fold). Experiment 2 uses the already-trained
Model B for stochastic inference only.

## Files

```
rebuttal/gpu_experiments/
├── README.md                    ← this file
├── checkpoint_note.md           ← Model A vs Model B disambiguation
├── spatial_kfold/
│   └── run_kfold.py             ← Experiment 1
└── uncertainty/
    ├── mc_dropout_inference.py  ← Experiment 2, step 1: inference
    └── plot_uncertainty.py      ← Experiment 2, step 2: figure
```

After running, the outputs land in those same directories alongside the
scripts (see each script's docstring for the full output list).

## Experiment 1 — Spatial 5-fold CV (≈ 15 GPU hours)

- **Script:** `spatial_kfold/run_kfold.py`
- **Command:**
  ```bash
  cd /home/valerian/SGTPublication
  python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py
  ```
- **Outputs (for the manuscript):**
  - `spatial_kfold/kfold_results.md` — paste this Table into manuscript §2.5
  - `spatial_kfold/figure_kfold.png` — new Figure (supplement or main)
  - `spatial_kfold/kfold_predictions_all_folds.parquet` — per-row test
    predictions (lon, lat, OC_actual, OC_predicted, fold_id, year, altitude)
- **Answers:** R1.3, R3.6, R3.8

## Experiment 2 — MC Dropout uncertainty (≈ 3 GPU hours)

- **Scripts:**
  - `uncertainty/mc_dropout_inference.py` — run first; writes GeoTIFFs + parquet
  - `uncertainty/plot_uncertainty.py` — run second; builds the 3-panel figure
- **Commands:**
  ```bash
  cd /home/valerian/SGTPublication
  python rebuttal/gpu_experiments/uncertainty/mc_dropout_inference.py
  python rebuttal/gpu_experiments/uncertainty/plot_uncertainty.py
  ```
- **Outputs (for the manuscript):**
  - `uncertainty/SGT_1mil_2023_mean_mc30.tif` — MC mean prediction (UTM 32N, 250 m)
  - `uncertainty/SGT_1mil_2023_std_mc30.tif` — MC uncertainty (std), same grid
  - `uncertainty/figure_uncertainty_3panel.png` — Figure for §4.6 (300 dpi)
  - `uncertainty/figure_uncertainty_3panel.pdf` — vector for journal submission
- **Answers:** R3.9, R4.4

## Parallelisation — built in

Both scripts shard themselves across every visible CUDA device with no
extra launcher. Plain `python script.py` is enough:

- **Experiment 1** orchestrates one worker subprocess per GPU and
  schedules the 5 folds across them (4 folds in parallel, then fold 4
  on the first free GPU). Total wall time ≈ 2 × (single-fold time)
  on 4 GPUs.
- **Experiment 2** shards the 1.3 M Bavaria inference grid into
  equal-sized contiguous slices, one per GPU, and concatenates the
  shards back into a single GeoTIFF/parquet at the end. Total wall
  time ≈ (single-GPU time) / N_GPUs.

Override with `--gpus 0,1` (subset) or `--sequential` (debug). Per-GPU
stdout lands in `{spatial_kfold,uncertainty}/worker_logs/<id>_gpu_<g>.log`.

If you really want to run both experiments concurrently on disjoint
GPU subsets:

```bash
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py        --gpus 0,1 &
python rebuttal/gpu_experiments/uncertainty/mc_dropout_inference.py --gpus 2,3 &
wait
```

The k-fold checkpoints are saved per fold so a crash on fold N does
not invalidate folds 0..N-1.

## What to do with the outputs

- **`kfold_results.md`** → paste the table into manuscript §2.5 and cite
  the row in the responses to R1.3 and R3.6.
- **`figure_kfold.png`** → new Figure in the supplement (or main text if
  the editor agrees).
- **`figure_uncertainty_3panel.png`** → new Figure in manuscript §4.6.
- **All four** → referenced in the response letter with explicit page /
  line numbers.

## Important assumptions flagged in the scripts

Every `# ASSUMPTION:` comment in the two scripts marks a place where the
spec the user provided and the actual codebase disagree, or where a default
needed to be chosen. Skim them before launching — they are the only
places where the project lead might want to override the default.

The four notable ones:

1. **Optimizer / lr.** Spec said Adam @ 1e-3 + exponential decay.
   `train.py` uses Adam @ 2e-4 with no scheduler. Scripts follow `train.py`
   (the spec's overriding instruction was "match the original exactly").
2. **Log transform.** Spec said `log1p` / `expm1`. `train.py` uses
   `torch.log(OC + 1e-10)` / `np.exp(pred)`. Scripts follow `train.py`.
3. **Model class.** Spec referred to "the SGT 1.1M". `train.py` imports
   `SimpleSGT` but the saved 1.1 M checkpoints are `EnhancedSGT` —
   `running.py` has `from EnhancedSGT import EnhancedSGT as SimpleSGT`.
   Scripts use `EnhancedSGT` (verified by state-dict shape and parameter
   count = 1,121,637).
4. **Normalisation in MC dropout.** The MC accumulator is in *original
   SOC units* (the log transform is inverted inside the MC loop). The
   spec said this explicitly; flagging here only to note it.

## Sanity checks before running

```bash
# 1. confirm Model B exists at the documented path
ls -la /home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/

# 2. confirm model_ready_dataset.parquet exists
ls -la /home/valerian/SGTPublication/rebuttal/model_ready_dataset.parquet

# 3. confirm rasterio + pyproj are importable (for Experiment 2 GeoTIFFs)
python -c "import rasterio, pyproj; print(rasterio.__version__, pyproj.__version__)"

# 4. confirm the EnhancedSGT module imports and parameter count matches
python -c "import sys; sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer'); from EnhancedSGT import EnhancedSGT; m = EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5, d_model=128); print(sum(p.numel() for p in m.parameters() if p.requires_grad))"
# expected: 1121637
```
