# Audited rebuttal runs

This folder collects **fully-auditable re-trainings** of the SGT model with
the hyperparameters and recipe locked to the Model A v2 audit (see
[`../table_2_corrected.md`](../table_2_corrected.md) and
[`../proper_r2_all_projects_FULL.md`](../proper_r2_all_projects_FULL.md)).

Every subfolder under here is named:

```
SGT_{small|big}_audited_{YYYYMMDD_HHMMSS}_{git_sha}/
```

— so any run can be unambiguously identified from its directory name alone.
Inside each run folder:

```
SGT_{size}_audited_{ts}_{sha}/
├── README.md                          ← human-readable lineage
├── experiment_config.json             ← full args, hyperparams, lineage
├── data/                              ← train/val parquets + norm stats
│   ├── train_val_data_run_1.parquet
│   ├── train_val_data_run_2.parquet
│   ├── ...
│   └── normalization_stats_run_*.pkl
├── models/                            ← per-run best checkpoints
│   ├── SGT_{size}_audited_run1_R2_*.pth
│   ├── SGT_{size}_audited_run2_R2_*.pth
│   └── ...
├── results/                           ← per-run residual analyses
└── EXPERIMENT_SUMMARY.txt
```

Every checkpoint also carries `rebuttal_audited: True`, `git_sha`, the model
class name, and the full args dict in its serialized payload — so any single
`.pth` is self-describing without needing the surrounding files.

## How to launch a fresh audited run

From `SOCmapping/SpatiotemporalGatedTransformer/`:

```bash
# Big model (Model A architecture, EnhancedSGT:
#   d_model=128, num_heads=4, num_encoder_layers=3, expansion_factor=4
#   → 1,120,546 trainable params; verified against saved .pth)
accelerate launch --num_processes 4 train.py \
    --rebuttal --model-size big --num-runs 4

# Small model (SimpleSGT: d_model=128, num_heads=2, 1 transformer layer
#   → 360,593 trainable params)
accelerate launch --num_processes 4 train.py \
    --rebuttal --model-size small --num-runs 4
```

The `--rebuttal` flag locks these hyperparameters automatically (you do
not need to pass them):

| | locked value |
|---|---|
| `--lr` | `2e-4` |
| `--dropout_rate` | `0.3` |
| `--loss_type` | `mse` |
| `--target_transform` | `normalize` |
| `--target-val-ratio` | `0.08` |
| `--distance-threshold` | `1.2` km |
| `--target-fraction` | `0.75` |
| `--use_validation` | `True` |
| `--save_train_and_val` | `True` |
| `--num_heads` | **`4`** for big / `2` for small |
| `--num_layers` | **`3`** for big (= EnhancedSGT `num_encoder_layers=3`) / `1` for small |
| **trainable params** | **big = 1,120,546** / **small = 360,593** |
| effective batch | `num_processes × per-gpu-batch-size × accum_steps` (auto-scales to 2048) |

You can still control parallelism via `--num_processes` / `--per-gpu-batch-size`,
and the number of runs via `--num-runs`.

## Reproducibility contract

Every audited run captures, in `experiment_config.json`:

- `git_sha` of the SOCmapping repo at training time
- `model_class` (`"EnhancedSGT"` or `"SimpleSGT"`)
- `model_size` (`"big"` / `"small"`)
- `audited_hyperparameters` dict (the locked values listed above)
- `args` dict (full CLI args after preset overrides)
- `lineage` string pointing to the historic Model A v2 it reproduces

The `README.md` inside each run folder reproduces this in a human-readable
form including the exact re-run command.

## Expected results

For the **big** model (1mil, normalize, MSE) we expect to reproduce
within run-to-run variance:

| | proper R² | Pearson r² | RMSE (g/kg) | bias (g/kg) | RPD |
|---|---|---|---|---|---|
| **Target (Model A v2, 2025-05-26)** | 0.594 | 0.626 | 4.76 | +1.13 | 1.57 |
| 95 % bootstrap CI | [0.450, 0.693] | [0.526, 0.712] | [4.18, 5.33] | [0.88, 1.37] | — |

A successful audited run should land its best run's val R² (proper, from
`residualsStudy.py`) inside that CI. If it lands outside, investigate
before publishing.

For the **small** model (SimpleSGT, 360,593 params) we don't have a
direct historic comparison — this is the architecture-ablation row, run
specifically to support the "the GRN-blocks + multi-layer transformer
matter" framing in the rebuttal. Expect moderately worse than big (the
big model has ~3× more parameters AND the BatchNorm stack the v1 paper
ablated against).

A note on why "small" is still 360 k and not the ~50 k I earlier mis-stated:
the bulk of SimpleSGT's parameters come from the `(feature_dim=512 ×
d_model=128)` projection inside the GRN block + the final `(time_steps ×
d_model = 640 → 64 → 1)` head, not from the single transformer encoder
layer.

## Post-training: residual study + uploading

```bash
# Residual study for each saved .pth (writes analysis_results.pkl with both
# proper R² and Pearson r² since e414a47).
python residualsStudy.py \
    --model-path rebuttal/audited_runs/SGT_<size>_audited_<ts>_<sha>/models/SGT_<size>_audited_run1_R2_*.pth

# (Optional) Upload to the SOCrebuttal HF dataset alongside the other
# rebuttal artifacts.
python rebuttal/gpu_experiments/upload_to_hf.py
```
