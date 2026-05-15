# Day log — 2026-05-15

**Manuscript:** GEODER-D-26-01032 (Geoderma revision)
**Deadline:** 2026-05-29 (14 days out)
**Commits:** 9 today, `6507975..4a14800`, all on `main`, 9 ahead of `origin/main` as of EOD.

This log is thematic (changes grouped by concept, not chronologically). For the previous session's narrative see [`session_log_2026-05-13.md`](session_log_2026-05-13.md); for the artifact index see [`session_log.md`](session_log.md).

---

## 1. Overview

Three files touched: `SpatiotemporalGatedTransformer/train.py`, `SpatiotemporalGatedTransformer/balancedDataset.py`, and `rebuttal/gpu_experiments/spatial_kfold/run_kfold.py`. Two changes fix bugs that were silently degrading behaviour; six add new CLI controls (epoch override, LR schedulers, BN sync); one is a full rewrite of `run_kfold.py` to use the new `train.py` recipe across 10 latitude-decile folds.

After this session the training pipeline gives:
- balanced ~18 k-row train half (was: 15 k unbalanced — orphaned-call bug),
- held-out test set with metrics per epoch (renamed from `val_*`),
- cosine / exponential / plateau LR schedules with proper DDP synchronization,
- SyncBatchNorm option for cross-`num_processes` reproducibility,
- `run_kfold.py` that mirrors `train.py`'s CLI 1:1 and shares its training loop.

---

## 2. Group A — Bugs fixed (3 commits)

### A.1 — `6507975` rebalance train half AFTER spatial split + rename val→test

**Bug.** `train.py:695` computed `create_balanced_dataset(df, use_validation=False)` to produce a ~20 k balanced train set, but `train.py:705` immediately overwrote `train_df` with the spatial-aware split returned by `create_validation_train_sets(df, ...)`. So in the default `--use_validation=True` path, the balanced df was thrown away and the model trained on the unbalanced ~15 k post-split train half. The user's "we want 20 k like before" assumption was correct in the codepath — just unreachable via the default flag.

**Fix.** Reordered the pipeline so the spatial test split runs first (no leakage), then `create_balanced_dataset` rebalances only the train half via the same 128-qcut / `min_ratio=3/4` per-OC-bin logic. Runtime banner now prints `Run N train_df rebalanced: 15194 → ~18000 (+2806 from per-bin oversample); test_df held out: 1321`.

**Rename.** Every `validation` / `val` reference → `test` (since the held-out set always served as a test set, not a validation set). Spans:
- CLI: `--use_validation` → `--use_test`, `--target-val-ratio` → `--target-test-ratio`, `--save_train_and_val` → `--save_train_and_test`.
- Vars: `val_df`/`val_loader`/`val_R²`/`val_outputs`/`val_targets`/etc. → `test_*`.
- Wandb keys: `val_loss` → `test_loss`, `val_size` → `test_size`, `use_validation` → `use_test`.
- Files: `final_validation_df.parquet` → `final_test_df.parquet`, `train_val_data_run_N.parquet` → `train_test_data_run_N.parquet`.
- Function: `create_validation_train_sets` → `create_test_train_sets`. `create_balanced_dataset`'s internal `use_validation` arg renamed to `with_test`.

### A.2 — `e688d5b` fix `--use_test` / `--save_train_and_test` bool flags

**Bug.** Argparse with `type=bool` is a footgun. The flag required a value (`--use_test True/False`), so passing it alone errored `expected one argument`. Worse, `--use_test False` evaluated to `True` because `bool("False")` is `True` in Python — the only string that argparse-bool coerces to `False` is the empty string.

**Fix.** Switched both flags to `action=argparse.BooleanOptionalAction` (Python 3.9+). Bare `--use_test` enables, `--no-use_test` disables. Same for `--save_train_and_test`.

### A.3 — `ff72685` gate `init_process_group` on `WORLD_SIZE > 1`

**Bug.** `train.py:41` unconditionally called `torch.distributed.init_process_group(backend='nccl', ...)` at module load. Fine for `accelerate launch --multi_gpu` (which sets `RANK`/`WORLD_SIZE`/etc. via torchrun), but single-process launches via `accelerate`'s `simple_launcher` left those env vars unset → `ValueError: environment variable RANK expected, but not set`.

**Fix.** Guarded with `if int(os.environ.get("WORLD_SIZE", "1")) > 1:`. Single-process `python train.py` and `accelerate launch --num_processes 1 train.py` now work. Multi-GPU launches unchanged (Accelerator initializes the process group when prepared).

---

## 3. Group B — New training CLI controls (4 commits)

### B.1 — `c2c1d28` `--num-epochs` CLI override

`num_epochs` was hardcoded in `config.py:28` (270 with test, 320 without). To bump it you had to edit config. Added `--num-epochs N`; defaults preserved when unset. CLI value wins over both config defaults.

### B.2 — `7008698` cosine / exponential LR scheduler

`train.py` had no LR scheduler. Added:

| flag | does what |
|---|---|
| `--lr-scheduler {none, cosine, cosine_warm_restarts, exponential}` | choose schedule (default `none` preserves old behaviour) |
| `--lr-min` | floor / `eta_min` for cosine schedules |
| `--lr-gamma` | per-epoch multiplier for exponential |
| `--lr-restart-T0` | cycle length for warm restarts |

Scheduler steps once per epoch after eval/logging. Current LR logged to wandb (`lr` key) and printed in the per-epoch banner. Built before `accelerator.prepare()` so accum_steps doesn't shift the step cadence.

### B.3 — `c85f3ce` `plateau` LR scheduler (ReduceLROnPlateau)

Cosine over 1000 epochs decays "too aggressively" in the last 10 % (lr falls below 5 % of starting lr after epoch 900). Added an adaptive alternative:

| flag | does what |
|---|---|
| `--lr-scheduler plateau` | watch a metric, halve LR when it plateaus |
| `--plateau-monitor {pearson_r2, r_squared, test_loss, mse, rmse, mae}` | metric watched (default `pearson_r2`) |
| `--plateau-patience N` | epochs without improvement before LR cut (default 20) |
| `--plateau-factor F` | multiplier on trigger (default 0.5) |

Mode (`max`/`min`) auto-selected from monitor name. Cross-rank consistency: metric is computed on rank 0, broadcast via `accelerator.gather()[0]` before stepping, NaN-guarded.

### B.4 — `47267ac` add `mse` to `--plateau-monitor` choices

`mse` was being computed and logged per-epoch but missing from the argparse `choices` enum. Passing `--plateau-monitor mse` silently fell back to `pearson_r2` via `.get(default)`. Closed the gap. (Practical note: `mse` and `rmse` produce identical plateau decisions because they're monotonically related.)

---

## 4. Group C — Cross-`num_processes` reproducibility (1 commit)

### C.1 — `4a14800` `--sync-bn` flag (SyncBatchNorm)

**Problem.** `EnhancedSGT.spatial_encoder` has 3 `nn.BatchNorm2d` layers. PyTorch's default BN computes statistics per-GPU, not across ranks. When `num_processes` changes (4×512 → 2×1024 → same effective batch 2048), each GPU's BN sees a different sample count after the `B*T` flatten → different running mean/var → different forward activations → divergent training trajectories over 1000 epochs.

**Fix.** `--sync-bn` calls `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` before optimizer build. BN running stats then all-reduce across ranks every step; per-GPU batch size no longer affects BN behaviour. ~5–10 % slower per step. Default off (preserves bit-equivalence with prior runs that didn't use it). Recommended whenever comparing reruns across different `num_processes` values.

---

## 5. Group D — `run_kfold.py` rewrite (1 commit)

### D.1 — `3d36ecd` kfold: latitude deciles, train.py recipe, sequential single-GPU

Old `run_kfold.py` (1207 lines) had its own copy of the training loop (drifted from `train.py`'s new features), hardcoded recipe constants, a `subprocess.Popen` orchestrator that sharded folds across 4 GPUs, and 5 equal-latitude-**span** folds (fold sample counts ranged 2.5–4 k, unbalanced).

New `run_kfold.py` (778 lines):

| change | from → to |
|---|---|
| Fold geometry | 5 equal-latitude-**span** strips → **10 latitude-decile** strips (np.quantile on `GPS_LAT`, ~10 % of points per fold) |
| Training loop | hand-coded copy → **imports `train_model` from `train.py`** so recipe stays in sync (cosine LR, gradient accumulation, log/normalize inverse, R²-in-SOC) |
| CLI surface | env-var-only → **mirrors `train.py` 1:1** plus k-fold-specific flags |
| Multi-GPU | subprocess.Popen orchestrator → **sequential single-GPU** (cleaner; user can still run multiple `--fold N` in parallel windows) |
| Wandb side effect | independent runs → `os.environ.setdefault('WANDB_MODE', 'disabled')` before `import wandb` so the imported `train_model`'s `wandb.log` calls are no-ops in kfold mode (no account needed) |

K-fold-specific flags: `--num-folds N` (default 10), `--fold-buffer-km KM` (default 1.2), `--fold N` (single-fold mode for debugging), `--seed-base`, `--rebalance-n-bins`, `--rebalance-min-ratio`, `--skip-figure`.

Output schema preserved: `fold_{i}_best.pth`, `fold_{i}_metrics.json`, `kfold_predictions_all_folds.parquet`, `kfold_results.md`, `kfold_results_summary.json`, `figure_kfold.png` — `progress.py` and downstream aggregators don't break.

---

## 6. New CLI flag reference

```bash
# Epochs
--num-epochs N

# LR schedules
--lr-scheduler none|cosine|cosine_warm_restarts|exponential|plateau
--lr-min FLOAT
--lr-gamma FLOAT
--lr-restart-T0 INT

# Plateau-specific
--plateau-monitor pearson_r2|r_squared|test_loss|mse|rmse|mae
--plateau-patience N
--plateau-factor F

# Multi-process reproducibility
--sync-bn

# Bool flags (proper BooleanOptionalAction)
--use_test / --no-use_test
--save_train_and_test / --no-save_train_and_test

# K-fold-only (run_kfold.py)
--num-folds N
--fold-buffer-km KM
--fold N
--seed-base INT
--rebalance-n-bins N
--rebalance-min-ratio F
--skip-figure
```

---

## 7. Launch commands

All commands assume:

```bash
cd /workspace/SOC/SOCmapping/SpatiotemporalGatedTransformer
source /workspace/SOC/SOCmapping/rebuttal/gpu_experiments/.venv/bin/activate
export SOC_DATA_DIR=/workspace/SOC/Data/Data
# Optional — keep wandb online if you've already done `wandb login`:
# unset WANDB_MODE
# Otherwise disable to skip the interactive prompt:
export WANDB_MODE=disabled
```

### 7.1 — 4 GPUs (canonical recipe — matches the wandb config from `7008698`)

Effective batch = 4 × 512 × 1 = 2048. This is the reference run.

```bash
accelerate launch \
    --num_processes 4 --num_machines 1 --multi_gpu \
    --mixed_precision no --dynamo_backend no \
    train.py \
    --model-size big --num_heads 4 --num_layers 3 \
    --hidden_size 128 --dropout_rate 0.3 \
    --lr 2e-4 \
    --per-gpu-batch-size 512 --effective-batch-size 2048 \
    --loss_type l1 --target_transform log \
    --num-runs 5 --num-epochs 1000 \
    --use_test --save_train_and_test \
    --lr-scheduler cosine --lr-min 1e-6
```

Expected banner:
```
[gradient-accumulation]  num_gpus=4  per_gpu_batch=512  accum_steps=1  effective_batch=2048
LR schedule: cosine  (lr=0.0002, lr_min=1e-06, ..., T_max=1000)
```

### 7.2 — 2 GPUs (option A: per-gpu doubled, accum=1)

Effective batch = 2 × 1024 × 1 = 2048. Largest per-GPU batch; uses ~2 GB VRAM each.

```bash
accelerate launch \
    --num_processes 2 --num_machines 1 --multi_gpu \
    --mixed_precision no --dynamo_backend no \
    train.py \
    --model-size big --num_heads 4 --num_layers 3 \
    --hidden_size 128 --dropout_rate 0.3 \
    --lr 2e-4 \
    --per-gpu-batch-size 1024 --effective-batch-size 2048 \
    --loss_type l1 --target_transform log \
    --num-runs 5 --num-epochs 1000 \
    --use_test --save_train_and_test \
    --lr-scheduler cosine --lr-min 1e-6
```

Expected banner:
```
[gradient-accumulation]  num_gpus=2  per_gpu_batch=1024  accum_steps=1  effective_batch=2048
```

### 7.3 — 2 GPUs (option B: closest reproduction of the 4-GPU run)

Effective batch = 2 × 512 × 2 = 2048. Per-GPU batch matches the 4-GPU run, so BN sees the same per-rank sample count. Add `--sync-bn` to align BN stats across ranks for the cleanest cross-config comparison.

```bash
accelerate launch \
    --num_processes 2 --num_machines 1 --multi_gpu \
    --mixed_precision no --dynamo_backend no \
    train.py \
    --model-size big --num_heads 4 --num_layers 3 \
    --hidden_size 128 --dropout_rate 0.3 \
    --lr 2e-4 \
    --per-gpu-batch-size 512 --effective-batch-size 2048 \
    --loss_type l1 --target_transform log \
    --num-runs 5 --num-epochs 1000 \
    --use_test --save_train_and_test \
    --lr-scheduler cosine --lr-min 1e-6 \
    --sync-bn
```

Expected banner:
```
[gradient-accumulation]  num_gpus=2  per_gpu_batch=512  accum_steps=2  effective_batch=2048
SyncBatchNorm: enabled (num_processes=2)
```

### 7.4 — 1 GPU

Effective batch = 1 × 2048 × 1 = 2048. Single micro-batch holds the whole effective batch; ~2.5 GB VRAM. No DDP.

```bash
accelerate launch \
    --num_processes 1 --num_machines 1 \
    --mixed_precision no --dynamo_backend no \
    train.py \
    --model-size big --num_heads 4 --num_layers 3 \
    --hidden_size 128 --dropout_rate 0.3 \
    --lr 2e-4 \
    --per-gpu-batch-size 2048 --effective-batch-size 2048 \
    --loss_type l1 --target_transform log \
    --num-runs 5 --num-epochs 1000 \
    --use_test --save_train_and_test \
    --lr-scheduler cosine --lr-min 1e-6
```

Note: no `--multi_gpu`. Works thanks to commit `ff72685` (without it, you'd need `--multi_gpu` even with one process, or `python train.py …` would crash at `init_process_group`).

### 7.5 — Wall-clock estimates

| config | per-step | per-epoch (~9 steps) | full job (5 runs × 1000 ep) |
|---|---|---|---|
| 4 × 512 | ~80 ms | ~700 ms | ~1.0 h × 5 ≈ **5 h** |
| 2 × 1024 (A) | ~150 ms | ~1.4 s | ~2.0 h × 5 ≈ **10 h** |
| 2 × 512 × 2 + sync-bn (B) | ~170 ms | ~1.5 s | ~2.1 h × 5 ≈ **10.5 h** |
| 1 × 2048 | ~280 ms | ~2.5 s | ~3.5 h × 5 ≈ **17.5 h** |

(L4-class GPUs; rough scaling — measure your actual once for the first epoch and extrapolate.)

### 7.6 — Variants worth trying

```bash
# Plateau scheduler instead of cosine
... --lr-scheduler plateau --plateau-monitor pearson_r2 \
    --plateau-patience 30 --plateau-factor 0.5 --lr-min 1e-6

# Higher cosine floor (less aggressive end-of-run decay)
... --lr-scheduler cosine --lr-min 5e-5

# Plateau on r_squared (more honest than pearson_r2 — penalizes bias drift)
... --lr-scheduler plateau --plateau-monitor r_squared --plateau-patience 30

# Single-fold debug of run_kfold
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py \
    --fold 0 --num-folds 10 --num-epochs 50 \
    --lr-scheduler cosine --lr-min 1e-6 \
    --per-gpu-batch-size 2048 --effective-batch-size 2048 \
    --loss_type l1 --target_transform log

# Full 10-fold k-fold (sequential, ~12 h on single GPU)
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py \
    --model-size big --num_heads 4 --num_layers 3 \
    --hidden_size 128 --dropout_rate 0.3 \
    --lr 2e-4 \
    --per-gpu-batch-size 2048 --effective-batch-size 2048 \
    --loss_type l1 --target_transform log \
    --num-epochs 1000 \
    --lr-scheduler cosine --lr-min 1e-6 \
    --num-folds 10 --fold-buffer-km 1.2
```

---

## 8. Behaviour change vs prior published runs

The `--rebuttal` preset and the **default** behaviour both still match Model A v2 when invoked as they were before. The new flags (Group B + C) are opt-in.

**The one non-opt-in change:** the rebalance-after-split fix (A.1). Before this session, default training used ~15 k unbalanced train rows (the orphaned-call bug); now it uses ~18 k balanced. Comparing today's results to commits before `6507975` is therefore not apples-to-apples. To reproduce the pre-fix behaviour exactly, check out `1bf27b0` or earlier.

---

## 9. State at EOD

```
4a14800 SGT/kfold: add --sync-bn flag (SyncBatchNorm)              ← HEAD
47267ac SGT/kfold: add 'mse' to --plateau-monitor choices
c85f3ce SGT/kfold: add 'plateau' LR scheduler (ReduceLROnPlateau)
3d36ecd kfold: rewrite — latitude deciles, train.py recipe, sequential single-GPU
ff72685 SGT: gate init_process_group on WORLD_SIZE > 1
7008698 SGT: add cosine / exponential LR scheduler with CLI controls
c2c1d28 SGT: add --num-epochs CLI override
e688d5b SGT: fix --use_test / --save_train_and_test bool flags
6507975 SGT: rebalance train half AFTER spatial split + rename validation→test
1bf27b0 audited training presets: big = 1,120,546 params           ← origin/main
```

- **9 commits ahead of `origin/main`.** Push pending (auto-mode block on direct-to-`main` push from sandbox).
- MC dropout (Experiment 2): complete, uploaded to `ValerianFourel/SOCrebuttal` (from `2026-05-13`).
- K-fold (Experiment 1): scripts rewritten with new geometry + recipe; **not yet re-run on Runpod** with the new configuration.
- 4-GPU reference run: in progress / partially complete (the L4 wandb config from `7008698` is the baseline).

---

## 10. Lessons

1. **`dict.get(k, default)` evaluates `default` eagerly** — already burned us once in [`session_log_2026-05-13.md`](session_log_2026-05-13.md). This time it bit us again: `--plateau-monitor` silently fell back to `pearson_r2` when an unrecognized name (`mse`) was passed. Argparse `choices` would have caught this earlier; lesson: any enum-like CLI argument should use `choices=[...]`.
2. **`type=bool` in argparse is a footgun.** Use `action=argparse.BooleanOptionalAction` (Python 3.9+) for boolean flags. `bool("False")` is `True`.
3. **`torch.distributed.init_process_group()` at module top-level breaks single-process launches.** Guard with `WORLD_SIZE > 1` — Accelerator handles initialization for the multi-process case via torchrun env vars.
4. **PyTorch's default `BatchNorm2d` is per-GPU.** Cross-`num_processes` reproduction (or even close approximation) requires `SyncBatchNorm`. The discrepancy compounds over hundreds of epochs.
5. **Cross-rank metric values for `ReduceLROnPlateau`** must be synced (`accelerator.gather()[0]`) — otherwise rank 0 sees the metric, rank 1+ see NaN, and DDP ranks diverge in their LR step decision.
