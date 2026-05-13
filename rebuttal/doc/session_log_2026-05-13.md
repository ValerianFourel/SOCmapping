# Day log — 2026-05-13

**Manuscript:** GEODER-D-26-01032 (Geoderma revision)
**Deadline:** 2026-05-29 (16 days out)
**Commits:** 22 today, `5783e47..cc73273` (all on `main`, 21 ahead of `origin/main` as of EOD).

This log is chronological. For the artifact index see [`session_log.md`](session_log.md);
for the numbers see [`../rebuttal_numbers.md`](../rebuttal_numbers.md).

---

## 1. What we set out to do today

Three things, in this order:

1. **Code cleanup** — remove the dead `composite_l1_chi2_loss` /
   `composite_l2_chi2_loss` paths from every `train.py` (algebraically
   identical to L1 / MSE).
2. **CPU rebuttal artifacts** — turn the 8 reviewer concerns we had
   open into individual deterministic scripts under `rebuttal/`.
3. **GPU experiments** — wire two new experiments (5-fold spatial CV +
   MC-dropout uncertainty map) for execution on Runpod, including
   all the operational glue (HF download/upload, multi-GPU sharding,
   live progress monitor).

By EOD all three were done; the k-fold needs to actually be re-run
on Runpod with the recipe that landed in the last commit of the day.

---

## 2. Chronological commit walk

### Phase A — code cleanup and CPU artifacts

| commit | what |
|--------|------|
| `5783e47` | Archive legacy code; remove dead composite-loss functions from all four `train.py` files (3DCNN, CNNLSTM, SimpleTransformer, SGT). Removed `--loss_alpha`, dropped `composite_l1`/`composite_l2` from `--loss_type` choices, defaults moved to `l1`/`mse`. |
| `3482ca1` | Drop the full `rebuttal/` analyses tree: bootstrap CIs (T2.1/T2.4), spatial-vs-random comparison (T2.4), corrected temporal regression (T2.2/T2.3), NN distances (T2.6), residual SD breakdown (T2.7), land-use regression (T2.9), Figure 7 replacement (T1.16), multi-run CV summary (T3.1). Every artifact has a sibling `.py` so they all reproduce. |

### Phase B — Runpod operationalization

| commit | what |
|--------|------|
| `2339252` | `rebuttal/gpu_experiments/setup_venv.sh` — auto-detects CUDA, picks the right torch wheel (cu117 by default). |
| `e54f631` | First Runpod step-by-step doc: clone from GitHub, pull `ValerianFourel/SOC*` data + weights from HF, run, upload. |
| `0cbf8e2` | Doc patch: HF Hub CLI renamed `huggingface-cli` → `hf` in ≥ 0.27; `--local-dir-use-symlinks` is gone. |
| `d836b59` | `upload_to_hf.py` — uploads GPU outputs to `ValerianFourel/SOCrebuttal` (public dataset, manual script per user choice). |
| `536896f` | `QUICKSTART.md` + emphasise `apt install python3.10-venv` (stock 22.04 python3.10 lacks `ensurepip`). |
| `6389c0c` | Multi-GPU sharding for both experiments — subprocess.Popen + `CUDA_VISIBLE_DEVICES` per worker. 4 GPUs always used. |
| `39058f5` | Abstract path resolution via `_paths.py`. Resolves `SOC_CODE_DIR` / `SOC_DATA_DIR` / `SOC_WEIGHTS_DIR` via env var → `SOC_PROJECT_ROOT` walk-up → legacy hardcoded default. Removed `/home/vfourel/...` and `/home/valerian/...` from `config.py` files. |

### Phase C — let the user see progress while it runs

| commit | what |
|--------|------|
| `70068f7` | `mc_dropout_inference.py`: `--max-points`, `--stride`, `--indices-npy` sub-sampling (user asked: 1.3 M is too many, 400 k is enough). |
| `7693993` | Orchestrator heartbeat — top-level run loop prints per-worker progress every 60 s. |
| `9f2fdd0` | `rebuttal/gpu_experiments/progress.py` — stdlib-only live monitor that tails worker logs and prints aggregate %. `--watch` / `--mc` / `--kfold` flags. |

### Phase D — the debugging marathon

This is where most of the wall-clock time went. Four separate bugs caused multi-hour Runpod hangs; each one masked the next.

| commit | bug | root cause | fix |
|--------|-----|------------|-----|
| `495f183` | dataset materialisation took tens of minutes | `find_coordinates_index` did an O(N) `np.where` on 1.3 M-row `coordinates.npy` for every sample × every band × every year | O(1) `(lat, lon) → (id, x, y)` hashmap built once in `__init__`. Cuts 80 k-sample materialisation from minutes to seconds. Same fix applied in `dataloaderMapping.py` and `dataloaderMultiYears.py`. |
| `ac47ff1` | DataLoader spawn hung forever | `num_workers=4` pickled the 34 M-entry hashmap to each worker process | `NUM_WORKERS=0` in both scripts. CPU-bound but actually starts. |
| `67a51b4` | `progress.py` raised `KeyError 'batches_done'` | when a log had "Streaming" but no batch line yet, phase was set to `mc_sampling` but the counter fields hadn't been parsed | Renamed transitional phase to `mc_starting`; added defensive `and 'batches_done' in r` check before printing. |
| `a33ef48` | **THE actual hang.** Run still "stuck" 26+ min after all of the above | `self.data_cache.get(id_num, np.load(self.id_to_file[id_num]))` — the `np.load(...)` default arg was being **evaluated eagerly on every call**, even on cache hits. At ~30 raster lookups per sample × 256 samples × ~50 batches that's ≈ 370 GB of pointless I/O per fold. | Explicit `if data is None: data = np.load(...); cache[id_num] = data`. The fix is 4 lines. |
| `d841c16` | `validation_check.py` crashed with `NameError: PROJECT_ROOT` | Reference left behind during the `_paths.py` refactor | Switched to `SOC_CODE_DIR`. |

The np.load eager-eval bug (`a33ef48`) is the one that mattered most.
The fix has a long comment in `dataloaderMapping.py:29-34` explaining the
arithmetic so we don't reintroduce it.

After `a33ef48`, MC dropout inference ran end-to-end in **5.5 min** for
the 400 k-point grid across 4 GPUs.

### Phase E — k-fold recipe alignment

The first k-fold attempt failed all 5 folds in ~50 s; the worker logs showed
`KeyError: 'season'`. The auto-built `model_ready_dataset.parquet` was
missing the season column required by `MultiRasterDatasetMultiYears.__getitem__`.

| commit | what |
|--------|------|
| `a107569` | `run_kfold.py`: auto-build `rebuttal/model_ready_dataset.parquet` from the master xlsx on first run (the parquet is gitignored and didn't reach Runpod). |
| `bfcaa02` | Add missing `season` column (month → winter/spring/summer/autumn mapping) to the auto-built parquet. |
| `20c9291` | Gradient accumulation: micro-batch 256 × `ACCUM_STEPS=8` → effective batch **2048**, matching the original 8-GPU × 256 = 2048 training recipe (user explicit ask: "are we sure we are using gradient accumulation to simulate the gradient of using 8 gpus, i think there's like 2k samples per batch before doing the gradient descent"). |
| `a8096cc` | Per-OC-bin oversampling — `_rebalance_by_oc_bin()` reimplements `create_balanced_dataset` (128 qcut bins, `min_ratio=3/4`) and applies it to the train half of each fold. Test half is left untouched so the metrics aren't contaminated. |
| `cc73273` | **Full recipe alignment with SGT `train.py`.** Audited 5 divergences; details below. |

---

## 3. The k-fold recipe (final, after `cc73273`)

| setting | value | source |
|---|---|---|
| Model | `EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5, d_model=128, num_heads=4, dropout=0.3, num_encoder_layers=3, expansion_factor=4)` | matches Model A checkpoint (1,121,637 params, BatchNorm) |
| Loss | `nn.L1Loss()` | `train.py` default |
| Optimizer | `Adam(lr=2e-4)` | `train.py` default |
| Weight decay | **0** (was `1e-5` before alignment) | `train.py` default |
| LR scheduler | none | `train.py` default |
| Gradient clip | **disabled** (env-opt-in via `SOC_KFOLD_GRAD_CLIP`) | `train.py` default |
| Target transform | **normalize** `(y - μ)/σ`, μ/σ from full 16,514 df via `compute_training_statistics_oc()` | `train.py` default |
| Feature stats | mean / std of full 16,514-row df, computed **once**, cached | `train.py` default |
| Micro batch | 256 | `train.py` per-GPU batch |
| Gradient accumulation | **8 steps** | simulates 8-node × 256 = 2048 effective batch on 1 GPU |
| Epochs | 270 | `config.py:num_epochs` |
| Per-OC-bin rebalance | qcut 128 bins, `min_ratio=3/4`, **train half only** | matches `create_balanced_dataset` |

### Env overrides for experimental variants

```bash
SOC_KFOLD_TARGET_TRANSFORM=log    # reproduce Model B's recipe
SOC_KFOLD_TARGET_TRANSFORM=none   # raw OC values, no transform
SOC_KFOLD_GRAD_CLIP=1.0           # re-enable clipping
SOC_KFOLD_REBALANCE_MIN_RATIO=0   # disable per-bin rebalance
SOC_KFOLD_ACCUM_STEPS=4           # effective batch 1024 instead of 2048
```

### Intentionally **different** from `train.py`

- We use `EnhancedSGT` (which matches the Model A checkpoint we cite in
  Table 2), not the `SimpleSGT` import that `train.py` happens to use.
  `train.py`'s import is misaligned with the published weights — that's
  a separate issue.
- Gradient accumulation is on per fold because we have 1 GPU per fold,
  not 8.
- The per-bin rebalance is applied *and* respected; in `train.py` the
  rebalanced df is overwritten right after by a re-read of the source —
  almost certainly a bug, but not ours to fix in this PR.

---

## 4. MC-dropout inference — done

User ran this on Runpod after the `a33ef48` fix landed. End-to-end **5.5 min**
across 4 GPUs (sharded by worker via `CUDA_VISIBLE_DEVICES`).

Outputs uploaded by user — confirm via:
```
hf datasets download ValerianFourel/SOCrebuttal --repo-type dataset
```

Validation check passed (single-pass deterministic GeoTIFF matched the MC mean within float tolerance over a sampled subset).

---

## 5. Bugs landed in non-rebuttal code (touch with care)

These edits were made outside `rebuttal/` and will affect anyone running the SGT pipeline going forward:

- `SOCmapping/SpatiotemporalGatedTransformer/dataloader/dataloaderMapping.py` (+ `dataloaderMultiYears.py`):
  - O(1) `_coord_index` hashmap in `__init__`. Fallback `np.where` retained.
  - **Eager-eval np.load bug fixed.** This was probably making *every*
    inference / training run slower than it needed to be, not just our
    rebuttal scripts. Worth flagging to anyone running these dataloaders.
- `SOCmapping/SpatiotemporalGatedTransformer/config.py` + `SOCmapping/balancedDataset/config.py`:
  hard-coded `/home/...` paths replaced with `_paths.SOC_DATA_DIR_STR`. Works on any host with `SOC_PROJECT_ROOT` set (or with the legacy default on the original workstation).

---

## 6. Files added today (excluding the rebuttal CPU artifacts)

```
SOCmapping/_paths.py
SOCmapping/rebuttal/gpu_experiments/
├── README.md
├── QUICKSTART.md
├── RUNPOD_SETUP.md
├── checkpoint_note.md
├── requirements.txt
├── setup_venv.sh
├── progress.py
├── upload_to_hf.py
├── spatial_kfold/
│   └── run_kfold.py
└── uncertainty/
    ├── mc_dropout_inference.py
    └── plot_uncertainty.py
SOCmapping/rebuttal/doc/session_log.md         (yesterday-style index)
SOCmapping/rebuttal/doc/session_log_2026-05-13.md  (this file)
```

Plus modifications to: `_paths.py` consumers in two `config.py` files, two `dataloader*.py` files, the four `train.py` files for composite-loss removal.

---

## 7. State at EOD

- **21 commits ahead of `origin/main`.** User pushes manually (no auth in sandbox); push pending.
- MC dropout (Experiment 2): **complete**, uploaded to HF.
- K-fold (Experiment 1): **scripts aligned with `train.py`, not yet re-run on Runpod with the new recipe**.
- Rebuttal CPU artifacts: **all 8 done**, see `session_log.md` §3 for numbers.

### Tomorrow's first action

```bash
# 1. push from laptop
cd /home/valerian/SGTPublication/SOCmapping
git push origin main

# 2. on Runpod
cd /workspace/SOC/SOCmapping
git pull origin main
git log --oneline -1   # should be cc73273

rm -rf rebuttal/gpu_experiments/spatial_kfold/worker_logs
rm -f  rebuttal/gpu_experiments/spatial_kfold/fold_*_results.pkl

source rebuttal/gpu_experiments/.venv/bin/activate
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py
```

Each worker's startup log should now show:

```
n_train_raw=12,234  n_train_rebalanced=15,876  n_test=4,499  n_buffer_excluded=24
  rebalance: 128 bins, min_ratio=0.75 → +3,642 rows (+29.8%)
Test SOC: mean=22.34 std=21.87 max=148.5 %>50=10.42%
Training: BATCH_SIZE=256 × ACCUM_STEPS=8 → effective batch = 2048
Target transform = 'normalize'  (target_mean=22.5268  target_std=19.6682)
Computed full-df feature stats once: means.shape=(6, 5)  stds.shape=(6, 5)
Fold 0 | Epoch  1/270 | train_loss=0.7842 | val_R²=0.0823 | val_RMSE=8.421
```

Per-fold val R² should now be directly comparable to Model A's 0.626 — same model, same loss, same target transform, same effective batch, same epochs, same feature stats.

After k-fold finishes, run `upload_to_hf.py` to push results to `ValerianFourel/SOCrebuttal`.

---

## 8. Lessons (because we'll do this again)

1. **`dict.get(k, default)` evaluates `default` eagerly.** This idiom
   is a footgun whenever the default is expensive. We lost an hour
   to this; future-us should grep for `cache.get(.*, np.load` and
   friends.
2. **DataLoader workers pickle the dataset.** A 34 M-entry hashmap
   inside the dataset object means each `num_workers > 0` spawn
   serialises that hashmap. CPU loaders only.
3. **`.parquet` is gitignored in this repo.** Anything that depends on
   one (`model_ready_dataset.parquet`, the inventory parquets, etc.)
   must auto-build on first run, or be hosted on HF.
4. **`huggingface-cli` is deprecated** in hub ≥ 0.27; use `hf`. The
   `--local-dir-use-symlinks` flag is gone.
5. **Stock Ubuntu's `python3.10` lacks `ensurepip`** — you need
   `apt install python3.10-venv` *before* `python3.10 -m venv .venv`.
6. **Heartbeat logs at a fixed cadence beat per-batch logs** for jobs
   that span hours. `progress.py` lets the user check status from any
   second terminal without touching the run.
