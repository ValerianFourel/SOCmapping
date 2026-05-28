# SOC Mapping — reproduce-may2025 session log

Working directory: `/home/valerian/SGTPublication/SOCmapping`
Branch: `reproduce-may2025`
Goal: reproduce the May 2025 historical TFT training run (commit `8dce131`) and
incrementally extend it for the current Bavaria soil-OC mapping work, while
keeping every change behind a CLI flag so historical behaviour is recoverable.

---

## Commits in chronological order (oldest first within this session)

| Commit | Subject |
|---|---|
| `412a4f8` | Fix R² to proper coefficient of determination + keep Pearson² alongside |
| `2178ba1` | Anti-overfit bundle — AdamW weight decay + early stopping |
| `f28696b` | Dataset bundle — stratified split + KS p-value gate + spatial K-fold CV |
| `ae374d9` | Augmentation bundle — spatial flip/rotate + temporal dropout + mixup |
| `831f317` | Model/eval bundle — LayerDrop (stochastic depth) + test-time aug |
| `6f65e6d` | `--rebalance-train` flag — per-OC-bin upsampling AFTER the split |
| `c2a7e3a` | Fix pyarrow ArrowTypeError when writing stratified-split parquets |
| `524aca6` | Fix stratified split — 3-stage oversample/filter/truncate |
| `386782c` | Fix DDP rank divergence in early-stopping (gather decision across ranks) |
| `d5f1e7e` | Remove satellite-unsafe augmentations (mixup, temporal-dropout) |
| `e14b6ec` | Enable `find_unused_parameters=True` for LayerDrop runs |
| `b2c2bda` | TFT --big: add learnable `output_scale` to widen initial prediction range (Tier 1) |
| `5741e42` | TFT --big: stack Tier 2 wider-init on top of Tier 1 `output_scale` |
| `eb73a99` | TFT --big: expose Tier-1 + Tier-2 as `--output-scale-init` / `--head-init-std` |
| `dcdef5d` | TFT: log current LR to wandb per-batch and per-epoch |
| `2c1a5f5` | TFT: make config-B the default split (stratified + KS gate, 10% test) |
| `11221b0` | TFT: strict train/test distribution matching (mean + std + mode gates) |
| `06eb0cf` | TFT: `MAX_OC=90`, `--test-oc-max=50`; outliers reserved for train only |
| `a3cdb68` | TFT: expose `MAX_OC` as `--max-oc` CLI flag |
| `231bef4` | TFT: add `--loss_type chi2` (real Pearson/Neyman chi-squared) |

---

## What each commit did

### `412a4f8` — proper R²
The earlier metric was Pearson r² (square of correlation), not R²
(coefficient of determination). The two diverge when predictions are
biased — Pearson r² stays high even if the model is systematically
off. Switched the reported `r_squared` to `1 - SS_res / SS_tot`, kept
`pearson_r2` as a sidecar metric for continuity with the May-2025 logs.

### `2178ba1` — anti-overfit bundle
- `--weight-decay`: AdamW decoupled weight decay (0 disables).
- `--early-stop-patience`: stop run if val R² hasn't improved for N epochs (0 disables).
- `--grad-clip`: optional gradient-norm clip.
- `--lr-scheduler {none, cosine, warmup_cosine}`, `--lr-min`,
  `--lr-warmup-epochs`, `--lr-warmup-start-factor`.

### `f28696b` — dataset bundle
- `--split-mode {legacy, stratified}`. `stratified` does OC-quantile-bin
  candidate sampling, spatial buffer, per-bin truncation.
- `--ks-pvalue-min`: reject splits whose train/val OC distributions are
  KS-distinguishable (gate on retry loop).
- `--kfold N`: spatial K-fold CV (latitude-decile blocks).

### `ae374d9` then `d5f1e7e` — augmentation bundle (then trimmed)
Initially shipped spatial flip+rotate (D4), temporal dropout, and mixup.
User pushed back: satellite covariates aren't sample-independent, so
temporal dropout and mixup produce physically-impossible inputs (e.g.
half a year of "darkness", or 50% Munich + 50% Rosenheim). Kept only
`--aug-spatial-flip` (D4 symmetries are exact for scalar bands).

### `831f317` — model/eval bundle
- `--layer-drop-prob`: stochastic depth on the transformer encoder
  (EnhancedTFT only).
- `--tta`: at eval, average outputs over 4 rotations of the input.

### `6f65e6d` — rebalance-train
Per-OC-bin upsampling AFTER the train/val split (independent dimension
from split mode). `--rebalance-min-ratio` (default 0.75): every bin
gets duplicated samples until it has ≥ ratio × max_bin_count rows.

### `c2a7e3a` — pyarrow fix
The split function writes train/val to parquet for inspection. POINTID
column is `object` dtype with mixed int/str values after the
concat/loc indexing, which pyarrow refuses. Coerce object columns to
str at write time only; in-memory dataframes keep their dtypes.

### `524aca6` — 3-stage stratified split
Earlier stratified split returned ~98 val points (target was 1,651).
Rewrote as: oversample 4× target per OC bin, apply spatial buffer
(rejects rejoin train), then truncate to `per_bin_target` per bin
(excess survivors also rejoin train). Mirrors the legacy
`create_validation_train_sets` adaptive-loop behaviour.

### `386782c` — DDP rank divergence on early-stop
Early-stop decision was computed only on rank 0, so other ranks didn't
break out of the epoch loop → NCCL allgather timeouts at the start of
the next run's `accelerator.prepare()`. Broadcast the stop decision via
`accelerator.gather`.

### `e14b6ec` — find_unused_parameters
LayerDrop drops entire encoder layers stochastically, so on some
batches some parameters don't get gradients. DDP's default
`find_unused_parameters=False` raises "params not receiving grad". Set
it to True conditionally when `--layer-drop-prob > 0`.

### `b2c2bda`, `5741e42` — Tier 1 + Tier 2 mean-collapse fixes
For heavy-tailed OC targets, the model output collapsed to the mean
(predictions all in a narrow band around 22 g/kg) because L1 loss
doesn't penalize tail under-prediction enough.

- **Tier 1**: `self.output_scale = nn.Parameter(torch.tensor(3.0))` on
  `EnhancedTFT`. Multiplies the head's output. Learnable; Adam adjusts.
- **Tier 2**: Re-init the head's final `Linear(d_model//4, 1)` with
  `nn.init.normal_(weight, std=1.0)` (instead of Kaiming-uniform), so
  intrinsic output std jumps from ~1.4 z-units to ~5 z-units at init.

Stacked: initial output std ≈ 5 × 3 = 15 z-units → covers the full
LUCAS OC range from epoch 0.

### `eb73a99` — Tier 1+2 as CLI flags
`--output-scale-init` (default 3.0), `--head-init-std` (default 1.0).
Both `--model-size big` only. Pass 1.0 / 0 to disable.

### `dcdef5d` — LR to wandb
`optimizer.param_groups[0]['lr']` logged per-batch and per-epoch so the
warmup ramp and cosine decay are visible alongside loss / R² in wandb.

### `2c1a5f5` — config-B as default
Empirical analysis on the actual xlsx showed the original
"stratified split + KS gate 0.0" produced a heavily biased test set
(test mean 34 vs train 21 g/kg, KS p ≈ 0). Stratified split's stage-3
ranks survivors by spatial-distance descending, which systematically
pulls the high-OC tail (peatlands, alpine) into test. KS gate at 0.05
fixes it. Flipped three defaults:

| flag | old → new |
|---|---|
| `--split-mode` | `legacy` → `stratified` |
| `--target-val-ratio` | `0.08` → `0.10` |
| `--ks-pvalue-min` | `0.0` → `0.05` |

### `11221b0` — strict mean/std/mode gates
KS p ≥ 0.05 still allowed 6% mean drift, 9% std drift, 11 g/kg mode
shift. Added three more gates on top:

| flag | default |
|---|---|
| `--match-mean-tol` | `0.02` (max 2% mean drift, normalized by global mean) |
| `--match-std-tol` | `0.05` (max 5% std drift, normalized by global std) |
| `--match-mode-tol` | `1.0` (max 1 g/kg KDE-mode diff, fixed-bandwidth Gaussian KDE on [0, 60]) |
| `--split-max-retries` | `100` (was 20, room for tighter gates) |

Also lowered default `--distance-threshold` from `1.2` → `0.5`. The
1.2 km buffer was so strict that only spatially-isolated high-OC
points could clear it, biasing the test set. 0.5 km still prevents
same-pixel leakage on a 250 m raster (2 pixels apart).

Result: all four gates clear on first retry with default seed.
Train/test means within 1.5%, modes identical at 14.25 g/kg.

### `06eb0cf` — MAX_OC=90 + test-oc-max=50
Two coupled changes:
- `MAX_OC = 90` in `config.py` (was 150). Filters the raw xlsx more
  aggressively. 16,514 → 16,143 points.
- `--test-oc-max 50` (new flag, default 50). Points with OC > cap are
  pulled out of the split pool BEFORE stratification and reattached to
  train AFTER. So test never contains outliers but the model still
  trains on them. `mean_tol` / `std_tol` are evaluated on
  train[OC≤cap] vs test (apples-to-apples). KS / mode-tol still
  compare full train vs test.

Empirical result (seed 42): mode train = test = 13.846 g/kg, test std
8.915 vs train≤50 std 8.594 (test +3.7% broader within range), test
has zero outliers, train (full) keeps all 803 of them. KS p = 0.345.

### `a3cdb68` — --max-oc as a CLI flag
`MAX_OC` was a `config.py` constant used in five places in `train.py`
(filter_dataframe calls, wandb config, checkpoint filename,
`compute_training_statistics_oc`). Replaced all with `args.max_oc`,
default reads from `config.MAX_OC` (90 g/kg). Both `--max-oc` and
`--test-oc-max` are now independently CLI-tunable.

### `231bef4` — real Pearson/Neyman chi-squared
The existing `composite_l1_chi2_loss` and `composite_l2_chi2_loss` are
not real chi-squared — `composite_l2` is MSE rescaled to itself,
`composite_l1` uses an exp-weighted squared error. Added
`neyman_chi2_loss`:

    L = mean( (y_pred_raw - y_true_raw)^2 / max(y_true_raw, eps) )

in original OC g/kg units (inverse-transforms predictions back from
z-score / log space; gradients flow through the inverse-transform).
Wired as `--loss_type chi2`, with `--chi2-eps 1.0` for the denominator
floor. Sanity-tested with 4 synthetic samples:
- perfect pred: 0
- under-predict 80 by 30: 2.81 = (30²/80)/4
- under-predict 5 by 30: 45.0 = (30²/5)/4 (correct: classical χ²
  weights low-target errors MORE than tail errors)

---

## Where the algorithm currently sits

`reproduce-may2025` HEAD = `231bef4`. The bare invocation:

```bash
python train.py --use_validation --model-size big --num-runs 1 --num_epochs 200 ...
```

now produces a Bavaria-2007-2023 split with:
- 14,609 train / 1,534 test (test ratio 10%)
- OC filter: ≤ 90 g/kg (global), ≤ 50 g/kg (test only, outliers in train)
- KS p ≥ 0.05, mean diff ≤ 2%, std diff ≤ 5%, mode diff ≤ 1 g/kg
- Spatial buffer ≥ 0.5 km between test and train
- Train modal OC = test modal OC = 13.85 g/kg
- Test std > train(≤50) std (broader shape within the comparable range)

And the model:
- `EnhancedTFT` 1.12M params, transformer encoder × 3 layers
- Tier-1 output_scale init 3.0, Tier-2 head init std 1.0
- L1 loss (`--loss_type l1`), `--target_transform normalize`
- Cosine LR with warmup if `--lr-scheduler warmup_cosine`

To recover the historical (May 2025, commit `8dce131`) defaults:
```bash
--split-mode legacy --max-oc 150 --test-oc-max 0 --ks-pvalue-min 0 \
--match-mean-tol 0 --match-std-tol 0 --match-mode-tol 0 \
--distance-threshold 1.2 --target-val-ratio 0.08 \
--output-scale-init 1.0 --head-init-std 0
```

---

## Pod patches generated this session
(base64-encoded `git format-patch -1`, ready for `git am`)

| Patch | Bytes | Commit |
|---|---|---|
| `/tmp/tft_tier12_flags.patch.b64` | 9,728 | `eb73a99` |
| `/tmp/tft_wandb_lr.patch.b64` | 2,840 | `dcdef5d` |
| `/tmp/tft_default_splitB.patch.b64` | 7,984 | `2c1a5f5` |
| `/tmp/tft_strict_match.patch.b64` | 20,900 | `11221b0` |
| `/tmp/tft_maxoc90_testcap50.patch.b64` | 15,284 | `06eb0cf` |
| `/tmp/tft_max_oc_flag.patch.b64` | 7,136 | `a3cdb68` |
| `/tmp/tft_chi2_loss.patch.b64` | 10,852 | `231bef4` |

Apply each via:
```bash
cd ~/SOCmapping && git checkout reproduce-may2025 && \
echo 'PASTE_BASE64' | base64 -d | git am
```
