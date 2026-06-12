---
license: cc-by-4.0
pretty_name: "SOC Rebuttal — 43-band spatial-CV results (Geoderma GEODER-D-26-01032)"
tags:
  - soil-organic-carbon
  - digital-soil-mapping
  - remote-sensing
  - spatial-cross-validation
  - transformer
  - bavaria
---

# SOCrebuttal — Bavaria SOC mapping, 43-band spatial cross-validation results

Companion data + code for the Geoderma revision **GEODER-D-26-01032**
("Spatiotemporal Gated Transformer for soil organic carbon mapping"). This
repository holds the **canonical experiment** that supersedes the first
submission, the **trained production models**, the **per-fold spatial-CV
results**, the **Bavaria-wide prediction maps**, and the **figure code** so the
figures can be downloaded and regenerated end-to-end (styling is meant to be
finished locally).

> **Source of truth.** Every number quoted below is read from the files in
> `sweep/` by the figure scripts — nothing is hand-entered. If a figure and
> this README ever disagree, the files in `sweep/` win.

---

## The canonical experiment

| Setting | Value |
|---|---|
| Covariates | **43-band "full_extended"** stack (20 revision bands + Tier-1/2/3: Landsat SRC, terrain, climate/phenology) |
| Target cap | `oc150` (SOC capped at 150 g/kg) |
| Spatial CV | **longitude-blocked (split-axis `lon`), 10-fold**, leave-one-block-out with a 1.2 km train/test buffer |
| Ranking score | `score = mean_R² − ½·SD(R²)` (rewards stable folds) |
| Samples | 16,514 LUCAS/LfL/LfU topsoil points, 5-year covariate windows |

This is a **strict spatial-generalization test**: whole west–east strips of
Bavaria are held out, so a model is scored on regions it never saw.

---

## Headline results

**1. The compact gated SGT wins.** Best per family at the matched
43-band / oc150 / lon / 10-fold protocol:

| Rank | Family | Best config | R² (mean ± SD) | params |
|---|---|---|---|---|
| 1 | **SGT** (gated CNN+GRN+Transformer) — flagship | `small_d32_h2_L1` | **0.387 ± 0.084** | ~84k |
| 2 | CNN-Transformer (vanilla, no gate) | `vanilla_d64_h4_L1` | 0.369 ± 0.108 | ~115k |
| 3 | Random Forest | `rf_deep` | 0.342 ± 0.095 | — |
| 4 | Simple Transformer (no CNN) | `simpletransformer_d64_h4_L1` | 0.331 | ~11M |
| 5 | XGBoost | `xgb_deep` | 0.317 ± 0.107 | — |
| 6 | CNN-LSTM | `cnnlstm_d64_h4_L1` | 0.260 | ~95k |
| 7 | 3D-CNN | `3dcnn_d64_h4_L1` | −0.459 ± 0.291 | ~30k |

**2. "Smaller is better."** Within SGT, R² declines monotonically with size
(84k → 0.387, 168k → 0.369, 366k → 0.344); across the 0.3–0.6M band R² sits flat
at ~0.34–0.35 regardless of family; past ~500k, adding depth/width collapses it
(L2 → ~0.18–0.27, L3 → negative). An **84k-param** model beats an **~11M** simple
transformer. Under strict spatial CV, extra capacity overfits the spatial
structure rather than helping.

**3. The CNN front-end is what separates attention from trees.** Vanilla (CNN +
Transformer, 0.369) > Random Forest (0.342) > Simple Transformer (no CNN, 0.331)
— i.e. a tree ensemble beats the CNN-free transformer; the convolutional encoder
is the difference-maker.

**4. The gate now earns its place (a reversal).** Gated SGT (0.387) edges the
no-gate vanilla (0.369). The margin is ~0.4 fold-SD, so the defensible claim is
*"scaling up does not help and slightly hurts; the compact gated model is at
least as good and more robust"* — figures show per-fold SD error bars accordingly.

### Caveats encoded in the data (do not strip)

- **`oc90` numbers are not comparable.** Simple-transformer and CNN-LSTM top
  their families only on the `oc90` sub-sweep (SOC capped at 90 g/kg), which
  mechanically lowers RMSE / inflates RPIQ. Use the **`oc150`** values above.
- **Two matched-~500k configs were never run:** `sgt d192_h4_L1` (~605k) and
  `lightweight_transformer d192_h4_L1` (~567k). Figures mark them as projected /
  not-run — they are **not** real measured points.

---

## Repository layout

```
README.md                              this notice
code/                                  figure-generation code (regenerate everything)
  figstyle.py  figdata.py              shared style + honest data loaders
  figFA_param_scaling.py  figFB_family_ranking.py  figFC_perfold_heatmap.py
  figFD_flagship_map.py   figA18_family_map_grid.py
  fig07_…  fig08_…  fig09_…  fig11_…  fig12_…  fig13_…  fig16_17_…
  run_all_figures.py                   driver -> figures/out/FIGURE_MANIFEST.md
  param_counts.py                      trainable-param counts per config

sweep/<group>/<tag>/                   the per-fold spatial-CV RESULTS
  kfold_results_summary.json           across-fold + per-fold R²/RMSE/RPIQ + fold geometry
  kfold_results.md
  fold_<i>_predictions.parquet         hold-out predictions per fold
  kfold_predictions_all_folds.parquet  pooled hold-out predictions

final_models/checkpoints/<run>/        the production fits (to *check* the models)
  final_model.pth | final_model.joblib | final_model.json
  stats.json  config.json  train_log.txt

maps/_locations_400000rand_seed42/     Bavaria mapping after training on all data
  <run>_2023_predictions.parquet/.png/.json
  combined_2023_maps.png  combined_2023_predictions.parquet  combined_2023_summary.json

figures/out/                           rendered figures (PDF + PNG) + FIGURE_MANIFEST.md
```

Production run-names (suffix `_extband` = 43-band, `_20band` = 20-band trees):
`sgt_d32_h2_L1_extband`, `vanilla_transformer_d64_h4_L1_extband`,
`simpletransformer_d64_h4_L1_extband`, `cnnlstm_d64_h4_L1_extband`,
`3dcnn_d64_h4_L1_extband`, `rf_deep_20band`, `xgb_deep_20band`.

---

## Download

```bash
pip install huggingface_hub
# whole dataset:
huggingface-cli download ValerianFourel/SOCrebuttal --repo-type dataset --local-dir SOCrebuttal
# or just the pieces you need:
huggingface-cli download ValerianFourel/SOCrebuttal --repo-type dataset \
    --include "sweep/**" "maps/**" "code/**" --local-dir SOCrebuttal
```
or in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download('ValerianFourel/SOCrebuttal', repo_type='dataset',
                  local_dir='SOCrebuttal')
```

---

## Regenerate the figures (then style locally)

The figure code is self-contained: `figdata.py` reads the `sweep/` summaries and
fold parquets directly (no other repo files needed for the data-driven figures),
and every metric is read from disk. After downloading:

```bash
cd SOCrebuttal/code
pip install matplotlib numpy pandas scipy pyarrow   # + pyproj for true UTM maps (optional)

# regenerate ALL figures from the downloaded data:
python run_all_figures.py \
    --sweep-dir ../sweep \
    --maps-dir  ../maps/_locations_400000rand_seed42 \
    --axis lon --bands 43 --max-oc 150 \
    --out-dir   ../figures/out

# or a single figure (each reads the same flags):
python figFA_param_scaling.py --sweep-dir ../sweep --axis lon --bands 43 --max-oc 150
python figFB_family_ranking.py --sweep-dir ../sweep --axis lon --bands 43 --max-oc 150
python fig11_residual_map.py  --sweep-dir ../sweep --axis lon --bands 43 --max-oc 150
```

Each script writes a `.pdf` (vector) and `.png` (300 dpi); `run_all_figures.py`
also writes `figures/out/FIGURE_MANIFEST.md` (per-figure provenance + a
DISCREPANCIES section). **Styling is meant to be finished locally** — edit
`code/figstyle.py` (palette, fonts, sizes, map projection) and re-run; all
figures share that one module.

**Notes**
- Maps (`figFD_*`, `figA18_*`) need the `maps/` parquets; without them they log
  `BLOCKED` and skip rather than fabricate.
- The parameter annotations in `figFA`/`figFB` and the architecture diagram
  `fig08` need the model classes (`param_counts.py` imports them). Those live in
  the full code repository — without them the scripts degrade gracefully (param
  labels show `-`/`n/a`); the data-driven figures are unaffected.
- `pyproj` is optional: present → maps reproject to UTM 32N (metres, true scale
  bar); absent → lon/lat with an aspect correction.

---

## What changed vs the originally compiled PDF (discrepancies)

The first submission's prose/Table 2/Table 4 reflect an **older** experiment.
`figures/out/FIGURE_MANIFEST.md` lists these in full; in brief:

1. **Experiment:** 6-band / latitude-decile / Vanilla-215k / R²≈0.20 → **43-band /
   lon-blocked / SGT-84k / R²≈0.34–0.39**.
2. **Gate:** "GRN gate is dead weight" → **gated SGT 0.387 > vanilla 0.369**.
3. **Param counts:** Simple Transformer 2.4M → **~11M**; Vanilla 215k → ~115k/317k
   (d64/d128, 43-band).
4. **Covariates:** the 6-covariate Table 1 / §2.2 description → **43 bands**.

These are the manuscript-text updates that accompany this data drop.

---

## Citation / contact

Fourel, V. et al., *Spatiotemporal Gated Transformer for soil organic carbon
mapping in Bavaria* (Geoderma, under revision, GEODER-D-26-01032). Model code:
`github.com/ValerianFourel/SOCmapping`.
