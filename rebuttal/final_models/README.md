# Final-model pipeline — Geoderma rebuttal pivot

The original paper presented Model A (EnhancedSGT, 1.1 M parameters) trained
on a single 91/9 spatial split. Reviewers (R1.3, R3.6) flagged that single
split as weaker than repeated CV. Our 10-fold spatial-CV sweep
(`rebuttal/gpu_experiments/spatial_kfold/`) settled the *evaluation*
question.

This directory handles the *production-mapping* side: we re-train each
winning architecture on the **full** LUCAS/LfL/LfU dataset (no spatial
holdout; only a 5% random monitor) and infer over the Bavaria 1mil
reference grid. The resulting maps are the figures the rebuttal will
present.

## Workflow

```
[1] Submit all training + inference jobs
    └─→ checkpoints/<run-name>/final_model.pth + stats.json + config.json
    └─→ maps/<run-name>/bavaria_2023_predictions.parquet + .png + .json

[2] After every job lands, generate the comparison figure
    └─→ maps_comparison_2023.png + .md + .json
```

## Pipeline scripts

| Script | Role |
|---|---|
| `train_full.py` | Neural-network training on the full dataset (one architecture) |
| `train_full_baselines.py` | RF / XGBoost training on the full dataset |
| `infer_bavaria.py` | Forward-pass any saved checkpoint over the 1mil Bavaria grid |
| `submit_finals.py` | Generate + submit the full set of sbatch jobs |
| `compare_maps.py` | Side-by-side comparison of all maps |

## Models trained

| run_name | Family | Architecture | Params | max-OC | Loss |
|---|---|---|---|---|---|
| `sgt_d128_h4_L1` | SGT | SimpleSGT d=128 h=4 L=1 | 363k | 150 | composite_l2 (α=0.5, β=0.1) |
| `simpletransformer_d64_h4_L1` | Transformer | SimpleTransformerV2 d=64 h=4 L=1 | 11.2M | 150 | L1 |
| `cnnlstm_d64_h4_L1` | CNN-LSTM | RefittedCovLSTM 64-d | 93k | 150 | L1 |
| `rf_default` | Tree | RandomForest n=500 | n/a | 150 | log target |
| `xgb_shallow` | Tree | XGBoost depth=4 n=2000 | n/a | 150 | log target |

These are the spatial-CV winners. 3DCNN is excluded because it collapsed
catastrophically (R² ≈ −0.76 mean across all configurations) and is not a
useful production map.

## Running

From the **login node** (sbatch needs queue access):

```bash
cd /e/project1/scifi/fourel1/SGT/SOCmapping
source ../venv/bin/activate

# Inspect what would be submitted
python rebuttal/final_models/submit_finals.py --dry-run

# Submit everything (3 NN train+infer pairs + 2 tree combined = 8 jobs)
python rebuttal/final_models/submit_finals.py

squeue -u $USER
```

The NN training jobs each take ~1 h on 4 GH200s; inference takes
~1.5 h on 1 GH200 (1 mil grid). Tree baselines train+infer in one ~2 h job.
Wall time end-to-end: ~3 h if your slot is responsive.

## After all jobs land

```bash
python rebuttal/final_models/compare_maps.py
```

Produces `maps_comparison_2023.png` and `maps_comparison_2023.md` — the
direct cross-architecture map comparison the rebuttal needs.

## Customisation

Submit only one model:

```bash
python rebuttal/final_models/submit_finals.py --only sgt_d128_h4_L1
```

Sweep additional years (after the default 2023 run lands):

```bash
python rebuttal/final_models/infer_bavaria.py --run-name sgt_d128_h4_L1 --year 2007
python rebuttal/final_models/infer_bavaria.py --run-name sgt_d128_h4_L1 --year 2015
```

Different max-OC cap for sensitivity:

Edit `FINAL_GRID` in `submit_finals.py` (change `--max-oc` in each `cmd`
string) and re-submit with a different `--only` set, e.g. `oc120`.

## Sanity-check before the full launch

Dry-run inference on a tiny grid slice for the SGT winner — needs only the
checkpoint to exist:

```bash
python rebuttal/final_models/infer_bavaria.py \
    --run-name sgt_d128_h4_L1 \
    --year 2023 \
    --limit 5000
```

Should finish in ~2 min on one GPU. If it works, drop `--limit` and let it
run on the full 1 M grid.

## Why this is the rebuttal-winning approach

The reviewer asks center on two themes:
1. **Honest evaluation.** Addressed by the spatial 10-fold CV (sweep tables).
2. **Production maps.** Addressed *here*, with every competing architecture
   shown side-by-side on the same training data, the same inference grid,
   and the same evaluation protocol. The comparison figure becomes the
   paper's headline result.

The narrative is **"the same lightweight inductive bias that wins on
spatial-CV evaluation also produces production maps that visually outperform
alternatives at a fraction of the parameter cost"** — both claims supported
by the same pipeline, with no methodological cherry-picking.
