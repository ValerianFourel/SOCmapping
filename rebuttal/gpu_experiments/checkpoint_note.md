# Checkpoint note — Models A vs B

Two existing checkpoints carry the SGT 1.1 M weights for this paper. They
serve different purposes and must not be confused.

## Model A — evaluation model (91/9 spatial split)

- **Path:** `Weights-ResidualsModels-MappingInference-SOCmapping/`
  `TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/`
  `TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_`
  `TIME_END_2023_TRANSFORM_normalize_LOSS_composite_l2_R2_0.6909.pth`
- **Trained on:** ~15,128 samples (91% of 16,514)
- **Held-out val:** ~1,386 candidate samples; after the 1.2 km buffer that
  trims to **1,359** rows
- **Confirmed val metrics:** R² = 0.6258, RMSE = 4.758, MAE = 2.791,
  RPIQ = 1.051, val n = 1,359
- **Bootstrap 95% CI** (`rebuttal/bootstrap_results.md`):
  R² [0.526, 0.712], RMSE [4.18, 5.33]
- **Architecture verified by checkpoint inspection:** EnhancedSGT, 1,121,637
  trainable parameters, BatchNorm2d in spatial encoder, LayerNorm in GRN
  blocks, 99 state-dict keys with `module.` prefix from Accelerate.
- **Loss / transform actually used:** `LOSS_composite_l2` ≡ MSE (per the
  loss-collapse finding in `rebuttal_numbers.md §1`) + `TRANSFORM_normalize`
  (per-target standardisation).
- **Purpose:** Table 2 metrics in the paper. Used as the comparison row in
  `kfold_results.md`. **NOT** used for maps or uncertainty.

## Model B — mapping model (full data)

- **Path:** `Weights-ResidualsModels-MappingInference-SOCmapping/`
  `TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/`
  `TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_`
  `TIME_END_2023_TRANSFORM_log_LOSS_l1_R2_1.0000.pth`
- **Trained on:** all 16,514 samples (no held-out validation; `use_validation
  = False` in `train.py`, so the saved R² = 1.0000 in the filename is the
  placeholder, not a real metric)
- **Architecture verified by checkpoint inspection:** identical to Model A —
  EnhancedSGT, 1,121,637 parameters, same `model_config` block.
- **Loss / transform actually used:** L1 loss on `torch.log(OC + 1e-10)`
  target (`TRANSFORM_log`, `LOSS_l1`). Inverse at inference: `np.exp(pred)`.
- **Purpose:** produced Figures 14/15 (2023 Bavaria SOC maps in the paper)
  and is the **operational deployment model**. Used by Experiment 2 (MC
  dropout uncertainty map). **NOT** used for Table 2 metrics.

## Experiment 1 (spatial 5-fold CV)

- Uses **neither** A nor B as a starting point.
- Each of the 5 folds **trains from scratch** with the same recipe as
  Model B (EnhancedSGT, L1 loss on log + 1e-10 target, Adam @ lr=2e-4,
  270 epochs, batch 256, dropout 0.3, per-fold feature normalisation
  fit on the in-fold training rows).
- The final row of `kfold_results.md` cites Model A's metrics
  (R² 0.626 / RMSE 4.758 / MAE 2.791 / RPIQ 1.051) for direct
  comparison only.

## Experiment 2 (MC dropout uncertainty)

- Uses **Model B only**.
- Selective dropout activation: `model.eval()` then `module.train()` for
  every `nn.Dropout` (BatchNorm must stay in eval mode — population
  statistics).
- 30 stochastic forward passes per Bavaria-grid point; Welford
  accumulator tracks mean and variance in original SOC units (the
  log-transform is inverted via `torch.exp` inside the MC loop).
- Validation check: the MC mean should correlate with the single-pass
  Figures-14/15 prediction at Pearson r > 0.99 (sanity check, not a
  metric).
