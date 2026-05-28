# SGT R² ≈ 0.25 diagnostic — staged report

Branch reference: `main`. Files audited (read from `git show main:...`,
not the working tree since `reproduce-may2025` doesn't contain the SGT
package):

- `SpatiotemporalGatedTransformer/train.py` (1,231 lines)
- `SpatiotemporalGatedTransformer/EnhancedSGT.py` (187 lines)
- `SpatiotemporalGatedTransformer/balancedDataset.py` (332 lines)
- `SpatiotemporalGatedTransformer/config.py` (134 lines)
- `SpatiotemporalGatedTransformer/dataloader/dataloaderMultiYears.py` (348 lines)
- `SpatiotemporalGatedTransformer/residualsStudy.py`

---

## SECTION 1 — R² COMPUTATION AUDIT

### Call-site table

| # | Site | Computation | y_true scale | y_pred scale | SStot reference | Notes |
|---|---|---|---|---|---|---|
| 1 | `train.py:329-331` (per-epoch test eval) | Manual `1 - ss_res / ss_tot` | original OC g/kg | original OC g/kg | **eval (test) subset variance** | matches scikit-learn `r2_score` |
| 2 | `residualsStudy.py:143-145` (post-hoc) | Manual `1 - ss_res / ss_tot` | original OC g/kg | original OC g/kg | **eval subset variance** | imports sklearn but uses manual formula |
| 3 | `train.py:328` and `:142` | Pearson r² (`corr ** 2`) reported alongside | same as above | same as above | n/a | label `pearson_r2` distinct from `r_squared` |

Both R² sites compute the textbook coefficient of determination. The
formula `1 - SS_res / SS_tot` with `SS_tot = Σ(y - ȳ_eval)²` is exactly
what scikit-learn's `r2_score` returns. They do **not** use a
"reference variance from the full dataset" — both denominators are
computed on the eval subset only, which is correct.

### Sub-question answers

**a) sklearn r2_score vs manual?**
Both manual (`1.0 - ss_res / ss_tot`). residualsStudy.py imports
`sklearn.metrics.r2_score` on line 10 but does not call it — uses the
manual formula on lines 143-145. The two are mathematically identical.

**b) y_true and y_pred scales?**
Both back-transformed to raw OC g/kg before metric computation.
- `train.py:309-318` — if `--target_transform log`, applies `np.exp` to
  both outputs and targets; if `normalize`, applies `* std + mean` to
  both. Then `r_squared` on those raw values.
- `residualsStudy.py:94-128` — same inverse path. `original_targets`
  is captured before any transformation (line 104), `outputs` is back-
  transformed inline (lines 117-120). The variable `targets_transformed`
  on line 108-112 is computed but **never used downstream** (dead code,
  harmless).

**c) Back-transformation applied to both?**
Yes, symmetrically. Same inverse applied to outputs and targets at the
same site.

**d) SStot from eval subset or full dataset?**
**Eval subset.** `np.mean(original_test_targets)` (`train.py:330`) and
`np.mean(targets)` inside `calculate_metrics` (`residualsStudy.py:144`).
This is correct per scikit-learn.

**e) Same scale (g/kg vs %)?**
Both in g/kg. Source: `MAX_OC = 150` in `config.py:27` is incompatible
with % units (which would be 0-10). The OC column is loaded as numeric
without unit conversion. No scale mismatch between predictions and
targets.

### Asymmetry in forward/inverse transforms (minor, sub-bug)

`train.py:294` forward: `targets = torch.log(targets + 1e-10)`
`train.py:311` inverse: `original_test_outputs = np.exp(test_outputs_all)`

Mathematical inverse of `log(y + ε)` is `exp(y_log) - ε`, not just
`exp(y_log)`. The discrepancy is `1e-10 g/kg`, ~12 orders of magnitude
below typical OC values — **completely negligible**. Same pattern for
`normalize`: forward divides by `(σ + 1e-10)`, inverse multiplies by
`σ`. Same negligibility.

Not a bug worth fixing; flagging only for completeness.

### Statistics-source leak (not a metric leak)

`train.py:481-488` — `compute_training_statistics_oc()` computes
`target_mean / target_std` on the **full filtered dataset**
(`filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)`), not on the
train subset. This means test set rows contribute to the normalization
constants.

Impact on R²: **zero**, because both predictions and targets are
back-transformed by the same `target_mean / target_std` constants —
the R² computation is invariant to a shared affine transform. Impact
on training: the loss landscape sees data normalized by stats that
include test-set OC values; arguably a soft leak but it does not
inflate the reported R² number. Flagging but not pursuing further
unless the user wants me to.

### Train-R² is NOT computed during training

The training loop computes R² only on the test set
(`train.py:282-335`). There is no per-epoch train-set R² evaluation,
so we cannot read off "train R² vs val R²" from the training logs
without re-running inference. residualsStudy.py does compute train R²
post-hoc (`:584` calls `run_inference` on `train_loader`), so that
number is available if a checkpoint exists.

This matters for Section 4 (model collapse diagnostics) — we'll need
residualsStudy outputs or a fresh inference run to compare train vs
val R².

### Conclusion for Section 1

R² is computed **correctly** on the OC g/kg scale, with proper inverse-
transformation applied symmetrically to predictions and targets, using
the eval subset's own variance as the denominator. The 0.25 number is
not a metric-computation artifact.

**No fixes recommended in this section.** Two minor observations:
(i) target_mean / target_std are computed on the full dataset (soft
leak, doesn't affect reported R²); (ii) train R² is not logged
per-epoch, which limits one diagnostic we want in Section 4.

---

*Sections 2–8 pending. Stop here per the user's "STOP after each
section" instruction.*
