#!/usr/bin/env python3
"""
bootstrap_cis.py — Task 1 of the Geoderma rebuttal.

Loads the TFT 1mil composite_l2_v2 run-1 train/val parquet, inspects schema,
gives descriptive stats for the train and validation splits, then bootstraps
95% CIs for R², RMSE, MAE and RPIQ.

The parquet itself stores only the actuals + split labels (no predictions),
so the predictions/targets used for the bootstrap are pulled from the
matching analysis_results.pkl produced by the residualsStudy step for the
same model run.

Outputs:
    rebuttal/bootstrap_results.json
    rebuttal/bootstrap_results.md

Run with the project venv:
    /home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python \
        /home/valerian/SGTPublication/rebuttal/bootstrap_cis.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_PATH = Path(
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/'
    'TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/'
    'train_val_data_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_'
    'TRANSFORM_normalize_LOSS_composite_l2.parquet'
)

# Predictions for the same model run (residualsStudy output)
PRED_PATH = Path(
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/'
    'Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/'
    'analysis_results.pkl'
)

OUT_DIR = Path(__file__).resolve().parent   # rebuttal/ next to this script
N_BOOT = 1000
RNG_SEED = 20260513     # rebuttal deadline date — fixed for reproducibility
HIGH_THRESHOLDS = (50.0, 120.0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def descriptive(series: pd.Series) -> dict:
    """Descriptive stats for one numeric series of SOC values."""
    s = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
    if s.size == 0:
        return {'n': 0}
    q25, q50, q75 = np.percentile(s, [25, 50, 75])
    return {
        'n': int(s.size),
        'mean': float(np.mean(s)),
        'sd': float(np.std(s, ddof=1)),
        'median': float(q50),
        'q25': float(q25),
        'q75': float(q75),
        'iqr': float(q75 - q25),
        'max': float(s.max()),
        'min': float(s.min()),
        'pct_gt_50': float(100.0 * (s > 50.0).mean()),
        'pct_gt_120': float(100.0 * (s > 120.0).mean()),
        'n_gt_50': int((s > 50.0).sum()),
        'n_gt_120': int((s > 120.0).sum()),
    }


def point_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Compute regression metrics. Reports BOTH:
      - `r2` (proper) = COEFFICIENT OF DETERMINATION = 1 - SS_res / SS_tot
        — what scikit-learn r2_score returns; sensitive to prediction bias
      - `pearson_r2` = squared Pearson correlation — what v1 paper called
        "R²"; invariant to additive/multiplicative shifts
    The two agree closely when predictions are unbiased; diverge when bias
    is large (spatial extrapolation typically causes this)."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = pred - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    if pred.std() > 0 and actual.std() > 0:
        rho = float(np.corrcoef(pred, actual)[0, 1])
    else:
        rho = 0.0
    pearson_r2 = rho ** 2
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    q25, q75 = np.percentile(actual, [25, 75])
    iqr = q75 - q25
    rpiq = float(iqr / rmse) if rmse > 0 else float('inf')
    return {'r2': float(r2), 'pearson_r2': float(pearson_r2),
            'pearson_r': float(rho), 'rmse': rmse, 'mae': mae, 'rpiq': rpiq,
            'bias_pred_minus_actual': bias, 'n': int(pred.size)}


def bootstrap_cis(pred: np.ndarray, actual: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = pred.size
    keys = ('r2', 'pearson_r2', 'pearson_r', 'rmse', 'mae', 'rpiq',
            'bias_pred_minus_actual')
    samples = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = point_metrics(pred[idx], actual[idx])
        for k in keys:
            samples[k][b] = m[k]
    cis = {}
    for k, vals in samples.items():
        cis[k] = {
            'mean': float(np.mean(vals)),
            'sd': float(np.std(vals, ddof=1)),
            'median': float(np.median(vals)),
            'ci95_lo': float(np.percentile(vals, 2.5)),
            'ci95_hi': float(np.percentile(vals, 97.5)),
        }
    cis['_boot_meta'] = {'n_boot': n_boot, 'seed': seed, 'n_samples': int(n)}
    return cis


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {'parquet_path': str(PARQUET_PATH),
                 'predictions_path': str(PRED_PATH)}

    # --- Load parquet ---
    df = pd.read_parquet(PARQUET_PATH)
    out['columns'] = list(df.columns)
    out['shape'] = list(df.shape)
    out['dtypes'] = {c: str(t) for c, t in df.dtypes.items()}

    print(f'Parquet shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'Split column: dataset_type — counts: {df["dataset_type"].value_counts().to_dict()}')
    print(f'Actual SOC column: OC')
    print('Predicted column: NOT in parquet (this file is a split snapshot, not predictions).\n')

    out['split_indicator_column'] = 'dataset_type'
    out['actual_column'] = 'OC'
    out['predicted_column_in_parquet'] = None
    out['note'] = ('Parquet holds the actuals + GPS_LONG/GPS_LAT/year/OC/survey_date/season/'
                   'bin/dataset_type only. Predictions for the same model run are loaded '
                   'from analysis_results.pkl produced by residualsStudy.py.')

    # --- Descriptive stats per split ---
    train_df = df[df['dataset_type'] == 'train']
    val_df = df[df['dataset_type'] == 'val']
    out['train_desc'] = descriptive(train_df['OC'])
    out['val_desc'] = descriptive(val_df['OC'])
    print(f'Train n: {out["train_desc"]["n"]}  Val n: {out["val_desc"]["n"]}')

    # --- Load predictions/targets pickle ---
    with open(PRED_PATH, 'rb') as f:
        ar = pickle.load(f)
    val_pred = np.asarray(ar['val_results']['predictions'], dtype=float)
    val_actual = np.asarray(ar['val_results']['targets'], dtype=float)
    train_pred = np.asarray(ar['train_results']['predictions'], dtype=float)
    train_actual = np.asarray(ar['train_results']['targets'], dtype=float)

    # Sanity: descriptive stats on pickle val targets, for cross-check vs parquet val OC
    out['val_pickle_targets_desc'] = descriptive(pd.Series(val_actual))
    out['train_pickle_targets_desc'] = descriptive(pd.Series(train_actual))

    # --- Point metrics (no bootstrap) ---
    out['val_point_metrics'] = point_metrics(val_pred, val_actual)
    out['train_point_metrics'] = point_metrics(train_pred, train_actual)

    # --- Bootstrap CIs on validation ---
    print(f'Bootstrapping validation set ({val_pred.size} samples, '
          f'{N_BOOT} iterations, seed={RNG_SEED})…')
    out['val_bootstrap'] = bootstrap_cis(val_pred, val_actual, N_BOOT, RNG_SEED)
    # Also report training-set bootstrap for completeness
    print(f'Bootstrapping training set ({train_pred.size} samples, '
          f'{N_BOOT} iterations)…')
    out['train_bootstrap'] = bootstrap_cis(train_pred, train_actual, N_BOOT, RNG_SEED + 1)

    # --- Persist JSON ---
    OUT_JSON = OUT_DIR / 'bootstrap_results.json'
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nWrote {OUT_JSON}')

    # --- Persist Markdown ---
    md = []
    md.append('# Bootstrap CIs — TFT 1 mil, composite_l2_v2, run 1')
    md.append('')
    md.append(f'_Source parquet:_ `{PARQUET_PATH}`')
    md.append('')
    md.append(f'_Predictions source:_ `{PRED_PATH}`')
    md.append('')
    md.append('## Column layout')
    md.append('')
    md.append('| column | dtype | notes |')
    md.append('|--------|-------|-------|')
    for c, t in out['dtypes'].items():
        note = ''
        if c == 'dataset_type':
            note = 'split indicator (`train` / `val`)'
        elif c == 'OC':
            note = 'actual SOC concentration (g/kg) — the regression target'
        elif c.startswith('GPS_'):
            note = 'geographic coordinate'
        elif c == 'year':
            note = 'sample year'
        md.append(f'| `{c}` | {t} | {note} |')
    md.append('')
    md.append('The parquet has **no** `predicted` column — it is the train/val split '
              'snapshot stored at training time. Predicted vs. actual pairs for the '
              'same model live in `analysis_results.pkl` (residualsStudy output), and '
              'are loaded from there for the bootstrap.')
    md.append('')

    md.append('## Descriptive statistics by split')
    md.append('')
    md.append('| metric | train | val |')
    md.append('|--------|-------|-----|')
    keys = [('n', '{:d}'), ('mean', '{:.3f}'), ('sd', '{:.3f}'),
            ('median', '{:.3f}'), ('q25', '{:.3f}'), ('q75', '{:.3f}'),
            ('iqr', '{:.3f}'), ('min', '{:.3f}'), ('max', '{:.3f}'),
            ('n_gt_50', '{:d}'), ('pct_gt_50', '{:.2f}%'),
            ('n_gt_120', '{:d}'), ('pct_gt_120', '{:.2f}%')]
    for k, fmt in keys:
        t = out['train_desc'].get(k)
        v = out['val_desc'].get(k)
        md.append(f'| `{k}` | {fmt.format(t)} | {fmt.format(v)} |')
    md.append('')
    md.append('### Cross-check — same stats on the pickled targets (predictions file)')
    md.append('')
    md.append('| metric | train (pickle) | val (pickle) |')
    md.append('|--------|----------------|--------------|')
    for k, fmt in keys:
        t = out['train_pickle_targets_desc'].get(k)
        v = out['val_pickle_targets_desc'].get(k)
        md.append(f'| `{k}` | {fmt.format(t)} | {fmt.format(v)} |')
    md.append('')
    md.append('(Pickle counts can differ from parquet counts by a handful: the parquet '
              'is the post-split snapshot at training time, the pickle records the '
              'samples the residualsStudy could actually re-encode at inference.)')
    md.append('')

    md.append('## Point estimates')
    md.append('')
    md.append('**Two flavours of R² are reported throughout this file:**')
    md.append('')
    md.append('- `r2` — **coefficient of determination** = 1 − SS_res / SS_tot. '
              'Scientifically meaningful "R²"; sensitive to prediction bias. '
              'What scikit-learn `r2_score` returns.')
    md.append('- `pearson_r2` — squared Pearson correlation. What the v1 manuscript '
              'and wandb panels called "R²"; invariant to additive/multiplicative '
              'shifts in predictions.')
    md.append('')
    md.append('| metric | train | val |')
    md.append('|--------|-------|-----|')
    for k in ('n', 'r2', 'pearson_r2', 'pearson_r', 'rmse', 'mae', 'rpiq', 'bias_pred_minus_actual'):
        tv = out['train_point_metrics'][k]
        vv = out['val_point_metrics'][k]
        if k == 'n':
            md.append(f'| `{k}` | {tv} | {vv} |')
        else:
            md.append(f'| `{k}` | {tv:.4f} | {vv:.4f} |')
    md.append('')

    md.append(f'## Bootstrap 95% CIs (n_boot={N_BOOT}, seed={RNG_SEED})')
    md.append('')
    md.append('Resampling with replacement on the **validation set** '
              f'({out["val_point_metrics"]["n"]} samples).')
    md.append('')
    md.append('| metric | mean | SD | median | 95% CI lo | 95% CI hi |')
    md.append('|--------|------|----|--------|-----------|-----------|')
    for k in ('r2', 'pearson_r2', 'rmse', 'mae', 'rpiq', 'bias_pred_minus_actual'):
        v = out['val_bootstrap'][k]
        md.append(f'| `{k}` | {v["mean"]:.4f} | {v["sd"]:.4f} | {v["median"]:.4f} | '
                  f'{v["ci95_lo"]:.4f} | {v["ci95_hi"]:.4f} |')
    md.append('')
    md.append(f'### Training-set bootstrap (for completeness, '
              f'{out["train_point_metrics"]["n"]} samples)')
    md.append('')
    md.append('| metric | mean | SD | median | 95% CI lo | 95% CI hi |')
    md.append('|--------|------|----|--------|-----------|-----------|')
    for k in ('r2', 'pearson_r2', 'rmse', 'mae', 'rpiq', 'bias_pred_minus_actual'):
        v = out['train_bootstrap'][k]
        md.append(f'| `{k}` | {v["mean"]:.4f} | {v["sd"]:.4f} | {v["median"]:.4f} | '
                  f'{v["ci95_lo"]:.4f} | {v["ci95_hi"]:.4f} |')
    md.append('')
    OUT_MD = OUT_DIR / 'bootstrap_results.md'
    OUT_MD.write_text('\n'.join(md))
    print(f'Wrote {OUT_MD}')


if __name__ == '__main__':
    main()
