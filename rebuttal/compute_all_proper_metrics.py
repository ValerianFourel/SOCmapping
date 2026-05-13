#!/usr/bin/env python3
"""
compute_all_proper_metrics.py — Walks the full Weights HF repo (or any
directory containing analysis_results.pkl files), recomputes ALL the
regression metrics with the proper definitions, and writes a
markdown + json summary.

Metrics reported:
  - n
  - pearson_r            — Pearson correlation
  - pearson_r2           — squared Pearson correlation (v1 manuscript "R²")
  - r2                   — COEFFICIENT OF DETERMINATION = 1 − SS_res / SS_tot
  - rmse                 — root mean squared error
  - mae                  — mean absolute error
  - bias                 — mean(pred − actual)
  - sd_y                 — SD(actual) on the same split
  - iqr_y                — IQR(actual)
  - rmse_over_sd_y       — RMSE / SD(y), ≤ 1 means useful (= 1 / RPD)
  - rpd                  — SD(y) / RMSE; calibrated so RPD=1 is mean predictor
  - rpiq                 — IQR(y) / RMSE; the v1 metric (uncalibrated)
  - rpiq_mean_predictor  — what a mean-only predictor would score for RPIQ
  - is_useful            — boolean: r2 > 0 (better than predicting the mean)
  - is_leakage           — model_path contains _R2_1.0000 (no held-out val)

Usage:
    python rebuttal/compute_all_proper_metrics.py [root_dir]

If root_dir is not given, defaults to scanning both local Weights/ and
the HF snapshot dir if present.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np


SEARCH_ROOTS = [
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping',
    '/home/valerian/SGTPublication/_hf_full_weights_repo',
    '/home/valerian/SGTPublication/residual_Maps_Bavaria_1milTFT',
    '/home/valerian/SGTPublication/residual_Maps_Bavaria_360kTFT',
    '/home/valerian/SGTPublication/residual_Maps_Bavaria_v3_2milSimpleTransformer',
    '/home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer/residual_analysis',
]

OUT_DIR = Path('/home/valerian/SGTPublication/SOCmapping/rebuttal')
OUT_MD = OUT_DIR / 'proper_r2_all_projects_FULL.md'
OUT_JSON = OUT_DIR / 'proper_r2_all_projects_FULL.json'


def compute_full_metrics(pred, actual):
    """Return a dict with every metric of interest. None if not computable."""
    pred = np.asarray(pred, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    mask = np.isfinite(pred) & np.isfinite(actual)
    p, a = pred[mask], actual[mask]
    if len(p) < 2 or a.std(ddof=0) < 1e-12:
        return None
    if p.std(ddof=0) < 1e-12:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(p, a)[0, 1])
    pearson_r2 = pearson_r ** 2
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(ss_res / len(p)))
    mae = float(np.mean(np.abs(p - a)))
    bias = float((p - a).mean())
    sd_y = float(a.std(ddof=1))
    q25, q75 = np.percentile(a, [25, 75])
    iqr_y = float(q75 - q25)
    rmse_over_sd_y = rmse / sd_y if sd_y > 0 else float('inf')
    rpd = sd_y / rmse if rmse > 0 else float('inf')
    rpiq = iqr_y / rmse if rmse > 0 else float('inf')
    rpiq_mean_predictor = iqr_y / sd_y if sd_y > 0 else float('nan')
    return {
        'n': int(len(p)),
        'pearson_r': pearson_r,
        'pearson_r2': pearson_r2,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'sd_y': sd_y,
        'iqr_y': iqr_y,
        'rmse_over_sd_y': rmse_over_sd_y,
        'rpd': rpd,
        'rpiq': rpiq,
        'rpiq_mean_predictor': rpiq_mean_predictor,
        'is_useful': bool(r2 > 0),
    }


_ARCH_PATTERNS = (
    ('TFT/SGT', r'TemporalFusionTransformer|residual_analysis_TFT|residual_Maps_Bavaria_1milTFT|residual_Maps_Bavaria_360kTFT|SpatiotemporalGatedTransformer|TFT360'),
    ('SimpleTransformer', r'2mil_SimpleTransformer|20k_SimpleTransformer|residual_Maps_Bavaria_v3_2milSimpleTransformer|SimpleTransformer'),
    ('CNN-LSTM', r'cnnlstm|CNNLSTM|CNN_?LSTM'),
    ('3DCNN', r'3DCNN|3dcnn'),
    ('2DCNN', r'2DCNN|2dcnn|resnet2dcnn'),
    ('XGBoost', r'xgboost|XGBoost'),
    ('RandomForest', r'(?<![A-Za-z])RF(?![A-Za-z])|RandomForest'),
)


def parse_run_meta(path, model_path=''):
    """Best-effort extraction of architecture / dataset size / transform /
    loss / validation flag. Tries both the path on disk AND the saved
    model_path (read out of the pickle) — the latter often has richer
    transform/loss tokens for rows in residual_Maps_Bavaria_*/."""
    text = (path + ' ' + str(model_path)).replace(' ', '')
    arch = '?'
    for name, pat in _ARCH_PATTERNS:
        if re.search(pat, text):
            arch = name
            break
    size_m = re.search(r'(\d+k|\d+mil|360k|1mil|2mil|20k)', text)
    size = size_m.group(1) if size_m else ''
    transform = '?'
    if re.search(r'TRANSFORM_log|_log_LOSS|transform_log', text):
        transform = 'log'
    elif re.search(r'TRANSFORM_normalize|transform_normalize|normalize', text):
        transform = 'normalize'
    elif re.search(r'TRANSFORM_none', text):
        transform = 'none'
    loss = '?'
    if re.search(r'LOSS_l1|loss_l1', text):
        loss = 'L1'
    elif re.search(r'LOSS_mse|loss_mse', text):
        loss = 'MSE'
    elif re.search(r'composite_l2|compositel2', text):
        loss = 'composite_l2 (≡MSE)'
    return {'arch': arch, 'size': size, 'transform': transform, 'loss': loss}


def is_data_leakage(model_path):
    """Detect the 'use_validation=False' / AllDataFit / fullDatasetRun placeholder.
    The v1 codebase tags such checkpoints with _R2_1.0000 in the filename."""
    if not model_path:
        return False
    return '_R2_1.0000' in model_path or 'AllDataFit' in model_path or 'fullDatasetRun' in model_path


def main():
    roots = sys.argv[1:] if len(sys.argv) > 1 else SEARCH_ROOTS
    files = []
    for r in roots:
        if not os.path.exists(r):
            continue
        files += glob.glob(os.path.join(r, '**', 'analysis_results.pkl'), recursive=True)
    files = sorted(set(files))
    print(f'Found {len(files)} analysis_results.pkl files across {len(roots)} roots', flush=True)

    rows = []
    for path in files:
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            print(f'ERROR: {path}: {e}')
            continue
        mp = str(d.get('model_path', ''))
        meta = parse_run_meta(path, mp)
        leakage = is_data_leakage(mp)
        for split in ('train_results', 'val_results'):
            sub = d.get(split, {})
            if 'predictions' not in sub or 'targets' not in sub:
                continue
            try:
                m = compute_full_metrics(sub['predictions'], sub['targets'])
            except Exception as e:
                print(f'ERROR in {path} / {split}: {e}')
                continue
            if m is None:
                continue
            row = {
                'path': path,
                'rel_path': path.replace('/home/valerian/SGTPublication/', ''),
                'split': split.replace('_results', ''),
                'model_path': str(d.get('model_path', '')),
                'is_leakage': bool(leakage),
                **meta,
                **m,
            }
            rows.append(row)

    print(f'\nProduced {len(rows)} (split × file) rows.\n', flush=True)

    # Dedup: identical (n, pearson_r2 rounded to 3 dp, rmse to 1 dp) within
    # same split. Tighter rounding than before so float-noise duplicates
    # like 0.5680 vs 0.5681 collapse. When picking the survivor, prefer
    # whichever row has the most metadata tokens recovered (i.e., fewer '?').
    def metadata_richness(r):
        return -sum(v == '?' or v == '' for v in (r.get('arch'), r.get('size'),
                                                    r.get('transform'), r.get('loss')))

    seen = OrderedDict()
    for r in sorted(rows, key=lambda x: (x['split'], metadata_richness(x), -x['r2'])):
        key = (r['split'], r['n'], round(r['pearson_r2'], 3), round(r['rmse'], 1))
        if key not in seen:
            seen[key] = r
            seen[key]['_aliases'] = [r['rel_path']]
        else:
            seen[key]['_aliases'].append(r['rel_path'])
    dedup_rows = list(seen.values())
    print(f'After dedup: {len(dedup_rows)} unique (split × identity) rows', flush=True)

    # Write JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({'rows': dedup_rows, 'all_rows': rows}, indent=2, default=str))
    print(f'Wrote {OUT_JSON}', flush=True)

    # Write markdown
    md = []
    md.append('# Full proper-R² leaderboard — every model in `ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping`')
    md.append('')
    md.append(f'_Source roots:_ {len(roots)} local directories incl. fresh HF snapshot at `_hf_full_weights_repo/`. '
              f'Scan found {len(files)} `analysis_results.pkl` files; '
              f'{len(rows)} (split × file) entries; '
              f'**{len(dedup_rows)} unique (split × identity)** rows after deduping mirror copies.')
    md.append('')
    md.append('## Metric glossary')
    md.append('')
    md.append('| metric | formula | interpretation |')
    md.append('|---|---|---|')
    md.append('| **`r2` (proper R²)** | `1 − Σ(a−p)² / Σ(a−ā)²` | 0 = mean-predictor; **negative = worse than predicting the mean** |')
    md.append('| `pearson_r2` | `corr(p,a)²` | What v1 paper called "R²"; bias-invariant |')
    md.append('| `rmse` | `√mean((p−a)²)` | absolute error in OC units (g/kg) |')
    md.append('| `mae` | `mean(|p−a|)` | robust absolute error (g/kg) |')
    md.append('| `bias` | `mean(p−a)` | systematic offset (g/kg) |')
    md.append('| `sd_y` / `iqr_y` | `SD / IQR of actual y` | spread of the target on this split |')
    md.append('| **`rmse_over_sd_y`** | `RMSE / SD(y)` | `< 1` = useful; `= 1` = mean-predictor; `> 1` = worse |')
    md.append('| **`rpd`** | `SD(y) / RMSE` | calibrated; `1` = mean-predictor; **`≥ 2` = good (DSM convention)** |')
    md.append('| `rpiq` | `IQR(y) / RMSE` | v1 metric; has non-zero floor at `rpiq_mean_predictor` |')
    md.append('| `rpiq_mean_predictor` | `IQR(y) / SD(y)` | what `rpiq` would be for a constant-mean predictor — the actual "zero-signal" floor |')
    md.append('| `is_useful` | `r2 > 0` | quick boolean — is this model better than predicting the mean? |')
    md.append('| `is_leakage` | model trained on full data, "val" is in-pool | `_R2_1.0000` in saved checkpoint name |')
    md.append('')

    # Val rows table — sort by proper R² desc
    md.append('## Val-set leaderboard (deduplicated, ranked by proper R²)')
    md.append('')
    md.append('| # | Architecture | Size | Transform | Loss | n_val | Pearson r² | **R² (proper)** | RMSE | bias | SD(y) | RMSE/SD | RPD | RPIQ | useful? | leak? |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    val_rows = [r for r in dedup_rows if r['split'] == 'val']
    val_rows.sort(key=lambda r: r['r2'], reverse=True)
    for i, r in enumerate(val_rows, 1):
        flag_useful = '✅' if r['is_useful'] else '❌'
        flag_leak = '⚠️ leak' if r['is_leakage'] else ''
        md.append(
            f'| {i} | {r["arch"]} | {r["size"]} | {r["transform"]} | {r["loss"]} '
            f'| {r["n"]} | {r["pearson_r2"]:.4f} | **{r["r2"]:+.4f}** '
            f'| {r["rmse"]:.2f} | {r["bias"]:+.2f} | {r["sd_y"]:.2f} '
            f'| {r["rmse_over_sd_y"]:.2f} | {r["rpd"]:.2f} | {r["rpiq"]:.2f} '
            f'| {flag_useful} | {flag_leak} |'
        )
    md.append('')

    # Same for train
    md.append('## Train-set rows (for completeness)')
    md.append('')
    md.append('| # | Architecture | Size | Transform | Loss | n_train | Pearson r² | R² (proper) | RMSE | bias | RPD | RPIQ |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    train_rows = [r for r in dedup_rows if r['split'] == 'train']
    train_rows.sort(key=lambda r: r['r2'], reverse=True)
    for i, r in enumerate(train_rows, 1):
        md.append(
            f'| {i} | {r["arch"]} | {r["size"]} | {r["transform"]} | {r["loss"]} '
            f'| {r["n"]} | {r["pearson_r2"]:.4f} | {r["r2"]:+.4f} '
            f'| {r["rmse"]:.2f} | {r["bias"]:+.2f} '
            f'| {r["rpd"]:.2f} | {r["rpiq"]:.2f} |'
        )
    md.append('')

    md.append('## Notes')
    md.append('')
    md.append('- **`leak`** rows are re-evaluations of an "AllDataFit" / "fullDatasetRun" '
              'model (trained with `use_validation=False`) on rows that were inside its '
              'training pool. They look better than the corresponding parent file but '
              '**must not be cited as held-out R²**.')
    md.append('- **`useful`** is just `r2 > 0`. A model worse than predicting the mean '
              'has negative R² and `useful = ❌`. Such models also have `RPD < 1` and '
              '`RMSE/SD > 1`.')
    md.append('- **`rpiq_mean_predictor`** typically falls in 0.6–0.8 on Bavarian SOC '
              'val sets; **RPIQ values ≤ ~0.8 are at or below the mean-predictor floor.** '
              'RPIQ ≥ 2 = "good" by DSM convention.')
    md.append('')
    OUT_MD.write_text('\n'.join(md))
    print(f'Wrote {OUT_MD}', flush=True)


if __name__ == '__main__':
    main()
