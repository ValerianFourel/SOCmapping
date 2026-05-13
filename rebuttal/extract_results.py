#!/usr/bin/env python3
"""
extract_results.py

Reproducible extraction of every numerical result already saved for the
Geoderma SGT rebuttal. Walks four root directories, catalogs every file,
loads every parquet / pkl / csv it finds, and recomputes per-run metrics
(R2, RMSE, MAE, RPIQ) directly from the predictions/targets stored in
each `analysis_results.pkl` (and from any CSVs containing predictions).

Outputs:
    rebuttal/results_inventory.md   - markdown audit log (this file overwrites)
    rebuttal/results_summary.json   - machine-readable summary

Run with the project venv that has pandas/numpy/pyarrow:
    /home/valerian/SGTPublication/SOCmapping/BaselinesXGBoostAndRF/.venv/bin/python \
        /home/valerian/SGTPublication/rebuttal/extract_results.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOTS = [
    Path('/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping'),
    Path('/home/valerian/SGTPublication/residual_Maps_Bavaria_1milTFT'),
    Path('/home/valerian/SGTPublication/residual_Maps_Bavaria_360kTFT'),
    Path('/home/valerian/SGTPublication/residual_Maps_Bavaria_v3_2milSimpleTransformer'),
]
OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal')
OUT_MD = OUT_DIR / 'results_inventory.md'
OUT_JSON = OUT_DIR / 'results_summary.json'

# File-extension filters we care about
DATA_EXTS = {'.parquet', '.pkl', '.csv', '.json', '.txt'}
SKIP_DIR_NAMES = {'.git', '__pycache__', 'wandb', '.venv'}

# Thresholds for high-SOC tail (g/kg)
HIGH_SOC_THRESHOLDS = (50.0, 120.0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fsize(p: Path) -> str:
    try:
        b = p.stat().st_size
    except OSError:
        return '?'
    for unit in ('B', 'K', 'M', 'G'):
        if b < 1024 or unit == 'G':
            return f'{b:6.1f}{unit}' if unit != 'B' else f'{b:6d}B'
        b /= 1024
    return f'{b}G'


def walk_files(root: Path):
    """Yield every interesting file under root (skipping .git / wandb / venv / pycache)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in DATA_EXTS:
                yield p


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    """R2 (Pearson^2), RMSE, MAE, RPIQ. Robust to NaN/empty arrays."""
    pred = np.asarray(pred, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[mask]
    actual = actual[mask]
    n = pred.size
    out = {'n': int(n)}
    if n < 2:
        out.update({'r2': float('nan'), 'rmse': float('nan'),
                    'mae': float('nan'), 'rpiq': float('nan'),
                    'bias': float('nan')})
        return out
    err = pred - actual
    out['rmse'] = float(np.sqrt(np.mean(err ** 2)))
    out['mae'] = float(np.mean(np.abs(err)))
    out['bias'] = float(np.mean(err))
    if np.std(pred) < 1e-12 or np.std(actual) < 1e-12:
        out['r2'] = 0.0
    else:
        rho = float(np.corrcoef(pred, actual)[0, 1])
        out['r2'] = float(rho ** 2)
    q1, q3 = np.percentile(actual, [25, 75])
    iqr = q3 - q1
    out['rpiq'] = float(iqr / out['rmse']) if out['rmse'] > 0 else float('inf')
    out['actual_mean'] = float(np.mean(actual))
    out['actual_std'] = float(np.std(actual))
    out['actual_min'] = float(np.min(actual))
    out['actual_max'] = float(np.max(actual))
    out['pred_mean'] = float(np.mean(pred))
    out['pred_std'] = float(np.std(pred))
    return out


def fmt_metrics(m: dict) -> str:
    if not m or np.isnan(m.get('r2', float('nan'))):
        return f"n={m.get('n', 0)} (insufficient data)"
    return (f"n={m['n']}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
            f"MAE={m['mae']:.4f}  RPIQ={m['rpiq']:.4f}  bias={m['bias']:+.4f}")


def safe_load_pickle(p: Path):
    """Load pickle, return (obj, err)."""
    try:
        with open(p, 'rb') as f:
            return pickle.load(f), None
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'


def tensor_to_np(x):
    """Convert torch.Tensor → ndarray without importing torch (works on raw tensor objects)."""
    try:
        import torch  # noqa
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(x, 'numpy'):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)


# --------------------------------------------------------------------------
# Per-file handlers
# --------------------------------------------------------------------------
def summarize_parquet(p: Path) -> dict:
    out = {'path': str(p), 'kind': 'parquet'}
    try:
        df = pd.read_parquet(p)
    except Exception as e:  # noqa: BLE001
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    out['shape'] = list(df.shape)
    out['columns'] = list(df.columns)
    out['dtypes'] = {c: str(t) for c, t in df.dtypes.items()}

    # Split breakdown if present
    split_col = next((c for c in ('dataset_type', 'split', 'subset') if c in df.columns), None)
    if split_col is not None:
        out['split_col'] = split_col
        out['split_counts'] = {str(k): int(v)
                               for k, v in df[split_col].value_counts().items()}

    # Descriptive stats for the interesting numeric columns
    target_cols = [c for c in df.columns
                   if c.lower() in {'oc', 'soc', 'target', 'actual', 'true_values'}]
    pred_cols = [c for c in df.columns
                 if c.lower() in {'predicted', 'prediction', 'predictions',
                                  'predicted_oc', 'predicted_soc'}]
    resid_cols = [c for c in df.columns if 'residual' in c.lower()]
    geo_cols = [c for c in df.columns
                if c.lower() in {'gps_long', 'gps_lat', 'longitude', 'latitude',
                                 'altitude', 'distance'}]
    year_cols = [c for c in df.columns if c.lower() == 'year']
    interesting = target_cols + pred_cols + resid_cols + geo_cols + year_cols
    if interesting:
        desc = df[interesting].describe(include='all').to_dict()
        # Strip non-JSON-serialisable entries
        for col, vals in desc.items():
            for k in list(vals.keys()):
                v = vals[k]
                if isinstance(v, (np.integer, np.floating)):
                    vals[k] = float(v)
                elif isinstance(v, pd.Timestamp):
                    vals[k] = str(v)
                elif pd.isna(v):
                    vals[k] = None
        out['describe'] = desc

    # Per-year mean/n for OC
    if 'year' in df.columns and 'OC' in df.columns:
        agg = df.groupby('year')['OC'].agg(['mean', 'count']).to_dict('index')
        out['per_year'] = {int(y): {'mean': float(v['mean']),
                                    'count': int(v['count'])}
                           for y, v in agg.items()}

    # High-SOC tail among the *validation* slice if dataset_type / split exists
    if split_col is not None and 'OC' in df.columns:
        val_mask = df[split_col].astype(str).str.lower().isin({'val', 'validation', 'valid'})
        val = df.loc[val_mask, 'OC']
        n_val = int(val.size)
        out['val_n'] = n_val
        if n_val:
            out['high_soc_validation'] = {
                f'>{t}': {'n': int((val > t).sum()),
                          'pct': float(100.0 * (val > t).mean())}
                for t in HIGH_SOC_THRESHOLDS}
        train_mask = df[split_col].astype(str).str.lower().isin({'train', 'training'})
        out['train_n'] = int(train_mask.sum())

    # If predicted *and* actual exist in same parquet → metrics
    if target_cols and pred_cols:
        actual = df[target_cols[0]].to_numpy()
        pred = df[pred_cols[0]].to_numpy()
        out['metrics_overall'] = compute_metrics(pred, actual)
        if split_col is not None:
            split_metrics = {}
            for s in df[split_col].dropna().unique():
                sub = df[df[split_col] == s]
                split_metrics[str(s)] = compute_metrics(
                    sub[pred_cols[0]].to_numpy(), sub[target_cols[0]].to_numpy())
            out['metrics_by_split'] = split_metrics
    return out


def summarize_pkl(p: Path) -> dict:
    out = {'path': str(p), 'kind': 'pkl'}
    obj, err = safe_load_pickle(p)
    if err:
        out['error'] = err
        return out
    out['type'] = type(obj).__name__
    if isinstance(obj, dict):
        out['keys'] = list(obj.keys())
        # Case A: analysis_results.pkl  (preds + targets per split)
        if {'train_results', 'val_results'} <= set(obj.keys()):
            splits = {}
            for sk in ('train_results', 'val_results'):
                r = obj.get(sk, {})
                if isinstance(r, dict) and 'predictions' in r and 'targets' in r:
                    pred = np.asarray(r['predictions'])
                    actual = np.asarray(r['targets'])
                    m = compute_metrics(pred, actual)
                    # Carry over pre-stored metrics for verification
                    stored = r.get('metrics', {})
                    stored_clean = {k: (float(v) if isinstance(v, (int, float, np.floating))
                                        else None)
                                    for k, v in stored.items()
                                    if not isinstance(v, np.ndarray)}
                    # High-SOC tail
                    high = {f'>{t}': {'n': int((actual > t).sum()),
                                       'pct': float(100.0 * (actual > t).mean())}
                            for t in HIGH_SOC_THRESHOLDS} if actual.size else {}
                    splits[sk] = {'metrics_recomputed': m,
                                  'metrics_stored': stored_clean,
                                  'high_soc_actual': high}
            out['analysis_results'] = splits
            out['model_path'] = str(obj.get('model_path', ''))
            stats = obj.get('stats', {})
            if isinstance(stats, dict):
                out['stats_target_mean'] = (float(stats['target_mean'])
                                            if 'target_mean' in stats else None)
                out['stats_target_std'] = (float(stats['target_std'])
                                           if 'target_std' in stats else None)
                fm = stats.get('feature_means')
                fs = stats.get('feature_stds')
                if fm is not None:
                    arr = tensor_to_np(fm)
                    out['feature_means_shape'] = list(arr.shape)
                    out['feature_means_summary'] = {
                        'min': float(arr.min()), 'max': float(arr.max()),
                        'mean': float(arr.mean())}
                if fs is not None:
                    arr = tensor_to_np(fs)
                    out['feature_stds_shape'] = list(arr.shape)
                    out['feature_stds_summary'] = {
                        'min': float(arr.min()), 'max': float(arr.max()),
                        'mean': float(arr.mean())}
            return out

        # Case B: normalization_stats_*.pkl  (no predictions)
        if {'feature_means', 'feature_stds'} <= set(obj.keys()):
            fm = tensor_to_np(obj['feature_means'])
            fs = tensor_to_np(obj['feature_stds'])
            out['feature_means_shape'] = list(fm.shape)
            out['feature_stds_shape'] = list(fs.shape)
            out['feature_means_summary'] = {'min': float(fm.min()),
                                            'max': float(fm.max()),
                                            'mean': float(fm.mean())}
            out['feature_stds_summary'] = {'min': float(fs.min()),
                                           'max': float(fs.max()),
                                           'mean': float(fs.mean())}
            if 'target_mean' in obj:
                out['target_mean'] = float(obj['target_mean'])
            if 'target_std' in obj:
                out['target_std'] = float(obj['target_std'])
            # Per-channel means / stds (squeeze the time axis)
            try:
                channel_means = fm.mean(axis=tuple(range(1, fm.ndim))) if fm.ndim > 1 else fm
                channel_stds = fs.mean(axis=tuple(range(1, fs.ndim))) if fs.ndim > 1 else fs
                out['channel_means'] = [float(x) for x in np.asarray(channel_means).ravel()]
                out['channel_stds'] = [float(x) for x in np.asarray(channel_stds).ravel()]
            except Exception as e:  # noqa: BLE001
                out['channel_means_err'] = str(e)
            return out

        # Other dict pickles
        out['preview'] = {k: type(v).__name__ for k, v in obj.items()}
    else:
        out['repr'] = str(obj)[:200]
    return out


def summarize_csv(p: Path) -> dict:
    out = {'path': str(p), 'kind': 'csv'}
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    out['shape'] = list(df.shape)
    out['columns'] = list(df.columns)
    # If small (metrics table) → embed fully
    if df.shape[0] <= 40 and df.shape[1] <= 12:
        out['table'] = df.to_dict(orient='records')
    # Compute metrics if predictions + actuals are in this CSV
    cols_l = {c.lower(): c for c in df.columns}
    actual_key = next((cols_l[k] for k in
                       ('true_values', 'actual', 'oc', 'soc', 'target', 'y_true')
                       if k in cols_l), None)
    pred_key = next((cols_l[k] for k in
                     ('predictions', 'predicted', 'prediction', 'predicted_oc',
                      'predicted_soc', 'y_pred')
                     if k in cols_l), None)
    if actual_key and pred_key:
        out['metrics_overall'] = compute_metrics(df[pred_key].to_numpy(),
                                                  df[actual_key].to_numpy())
    return out


def summarize_text_metrics(p: Path) -> dict:
    """Heuristic: keep training_metrics_*.txt, EXPERIMENT_SUMMARY.txt content (short)."""
    out = {'path': str(p), 'kind': 'txt'}
    try:
        with open(p, 'r', errors='replace') as f:
            txt = f.read()
        out['size'] = len(txt)
        # Pull a small preview window (first 50 lines)
        lines = txt.splitlines()
        out['preview'] = '\n'.join(lines[:80])
    except Exception as e:  # noqa: BLE001
        out['error'] = f'{type(e).__name__}: {e}'
    return out


def summarize_json(p: Path) -> dict:
    out = {'path': str(p), 'kind': 'json'}
    try:
        with open(p, 'r') as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    if isinstance(data, dict):
        out['keys'] = list(data.keys())[:30]
        # Heuristic: extract config-style top-level scalars
        flat = {}
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                flat[k] = v
            elif isinstance(v, list) and len(v) < 20:
                flat[k] = v
        out['flat'] = flat
    elif isinstance(data, list):
        out['list_len'] = len(data)
    return out


# --------------------------------------------------------------------------
# Main pass
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory: dict = {'roots': [str(r) for r in ROOTS], 'files': {}}
    per_kind: dict = {'parquet': [], 'pkl': [], 'csv': [], 'json': [], 'txt': []}
    errors: list = []

    all_files = []
    for root in ROOTS:
        if not root.exists():
            errors.append(f'MISSING ROOT: {root}')
            continue
        all_files.extend(walk_files(root))
    all_files.sort()
    print(f'Scanning {len(all_files)} files…', flush=True)

    # Limit JSON/TXT to ones that actually look interesting (avoid wandb noise)
    interesting_txt_keywords = ('training_metrics', 'EXPERIMENT_SUMMARY',
                                'training_summary', 'split_stats',
                                'training_metrics_summary')
    interesting_json_keywords = ('experiment_config', 'detailed_metrics',
                                 'config.json', 'metrics.json',
                                 'training_summary')

    for p in all_files:
        ext = p.suffix.lower()
        try:
            if ext == '.parquet':
                rec = summarize_parquet(p)
                per_kind['parquet'].append(rec)
            elif ext == '.pkl':
                rec = summarize_pkl(p)
                per_kind['pkl'].append(rec)
            elif ext == '.csv':
                rec = summarize_csv(p)
                per_kind['csv'].append(rec)
            elif ext == '.json':
                if any(kw in p.name for kw in interesting_json_keywords):
                    rec = summarize_json(p)
                    per_kind['json'].append(rec)
            elif ext == '.txt':
                if any(kw in p.name for kw in interesting_txt_keywords):
                    rec = summarize_text_metrics(p)
                    per_kind['txt'].append(rec)
        except Exception as e:  # noqa: BLE001
            errors.append(f'{p}: {type(e).__name__}: {e}')
            traceback.print_exc()

    # Group files by parent dir for the inventory section
    by_dir: dict = {}
    for p in all_files:
        parent = str(p.parent)
        by_dir.setdefault(parent, []).append(p)

    # ----- Write markdown -----
    md_lines: list[str] = []
    md_lines.append('# Geoderma SGT Rebuttal — Results Inventory')
    md_lines.append('')
    md_lines.append(f'_Generated by `rebuttal/extract_results.py`. Walks {len(ROOTS)} roots, '
                    f'inventories {len(all_files)} data files._')
    md_lines.append('')
    md_lines.append('Roots scanned:')
    for r in ROOTS:
        md_lines.append(f'- `{r}` {"(exists)" if r.exists() else "(MISSING)"}')
    md_lines.append('')
    md_lines.append(f'File-kind counts: ' +
                    ', '.join(f'**{k}**={len(v)}' for k, v in per_kind.items()))
    md_lines.append('')
    if errors:
        md_lines.append(f'⚠️ {len(errors)} loader errors collected — see end of file.')
        md_lines.append('')

    # ---- Section 1: directory inventory ----
    md_lines.append('---')
    md_lines.append('## 1. Directory inventory')
    md_lines.append('')
    md_lines.append('Each block below is one directory containing data files of interest. '
                    'Sizes given in bytes/KiB/MiB.')
    md_lines.append('')
    for d in sorted(by_dir):
        md_lines.append(f'### `{d}`')
        md_lines.append('')
        md_lines.append('| File | Size | Kind |')
        md_lines.append('|------|------|------|')
        for p in sorted(by_dir[d]):
            md_lines.append(f'| `{p.name}` | {fsize(p)} | {p.suffix.lstrip(".")} |')
        md_lines.append('')

    # ---- Section 2: parquet summaries ----
    md_lines.append('---')
    md_lines.append('## 2. Parquet files')
    md_lines.append('')
    md_lines.append(f'_{len(per_kind["parquet"])} parquet files inspected._')
    md_lines.append('')
    for rec in per_kind['parquet']:
        md_lines.append(f'### `{rec["path"]}`')
        if 'error' in rec:
            md_lines.append(f'  ❌ load error: `{rec["error"]}`')
            md_lines.append('')
            continue
        md_lines.append(f'- shape: **{rec["shape"][0]} rows × {rec["shape"][1]} cols**')
        md_lines.append(f'- columns: `{rec["columns"]}`')
        md_lines.append('- dtypes:')
        for c, t in rec['dtypes'].items():
            md_lines.append(f'    - `{c}`: {t}')
        if 'split_col' in rec:
            md_lines.append(f'- split column: `{rec["split_col"]}` — '
                            f'{rec["split_counts"]}')
        if 'train_n' in rec or 'val_n' in rec:
            md_lines.append(f'- train n: {rec.get("train_n", 0)}, '
                            f'val n: {rec.get("val_n", 0)}')
        if 'describe' in rec:
            md_lines.append('- descriptive stats for OC/geo/year columns:')
            md_lines.append('')
            md_lines.append('  | column | count | mean | std | min | 25% | 50% | 75% | max |')
            md_lines.append('  |--------|-------|------|-----|-----|-----|-----|-----|-----|')
            for c, vals in rec['describe'].items():
                row = [vals.get('count'), vals.get('mean'), vals.get('std'),
                       vals.get('min'), vals.get('25%'), vals.get('50%'),
                       vals.get('75%'), vals.get('max')]
                row = ['—' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v))
                       for v in row]
                md_lines.append(f'  | `{c}` | {" | ".join(row)} |')
            md_lines.append('')
        if 'high_soc_validation' in rec:
            md_lines.append('- validation-slice high-SOC tail:')
            for k, v in rec['high_soc_validation'].items():
                md_lines.append(f'    - OC {k} g/kg: n={v["n"]} ({v["pct"]:.2f}%)')
        if 'per_year' in rec:
            md_lines.append('- per-year OC mean / count:')
            md_lines.append('')
            md_lines.append('  | year | mean OC | n |')
            md_lines.append('  |------|---------|---|')
            for y in sorted(rec['per_year']):
                pv = rec['per_year'][y]
                md_lines.append(f'  | {y} | {pv["mean"]:.3f} | {pv["count"]} |')
            md_lines.append('')
        if 'metrics_overall' in rec:
            md_lines.append(f'- **metrics (predictions ⨯ actuals in same file)**: '
                            f'{fmt_metrics(rec["metrics_overall"])}')
            for s, m in rec.get('metrics_by_split', {}).items():
                md_lines.append(f'    - split=`{s}`: {fmt_metrics(m)}')
        md_lines.append('')

    # ---- Section 3: pkl summaries ----
    md_lines.append('---')
    md_lines.append('## 3. Pickle files')
    md_lines.append('')
    md_lines.append(f'_{len(per_kind["pkl"])} pickle files inspected._')
    md_lines.append('')
    md_lines.append('Pickles fall into two families: **(a) `normalization_stats_*.pkl`** '
                    '(per-channel feature means/stds + target mean/std) and '
                    '**(b) `analysis_results.pkl`** (predictions + targets + pre-stored '
                    'train/val metrics).')
    md_lines.append('')
    for rec in per_kind['pkl']:
        md_lines.append(f'### `{rec["path"]}`')
        if 'error' in rec:
            md_lines.append(f'  ❌ load error: `{rec["error"]}`')
            md_lines.append('')
            continue
        md_lines.append(f'- root type: `{rec.get("type")}` — keys: `{rec.get("keys")}`')
        # analysis_results format
        if 'analysis_results' in rec:
            md_lines.append(f'- model path (from pickle): `{rec.get("model_path", "")}`')
            if rec.get('stats_target_mean') is not None:
                md_lines.append(f'- target mean / std (saved): '
                                f'{rec["stats_target_mean"]:.4f} / '
                                f'{rec["stats_target_std"]:.4f}')
            if 'feature_means_shape' in rec:
                fms = rec['feature_means_summary']
                md_lines.append(f'- feature_means: shape {rec["feature_means_shape"]}  '
                                f'min={fms["min"]:.4g} max={fms["max"]:.4g} '
                                f'mean={fms["mean"]:.4g}')
            if 'feature_stds_shape' in rec:
                fss = rec['feature_stds_summary']
                md_lines.append(f'- feature_stds: shape {rec["feature_stds_shape"]}  '
                                f'min={fss["min"]:.4g} max={fss["max"]:.4g} '
                                f'mean={fss["mean"]:.4g}')
            for sk, sv in rec['analysis_results'].items():
                md_lines.append(f'- **{sk}**:')
                mr = sv['metrics_recomputed']
                ms = sv['metrics_stored']
                md_lines.append(f'    - recomputed: {fmt_metrics(mr)}')
                if ms:
                    keep = {k: ms[k] for k in ('r2', 'rmse', 'mae', 'rpiq',
                                               'bias', 'mean_residual',
                                               'std_residual', 'residual_iqr')
                            if k in ms}
                    md_lines.append(f'    - stored:     '
                                    + ', '.join(f'{k}={v:.4f}' for k, v
                                                in keep.items()
                                                if v is not None))
                hs = sv.get('high_soc_actual', {})
                if hs:
                    md_lines.append('    - high-SOC tail (actual):')
                    for k, v in hs.items():
                        md_lines.append(f'        - OC {k} g/kg: n={v["n"]} '
                                        f'({v["pct"]:.2f}%)')
        elif 'feature_means_shape' in rec:
            md_lines.append(f'- feature_means shape: {rec["feature_means_shape"]}')
            md_lines.append(f'- feature_stds shape: {rec["feature_stds_shape"]}')
            fms = rec['feature_means_summary']
            fss = rec['feature_stds_summary']
            md_lines.append(f'- feature_means: min={fms["min"]:.4g} '
                            f'max={fms["max"]:.4g} mean={fms["mean"]:.4g}')
            md_lines.append(f'- feature_stds:  min={fss["min"]:.4g} '
                            f'max={fss["max"]:.4g} mean={fss["mean"]:.4g}')
            if 'target_mean' in rec:
                md_lines.append(f'- target_mean / target_std: '
                                f'{rec["target_mean"]:.4f} / {rec["target_std"]:.4f}')
            if 'channel_means' in rec:
                md_lines.append('- per-channel mean (collapsed over non-channel axes):')
                cm = rec['channel_means']
                md_lines.append(f'    `{[round(x, 3) for x in cm]}`')
                cs = rec['channel_stds']
                md_lines.append('- per-channel std (collapsed over non-channel axes):')
                md_lines.append(f'    `{[round(x, 3) for x in cs]}`')
        else:
            md_lines.append(f'- preview: {rec.get("preview")}')
        md_lines.append('')

    # ---- Section 4: csv summaries (performance_metrics & predictions) ----
    md_lines.append('---')
    md_lines.append('## 4. CSV files')
    md_lines.append('')
    md_lines.append(f'_{len(per_kind["csv"])} CSV files inspected._ '
                    'Tables small enough are embedded in full.')
    md_lines.append('')
    for rec in per_kind['csv']:
        md_lines.append(f'### `{rec["path"]}`')
        if 'error' in rec:
            md_lines.append(f'  ❌ load error: `{rec["error"]}`')
            md_lines.append('')
            continue
        md_lines.append(f'- shape: **{rec["shape"][0]} × {rec["shape"][1]}** — '
                        f'columns: `{rec["columns"]}`')
        if 'metrics_overall' in rec:
            md_lines.append(f'- **metrics (from CSV preds × actuals)**: '
                            f'{fmt_metrics(rec["metrics_overall"])}')
        if 'table' in rec and rec['table']:
            cols = list(rec['table'][0].keys())
            md_lines.append('')
            md_lines.append('  | ' + ' | '.join(cols) + ' |')
            md_lines.append('  |' + '|'.join(['---'] * len(cols)) + '|')
            for row in rec['table']:
                cells = []
                for c in cols:
                    v = row[c]
                    if isinstance(v, float):
                        cells.append(f'{v:.4f}')
                    else:
                        cells.append(str(v))
                md_lines.append('  | ' + ' | '.join(cells) + ' |')
            md_lines.append('')
        md_lines.append('')

    # ---- Section 5: cross-run comparison ----
    md_lines.append('---')
    md_lines.append('## 5. Cross-run comparison — all metrics in one table')
    md_lines.append('')
    md_lines.append('Each row is one model run / split for which we have predictions vs. '
                    'actuals saved. **R²** is the squared Pearson correlation, '
                    'RMSE/MAE are on the original SOC scale (g/kg), '
                    'RPIQ = (Q3 − Q1) / RMSE.')
    md_lines.append('')
    md_lines.append('> **Bias sign convention.** The `Bias` column here is `mean(prediction − actual)` '
                    '(positive = model over-predicts SOC). Some pre-stored '
                    '`analysis_results.pkl` files instead define `bias = mean(actual − prediction)`; '
                    'in §3 the “stored” line uses that opposite sign. The two have identical magnitude.')
    md_lines.append('')
    md_lines.append('| Source | Split | n | R² | RMSE | MAE | RPIQ | Bias |')
    md_lines.append('|--------|-------|---|----|------|-----|------|------|')
    cross_rows: list[dict] = []
    for rec in per_kind['pkl']:
        if 'analysis_results' not in rec:
            continue
        rel = str(Path(rec['path']).relative_to(Path('/home/valerian/SGTPublication')))
        for sk, sv in rec['analysis_results'].items():
            m = sv['metrics_recomputed']
            if np.isnan(m.get('r2', float('nan'))):
                continue
            row = {'source': rel, 'split': sk, **m}
            cross_rows.append(row)
            md_lines.append(f'| `{rel}` | {sk} | {m["n"]} | {m["r2"]:.4f} | '
                            f'{m["rmse"]:.4f} | {m["mae"]:.4f} | {m["rpiq"]:.4f} | '
                            f'{m["bias"]:+.4f} |')
    # Also include csv-derived metrics (e.g. SimpleTimeModel predictions)
    for rec in per_kind['csv']:
        if 'metrics_overall' in rec:
            rel = str(Path(rec['path']).relative_to(Path('/home/valerian/SGTPublication')))
            m = rec['metrics_overall']
            if np.isnan(m.get('r2', float('nan'))):
                continue
            cross_rows.append({'source': rel, 'split': 'csv', **m})
            md_lines.append(f'| `{rel}` | csv | {m["n"]} | {m["r2"]:.4f} | '
                            f'{m["rmse"]:.4f} | {m["mae"]:.4f} | {m["rpiq"]:.4f} | '
                            f'{m["bias"]:+.4f} |')
    md_lines.append('')

    # ---- Section 6: parquet-derived per-year + high-SOC summary ----
    md_lines.append('---')
    md_lines.append('## 6. Per-year and high-SOC tail (from parquet splits)')
    md_lines.append('')
    md_lines.append('Parquet snapshots only record actuals + split membership (no '
                    'predictions). The per-year SOC mean and the share of validation '
                    'samples in the right tail are computed here.')
    md_lines.append('')
    for rec in per_kind['parquet']:
        if 'per_year' not in rec and 'high_soc_validation' not in rec:
            continue
        md_lines.append(f'### `{rec["path"]}`')
        if 'high_soc_validation' in rec:
            md_lines.append('- validation high-SOC tail:')
            for k, v in rec['high_soc_validation'].items():
                md_lines.append(f'    - {k} g/kg: n={v["n"]} ({v["pct"]:.2f}%)')
        if 'per_year' in rec:
            md_lines.append('- per-year:')
            md_lines.append('')
            md_lines.append('  | year | mean OC | n |')
            md_lines.append('  |------|---------|---|')
            for y in sorted(rec['per_year']):
                pv = rec['per_year'][y]
                md_lines.append(f'  | {y} | {pv["mean"]:.3f} | {pv["count"]} |')
            md_lines.append('')
        md_lines.append('')

    # ---- Section 7: experiment summaries from txt + json ----
    if per_kind['txt'] or per_kind['json']:
        md_lines.append('---')
        md_lines.append('## 7. Experiment summary text / JSON')
        md_lines.append('')
        md_lines.append('Selected `training_metrics_*.txt`, `EXPERIMENT_SUMMARY.txt`, '
                        '`experiment_config.json` and `detailed_metrics.json` files. '
                        'Previews are the first 80 lines of each.')
        md_lines.append('')
        for rec in per_kind['txt']:
            md_lines.append(f'### `{rec["path"]}`')
            if 'error' in rec:
                md_lines.append(f'  ❌ load error: `{rec["error"]}`')
                md_lines.append('')
                continue
            md_lines.append('')
            md_lines.append('```')
            md_lines.append(rec.get('preview', ''))
            md_lines.append('```')
            md_lines.append('')
        for rec in per_kind['json']:
            md_lines.append(f'### `{rec["path"]}`')
            if 'error' in rec:
                md_lines.append(f'  ❌ load error: `{rec["error"]}`')
                md_lines.append('')
                continue
            md_lines.append(f'- keys: `{rec.get("keys")}`')
            flat = rec.get('flat', {})
            if flat:
                md_lines.append('- top-level scalars:')
                for k, v in flat.items():
                    md_lines.append(f'    - `{k}`: `{v}`')
            md_lines.append('')

    # ---- Errors at the end ----
    if errors:
        md_lines.append('---')
        md_lines.append('## ⚠️ Loader errors')
        md_lines.append('')
        for e in errors:
            md_lines.append(f'- `{e}`')
        md_lines.append('')

    # Write markdown + JSON
    OUT_MD.write_text('\n'.join(md_lines))
    summary = {
        'roots': [str(r) for r in ROOTS],
        'n_files': len(all_files),
        'per_kind_counts': {k: len(v) for k, v in per_kind.items()},
        'cross_run_metrics': cross_rows,
        'errors': errors,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nWrote {OUT_MD}  ({OUT_MD.stat().st_size:,} bytes)')
    print(f'Wrote {OUT_JSON} ({OUT_JSON.stat().st_size:,} bytes)')
    print(f'Cross-run metric rows: {len(cross_rows)}')
    if errors:
        print(f'Errors logged: {len(errors)} (see end of markdown)')


if __name__ == '__main__':
    main()
