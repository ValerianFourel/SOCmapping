#!/usr/bin/env python3
"""
rebuttal/figures/figdata.py — honest, single-source data access for the
revision figure set. Every figure reads metrics from disk through here; no
hardcoded numbers.

Reuses sweep_summarize.collect() (the exact parser behind the 297-row
ranking) so figures cannot drift from the table. Adds:
  - load_ranking(sweep_dir)         : ranking rows + score + canonical family + config_dir
  - select(rows, axis/bands/max_oc/family) and best_per_family(...)
  - load_fold_predictions(config_dir) : pooled hold-out preds (cols GPS_LAT,
                                        GPS_LONG, OC_actual, OC_predicted, fold_id, year)
  - fold_boundaries(config_dir)     : per-fold lat/lon lo/hi + geometry label + axis
  - Manifest                        : per-figure provenance + DISCREPANCIES log

Defaults point at the repo sweep dir; pass --sweep-dir on JUPITER where the
canonical 43-band / lon-blocked runs live (the local SOCrebuttal_HF/sweep
bundle is the STALE lat / 20-band experiment — same schema, wrong numbers).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]
KFOLD_DIR = SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'
sys.path.insert(0, str(KFOLD_DIR))

SWEEP_DIR_DEFAULT = KFOLD_DIR / 'sweep'
MAPS_DIR_DEFAULT = SOC_ROOT / 'rebuttal' / 'final_models' / 'maps' / '_locations_400000rand_seed42'
CKPT_DIR_DEFAULT = SOC_ROOT / 'rebuttal' / 'final_models' / 'checkpoints'

_FAM_MAP = {
    'sgt': 'sgt',
    'vanilla_transformer': 'vanilla',
    'simpletransformer': 'simpletransformer',
    'lightweight_transformer': 'lightweight',
    'cnnlstm': 'cnnlstm',
    '3dcnn': '3dcnn',
}


def family_of(row: dict) -> str:
    """Canonical family key (matches figstyle.FAMILIES)."""
    mn = (row.get('model_name') or '').lower()
    if mn in _FAM_MAP:
        return _FAM_MAP[mn]
    t = (row.get('tag') or '').lower()
    if 'xgb' in t or mn == 'xgb':
        return 'xgb'
    if 'rf' in t or mn == 'rf':
        return 'rf'
    mf = (row.get('model_family') or '').lower()
    return _FAM_MAP.get(mf, mf or 'unknown')


def load_ranking(sweep_dir=None) -> list[dict]:
    """All sweep config rows (status ok), each with score, family, config_dir.

    Sorted by score (mean_R2 - 0.5*SD) descending, like sweep_summarize.
    """
    import sweep_summarize as ss
    sd = Path(sweep_dir) if sweep_dir else SWEEP_DIR_DEFAULT
    ss.SWEEP_DIR = sd                          # collect() reads this module global
    rows = [r for r in ss.collect() if r.get('r2_mean') is not None]
    for r in rows:
        r['score'] = float(r['r2_mean']) - 0.5 * float(r.get('r2_std') or 0.0)
        r['family'] = family_of(r)
        grp = r.get('sweep_group') or ''
        r['config_dir'] = (sd / r['tag']) if grp in ('', '(top)') \
            else (sd / grp / r['tag'])
    rows.sort(key=lambda r: r['score'], reverse=True)
    return rows


def select(rows, *, axis=None, bands=None, max_oc=None, family=None,
           n_folds=None) -> list[dict]:
    out = []
    for r in rows:
        if axis is not None and r.get('axis') != axis:
            continue
        if bands is not None and str(r.get('bands')) != str(bands):
            continue
        if max_oc is not None and r.get('max_oc') not in (None, float(max_oc)):
            continue
        if family is not None and r.get('family') != family:
            continue
        if n_folds is not None and r.get('n_folds') != n_folds:
            continue
        out.append(r)
    return out


def best_per_family(rows, *, axis, bands, max_oc=None, families=None) -> dict:
    """Best (highest-score) row per family at a fixed protocol."""
    fams = families or ['sgt', 'vanilla', 'simpletransformer', 'lightweight',
                        'rf', 'xgb', 'cnnlstm', '3dcnn']
    best = {}
    for fam in fams:
        cand = select(rows, axis=axis, bands=bands, max_oc=max_oc, family=fam)
        if cand:
            best[fam] = max(cand, key=lambda r: r['score'])
    return best


def load_fold_predictions(config_dir) -> pd.DataFrame | None:
    """Pooled hold-out predictions for one config (all 10 folds concatenated).

    Prefers kfold_predictions_all_folds.parquet, else concatenates
    fold_*_predictions.parquet. Returns None if neither exists.
    """
    config_dir = Path(config_dir)
    allp = config_dir / 'kfold_predictions_all_folds.parquet'
    if allp.exists():
        return pd.read_parquet(allp)
    parts = sorted(config_dir.glob('fold_*_predictions.parquet'))
    if not parts:
        return None
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def fold_boundaries(config_dir) -> dict | None:
    """Per-fold geometry from kfold_results_summary.json.

    Returns {'axis': 'lon'|'lat', 'geometry': str, 'n_folds': int,
             'buffer_km': float, 'folds': [{fold_id, lo, hi, n_test, r2}, ...]}.
    Detects axis from split_axis or the lo/hi key names present.
    """
    import json
    p = Path(config_dir) / 'kfold_results_summary.json'
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    fr = j.get('fold_results', [])
    axis = j.get('split_axis')
    if not axis:
        k0 = fr[0] if fr else {}
        axis = 'lon' if 'lon_lo' in k0 else 'lat'
    lo_k, hi_k = f'{axis}_lo', f'{axis}_hi'
    folds = [{'fold_id': f.get('fold_id'), 'lo': f.get(lo_k), 'hi': f.get(hi_k),
              'n_test': f.get('n_test'), 'r2': f.get('r2')} for f in fr]
    return {'axis': axis, 'geometry': j.get('fold_geometry'),
            'n_folds': j.get('n_folds', len(fr)),
            'buffer_km': j.get('distance_threshold_km'), 'folds': folds}


class Manifest:
    """Accumulates per-figure provenance and discrepancies, writes markdown."""

    def __init__(self):
        self.rows = []          # (label, files, source, numbers, status)
        self.discrepancies = []  # (what, on_disk, manuscript, fig)

    def done(self, label, files, source, numbers):
        self.rows.append((label, files, source, numbers, 'DONE'))

    def block(self, label, reason, path=''):
        self.rows.append((label, '-', path, reason, 'BLOCKED'))
        print(f'[BLOCKED] {label}: {reason}  {path}', file=sys.stderr)

    def discrepancy(self, what, on_disk, manuscript, fig=''):
        self.discrepancies.append((what, on_disk, manuscript, fig))
        print(f'[DISCREPANCY] {what}: on_disk={on_disk} manuscript={manuscript} '
              f'(fig {fig})', file=sys.stderr)

    def write(self, path):
        path = Path(path)
        L = ['# FIGURE_MANIFEST', '',
             '| Figure | Output | Source | Key numbers | Status |',
             '|---|---|---|---|---|']
        for label, files, source, numbers, status in self.rows:
            f = files if isinstance(files, str) else ', '.join(map(str, files))
            L.append(f'| {label} | {f} | {source} | {numbers} | {status} |')
        L += ['', '## DISCREPANCIES', '',
              '| What | On disk | Manuscript | Dependent figure |',
              '|---|---|---|---|']
        for what, on_disk, manuscript, fig in self.discrepancies:
            L.append(f'| {what} | {on_disk} | {manuscript} | {fig} |')
        path.write_text('\n'.join(L) + '\n')
        print(f'[manifest] wrote {path}')
