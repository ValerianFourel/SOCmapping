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


def _collect_via_sweep_summarize(sd: Path):
    """Primary path (inside the repo): reuse sweep_summarize.collect() so the
    figures are byte-consistent with the published ranking. Raises on import
    failure (e.g. in a standalone HF-download bundle) -> caller falls back."""
    import sweep_summarize as ss
    ss.SWEEP_DIR = sd                          # collect() reads this module global
    return ss.collect()


# --- Vendored, dependency-free parser (mirrors sweep_summarize) so a bare HF
#     download of sweep/ regenerates figures without the rest of the repo. ----
import re as _re
_ARCH_RE = _re.compile(r'^(small_)?d(\d+)_h(\d+)_L(\d+)$')
_FAMILY_RE = _re.compile(r'^(vanilla_transformer|simpletransformer|'
                         r'lightweight_transformer|cnnlstm|3dcnn)_'
                         r'd(\d+)_h(\d+)_L(\d+)$')
_BASELINE_RE = _re.compile(r'^baseline_(rf|xgb)_(\w+)$')


def _parse_tag_standalone(tag: str):
    m = _ARCH_RE.match(tag)
    if m:
        return (('small' if m.group(1) else 'big'),
                int(m.group(2)), int(m.group(3)), int(m.group(4)), 'sgt')
    m = _FAMILY_RE.match(tag)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                m.group(1))
    m = _BASELINE_RE.match(tag)
    if m:
        return ('baseline', None, None, None, m.group(1))
    return None


def _bands_standalone(group: str) -> str:
    g = group or ''
    if '_6band' in g:
        return '6'
    if '_extband' in g or 'extended' in g:
        return '43'
    return '20'


_SEED_RE = _re.compile(r'_seed\d+')
_OC_RE = _re.compile(r'oc(\d+)')


def _strip_seed(group: str) -> str:
    """Collapse a seed-replica group to its seed-agnostic key, e.g.
    'lon_f10_oc150_ep100_l1_tp8_seed342_extband' ->
    'lon_f10_oc150_ep100_l1_tp8_seedAVG_extband'."""
    return _SEED_RE.sub('_seedAVG', group)


def _maxoc_from_group(group: str):
    """Infer the SOC cap from an 'oc150'/'oc120'/'oc90' token in the path."""
    m = _OC_RE.search(group or '')
    return float(m.group(1)) if m else None


def _collect_standalone(sd: Path):
    """Walk kfold_results_summary.json directly (no sweep_summarize needed).

    Captures both the *declared* protocol fold count (recipe/summary n_folds,
    e.g. 10 for longitude-deciles) and the number of folds that actually
    produced a valid R^2 — a canonical 10-fold config can legitimately ship
    8-9 valid folds if one fold failed, and we must keep it (it is still the
    10-fold protocol), not silently treat it as a different experiment."""
    import json
    rows = []
    skip = {'__pycache__', 'sbatch', 'slurm_logs', 'baseline_features'}
    for summ in sorted(Path(sd).rglob('kfold_results_summary.json')):
        cfg_dir = summ.parent
        rel = cfg_dir.relative_to(sd).parts
        if any(p in skip for p in rel):
            continue
        tag = rel[-1]
        group = '/'.join(rel[:-1]) or '(top)'
        parsed = _parse_tag_standalone(tag)
        if parsed is None:
            continue
        try:
            j = json.loads(summ.read_text())
        except Exception:
            continue
        ac = j.get('across_folds', {})
        recipe = j.get('recipe', {})
        fr = j.get('fold_results', [])
        axis = j.get('split_axis')
        if not axis:
            axis = 'lon' if (fr and 'lon_lo' in fr[0]) else 'lat'
        max_oc = recipe.get('max_oc')
        if max_oc is None:
            max_oc = _maxoc_from_group(group)
        rows.append({
            'tag': tag, 'sweep_group': group,
            'variant': parsed[0], 'd_model': parsed[1],
            'num_heads': parsed[2], 'num_layers': parsed[3],
            'model_name': parsed[4], 'model_family': recipe.get('model_family') or parsed[4],
            'model_size': recipe.get('model_size'),
            'band_arch': recipe.get('band_arch'),
            'seed_base': recipe.get('seed_base'),
            'axis': axis, 'bands': _bands_standalone(group),
            'n_bands': recipe.get('n_bands'), 'max_oc': max_oc,
            'loss': recipe.get('loss_type'),
            'n_folds': len(fr),
            'n_folds_declared': j.get('n_folds') or len(fr),
            'fold_geometry': j.get('fold_geometry'),
            'buffer_km': j.get('distance_threshold_km'),
            'fold_ids': [f.get('fold_id') for f in fr],
            'r2_mean': ac.get('r2_mean'), 'r2_std': ac.get('r2_std'),
            'rmse_mean': ac.get('rmse_mean'), 'mae_mean': ac.get('mae_mean'),
            'rpiq_mean': ac.get('rpiq_mean'),
            'r2_per_fold': [f.get('r2', float('nan')) for f in fr],
            'status': 'ok',
        })
    return rows


def _merge_seed_replicas(rows: list[dict]) -> list[dict]:
    """Collapse seed-replica configs (same protocol & architecture, differing
    only by `_seedNNN` in the group name) into ONE averaged row.

    Per the revision protocol the flagship metric is the AVERAGE over the
    seed replicas with the across-fold SD reported. We fold-wise average the
    per-fold R^2 across seeds (aligned by fold_id), then take the mean and
    sample SD of that averaged series. The merged row keeps `seed_dirs` (every
    replica's config dir) so prediction-level figures can pool the seeds.
    Singletons pass through unchanged.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (_strip_seed(r.get('sweep_group') or ''), r['tag'],
               r.get('n_folds_declared'))
        groups.setdefault(key, []).append(r)

    out = []
    for (sgroup, tag, _nf), members in groups.items():
        seeds = [m for m in members if m.get('seed_base') is not None]
        if len(members) < 2 or '_seedAVG' not in sgroup:
            out.extend(members)
            continue
        # align per-fold R^2 across replicas by fold_id
        by_fold: dict[int, list[float]] = {}
        for m in members:
            for fid, r2 in zip(m.get('fold_ids') or [], m.get('r2_per_fold') or []):
                if r2 is not None and np.isfinite(r2):
                    by_fold.setdefault(fid, []).append(float(r2))
        if not by_fold:
            out.extend(members)
            continue
        fids = sorted(by_fold)
        per_fold = [float(np.mean(by_fold[f])) for f in fids]
        base = dict(members[0])
        base['sweep_group'] = sgroup
        base['r2_per_fold'] = per_fold
        base['fold_ids'] = fids
        base['r2_mean'] = float(np.mean(per_fold))
        base['r2_std'] = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0
        base['n_folds'] = len(per_fold)
        base['n_seeds'] = len(members)
        base['seed_dirs'] = [m.get('config_dir') for m in members]
        base['rmse_mean'] = float(np.mean([m['rmse_mean'] for m in members
                                           if m.get('rmse_mean') is not None])) \
            if any(m.get('rmse_mean') is not None for m in members) else None
        base['is_seed_avg'] = True
        out.append(base)
    return out


def _config_dir(sd: Path, r: dict) -> Path:
    grp = r.get('sweep_group') or ''
    return (sd / r['tag']) if grp in ('', '(top)') else (sd / grp / r['tag'])


def load_ranking(sweep_dir=None, *, drop_quintile=True,
                 merge_seeds=True, drop_landuse=True) -> list[dict]:
    """All sweep config rows (status ok), each with score, family, config_dir.

    Deterministic, self-contained: walks kfold_results_summary.json directly
    (the vendored parser) so a bare HF download of sweep/ regenerates the
    figures byte-for-byte regardless of whether the rest of the repo is on the
    path. Sorted by score (mean_R2 - 0.5*SD) descending.

    Two canonical-protocol normalisations are applied by default:
      * drop_quintile — the secondary 5-fold longitude-QUINTILE runs (group
        name carries `_f5_`) are removed so they cannot out-rank the canonical
        10-fold (decile) runs when a figure asks for "best per family". The
        manuscript reports the 10-fold protocol.
      * merge_seeds  — seed-replica runs (`_seedNNN`) are averaged into one
        row (see _merge_seed_replicas) so the flagship SGT is the across-seed
        mean with its across-fold SD, not a cherry-picked single seed.
    """
    sd = Path(sweep_dir) if sweep_dir else SWEEP_DIR_DEFAULT
    raw = _collect_standalone(sd)
    rows = [r for r in raw if r.get('r2_mean') is not None]
    if drop_quintile:
        rows = [r for r in rows if '_f5_' not in (r.get('sweep_group') or '')]
    if drop_landuse:
        # Land-use-subset runs (e.g. cropland-only) share axis/bands/max_oc with
        # the full-domain runs but score very differently; excluded by default so
        # "best per family" on the full domain never picks a cropland config
        # (which would, e.g., show tree baselines at cropland R²~0.66 instead of
        # the full-domain ~0.34). Pass drop_landuse=False to include them.
        rows = [r for r in rows if 'cropland' not in (r.get('sweep_group') or '')]
    for r in rows:
        r['config_dir'] = _config_dir(sd, r)
    if merge_seeds:
        rows = _merge_seed_replicas(rows)
    for r in rows:
        r['score'] = float(r['r2_mean']) - 0.5 * float(r.get('r2_std') or 0.0)
        r['family'] = family_of(r)
        if 'config_dir' not in r:
            r['config_dir'] = _config_dir(sd, r)
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


def load_pooled_predictions(row_or_dir) -> pd.DataFrame | None:
    """Pooled hold-out predictions for a ranking row OR a config dir.

    For a seed-averaged flagship row (carries `seed_dirs`), the per-fold
    predictions are ENSEMBLED across seeds: within each fold the replicas are
    aligned on (GPS_LAT, GPS_LONG, OC_actual, year) and OC_predicted is
    averaged, mirroring how the seed-averaged R^2 is formed. Otherwise this is
    just load_fold_predictions(config_dir).
    """
    if isinstance(row_or_dir, dict):
        seed_dirs = row_or_dir.get('seed_dirs')
        if seed_dirs and len(seed_dirs) > 1:
            keys = ['GPS_LAT', 'GPS_LONG', 'OC_actual', 'year', 'fold_id']
            parts = []
            for d in seed_dirs:
                df = load_fold_predictions(d)
                if df is None:
                    continue
                kk = [c for c in keys if c in df.columns]
                parts.append(df.sort_values(kk).reset_index(drop=True))
            if not parts:
                return None
            base = parts[0].copy()
            preds = np.vstack([p['OC_predicted'].to_numpy(dtype=float)
                               for p in parts if len(p) == len(base)])
            base['OC_predicted'] = preds.mean(axis=0)
            return base
        return load_fold_predictions(row_or_dir.get('config_dir'))
    return load_fold_predictions(row_or_dir)


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
        axis = 'lon' if ('lon_lo' in k0 or 'edge_lo' in k0) else 'lat'

    def _edge(f, which):
        # canonical schema uses edge_lo/edge_hi (axis-agnostic); older runs
        # used lon_lo/lon_hi or lat_lo/lat_hi.
        for k in (f'edge_{which}', f'{axis}_{which}'):
            if f.get(k) is not None:
                return f.get(k)
        return None

    folds = [{'fold_id': f.get('fold_id'),
              'lo': _edge(f, 'lo'), 'hi': _edge(f, 'hi'),
              'centroid_lat': f.get('centroid_lat'),
              'centroid_lon': f.get('centroid_lon'),
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
