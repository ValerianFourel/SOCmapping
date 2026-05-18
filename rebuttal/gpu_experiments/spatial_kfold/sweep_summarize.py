#!/usr/bin/env python3
"""
sweep_summarize.py — rank all completed sweep configs and pick the winner.

Reads:  rebuttal/gpu_experiments/spatial_kfold/sweep/<tag>/kfold_results_summary.json
Prints: sorted table (mean R2 desc, then std asc, then RMSE asc).
Writes: rebuttal/gpu_experiments/spatial_kfold/sweep/sweep_ranking.md|.json

A config is "good" not just by highest mean R^2 but by stable R^2 across
folds. We rank by (mean R^2 - 0.5 * std R^2) as a robustness-adjusted score.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / 'sweep'

_TAG_RE = re.compile(r'^(small_)?d(\d+)_h(\d+)_L(\d+)$')


def parse_tag(tag: str) -> tuple[str, int, int, int] | None:
    m = _TAG_RE.match(tag)
    if not m:
        return None
    variant = 'small' if m.group(1) else 'big'
    return (variant, int(m.group(2)), int(m.group(3)), int(m.group(4)))


def collect():
    rows = []
    for d in sorted(SWEEP_DIR.iterdir()):
        if not d.is_dir() or d.name in ('sbatch', 'slurm_logs'):
            continue
        if parse_tag(d.name) is None:
            continue
        summary = d / 'kfold_results_summary.json'
        if not summary.exists():
            rows.append({'tag': d.name, 'status': 'no summary yet', 'r2_mean': None})
            continue
        try:
            j = json.loads(summary.read_text())
        except Exception as e:
            rows.append({'tag': d.name, 'status': f'parse error: {e}', 'r2_mean': None})
            continue
        ac = j.get('across_folds', {})
        recipe = j.get('recipe', {})
        v_d_h_L = parse_tag(d.name)
        fold_r2s = [r.get('r2', float('nan')) for r in j.get('fold_results', [])]
        rows.append({
            'tag': d.name,
            'variant':    v_d_h_L[0],
            'd_model':    v_d_h_L[1],
            'num_heads':  v_d_h_L[2],
            'num_layers': v_d_h_L[3],
            'epochs':     recipe.get('num_epochs'),
            'lr':         recipe.get('lr'),
            'max_oc':     recipe.get('max_oc'),
            'sampler':    recipe.get('sampler_mode'),
            'augment':    recipe.get('augment_train'),
            'n_folds':    len(fold_r2s),
            'r2_mean':    ac.get('r2_mean'),
            'r2_std':     ac.get('r2_std'),
            'rmse_mean':  ac.get('rmse_mean'),
            'rmse_std':   ac.get('rmse_std'),
            'mae_mean':   ac.get('mae_mean'),
            'rpiq_mean':  ac.get('rpiq_mean'),
            'r2_per_fold': fold_r2s,
            'status': 'ok',
        })
    return rows


def score(row: dict) -> float:
    """Robustness-adjusted ranking: penalize cross-fold variance."""
    m, s = row.get('r2_mean'), row.get('r2_std')
    if m is None:
        return -1e9
    return float(m) - 0.5 * float(s or 0.0)


def fmt(v, prec=4, default='—'):
    if v is None or (isinstance(v, float) and v != v):
        return default
    return f'{v:.{prec}f}' if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--top', type=int, default=None,
                    help='Only print the top N configs (default: all).')
    ap.add_argument('--no-write', action='store_true',
                    help='Print ranking but do not write sweep_ranking.{md,json}.')
    a = ap.parse_args()

    if not SWEEP_DIR.exists():
        print(f'[sweep] {SWEEP_DIR} does not exist yet — submit jobs first.',
              file=sys.stderr)
        sys.exit(1)

    rows = collect()
    if not rows:
        print(f'[sweep] no config subdirs found under {SWEEP_DIR}.',
              file=sys.stderr)
        sys.exit(1)

    ok = [r for r in rows if r.get('r2_mean') is not None]
    pending = [r for r in rows if r.get('r2_mean') is None]
    ok.sort(key=score, reverse=True)
    if a.top is not None:
        ok = ok[:a.top]

    # Console table
    print(f'\n{"rank":>4}  {"tag":<14}  {"params":>7}  {"R2 mean":>9}  '
          f'{"R2 std":>7}  {"RMSE":>7}  {"MAE":>7}  {"RPIQ":>6}  {"score":>7}  '
          f'{"per-fold R2"}')
    print('-' * 110)
    for i, r in enumerate(ok, 1):
        s = score(r)
        per_fold = ' '.join(fmt(v, 2) for v in r.get('r2_per_fold', [])[:10])
        print(f'{i:>4}  {r["tag"]:<14}  '
              f'{"":>7}  '
              f'{fmt(r["r2_mean"]):>9}  '
              f'{fmt(r["r2_std"]):>7}  '
              f'{fmt(r["rmse_mean"], 3):>7}  '
              f'{fmt(r["mae_mean"],  3):>7}  '
              f'{fmt(r["rpiq_mean"], 3):>6}  '
              f'{fmt(s):>7}  '
              f'[{per_fold}]')

    if pending:
        print(f'\n[sweep] {len(pending)} config(s) still pending '
              f'(no kfold_results_summary.json yet):')
        for r in pending:
            print(f'    {r["tag"]:<14}  {r["status"]}')

    if ok and not a.no_write:
        md = [f'# Architecture sweep ranking ({len(ok)} configs)', '']
        md.append('Ranked by `score = r2_mean − 0.5 × r2_std` '
                  '(rewards high mean, penalizes cross-fold variance).')
        md.append('')
        md.append('| Rank | tag | variant | d_model | heads | layers | R² mean | R² std | RMSE | MAE | RPIQ | score |')
        md.append('|------|-----|---------|---------|-------|--------|---------|--------|------|-----|------|-------|')
        for i, r in enumerate(ok, 1):
            md.append(f'| {i} | {r["tag"]} | {r["variant"]} | {r["d_model"]} | '
                      f'{r["num_heads"]} | {r["num_layers"]} | {fmt(r["r2_mean"])} | '
                      f'{fmt(r["r2_std"])} | {fmt(r["rmse_mean"], 3)} | '
                      f'{fmt(r["mae_mean"], 3)} | {fmt(r["rpiq_mean"], 3)} | '
                      f'{fmt(score(r))} |')
        md.append('')
        if ok:
            best = ok[0]
            md.append('## Recommended for full 300-epoch retrain')
            md.append('')
            md.append(f'`--model-size {best["variant"]} '
                      f'--hidden_size {best["d_model"]} '
                      f'--num_heads {best["num_heads"]} '
                      f'--num_layers {best["num_layers"]}`')
        (SWEEP_DIR / 'sweep_ranking.md').write_text('\n'.join(md))
        (SWEEP_DIR / 'sweep_ranking.json').write_text(
            json.dumps({'ranked': ok, 'pending': pending}, indent=2, default=str))
        print(f'\n[sweep] wrote {SWEEP_DIR}/sweep_ranking.md and .json')


if __name__ == '__main__':
    main()
