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

_ARCH_TAG_RE = re.compile(r'^(small_)?d(\d+)_h(\d+)_L(\d+)$')
_FAMILY_TAG_RE = re.compile(
    r'^(3dcnn|cnnlstm|simpletransformer|vanilla_transformer)'
    r'_d(\d+)_h(\d+)_L(\d+)$')
_BASELINE_TAG_RE = re.compile(r'^baseline_([a-z0-9]+)_(.+)$')


def parse_tag(tag: str) -> tuple[str, int | None, int | None, int | None, str] | None:
    """Return (variant, d, h, L, model_name).

    For SGT configs:    variant in {'big', 'small'}, model_name = 'sgt'.
    For other families: variant = '<family>', model_name = '<family>'
                        (e.g., 3dcnn, cnnlstm, simpletransformer).
    For baselines:      variant = 'baseline', d=h=L=None, model_name in {'xgb','rf',...}.
    Returns None if the tag isn't recognized.
    """
    m = _ARCH_TAG_RE.match(tag)
    if m:
        variant = 'small' if m.group(1) else 'big'
        return (variant, int(m.group(2)), int(m.group(3)), int(m.group(4)), 'sgt')
    m = _FAMILY_TAG_RE.match(tag)
    if m:
        fam = m.group(1)
        return (fam, int(m.group(2)), int(m.group(3)), int(m.group(4)), fam)
    m = _BASELINE_TAG_RE.match(tag)
    if m:
        return ('baseline', None, None, None, m.group(1))
    return None


_BAND_LABEL = {'original_6': '6', 'full_20': '20', 'full_extended': '43'}


def bands_category(recipe: dict, group: str) -> str:
    """6 / 20 / 43 band label. Prefers the recorded n_bands/bands_list (runs
    with the config manifest); falls back to the sweep-group suffix for older
    runs that predate it. No suffix on an old run = the 20-band default."""
    nb = recipe.get('n_bands')
    if nb in (6, 20, 43):
        return str(nb)
    bl = recipe.get('bands_list')
    if bl in _BAND_LABEL:
        return _BAND_LABEL[bl]
    g = group or ''
    if '_6band' in g:
        return '6'
    if '_extband' in g or 'extended' in g:
        return '43'
    return '20'


def collect():
    rows = []
    # Walk recursively so namespaced sweeps (sweep/oc120/<tag>) are merged
    # into the same ranking alongside flat tags (sweep/<tag>).
    skip_dirs = {'sbatch', 'slurm_logs', 'baseline_features'}
    seen_dirs = set()
    for summary in sorted(SWEEP_DIR.rglob('kfold_results_summary.json')):
        cfg_dir = summary.parent
        if cfg_dir in seen_dirs:
            continue
        if any(part in skip_dirs for part in cfg_dir.relative_to(SWEEP_DIR).parts):
            continue
        seen_dirs.add(cfg_dir)
        rel_parts = cfg_dir.relative_to(SWEEP_DIR).parts
        base_tag = rel_parts[-1]
        sweep_group = '/'.join(rel_parts[:-1]) or '(top)'
        parsed = parse_tag(base_tag)
        if parsed is None:
            continue
        try:
            j = json.loads(summary.read_text())
        except Exception as e:
            rows.append({'tag': base_tag, 'sweep_group': sweep_group,
                         'status': f'parse error: {e}', 'r2_mean': None})
            continue
        ac = j.get('across_folds', {})
        recipe = j.get('recipe', {})
        fold_r2s = [r.get('r2', float('nan')) for r in j.get('fold_results', [])]
        rows.append({
            'tag': base_tag,
            'sweep_group': sweep_group,
            'variant':    parsed[0],
            'd_model':    parsed[1],
            'num_heads':  parsed[2],
            'num_layers': parsed[3],
            'model_name': parsed[4],
            'epochs':     recipe.get('num_epochs'),
            'lr':         recipe.get('lr'),
            'max_oc':     recipe.get('max_oc'),
            'window':     recipe.get('window_size'),
            'axis':       j.get('split_axis', 'lat'),
            'bands':      bands_category(recipe, sweep_group),
            'n_bands':    recipe.get('n_bands'),
            'loss':       recipe.get('loss_type'),
            'model_family': recipe.get('model_family') or parsed[4],
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
    print(f'\n{"rank":>4}  {"group":<20}  {"tag":<22}  {"bnd":>3}  {"ax":>4}  '
          f'{"fld":>3}  {"R2 mean":>9}  {"R2 std":>7}  {"RMSE":>7}  {"MAE":>7}  '
          f'{"RPIQ":>6}  {"score":>7}  {"per-fold R2"}')
    print('-' * 150)
    for i, r in enumerate(ok, 1):
        s = score(r)
        per_fold = ' '.join(fmt(v, 2) for v in r.get('r2_per_fold', [])[:10])
        print(f'{i:>4}  {r.get("sweep_group", "(top)"):<20}  {r["tag"]:<22}  '
              f'{r.get("bands", "?"):>3}  {r.get("axis", "lat"):>4}  '
              f'{fmt(r.get("n_folds"), 0):>3}  '
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
        md.append('| Rank | group | tag | family | bands | axis | folds | win | loss | max_oc | variant | d_model | heads | layers | R² mean | R² std | RMSE | MAE | RPIQ | score |')
        md.append('|------|-------|-----|--------|-------|------|-------|-----|------|--------|---------|---------|-------|--------|---------|--------|------|-----|------|-------|')
        for i, r in enumerate(ok, 1):
            md.append(f'| {i} | {r.get("sweep_group", "(top)")} | {r["tag"]} | '
                      f'{r.get("model_family") or r["model_name"]} | '
                      f'{r.get("bands", "?")} | {r.get("axis", "lat")} | '
                      f'{fmt(r.get("n_folds"), 0, "—")} | {fmt(r.get("window"), 0, "—")} | '
                      f'{r.get("loss") or "—"} | {fmt(r.get("max_oc"), 0, "—")} | '
                      f'{r["variant"]} | '
                      f'{fmt(r["d_model"], 0, "—")} | '
                      f'{fmt(r["num_heads"], 0, "—")} | '
                      f'{fmt(r["num_layers"], 0, "—")} | '
                      f'{fmt(r["r2_mean"])} | {fmt(r["r2_std"])} | '
                      f'{fmt(r["rmse_mean"], 3)} | {fmt(r["mae_mean"], 3)} | '
                      f'{fmt(r["rpiq_mean"], 3)} | {fmt(score(r))} |')
        md.append('')

        # -- 20-band vs 6-band pair view --
        # Every (sweep_group, tag) that has both a "*_6band" namespace AND
        # a matching non-suffixed namespace is paired here for direct
        # comparison.
        def _strip_bands(g: str) -> tuple[str, str]:
            """Return (base_group, band_label). 'oc150_6band' → ('oc150', '6band');
            'oc150_extband' → ('oc150', '43band'); 'oc150' → ('oc150', '20band'
            [implicit default])."""
            if g.endswith('_6band'):
                return g[:-6], '6band'
            if g.endswith('_extband'):
                return g[:-8], '43band'
            if g.endswith('_20band'):
                return g[:-7], '20band'
            return g, '20band'   # assume default = 20-band

        pairs: dict[tuple[str, str], dict] = {}  # (base_group, tag) → {band → row}
        for r in ok:
            base, band = _strip_bands(r.get('sweep_group') or '')
            key = (base, r['tag'])
            pairs.setdefault(key, {})[band] = r

        paired = {k: v for k, v in pairs.items() if '6band' in v and '20band' in v}
        if paired:
            md.append('## 20-band vs 6-band paired comparison')
            md.append('')
            md.append(f'{len(paired)} (base sweep × tag) pairs have BOTH '
                      f'a 20-band and a 6-band run available. Δ R² and Δ '
                      f'fold-0 R² isolate the band-set effect for the '
                      f'same architecture and same evaluation protocol.')
            md.append('')
            md.append('| Base sweep | Tag | R² mean 20b | R² mean 6b | Δ (6b−20b) | Fold-0 20b | Fold-0 6b | Δ fold-0 |')
            md.append('|---|---|---|---|---|---|---|---|')
            for (base, tag), variants in sorted(paired.items()):
                r20 = variants['20band']; r6 = variants['6band']
                f0_20 = r20.get('r2_per_fold', [float('nan')])[0]
                f0_6  = r6.get('r2_per_fold',  [float('nan')])[0]
                d_mean = r6['r2_mean'] - r20['r2_mean']
                d_f0 = f0_6 - f0_20
                md.append(f'| {base} | {tag} | {r20["r2_mean"]:.4f} | '
                          f'{r6["r2_mean"]:.4f} | {d_mean:+.4f} | '
                          f'{f0_20:+.3f} | {f0_6:+.3f} | {d_f0:+.3f} |')
            md.append('')

        if ok:
            best = ok[0]
            md.append('## Recommended for full 300-epoch retrain')
            md.append('')
            if best['variant'] in ('big', 'small'):
                md.append(f'`--model-size {best["variant"]} '
                          f'--hidden_size {best["d_model"]} '
                          f'--num_heads {best["num_heads"]} '
                          f'--num_layers {best["num_layers"]}`')
            else:
                md.append(f'Top is a baseline (`{best["tag"]}`). '
                          f'Use the matching `run_baselines.py` flags to reproduce.')
        (SWEEP_DIR / 'sweep_ranking.md').write_text('\n'.join(md))
        (SWEEP_DIR / 'sweep_ranking.json').write_text(
            json.dumps({'ranked': ok, 'pending': pending}, indent=2, default=str))
        print(f'\n[sweep] wrote {SWEEP_DIR}/sweep_ranking.md and .json')


if __name__ == '__main__':
    main()
