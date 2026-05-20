#!/usr/bin/env python3
"""
publish_to_hf.py — push rebuttal artifacts to the Hugging Face Hub.

Designed for incremental publishing: re-running picks up any NEW sweep
configs, new final-model checkpoints, or new maps automatically, and
uploads only changed files (HF Hub deduplicates by hash). Use this same
script to "expand later" — train a new architecture, generate new maps,
re-run with the same --repo and only the deltas push.

The repo is organized by "bundle" (semantic category). Each bundle has
a source path on disk and a destination prefix in the Hub repo:

  sweep/                       spatial-CV k-fold sweep summaries (small JSON)
  sweep_checkpoints/           per-fold .pth weights (large, ~GBs)
  final_models/checkpoints/    production trained weights
  final_models/maps/           Bavaria 2023 SOC predictions
  final_models/                cross-architecture comparison figures
  code/                        architecture source + training scripts
  docs/                        REVISION_LOG, methodology notes
  README.md                    auto-generated repo overview

Usage on the cluster (after `huggingface-cli login`):

  # First-time publish: everything except the heavy sweep checkpoints
  python rebuttal/publish_to_hf.py --repo ValerianFourel/SOCrebuttal

  # Add the heavy bundle later when you want it
  python rebuttal/publish_to_hf.py --include sweep-checkpoints

  # Preview without uploading
  python rebuttal/publish_to_hf.py --dry-run

  # Push only the rebalanced maps + the new figure
  python rebuttal/publish_to_hf.py --include finals-maps,finals-figures

After the first push, expanding is a one-liner:
  # Trained a new architecture? Generated new maps?
  python rebuttal/publish_to_hf.py --include finals-checkpoints,finals-maps
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Walk up to SOCmapping (the file lives in SOCmapping/rebuttal/)
SOC_ROOT = HERE
while SOC_ROOT.parent != SOC_ROOT:
    if (SOC_ROOT / 'SpatiotemporalGatedTransformer').is_dir():
        break
    SOC_ROOT = SOC_ROOT.parent
else:
    print('[publish] could not locate SOCmapping root', file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print('[publish] huggingface_hub not installed. Install with:', file=sys.stderr)
    print('             pip install huggingface_hub', file=sys.stderr)
    print('         and authenticate with:', file=sys.stderr)
    print('             huggingface-cli login', file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bundle definitions — semantic groupings of artifacts. Each entry's
# `src` is a folder on disk; `dest` is its prefix inside the Hub repo.
#
# `allow_patterns` / `ignore_patterns` are glob lists passed to HfApi
# `.upload_folder`; both are relative to `src`. Globs use fnmatch syntax
# (** = recursive, * = single segment).
#
# `size` controls default --include behavior: 'large' is excluded unless
# explicitly requested.
# ---------------------------------------------------------------------------
BUNDLES = {
    'sweep': {
        'src':  SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold' / 'sweep',
        'dest': 'sweep',
        'desc': 'Spatial-CV k-fold sweep RESULTS (per-config '
                'kfold_results_summary.json, per-fold metrics, '
                'sweep_ranking, README). Excludes the heavy .pth files '
                '— request "sweep-checkpoints" for those.',
        'allow_patterns': ['**/*.json', '**/*.md', '**/*.csv', '**/*.txt',
                           '**/*.parquet'],
        'ignore_patterns': ['sbatch/**', 'slurm_logs/**', '**/*.pth',
                            'baseline_features/**'],
        'size': 'small',
    },
    'sweep-checkpoints': {
        'src':  SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold' / 'sweep',
        'dest': 'sweep_checkpoints',
        'desc': 'Per-fold .pth weights for ALL spatial-CV runs. Heavy '
                '(~GBs across hundreds of configs). Not included by '
                'default; pass --include sweep-checkpoints to upload.',
        'allow_patterns': ['**/*.pth'],
        'ignore_patterns': ['sbatch/**', 'slurm_logs/**'],
        'size': 'large',
    },
    'finals-checkpoints': {
        'src':  SOC_ROOT / 'rebuttal' / 'final_models' / 'checkpoints',
        'dest': 'final_models/checkpoints',
        'desc': 'Production-mapping trained models: .pth weights + '
                'stats.json + config.json + train_log.txt per run.',
        'ignore_patterns': ['**/__pycache__/**'],
        'size': 'medium',
    },
    'finals-maps': {
        'src':  SOC_ROOT / 'rebuttal' / 'final_models' / 'maps',
        'dest': 'final_models/maps',
        'desc': 'Bavaria 2023 SOC predictions: parquet + summary.json + '
                'map.png per architecture × per band variant × per '
                'sampling mode (rebalanced and non-rebalanced).',
        'ignore_patterns': ['infer_all_*.out', '**/__pycache__/**'],
        'size': 'medium',
    },
    'finals-figures': {
        'src':  SOC_ROOT / 'rebuttal' / 'final_models',
        'dest': 'final_models',
        'desc': 'Cross-architecture comparison figures: '
                'maps_comparison_<year>{.png,.md,.json} and the '
                'rebalanced variant.',
        'allow_patterns': ['maps_comparison_*.png',
                           'maps_comparison_*.md',
                           'maps_comparison_*.json'],
        'size': 'small',
    },
    'code-architectures': {
        'src':  SOC_ROOT / 'SpatiotemporalGatedTransformer',
        'dest': 'code/architectures/SpatiotemporalGatedTransformer',
        'desc': 'Spatiotemporal Gated Transformer family source: '
                'SimpleSGT, EnhancedSGT, VanillaSpatiotemporalTransformer, '
                'LightweightTransformer (the four core architectures from '
                'the rebuttal\'s 3-way ablation).',
        'allow_patterns': ['*.py'],
        'ignore_patterns': ['__pycache__/**', '*.pyc'],
        'size': 'small',
    },
    'code-rebuttal': {
        'src':  SOC_ROOT / 'rebuttal',
        'dest': 'code/rebuttal',
        'desc': 'Rebuttal scripts: spatial-CV orchestration, '
                'final-model training/inference, inspector tools, '
                'sweep orchestration, this publish script.',
        'allow_patterns': ['**/*.py', '**/*.sbatch'],
        'ignore_patterns': ['**/__pycache__/**',
                            'gpu_experiments/spatial_kfold/sweep/**',
                            'final_models/checkpoints/**',
                            'final_models/maps/**',
                            '**/sbatch/**',
                            '**/slurm_logs/**'],
        'size': 'small',
    },
    'docs': {
        'src':  SOC_ROOT / 'rebuttal',
        'dest': 'docs',
        'desc': 'REVISION_LOG.md plus any per-folder READMEs.',
        'allow_patterns': ['REVISION_LOG.md', '**/README.md', '*.md'],
        'ignore_patterns': ['**/sweep/**', '**/checkpoints/**', '**/maps/**'],
        'size': 'small',
    },
}

# Default: include everything not flagged 'large'. The heavy sweep
# checkpoints are opt-in.
DEFAULT_INCLUDE = [k for k, v in BUNDLES.items() if v['size'] != 'large']


def _dir_size_bytes(src: Path, allow=None, ignore=None) -> int:
    """Quick estimate of how much will get uploaded. Walks src recursively.

    For the dry-run / pre-upload size print. Not exact — doesn't honor
    every glob nuance — but good enough to tell GB-scale from MB-scale.
    """
    if not src.exists():
        return 0
    total = 0
    for p in src.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if ignore and any(_fnmatch_any(rel, ip) for ip in ignore):
            continue
        if allow and not any(_fnmatch_any(rel, ap) for ap in allow):
            continue
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _fnmatch_any(rel: str, pattern: str) -> bool:
    """fnmatch with ** support — single segment match isn't enough for
    our nested layouts."""
    import fnmatch
    # ** matches anything; convert to fnmatch's *
    if '**' in pattern:
        # naive but works for our patterns
        return fnmatch.fnmatch(rel, pattern.replace('**/', '*').replace('/**', '/*'))
    return fnmatch.fnmatch(rel, pattern)


def _human_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1024


def _generate_readme(included: list[str]) -> str:
    """Build the top-level README.md for the Hub repo. Reflects which
    bundles are present after this publish. Idempotent — re-run with a
    superset and the README is regenerated to match."""
    bundle_lines = []
    for name in included:
        b = BUNDLES[name]
        bundle_lines.append(f'- **`{b["dest"]}/`** — {b["desc"]}')
    bundle_block = '\n'.join(bundle_lines)
    return textwrap.dedent(f'''\
        # SOCrebuttal — Soil Organic Carbon Mapping (Geoderma rebuttal artifacts)

        Architecture comparison, production maps, and supporting analyses
        for Bavaria-wide soil organic carbon mapping at LUCAS data scale
        (~16,360 samples). Published as the artifact bundle accompanying
        the manuscript revision.

        ## Headline finding

        A **lightweight (~215k-parameter) CNN + Transformer hybrid** is the
        recommended architecture at this data scale (Vanilla in the
        comparison table):

        - **+0.23 R² gain** from the CNN spatial encoder over a
          parameter-matched transformer-alone baseline (Lightweight).
        - **4× lower cross-fold variance** than transformer-alone.
        - **~3× faster convergence** (median best-epoch 7-8 vs 20).
        - **No measurable benefit** from the gated-residual mechanism on
          top of CNN+Transformer (SGT and Vanilla tie at matched
          hyperparameters; SGT is 1.7× larger for no R² gain).
        - 30× larger pure-transformer (SimpleTransformerV2, 11M params)
          matches Vanilla in mean R² but offers no other advantages.

        See `final_models/maps_comparison_2023.png` for the
        cross-architecture production-map figure, and
        `final_models/maps_comparison_2023_rebal.png` for the
        KDE-rebalanced training variant.

        ## Repository structure

        {bundle_block}

        ## Methodology summary

        - **Spatial cross-validation**: 10-fold latitude-decile splits with
          a 1.2 km train/test buffer (Roberts 2017, Ploton 2020). The R²
          values reported in `sweep/sweep_ranking.md` are the honest
          generalization estimates.
        - **Production maps**: full-data training (95% train + 5% random
          monitor holdout — non-spatial, used only for best-epoch weight
          selection). All architectures inferred on the 1mil-point Bavaria
          reference grid for target year 2023 over a 5-year covariate
          window {{2019, …, 2023}}.
        - **Rebalanced production maps** (`*_rebal`): same architectures,
          retrained with KDE-inverse-density sample weighting on log(SOC)
          at α=0.5 (Yang et al. ICML 2021) to ensure organic-rich
          regions (Alpine peat, fen/bog) are not under-predicted.

        ## Reproducing

        All training, inference, and analysis scripts are in `code/`.
        Architecture source under `code/architectures/`; rebuttal pipeline
        scripts (run_kfold, train_full, infer_bavaria, sweep_submit,
        submit_finals, inspect_run, compare_maps) under `code/rebuttal/`.

        ## Citation

        Will be populated once the manuscript is accepted.

        ---

        Last updated: {os.environ.get("USER", "")}@{os.uname().nodename}
        ''')


def _check_bundle_paths(bundles: list[str]) -> None:
    """Print which bundle source dirs exist and warn about missing ones."""
    print(f'[publish] bundle source paths under {SOC_ROOT}:')
    for name in bundles:
        b = BUNDLES[name]
        ok = b['src'].exists()
        size = _dir_size_bytes(b['src'],
                                allow=b.get('allow_patterns'),
                                ignore=b.get('ignore_patterns')) if ok else 0
        marker = '✓' if ok else '✗ MISSING'
        rel = b['src'].relative_to(SOC_ROOT) if ok else b['src']
        print(f'    [{name:<22}] {marker:<10}  '
              f'{_human_bytes(size):>10}   {rel}')


def _do_upload(api: HfApi, repo: str, dry_run: bool) -> None:
    pass   # not used — main does it inline so progress prints are linear


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', type=str, default='ValerianFourel/SOCrebuttal',
                   help='Hub repo ID, e.g. ValerianFourel/SOCrebuttal.')
    p.add_argument('--repo-type', type=str, default='dataset',
                   choices=['dataset', 'model', 'space'],
                   help='Hub repo type. "dataset" is most flexible for '
                        'mixed-content rebuttal artifacts (maps, '
                        'checkpoints, code, docs). Default: dataset.')
    p.add_argument('--include', type=str, default=None,
                   help='Comma-separated bundles to include. Default: all '
                        'bundles except "sweep-checkpoints" (which is heavy '
                        'and rarely needed). Available: '
                        + ', '.join(BUNDLES.keys()))
    p.add_argument('--exclude', type=str, default=None,
                   help='Comma-separated bundles to skip.')
    p.add_argument('--dry-run', action='store_true',
                   help='Show which paths would upload (with size estimates) '
                        'and exit without touching the Hub.')
    p.add_argument('--commit-message', type=str, default=None,
                   help='Override the per-bundle commit message.')
    p.add_argument('--token', type=str, default=None,
                   help='HF Hub token. Default: HF_TOKEN env var, or '
                        'huggingface-cli login state.')
    p.add_argument('--private', action='store_true',
                   help='If the repo does not yet exist, create as private.')
    p.add_argument('--skip-readme', action='store_true',
                   help='Do not regenerate / upload the top-level README.md.')
    return p.parse_args()


def main():
    a = parse()

    # Resolve bundle selection
    if a.include:
        bundles = [b.strip() for b in a.include.split(',') if b.strip()]
    else:
        bundles = list(DEFAULT_INCLUDE)
    if a.exclude:
        for b in a.exclude.split(','):
            b = b.strip()
            if b in bundles:
                bundles.remove(b)
    unknown = [b for b in bundles if b not in BUNDLES]
    if unknown:
        print(f'[publish] unknown bundle(s): {unknown}', file=sys.stderr)
        print(f'          available: {list(BUNDLES.keys())}', file=sys.stderr)
        sys.exit(1)

    print(f'[publish] repo      = {a.repo}  (type={a.repo_type})')
    print(f'[publish] bundles   = {bundles}')
    print(f'[publish] dry_run   = {a.dry_run}')
    print()

    _check_bundle_paths(bundles)
    print()

    if a.dry_run:
        total = 0
        for name in bundles:
            b = BUNDLES[name]
            total += _dir_size_bytes(b['src'],
                                       allow=b.get('allow_patterns'),
                                       ignore=b.get('ignore_patterns'))
        print(f'[publish] DRY RUN total ≈ {_human_bytes(total)} '
              f'would upload')
        return

    # Authenticate + ensure repo exists
    api = HfApi(token=a.token or os.environ.get('HF_TOKEN'))
    try:
        api.create_repo(a.repo, repo_type=a.repo_type, exist_ok=True,
                         private=a.private)
    except HfHubHTTPError as e:
        print(f'[publish] create_repo failed: {e}', file=sys.stderr)
        sys.exit(1)
    print(f'[publish] repo OK: https://huggingface.co/{a.repo_type}s/{a.repo}')
    print()

    # Per-bundle upload (each is its own commit so partial progress is
    # never lost if the script is interrupted)
    for name in bundles:
        b = BUNDLES[name]
        if not b['src'].exists():
            print(f'[publish] [{name}] SKIP — source path missing')
            continue
        msg = a.commit_message or f'publish: {name} ({b["desc"][:80]})'
        print(f'[publish] [{name}] uploading {b["src"]}  →  {b["dest"]}/')
        try:
            api.upload_folder(
                folder_path=str(b['src']),
                path_in_repo=b['dest'],
                repo_id=a.repo,
                repo_type=a.repo_type,
                allow_patterns=b.get('allow_patterns'),
                ignore_patterns=b.get('ignore_patterns'),
                commit_message=msg,
            )
        except HfHubHTTPError as e:
            print(f'[publish] [{name}] FAILED: {e}', file=sys.stderr)
            continue
        print(f'[publish] [{name}] done.')

    # Regenerate the top-level README (always reflects the union of all
    # bundles published so far, conceptually — though only `included`
    # ones get listed in the structure block for this run).
    if not a.skip_readme:
        readme = _generate_readme(bundles)
        with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False) as f:
            f.write(readme)
            readme_path = f.name
        try:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo='README.md',
                repo_id=a.repo,
                repo_type=a.repo_type,
                commit_message='publish: regenerate top-level README',
            )
            print(f'[publish] README.md uploaded.')
        except HfHubHTTPError as e:
            print(f'[publish] README upload FAILED: {e}', file=sys.stderr)
        finally:
            os.unlink(readme_path)

    print()
    print(f'[publish] all done. View at: '
          f'https://huggingface.co/{a.repo_type}s/{a.repo}')


if __name__ == '__main__':
    main()
