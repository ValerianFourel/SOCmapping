#!/usr/bin/env python3
"""
rebuttal/final_models/hf_upload_results.py — upload the Geoderma revision
RESULTS to the HuggingFace dataset 'ValerianFourel/SOCrebuttal'.

  https://huggingface.co/datasets/ValerianFourel/SOCrebuttal

What goes up (clear remote layout, mirrors the on-disk tree):

  final_models/checkpoints/<run>/   the fits to *check* the models:
       final_model.pth | final_model.joblib | final_model.json
       stats.json, config.json, train_log.txt
  sweep/<group>/<tag>/               the per-fold spatial-CV RESULTS:
       kfold_results_summary.json, kfold_results.md,
       fold_*_predictions.parquet, kfold_predictions_all_folds.parquet
  maps/_locations_400000rand_seed42/  the mapping after training on all folds:
       <run>_*.parquet/.png/.json + combined_*.parquet/.png/.json
  figures/out/                       the generated revision figures.

This uploader is deliberately focused (one call per bundle, explicit
allow-globs) so the remote layout is predictable and re-runnable: HF Hub
deduplicates by hash, so re-running pushes only deltas.

CLI:
  --repo-id   default ValerianFourel/SOCrebuttal
  --root      default /home/valerian/SGTPublication/SOCmapping/rebuttal
  --token     HF token (falls back to env HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
  --include   one or more bundle names (checkpoints sweep maps figures); default all
  --dry-run   list what WOULD upload (no network, no token needed)
  --large     use upload_large_folder for big bundles (sweep/maps) instead of upload_folder

Smoke test (no token):
  python rebuttal/final_models/hf_upload_results.py --dry-run
"""
from __future__ import annotations
import argparse
import fnmatch
import os
import sys
from pathlib import Path

REPO_ID_DEFAULT = 'ValerianFourel/SOCrebuttal'
# Derive the rebuttal/ root from this script's location (this file lives at
# rebuttal/final_models/hf_upload_results.py) so it works on any machine —
# laptop, HoreKa, JUPITER — not just the original hardcoded laptop path.
ROOT_DEFAULT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Bundle definitions. `src` is relative to --root; `dest` is the prefix in
# the Hub repo. `allow` is a list of fnmatch globs (** = recursive) applied
# to the path relative to `src`; empty => everything (minus `ignore`).
# `size` flags which bundles benefit from upload_large_folder.
# --------------------------------------------------------------------------
def make_bundles(root: Path) -> dict:
    return {
        'checkpoints': {
            'src':  root / 'final_models' / 'checkpoints',
            'dest': 'final_models/checkpoints',
            'desc': 'Production fits to check the models: per-run weights '
                    '(final_model.pth|joblib|json), stats.json, config.json, '
                    'train_log.txt.',
            'allow': ['**/final_model.pth', '**/final_model.joblib',
                      '**/final_model.json', '**/stats.json',
                      '**/config.json', '**/train_log.txt'],
            'ignore': ['**/__pycache__/**'],
            'size': 'medium',
        },
        'sweep': {
            'src':  root / 'gpu_experiments' / 'spatial_kfold' / 'sweep',
            'dest': 'sweep',
            'desc': 'Spatial-CV k-fold RESULTS per <group>/<tag>: '
                    'kfold_results_summary.json, kfold_results.md, '
                    'fold_*_predictions.parquet, '
                    'kfold_predictions_all_folds.parquet.',
            'allow': ['**/kfold_results_summary.json', '**/kfold_results.md',
                      '**/fold_*_predictions.parquet',
                      '**/kfold_predictions_all_folds.parquet'],
            'ignore': ['**/__pycache__/**', '**/*.pth', 'sbatch/**',
                       'slurm_logs/**'],
            'size': 'large',
        },
        'maps': {
            'src':  root / 'final_models' / 'maps' / '_locations_400000rand_seed42',
            'dest': 'maps/_locations_400000rand_seed42',
            'desc': 'Bavaria mapping after training on all folds: per-model '
                    'and combined_* parquet/png/json.',
            'allow': ['*.parquet', '*.png', '*.json'],
            'ignore': ['**/__pycache__/**', 'infer_all_*.out'],
            'size': 'large',
        },
        'figures': {
            'src':  root / 'figures' / 'out',
            'dest': 'figures/out',
            'desc': 'Generated revision figures (PDF + PNG) and the manifest.',
            'allow': ['*.pdf', '*.png', '*.md'],
            'ignore': ['**/__pycache__/**'],
            'size': 'small',
        },
        'code': {
            'src':  root / 'figures',
            'dest': 'code',
            'desc': 'Figure-generation code so the bundle regenerates itself: '
                    'figstyle.py, figdata.py (standalone loaders), every '
                    'figXX/figFx script, run_all_figures.py.',
            'allow': ['*.py'],
            'ignore': ['**/__pycache__/**', 'out/**'],
            'size': 'small',
        },
    }


def make_extra_files(root: Path) -> list[tuple[Path, str]]:
    """Individual (src, dest-in-repo) files uploaded via upload_file:
    the dataset card (README.md) and param_counts.py (needed by figFA/figFB/
    fig08 param annotations)."""
    return [
        (root / 'HF_DATASET_README.md', 'README.md'),
        (root / 'gpu_experiments' / 'spatial_kfold' / 'param_counts.py',
         'code/param_counts.py'),
    ]


def _match_any(rel: str, patterns) -> bool:
    for p in patterns:
        # crude but sufficient ** support for our shallow layouts
        if '**' in p:
            if fnmatch.fnmatch(rel, p.replace('**/', '*').replace('/**', '/*')):
                return True
            if fnmatch.fnmatch(rel, p.replace('**/', '')):  # zero-depth case
                return True
        elif fnmatch.fnmatch(rel, p):
            return True
    return False


def _plan_files(src: Path, allow, ignore):
    """Return sorted list of (rel_posix, size_bytes) that WOULD be uploaded."""
    out = []
    if not src.exists():
        return out
    for p in sorted(src.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if ignore and _match_any(rel, ignore):
            continue
        if allow and not _match_any(rel, allow):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        out.append((rel, sz))
    return out


def _human(n: int) -> str:
    f = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if f < 1024 or unit == 'TB':
            return f'{f:.1f} {unit}'
        f /= 1024
    return f'{f:.1f} TB'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Upload the SOC rebuttal RESULTS to the HF dataset.')
    ap.add_argument('--repo-id', default=REPO_ID_DEFAULT)
    ap.add_argument('--root', type=Path, default=ROOT_DEFAULT,
                    help='rebuttal/ root holding final_models, gpu_experiments, figures')
    ap.add_argument('--token', default=None,
                    help='HF token; falls back to env HF_TOKEN / HUGGING_FACE_HUB_TOKEN')
    ap.add_argument('--include', nargs='+', default=None,
                    metavar='BUNDLE',
                    help='subset of {checkpoints, sweep, maps, figures, code}; default all')
    ap.add_argument('--dry-run', action='store_true',
                    help='list planned files; no network, no token needed')
    ap.add_argument('--large', action='store_true',
                    help="use upload_large_folder for 'large' bundles (sweep, maps)")
    ap.add_argument('--commit-message', default='Upload SOC rebuttal revision results')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bundles = make_bundles(Path(args.root))
    names = args.include or list(bundles)
    unknown = [n for n in names if n not in bundles]
    if unknown:
        print(f'[upload] unknown bundle(s): {unknown}; '
              f'valid: {list(bundles)}', file=sys.stderr)
        return 2

    # Build the plan first (used by both dry-run and real upload preview).
    plan = {}
    grand_files = grand_bytes = 0
    for name in names:
        b = bundles[name]
        src = b['src']
        if not src.exists():
            print(f'[upload] SKIP {name}: missing dir {src}', file=sys.stderr)
            plan[name] = None
            continue
        files = _plan_files(src, b['allow'], b['ignore'])
        if not files:
            print(f'[upload] SKIP {name}: no matching files under {src}',
                  file=sys.stderr)
            plan[name] = None
            continue
        plan[name] = files
        grand_files += len(files)
        grand_bytes += sum(s for _, s in files)

    # Individual extra files (README dataset card + param_counts.py). Tied to
    # the 'code' bundle being selected so a results-only --include skips them.
    extras = []
    if 'code' in names:
        for src, dest in make_extra_files(Path(args.root)):
            if src.exists():
                extras.append((src, dest, src.stat().st_size))
                grand_files += 1
                grand_bytes += src.stat().st_size
            else:
                print(f'[upload] SKIP extra {dest}: missing {src}', file=sys.stderr)

    print(f'== HF upload plan ==  repo=datasets/{args.repo_id}')
    for name in names:
        b = bundles[name]
        files = plan.get(name)
        if not files:
            print(f'  [{name}] -> {b["dest"]}/   (skipped — nothing to upload)')
            continue
        nbytes = sum(s for _, s in files)
        print(f'  [{name}] {b["src"]}  ->  {b["dest"]}/   '
              f'({len(files)} files, {_human(nbytes)})')
        for rel, sz in files:
            print(f'      {b["dest"]}/{rel}  ({_human(sz)})')
    for src, dest, sz in extras:
        print(f'  [extra] {src}  ->  {dest}   ({_human(sz)})')
    print(f'== total: {grand_files} files, {_human(grand_bytes)} ==')

    if args.dry_run:
        print('OK dry-run (no upload performed)')
        return 0

    if grand_files == 0:
        print('BLOCKED nothing to upload (all bundles missing/empty)')
        return 1

    # --- real upload from here ---
    token = (args.token or os.environ.get('HF_TOKEN')
             or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    if not token:
        print('BLOCKED no HF token (pass --token or set HF_TOKEN / '
              'HUGGING_FACE_HUB_TOKEN); re-run with --dry-run to preview',
              file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, upload_large_folder
    except ImportError:
        print('BLOCKED huggingface_hub not installed (pip install huggingface_hub)',
              file=sys.stderr)
        return 1

    api = HfApi(token=token)
    # Create the dataset repo if absent (idempotent).
    api.create_repo(repo_id=args.repo_id, repo_type='dataset', exist_ok=True)
    print(f'[upload] repo datasets/{args.repo_id} ready')

    uploaded = []
    for name in names:
        files = plan.get(name)
        if not files:
            continue
        b = bundles[name]
        use_large = args.large and b['size'] == 'large'
        # upload_large_folder ignores allow/ignore in old versions, so we pass
        # both forms; upload_folder is the safe default for the curated globs.
        try:
            if use_large:
                upload_large_folder(
                    repo_id=args.repo_id, repo_type='dataset',
                    folder_path=str(b['src']),
                    path_in_repo=b['dest'],
                    allow_patterns=b['allow'], ignore_patterns=b['ignore'],
                )
            else:
                api.upload_folder(
                    repo_id=args.repo_id, repo_type='dataset',
                    folder_path=str(b['src']),
                    path_in_repo=b['dest'],
                    allow_patterns=b['allow'], ignore_patterns=b['ignore'],
                    commit_message=f'{args.commit_message} [{name}]',
                )
            print(f'[upload] DONE {name} -> {b["dest"]}/ ({len(files)} files)')
            uploaded.append(name)
        except Exception as e:  # noqa: BLE001 — report and continue with next bundle
            print(f'[upload] FAILED {name}: {e}', file=sys.stderr)

    for src, dest, _sz in extras:
        try:
            api.upload_file(
                repo_id=args.repo_id, repo_type='dataset',
                path_or_fileobj=str(src), path_in_repo=dest,
                commit_message=f'{args.commit_message} [{dest}]',
            )
            print(f'[upload] DONE extra -> {dest}')
            uploaded.append(dest)
        except Exception as e:  # noqa: BLE001
            print(f'[upload] FAILED extra {dest}: {e}', file=sys.stderr)

    if not uploaded:
        print('BLOCKED no bundle uploaded successfully')
        return 1
    print(f'OK uploaded bundles: {uploaded} -> '
          f'https://huggingface.co/datasets/{args.repo_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
