"""Push Data_HF/ to a HuggingFace dataset repo.

Resolves symlinks at upload time, so the remote repo always contains
actual file content even if the local mirror uses symlinks.

One-time setup:
    pip install huggingface_hub
    huggingface-cli login        # paste your HF write token

Then:
    python SamplePoints/push_to_hf.py --repo-id <username>/<repo-name>

To create the repo on first push:
    python SamplePoints/push_to_hf.py --repo-id <username>/<repo-name> --create-repo
"""
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("ERROR: huggingface_hub not installed.")
    print("       pip install huggingface_hub")
    sys.exit(1)

DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / 'Data_HF'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', required=True,
                    help='HF dataset repo id, e.g. yourusername/sgt-bavaria-soc-2002-2023')
    ap.add_argument('--folder', type=Path, default=DEFAULT_FOLDER,
                    help=f'Local folder to upload (default {DEFAULT_FOLDER})')
    ap.add_argument('--create-repo', action='store_true',
                    help='Create the repo on HF first (idempotent — safe to pass even if it exists)')
    ap.add_argument('--private', action='store_true',
                    help='Create as private repo (only with --create-repo)')
    ap.add_argument('--commit-message', default='Push SGT Bavaria SOC data',
                    help='Commit message attached to this upload')
    args = ap.parse_args()

    if not args.folder.exists():
        sys.exit(f'ERROR: {args.folder} does not exist.\n'
                 'Run prepare_hf_export.py first to build it.')

    # Quick stat — show how much will be uploaded.
    n_files = sum(1 for _ in args.folder.rglob('*') if _.is_file() or _.is_symlink())
    print(f'Source:    {args.folder}')
    print(f'Repo:      {args.repo_id}  (type=dataset)')
    print(f'Files:     ~{n_files} (symlinks resolved on upload)\n')

    if args.create_repo:
        print(f'Creating dataset repo (idempotent, private={args.private}) …')
        create_repo(repo_id=args.repo_id, repo_type='dataset',
                     exist_ok=True, private=args.private)
        print('  done.')

    print(f'\nUploading via upload_large_folder (8 workers, chunked + resumable).')
    print('Re-run after a network drop to resume from the last completed chunk.\n')
    api = HfApi()
    # IMPORTANT: upload_folder does NOT traverse directory-level symlinks (so
    # Data_HF/'s dir symlinks would only catch the file-level .xlsx symlinks).
    # Use the real Data/ tree with allow/ignore patterns instead, and prefer
    # upload_large_folder which parallelizes + retries automatically.
    real_source = Path('/home/valerian/SGTPublication/Data')
    if args.folder == DEFAULT_FOLDER and real_source.exists():
        print(f'  source: {real_source}  (real Data/ — bypassing Data_HF/ to traverse all dirs)')
    else:
        real_source = args.folder
        print(f'  source: {real_source}  (using --folder as given)')

    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=str(real_source),
        repo_type='dataset',
        num_workers=8,
        allow_patterns=[
            'Coordinates1Mil/**',
            'OC_LUCAS_LFU_LfL_Coordinates_v2/**',
            'RasterTensorData/**',
            'LUCAS_LFU_*.xlsx',
            'README.md',
            'manifest.json',
        ],
        ignore_patterns=[
            'RasterBandsData/**',           # legacy v1 — not shipped
            'pipeline_state.json',           # machine-local
            '__pycache__/**',
            '*.bak*',
            '.DS_Store',
            'Thumbs.db',
        ],
    )
    print(f'\n✓ Uploaded.')
    print(f'  View at: https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()
