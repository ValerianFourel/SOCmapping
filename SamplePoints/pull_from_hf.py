"""Pull the SGT Bavaria SOC dataset from HuggingFace into a local folder.

One-time setup:
    pip install huggingface_hub
    # Public datasets work without login. Private ones need:
    huggingface-cli login

Then:
    python SamplePoints/pull_from_hf.py --repo-id <username>/<repo-name>

Defaults to writing into Data_HF_pulled/ alongside the project. Override
with --out. To make the pipeline pick it up:

    export SOC_DATA_DIR=$(realpath /path/to/Data_HF_pulled)

(or move/rename it to Data/).
"""
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. pip install huggingface_hub")
    sys.exit(1)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / 'Data_HF_pulled'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', required=True,
                    help='HF dataset repo id, e.g. yourusername/sgt-bavaria-soc-2002-2023')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'Local destination (default {DEFAULT_OUT})')
    ap.add_argument('--revision', default=None,
                    help='Optional revision (commit SHA or branch). Defaults to main.')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {args.repo_id} (rev={args.revision or "main"}) → {args.out}')
    print('(huggingface-cli supports resumable downloads; safe to Ctrl+C and re-run.)\n')
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type='dataset',
        local_dir=str(args.out),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f'\n✓ Downloaded to: {path}')
    print(f'\nTo wire it into the pipeline:')
    print(f'  export SOC_DATA_DIR={Path(path).resolve()}')
    print(f'or copy/symlink contents into your existing Data/ directory.')


if __name__ == '__main__':
    main()
