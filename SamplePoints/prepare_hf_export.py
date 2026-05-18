"""Build Data_HF/ — a HuggingFace-ready mirror of the data directory.

Produces an alongside-Data_HF/ folder that points (via symlinks by default)
at the source Data/ contents we want to publish. huggingface_hub.upload_folder
resolves symlinks on upload, so the remote repo gets actual file content
even though Data_HF/ stays small on local disk.

The original Data/ is never touched.

Use:
    python SamplePoints/prepare_hf_export.py             # symlinks (default)
    python SamplePoints/prepare_hf_export.py --mode copy # full copies (slower, doubles disk)
    python SamplePoints/prepare_hf_export.py --out /custom/path

What's included:
    Coordinates1Mil/                     — 1.3M Bavaria reference grid (CSV)
    LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx — yearly LUCAS soil samples
    LUCAS_LFU_Bavaria_OC_joint_data_modified.xlsx — seasonal-resolution joint LUCAS+LFU
    OC_LUCAS_LFU_LfL_Coordinates_v2/     — per-(band, year) projected GPS coords
    RasterTensorData/                    — per-band tile arrays (the actual raster data)

What's excluded:
    pipeline_state.json, *.bak, .DS_Store, RasterBandsData/ (legacy v1)
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import SOC_DATA_DIR_STR

SOC_DATA_DIR = Path(SOC_DATA_DIR_STR)
DEFAULT_OUT = SOC_DATA_DIR.parent / 'Data_HF'

# Top-level Data/ items to mirror.
INCLUDE = [
    'Coordinates1Mil',
    'LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx',
    'LUCAS_LFU_Bavaria_OC_joint_data_modified.xlsx',
    'OC_LUCAS_LFU_LfL_Coordinates_v2',
    'RasterTensorData',
]

# Ephemeral / machine-local files to skip even if they appear in INCLUDE dirs.
EXCLUDE_NAMES = {
    'pipeline_state.json',
    '.DS_Store',
    'Thumbs.db',
    '__pycache__',
}


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """Create dst pointing at src — symlink, copy, or already-OK."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return 'already-linked'
        if dst.is_symlink():
            dst.unlink()  # stale link, replace
        else:
            return f'EXISTS (skipping; remove manually to refresh): {dst}'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'symlink':
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        return 'symlinked'
    elif mode == 'copy':
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                             ignore=shutil.ignore_patterns(*EXCLUDE_NAMES))
        else:
            shutil.copy2(src, dst)
        return 'copied'
    raise ValueError(f'unknown mode {mode!r}')


def _bands_list_order():
    """Try to read bands_list_order from the TFT config so the manifest matches."""
    try:
        sys.path.insert(0, str(SOC_DATA_DIR.parent / 'SOCmapping' / 'TemporalFusionTransformer'))
        from config import bands_list_order  # type: ignore
        return list(bands_list_order)
    except Exception as ex:
        return f'(unavailable: {ex})'


def _write_readme(out: Path, manifest: dict) -> None:
    """Write a human-readable README.md tailored for the HF dataset page."""
    bands = manifest.get('bands_list_order')
    bands_str = ', '.join(bands) if isinstance(bands, list) else str(bands)
    n_bands = len(bands) if isinstance(bands, list) else '?'
    readme = f"""# SGT Bavaria — SOC mapping data ({n_bands} bands, 2002–2023)

Raster + ground-truth dataset for soil organic carbon (SOC) modelling
over Bavaria, derived from MODIS, ERA5-Land, CHIRPS, OpenLandMap,
SoilGrids 2.0, SRTM, and MERIT Hydro. Aligned 250 m grid in EPSG:4326,
12 tiles × 979×979 px per band, 22 years.

## Contents

```
Coordinates1Mil/                          # 1.3M Bavaria reference grid
LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx      # yearly LUCAS soil samples
LUCAS_LFU_Bavaria_OC_joint_data_modified.xlsx
OC_LUCAS_LFU_LfL_Coordinates_v2/
  StaticValue/Elevation/coordinates.npy
  YearlyValue/<band>/<year>/coordinates.npy   # per-(band, year) LUCAS pixel coords
RasterTensorData/
  StaticValue/Elevation/IDxN..S..W..E..npy    # 12 tiles
  YearlyValue/<band>/<year>/IDx*.npy          # 12 tiles per (band, year)
manifest.json                                  # this snapshot's metadata
```

## Bands ({n_bands} channels)

{bands_str}

## How to use

```python
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='{manifest.get("repo_id_hint", "<your-username>/<repo-name>")}',
    repo_type='dataset',
    local_dir='./Data',                # or wherever your pipeline expects it
    local_dir_use_symlinks=False,
)
```

Then point the SGT pipeline at the downloaded folder by setting:
```bash
export SOC_DATA_DIR=$(realpath ./Data)
```

## Provenance

Built by SOCmapping/SamplePoints/prepare_hf_export.py on
{manifest.get("created_at")} from {manifest.get("source_dir")}.

GEE export: `gee_download_all_bands.py --category curated --years 2002 2023`
→ 250 m / EPSG:4326 / region [W 7.1864, S 46.7109, E 14.3750, N 52.1028].
"""
    (out / 'README.md').write_text(readme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'Destination folder (default {DEFAULT_OUT})')
    ap.add_argument('--mode', choices=['symlink', 'copy'], default='symlink',
                    help='symlink (saves disk; HF upload still resolves to file content) '
                         '| copy (doubles disk; safer if you want to move the folder later)')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Building HF mirror: {args.out}')
    print(f'  source: {SOC_DATA_DIR}')
    print(f'  mode:   {args.mode}')
    print()

    n_done = n_skip = n_err = 0
    for name in INCLUDE:
        src = SOC_DATA_DIR / name
        dst = args.out / name
        if not src.exists():
            print(f'  · skip (missing source): {src}')
            n_skip += 1
            continue
        status = _link_or_copy(src, dst, args.mode)
        if status in ('symlinked', 'copied'):
            n_done += 1
            print(f'  ✓ {status:14}: {dst}')
        elif status == 'already-linked':
            n_skip += 1
            print(f'  · already-linked: {dst}')
        else:
            n_err += 1
            print(f'  ✗ {status}')

    # Manifest + README
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'mode': args.mode,
        'source_dir': str(SOC_DATA_DIR),
        'entries': INCLUDE,
        'bands_list_order': _bands_list_order(),
    }
    with (args.out / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2)
    _write_readme(args.out, manifest)
    print(f'\n  ✓ manifest.json + README.md written')

    print(f'\nSummary: {n_done} new, {n_skip} skipped, {n_err} errors')
    print(f'\nNext: push to HuggingFace with')
    print(f'  python SOCmapping/SamplePoints/push_to_hf.py --repo-id <username>/<repo-name>')


if __name__ == '__main__':
    main()
