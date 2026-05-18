"""Create placeholder symlinks for the 14 new bands → existing LAI directories.

PURPOSE: Lets the pipeline + dataloader be exercised end-to-end before the
real GEE export has been pulled. Each new band's RasterTensorData and
OC_LUCAS_LFU_LfL_Coordinates_v2 directory becomes a directory-level
symlink to LAI's. The dataloader will read real LAI data when it looks
up these bands — content is wrong, but the structure / shape / call
graph is fully exercised.

Usage:
    python SamplePoints/scaffold_synthetic_bands.py            # create
    python SamplePoints/scaffold_synthetic_bands.py --undo     # remove

This is for testing only. **Do not run after the real GEE data has
been cut and projected** — the script refuses to overwrite a real
(non-symlink) directory.
"""
import argparse
import sys
from pathlib import Path

# Path resolution — match the rest of the project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import SOC_DATA_DIR_STR

DATA = Path(SOC_DATA_DIR_STR)
TENSOR_ROOT = DATA / 'RasterTensorData' / 'YearlyValue'
COORDS_ROOT = DATA / 'OC_LUCAS_LFU_LfL_Coordinates_v2' / 'YearlyValue'

# 14 new bands. All point at LAI's structure (which has 2002-2023 +
# 12 tiles per year + coordinates.npy per year — confirmed on disk).
NEW_BANDS = [
    # Yearly
    'NDVI', 'EVI', 'Precipitation', 'AirTemperature',
    'SoilMoisture_layer1', 'SnowDepth',
    # Static-as-yearly (soil chemistry; CEC is now SoilGrids 2.0)
    'ClayContent_0_10cm', 'SandContent_0_10cm', 'pH_H2O_0_10cm',
    'BulkDensity_0_10cm', 'CEC_0_10cm',
    # Static-as-yearly (terrain derivatives)
    'Slope', 'Aspect', 'TWI',
]
ANCHOR_BAND = 'LAI'


def make_one(target: Path, source: Path) -> str:
    """Make `target` a directory-level symlink to `source`. Returns a status string."""
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return 'already-linked'
        if not target.is_symlink():
            return f'refuses-real-dir: {target} is a real directory; remove it manually if you want to replace with a synthetic symlink'
        # Symlink to a different path — replace.
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=True)
    return 'linked'


def undo_one(target: Path) -> str:
    if not target.exists() and not target.is_symlink():
        return 'absent'
    if not target.is_symlink():
        return f'refuses-real-dir: {target} is a real directory; not removing'
    target.unlink()
    return 'unlinked'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--undo', action='store_true',
                    help='Remove the synthetic symlinks (real dirs are untouched).')
    args = ap.parse_args()

    anchor_tensor = TENSOR_ROOT / ANCHOR_BAND
    anchor_coords = COORDS_ROOT / ANCHOR_BAND
    if not anchor_tensor.exists():
        print(f'ERROR: anchor {anchor_tensor} does not exist. Cannot scaffold.')
        sys.exit(1)
    if not anchor_coords.exists():
        print(f'ERROR: anchor {anchor_coords} does not exist. Cannot scaffold.')
        sys.exit(1)

    op = 'undo' if args.undo else 'create'
    print(f'Scaffold mode: {op}')
    print(f'Anchor band:   {ANCHOR_BAND}')
    print(f'New bands:     {len(NEW_BANDS)}')
    print()

    n_ok = n_skip = n_err = 0
    for band in NEW_BANDS:
        for root, anchor in [(TENSOR_ROOT, anchor_tensor), (COORDS_ROOT, anchor_coords)]:
            target = root / band
            if args.undo:
                status = undo_one(target)
            else:
                status = make_one(target, anchor)
            if status in ('linked', 'unlinked'):
                n_ok += 1
                print(f'  ✓ {status:18}  {target}')
            elif status in ('already-linked', 'absent'):
                n_skip += 1
                print(f'  · {status:18}  {target}')
            else:
                n_err += 1
                print(f'  ✗ {status}')

    print()
    print(f'Done. ok={n_ok}  skip={n_skip}  err={n_err}')
    if n_err:
        sys.exit(2)


if __name__ == '__main__':
    main()
