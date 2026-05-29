"""band_subsets.py — single source of truth for the --bands-list flag.

Three named subsets are supported by every pipeline script:

    original_6    The 6 covariates the original submission used:
                  Elevation, LAI, LST, MODIS_NPP, SoilEvaporation,
                  TotalEvapotranspiration.

    full_20       The 20 covariates of the revision band expansion (original 6
                  + 14). Pinned by name, so it stays 20 even after the stack is
                  extended (the project default since the SGT-config 6→20 commit).

    full_extended The full stack including the Tier 1/2/3 revision covariates
                  (Landsat SRC + multi-scale terrain + climate/phenology) —
                  i.e. every channel in bands_list_order.

Resolution always returns INDICES into the project-wide bands_list_order
(SpatiotemporalGatedTransformer/config.py). The original 6 sit at the
front of that list by construction, so original_6 indices are [0, 1, 2,
3, 4, 5] for the canonical bands_list_order. The implementation does the
lookup defensively (config.bands_list_order.index(b)) in case the order
changes.

`band_suffix()` returns the run-tag suffix ("_6band", "_20band", or
"_extband") that every output-naming function appends so the variants
don't collide.
"""
from __future__ import annotations


ORIGINAL_6_BANDS = [
    'Elevation', 'LAI', 'LST', 'MODIS_NPP',
    'SoilEvaporation', 'TotalEvapotranspiration',
]

# The 20-band revision stack (original 6 + 14). Pinned by NAME so `full_20`
# always resolves to exactly these 20 even after the Tier 1/2/3 covariates
# extend bands_list_order — i.e. `full_20` is NO LONGER "every channel".
FULL_20_BANDS = ORIGINAL_6_BANDS + [
    'NDVI', 'EVI', 'Precipitation', 'AirTemperature', 'SoilMoisture_layer1', 'SnowDepth',
    'ClayContent_0_10cm', 'SandContent_0_10cm', 'pH_H2O_0_10cm',
    'BulkDensity_0_10cm', 'CEC_0_10cm',
    'Slope', 'Aspect', 'TWI',
]


_EXT_NAMES  = {'full_extended', 'extended', 'full_all', 'all', 'ext', 'extband'}
_FULL_NAMES = {'full', 'full_20', '20', '20band'}
_ORIG_NAMES = {'original', 'original_6', 'orig', 'orig_6', '6', '6band'}


def get_band_indices(bands_list_name: str, full_bands_list: list[str]) -> list[int]:
    """Return the channel-dim indices to keep for the requested subset.

    Raises ValueError on an unrecognised name.
      full_extended -> every channel currently in bands_list_order (20 + Tier 1/2/3).
      full_20       -> the 20 revision bands (pinned by name; NOT "all channels").
      original_6    -> the 6 original-paper covariates.
    Resolved defensively via .index() so a changed bands_list_order order is tolerated.
    """
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _EXT_NAMES:
        return list(range(len(full_bands_list)))
    if n in _FULL_NAMES:
        missing = [b for b in FULL_20_BANDS if b not in full_bands_list]
        if missing:
            raise ValueError(
                f'full_20 bands {missing!r} not present in current '
                f'bands_list_order {full_bands_list!r}; cannot subset.')
        return [full_bands_list.index(b) for b in FULL_20_BANDS]
    if n in _ORIG_NAMES:
        missing = [b for b in ORIGINAL_6_BANDS if b not in full_bands_list]
        if missing:
            raise ValueError(
                f'original_6 bands {missing!r} not present in current '
                f'bands_list_order {full_bands_list!r}; cannot subset.')
        return [full_bands_list.index(b) for b in ORIGINAL_6_BANDS]
    raise ValueError(
        f'Unknown bands_list {bands_list_name!r}. '
        f'Choices: original_6 (6-band original-paper covariates), '
        f'full_20 (20-band revision expansion), '
        f'full_extended (20 + Tier 1/2/3 revision covariates).')


def band_suffix(bands_list_name: str) -> str:
    """Return the run-tag suffix corresponding to --bands-list."""
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _EXT_NAMES:
        return '_extband'
    if n in _FULL_NAMES:
        return '_20band'
    if n in _ORIG_NAMES:
        return '_6band'
    return f'_{n}'


def normalize_name(bands_list_name: str) -> str:
    """Canonicalize to either 'full_20' or 'original_6'."""
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _EXT_NAMES:
        return 'full_extended'
    if n in _FULL_NAMES:
        return 'full_20'
    if n in _ORIG_NAMES:
        return 'original_6'
    raise ValueError(f'Unknown bands_list {bands_list_name!r}')
