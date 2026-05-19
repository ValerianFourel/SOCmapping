"""band_subsets.py — single source of truth for the --bands-list flag.

Two named subsets are supported by every pipeline script:

    original_6   The 6 covariates the original submission used:
                 Elevation, LAI, LST, MODIS_NPP, SoilEvaporation,
                 TotalEvapotranspiration.

    full_20      All 20 covariates added during the revision band expansion
                 (the project default since the SGT-config 6→20 commit).

Resolution always returns INDICES into the project-wide bands_list_order
(SpatiotemporalGatedTransformer/config.py). The original 6 sit at the
front of that list by construction, so original_6 indices are [0, 1, 2,
3, 4, 5] for the canonical bands_list_order. The implementation does the
lookup defensively (config.bands_list_order.index(b)) in case the order
changes.

`band_suffix()` returns the run-tag suffix ("_6band" or "_20band") that
every output-naming function appends so the two variants don't collide.
"""
from __future__ import annotations


ORIGINAL_6_BANDS = [
    'Elevation', 'LAI', 'LST', 'MODIS_NPP',
    'SoilEvaporation', 'TotalEvapotranspiration',
]


_FULL_NAMES = {'full', 'full_20', '20', '20band', 'all'}
_ORIG_NAMES = {'original', 'original_6', 'orig', 'orig_6', '6', '6band'}


def get_band_indices(bands_list_name: str, full_bands_list: list[str]) -> list[int]:
    """Return the channel-dim indices to keep for the requested subset.

    Raises ValueError on an unrecognised name. Default behaviour (full_20)
    is to keep every channel, so all upstream code can call this
    unconditionally without affecting behaviour when the flag is absent.
    """
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _FULL_NAMES:
        return list(range(len(full_bands_list)))
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
        f'full_20 (20-band revision expansion).')


def band_suffix(bands_list_name: str) -> str:
    """Return the run-tag suffix corresponding to --bands-list."""
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _FULL_NAMES:
        return '_20band'
    if n in _ORIG_NAMES:
        return '_6band'
    return f'_{n}'


def normalize_name(bands_list_name: str) -> str:
    """Canonicalize to either 'full_20' or 'original_6'."""
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _FULL_NAMES:
        return 'full_20'
    if n in _ORIG_NAMES:
        return 'original_6'
    raise ValueError(f'Unknown bands_list {bands_list_name!r}')
