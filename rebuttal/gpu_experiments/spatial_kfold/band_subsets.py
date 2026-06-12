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


# Pinned-by-name 43-band extended stack (mirrors _bands.FULL_EXTENDED_BANDS) so
# `full_extended` resolves to exactly these 43 even after Tier-4 S2 SWIR extends
# bands_list_order to 45 — i.e. existing 43-band runs/checkpoints are unchanged.
_TIER_EXTENDED_BANDS = [
    'SRC_Blue', 'SRC_Green', 'SRC_Red', 'SRC_NIR', 'SRC_SWIR1', 'SRC_SWIR2',
    'SRC_RCC', 'SRC_BCC', 'SRC_NBR2', 'SRC_BSI', 'SRC_ExposureCount',
    'TPI_90', 'TPI_300', 'TPI_1000', 'TRI', 'Roughness',
    'ClimaticWaterBalance', 'SoilTemperature_layer1', 'FrostDays',
    'GrowingDegreeDays', 'NDVI_Amplitude', 'NDVI_Integral', 'EVI_Amplitude',
]
FULL_EXTENDED_BANDS = FULL_20_BANDS + _TIER_EXTENDED_BANDS
# Tier 4 — Sentinel-2 SWIR bare-soil composite (static); 45-band superset.
_TIER4_S2SWIR_BANDS = ['S2SRC_SWIR1', 'S2SRC_SWIR2']
FULL_EXTENDED_S2_BANDS = FULL_EXTENDED_BANDS + _TIER4_S2SWIR_BANDS


_EXT_NAMES   = {'full_extended', 'extended', 'full_all', 'all', 'ext', 'extband'}
_EXTS2_NAMES = {'full_extended_s2', 'extended_s2', 'exts2', 'exts2band', 's2band'}
_FULL_NAMES  = {'full', 'full_20', '20', '20band'}
_ORIG_NAMES  = {'original', 'original_6', 'orig', 'orig_6', '6', '6band'}


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
        # Pinned to the 43 named bands (NOT range(len)) so the S2-extended
        # 45-band bands_list_order does not silently grow `full_extended`.
        missing = [b for b in FULL_EXTENDED_BANDS if b not in full_bands_list]
        if missing:
            raise ValueError(
                f'full_extended bands {missing!r} not present in current '
                f'bands_list_order {full_bands_list!r}; cannot subset.')
        return [full_bands_list.index(b) for b in FULL_EXTENDED_BANDS]
    if n in _EXTS2_NAMES:
        missing = [b for b in FULL_EXTENDED_S2_BANDS if b not in full_bands_list]
        if missing:
            raise ValueError(
                f'full_extended_s2 bands {missing!r} not present in '
                f'bands_list_order; the Sentinel-2 SWIR bands are opt-in — set '
                f'SGT_BANDS_S2=1 (and ensure the S2 rasters are downloaded) so '
                f'config.py appends them. Present: {full_bands_list!r}.')
        return [full_bands_list.index(b) for b in FULL_EXTENDED_S2_BANDS]
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
    if n in _EXTS2_NAMES:
        return '_exts2band'
    if n in _EXT_NAMES:
        return '_extband'
    if n in _FULL_NAMES:
        return '_20band'
    if n in _ORIG_NAMES:
        return '_6band'
    return f'_{n}'


def normalize_name(bands_list_name: str) -> str:
    """Canonicalize to a known bands-list name."""
    n = bands_list_name.lower().strip().replace('-', '_')
    if n in _EXTS2_NAMES:
        return 'full_extended_s2'
    if n in _EXT_NAMES:
        return 'full_extended'
    if n in _FULL_NAMES:
        return 'full_20'
    if n in _ORIG_NAMES:
        return 'original_6'
    raise ValueError(f'Unknown bands_list {bands_list_name!r}')
