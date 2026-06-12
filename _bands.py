"""Canonical band groups + path builder — single source of truth for the band
stack, so the per-model config.py files (and band_subsets.py) don't drift.

Channel order is load-bearing: the original 6 stay first (6-channel
checkpoints depend on that order), then the 14 revision bands, then the
Tier 1/2/3 revision covariates appended here. Each config builds
`bands_list_order` = FULL_20_BANDS + TIER_EXTENDED_BANDS and appends the
matching YearlyValue paths to its SamplesCoordinates_Yearly / DataYearly
lists via build_yearly_paths().
"""
from __future__ import annotations

ORIGINAL_6_BANDS = [
    'Elevation', 'LAI', 'LST', 'MODIS_NPP', 'SoilEvaporation', 'TotalEvapotranspiration',
]
REVISION_14_BANDS = [
    'NDVI', 'EVI', 'Precipitation', 'AirTemperature', 'SoilMoisture_layer1', 'SnowDepth',
    'ClayContent_0_10cm', 'SandContent_0_10cm', 'pH_H2O_0_10cm',
    'BulkDensity_0_10cm', 'CEC_0_10cm',
    'Slope', 'Aspect', 'TWI',
]
FULL_20_BANDS = ORIGINAL_6_BANDS + REVISION_14_BANDS

# Tier 1/2/3 revision covariates. Order here is the appended channel order.
# All are stored under RasterTensorData/YearlyValue/<band>/ — the Tier-1 SRC
# and Tier-3 derivations are genuinely yearly; the Tier-2 terrain statics are
# materialized-as-yearly (symlinked) just like Slope/Aspect/TWI.
TIER1_SRC_BANDS = [
    'SRC_Blue', 'SRC_Green', 'SRC_Red', 'SRC_NIR', 'SRC_SWIR1', 'SRC_SWIR2',
    'SRC_RCC', 'SRC_BCC', 'SRC_NBR2', 'SRC_BSI', 'SRC_ExposureCount',
]
TIER2_TERRAIN_BANDS = ['TPI_90', 'TPI_300', 'TPI_1000', 'TRI', 'Roughness']
TIER3_DERIVED_BANDS = [
    'ClimaticWaterBalance', 'SoilTemperature_layer1', 'FrostDays',
    'GrowingDegreeDays', 'NDVI_Amplitude', 'NDVI_Integral', 'EVI_Amplitude',
]
TIER_EXTENDED_BANDS = TIER1_SRC_BANDS + TIER2_TERRAIN_BANDS + TIER3_DERIVED_BANDS
FULL_EXTENDED_BANDS = FULL_20_BANDS + TIER_EXTENDED_BANDS

# Tier 4 — Sentinel-2 SWIR bare-soil composite (B11 ~1.61 um, B12 ~2.19 um;
# COPERNICUS/S2_SR_HARMONIZED, 20 m native — the Zepp/Broeg/Tziolas exposed-soil
# SWIR signal for SOC). Sentinel-2 SR only covers ~2017+, so unlike the per-year
# Landsat SRC these are a MULTI-YEAR STATIC bare-soil composite (full spatial
# coverage, no 2002-2016 gap), materialized as the same value for every year
# window. Appended AFTER the 43-band stack so `full_extended` keeps its
# 43-channel meaning; the 45-channel superset is `full_extended_s2`. Opt-in via
# the SGT_BANDS_S2=1 env flag in the per-model config.py.
TIER4_S2SWIR_BANDS = ['S2SRC_SWIR1', 'S2SRC_SWIR2']
FULL_EXTENDED_S2_BANDS = FULL_EXTENDED_BANDS + TIER4_S2SWIR_BANDS

# Bands stored under StaticValue/<band>/ in the config path lists. Everything
# else is YearlyValue/<band>/ — including materialized-as-yearly statics
# (soil properties, terrain). Elevation and the S2 SWIR composite use StaticValue.
STATIC_BANDS = {'Elevation', 'S2SRC_SWIR1', 'S2SRC_SWIR2'}


def _tier(band: str) -> str:
    return 'StaticValue' if band in STATIC_BANDS else 'YearlyValue'


def build_yearly_paths(band_names, base_path_data):
    """Return (coords, data) path lists for `band_names`, matching the existing
    config convention exactly:
        OC_LUCAS_LFU_LfL_Coordinates_v2/<tier>/<band>   (sample/grid coords)
        RasterTensorData/<tier>/<band>                  (tile arrays)
    so the appended entries line up with bands_list_order index-for-index.
    """
    coords = [f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/{_tier(b)}/{b}'
              for b in band_names]
    data = [f'{base_path_data}/RasterTensorData/{_tier(b)}/{b}'
            for b in band_names]
    return coords, data


def build_1mil_coords(band_names, base_path_data):
    """Return the 1.3 M-grid inference coord paths for `band_names`, matching the
    config convention `Coordinates1Mil/<tier>/<band>`. Only the COORDS differ for
    1mil inference — the raster data list is the shared `DataYearly`
    (RasterTensorData/<tier>/<band>) — so this returns a single list that lines up
    with bands_list_order index-for-index.
    """
    return [f'{base_path_data}/Coordinates1Mil/{_tier(b)}/{b}'
            for b in band_names]
