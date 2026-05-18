"""Download SOC-prediction bands from Google Earth Engine, 2002–2023.

The default `--category curated` set is the 11 bands kept after the
SOC-leakage / categorical / temporal-coverage audit. All bands are
exported as annual composites (or static) GeoTIFFs at 250 m /
EPSG:4326 covering Bavaria, written to a Google Drive folder
`bavaria_bands_2002_2023/`.

Curated band set (this is what `--category curated` exports):

  yearly (11, 22 years each):
    modis  : NDVI, EVI, LAI, LST, MODIS_NPP,
             TotalEvapotranspiration                    (250-1000 m → 250 m)
    chirps : Precipitation                              (5566 m → bilinear)
    era5   : AirTemperature, SoilMoisture_layer1,
             SnowDepth, SoilEvaporation                  (11 km → bilinear)

  static (9, single image each):
    soil   : ClayContent_0_10cm, SandContent_0_10cm,
             pH_H2O_0_10cm, BulkDensity_0_10cm          (250 m, OpenLandMap)
             CEC_0_10cm                                  (250 m, SoilGrids 2.0)
    topo   : Elevation                                   (SRTM 30 m → 250 m)
             Slope, Aspect                               (server-side from SRTM)
             TWI                                        (server-side, ln(SCA/tan(slope)),
                                                        SCA from MERIT Hydro upa)

  → 11 × 22 + 9 = 251 export tasks for the curated set.

  This includes redownloading the 6 pre-existing bands (LAI, LST, MODIS_NPP,
  SoilEvaporation, TotalEvapotranspiration, Elevation) onto the same 250 m
  grid as the 14 new bands, producing pixel-perfect alignment across all 20
  channels.

Categories explicitly excluded from `--category all` (still callable
by explicit `--category`):
  • OrganicCarbon_Reference   — target leakage (SOC itself)
  • TextureUSDA               — categorical class index; bilinear resample meaningless
  • s1 (Sentinel-1 SAR)       — only 2014+; gap years for 2002–2023
  • s2 (Sentinel-2)           — only 2015+; gap years for 2002–2023

Categories included in `--category all` (uncurated full set):
  modis, chirps, era5, topo, soil

Run requirements:
    pip install earthengine-api
    earthengine authenticate                  # one-time, browser-based

Common invocations:
    python SamplePoints/gee_download_all_bands.py --dry-run
    python SamplePoints/gee_download_all_bands.py
    python SamplePoints/gee_download_all_bands.py --category modis
    python SamplePoints/gee_download_all_bands.py --category s2 --years 2015 2023
        (S1/S2 only valid 2014+/2015+; pass an explicit category)

The script submits Export.image.toDrive() tasks (one per variable per
year). They run server-side; monitor at https://code.earthengine.google.com/tasks
or via `earthengine task list`. Pull the resulting GeoTIFFs locally
with `pull_from_drive.py`, then `tiff_to_tiles.py` and
`project_lucas_coords.py`.

NOTE on resampling: ERA5-Land (11 km) and CHIRPS (5566 m) are coarser
than the 250 m export scale. The script forces bilinear resampling
on the source images so the exported TIFFs are smoothly upsampled
instead of GEE's default nearest-neighbor blocky output.
"""
import argparse
import sys
from pathlib import Path

# Don't fail at import time if ee is missing — only fail later if the
# user actually tries to submit (not for --dry-run, which is offline).
try:
    import ee
except ImportError:
    ee = None


def _require_ee():
    if ee is None:
        print("ERROR: earthengine-api not installed. Run:")
        print("    pip install earthengine-api && earthengine authenticate")
        sys.exit(1)


# ---------------------------------------------------------------------------
# AOI matches the EXISTING 12-tile layout in
#   Data/RasterTensorData/StaticValue/Elevation/IDxN..S..W..E..npy
# decoded by parsing the filenames. Each tile is 1.7986° × 1.7986°,
# 3 lat-rows × 4 lon-cols, total envelope:
#   N = 52.1028   S = 46.7109   W = 7.1864   E = 14.3750
# This buffers the actual Bavaria GPS extent (lon 8.0-13.9, lat 47.2-50.6)
# from the 1.3 M reference grid `Coordinates1Mil/coordinates_Bavaria_1mil.csv`.
# Keep the buffer so downstream samplePoints projection can place LUCAS
# points near the Bavaria edge without falling off a tile.
# ---------------------------------------------------------------------------
BAVARIA_BBOX = [7.1864, 46.7109, 14.3750, 52.1028]   # [W, S, E, N]
# Tile grid (3 rows × 4 cols), W-edge longitudes and N-edge latitudes
# of each row/col. Used by the tile-cutter to slice downloaded GeoTIFFs.
TILE_LAT_NORTH = [48.5095, 50.3062, 52.1028]   # N edge of each lat-row
TILE_LON_WEST  = [7.1864,  8.9831, 10.7797, 12.5763]  # W edge of each lon-col
TILE_DEG       = 1.7986   # tile width = height in degrees
TILE_PX        = 979      # pixels per tile side (matches existing rasters)


# ---------------------------------------------------------------------------
# Band definitions. Each entry:
#   collection_id   : EE asset ID
#   band            : band name(s) in the collection
#   reducer         : 'mean' / 'sum' / 'median' / None (static)
#   scale_native_m  : the source's native resolution (for documentation)
#   notes           : free-form
# ---------------------------------------------------------------------------
BANDS = {
    # ============================================================
    # MODIS bands (2000+ coverage; replaces & extends your existing 6)
    # ============================================================
    'modis': {
        'LAI': {
            'collection': 'MODIS/061/MCD15A3H',
            'band': 'Lai',
            'reducer': 'mean',
            'scale_native_m': 500,
            'notes': 'Leaf Area Index, 4-day composite, native units ×0.1',
        },
        'FPAR': {
            'collection': 'MODIS/061/MCD15A3H',
            'band': 'Fpar',
            'reducer': 'mean',
            'scale_native_m': 500,
            'notes': 'Fraction of absorbed PAR — complements LAI in dense canopies',
        },
        'LST': {
            # Renamed from 'LST_Day' to match the on-disk band name for the
            # existing pipeline. Same source: MOD11A2 day LST.
            'collection': 'MODIS/061/MOD11A2',
            'band': 'LST_Day_1km',
            'reducer': 'mean',
            'scale_native_m': 1000,
            'notes': 'Land Surface Temperature, day (Kelvin × 0.02)',
        },
        'LST_Night': {
            'collection': 'MODIS/061/MOD11A2',
            'band': 'LST_Night_1km',
            'reducer': 'mean',
            'scale_native_m': 1000,
            'notes': 'Land Surface Temperature, night (Kelvin × 0.02) — separate from the day-LST default',
        },
        'NDVI': {
            'collection': 'MODIS/061/MOD13Q1',
            'band': 'NDVI',
            'reducer': 'mean',
            'scale_native_m': 250,
            'notes': 'Normalized Difference Vegetation Index, 16-day, ×0.0001',
        },
        'EVI': {
            'collection': 'MODIS/061/MOD13Q1',
            'band': 'EVI',
            'reducer': 'mean',
            'scale_native_m': 250,
            'notes': 'Enhanced VI — less saturation than NDVI over dense canopy, ×0.0001',
        },
        'MODIS_NPP': {
            # Renamed from 'NPP' to match the on-disk band name for the existing pipeline.
            'collection': 'MODIS/061/MOD17A3HGF',
            'band': 'Npp',
            'reducer': 'mean',            # already annual; mean over 1-img/year is identity
            'scale_native_m': 500,
            'notes': 'Net Primary Productivity, ANNUAL native, kg C/m²/yr × 0.0001',
        },
        'GPP': {
            'collection': 'MODIS/061/MOD17A2H',
            'band': 'Gpp',
            'reducer': 'sum',
            'scale_native_m': 500,
            'notes': 'Gross Primary Productivity — additional photosynthesis signal',
        },
        'TotalEvapotranspiration': {
            # Renamed from 'ET' to match the on-disk band name for the existing pipeline.
            # Switched from MOD16A2 → MOD16A2GF (gap-filled v6.1): the non-gap-filled
            # `MODIS/061/MOD16A2` asset on GEE has missing coverage for ~2000-2020
            # (only 2021+ exports succeeded). The gap-filled variant covers the
            # full 2001-present archive with the same 8-day, 500 m, ET band.
            'collection': 'MODIS/061/MOD16A2GF',
            'band': 'ET',
            'reducer': 'sum',
            'scale_native_m': 500,
            'notes': 'Total Evapotranspiration (gap-filled MOD16A2GF), 8-day, mm × 0.1',
        },
        'PET': {
            'collection': 'MODIS/061/MOD16A2',
            'band': 'PET',
            'reducer': 'sum',
            'scale_native_m': 500,
            'notes': 'Potential ET — useful for water-balance / aridity index',
        },
        'BurnedArea': {
            'collection': 'MODIS/061/MCD64A1',
            'band': 'BurnDate',
            'reducer': 'max',
            'scale_native_m': 500,
            'notes': 'Burn-date mask — fire history affects SOC',
        },
    },

    # ============================================================
    # CHIRPS precipitation (1981+, daily 5 km — gold standard for precip)
    # ============================================================
    'chirps': {
        'Precipitation': {
            'collection': 'UCSB-CHG/CHIRPS/DAILY',
            'band': 'precipitation',
            'reducer': 'sum',
            'scale_native_m': 5566,
            'notes': 'Annual precipitation total, mm/year',
        },
    },

    # ============================================================
    # ERA5-Land (1950+, 9 km — climate reanalysis)
    # ============================================================
    'era5': {
        'AirTemperature': {
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'temperature_2m',
            'reducer': 'mean',
            'scale_native_m': 11132,
            'notes': '2m air temperature (Kelvin), annual mean',
        },
        'SoilMoisture_layer1': {
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'volumetric_soil_water_layer_1',
            'reducer': 'mean',
            'scale_native_m': 11132,
            'notes': 'Soil moisture 0-7 cm — primary SOC decomposition driver',
        },
        'SoilMoisture_layer2': {
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'volumetric_soil_water_layer_2',
            'reducer': 'mean',
            'scale_native_m': 11132,
            'notes': 'Soil moisture 7-28 cm',
        },
        'SnowDepth': {
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'snow_depth',
            'reducer': 'mean',
            'scale_native_m': 11132,
            'notes': 'Annual mean snow depth — proxy for cold-season insulation effects',
        },
        'SoilEvaporation': {
            # ERA5-Land monthly aggregates expose `evaporation_from_bare_soil_sum`
            # in m of water equivalent per month. Annual sum gives total bare-soil
            # evaporation. Used to match the on-disk `SoilEvaporation` band from the
            # existing pipeline so the redownload lands in the same folder structure.
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'evaporation_from_bare_soil_sum',
            'reducer': 'sum',
            'scale_native_m': 11132,
            'notes': 'ERA5-Land bare-soil evaporation, annual sum in m water equiv.',
        },
        'Precipitation': {
            # Replaces CHIRPS (which had no coverage north of 50°N — affected
            # ~13.5% of Bavaria sample points). ERA5-Land has global coverage
            # incl. high latitudes. Monthly aggregate gives total_precipitation_sum
            # in m water equivalent per month; annual sum gives annual rainfall
            # in m. Coarser native resolution (11 km) but reprojected with
            # bilinear to 250 m, same as the other ERA5-Land bands.
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'total_precipitation_sum',
            'reducer': 'sum',
            'scale_native_m': 11132,
            'notes': 'ERA5-Land total precipitation, annual sum (m water equiv.).',
        },
    },

    # ============================================================
    # Topography (SRTM 30m, static)
    # ============================================================
    'topo': {
        'Elevation': {
            # Renamed from 'Elevation_SRTM' to match the on-disk band name for the
            # existing pipeline. Lands in StaticValue/Elevation/ (not materialize-yearly)
            # — matches the dataloader's hardcoded 'Elevation' static-band check.
            'collection': 'USGS/SRTMGL1_003',
            'band': 'elevation',
            'reducer': None,
            'scale_native_m': 30,
            'notes': '30m SRTM DEM — replaces the pre-existing Elevation tiles with same-grid output',
        },
        # The next three are computed server-side from SRTM (slope/aspect) and
        # MERIT Hydro (upstream area for TWI). No 'collection' key — the
        # 'image_fn' builds the ee.Image directly. _static_image() honors that.
        'Slope': {
            'image_fn': lambda: _terrain_slope(),
            'band': 'slope',
            'reducer': None,
            'scale_native_m': 30,
            'notes': 'Slope (degrees) from ee.Terrain.slope(SRTM 30 m).',
        },
        'Aspect': {
            'image_fn': lambda: _terrain_aspect(),
            'band': 'aspect',
            'reducer': None,
            'scale_native_m': 30,
            'notes': 'Aspect (degrees 0-360, 0=N) from ee.Terrain.aspect(SRTM 30 m).',
        },
        'TWI': {
            'image_fn': lambda: _terrain_twi(),
            'band': 'twi',
            'reducer': None,
            'scale_native_m': 90,
            'notes': 'TWI = ln(upstream_area / tan(slope)). '
                     'upstream area from MERIT Hydro upa (km²); slope from SRTM 30 m. '
                     'Slope floored at 0.05° to avoid log(0).',
        },
    },

    # ============================================================
    # Static soils (OpenLandMap / ISRIC SoilGrids on GEE)
    # ============================================================
    'soil': {
        'ClayContent_0_10cm': {
            'collection': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'Clay fraction (%) at 10 cm — classic SOC protector',
        },
        'SandContent_0_10cm': {
            'collection': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'Sand fraction (%) at 10 cm',
        },
        'pH_H2O_0_10cm': {
            'collection': 'OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'pH in water at 10 cm × 10',
        },
        'BulkDensity_0_10cm': {
            'collection': 'OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'Bulk density of fine earth at 10 cm × 10 kg/dm³',
        },
        'CEC_0_10cm': {
            # OpenLandMap CEC asset was deprecated/inaccessible on GEE
            # (HTTP 400 "asset not found"). Switched to ISRIC SoilGrids 2.0
            # mirror at projects/soilgrids-isric/cec_mean which exposes
            # six depth bands; we pick cec_0-5cm_mean as the shallow layer.
            # The on-disk label stays `CEC_0_10cm` for naming continuity with
            # the other 0-10cm soil bands — 0-5cm is functionally equivalent
            # for SOC prediction at this resolution.
            'collection': 'projects/soilgrids-isric/cec_mean',
            'band': 'cec_0-5cm_mean',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'Cation Exchange Capacity, 0–5 cm (SoilGrids 2.0), mmol(c)/kg × 10',
        },
        # NOTE — these two are intentionally kept here for reachability via
        # --category soil_extra (see _CATEGORY_GROUPS below), but they are
        # NOT in the curated default. OrganicCarbon_Reference leaks the
        # target (SOC); TextureUSDA is a categorical class index and
        # bilinear resampling produces meaningless float averages.
    },
    'soil_extra': {
        'OrganicCarbon_Reference': {
            'collection': 'OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'OpenLandMap reference SOC at 10 cm — TARGET LEAKAGE, exclude unless explicitly testing as a prior.',
        },
        'TextureUSDA': {
            'collection': 'OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'USDA texture class (integer) at 10 cm — CATEGORICAL, requires one-hot.',
        },
    },

    # ============================================================
    # Sentinel-1 SAR (2014+, gaps 2002-2013)
    # ============================================================
    's1': {
        'S1_VV': {
            'collection': 'COPERNICUS/S1_GRD',
            'band': 'VV',
            'reducer': 'mean',
            'scale_native_m': 10,
            'notes': 'Sentinel-1 VV backscatter (dB), annual mean. ONLY 2014+. Surface moisture/roughness proxy.',
            'extra_filter': lambda c: c.filter(ee.Filter.eq('instrumentMode', 'IW'))
                                       .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                                       .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')),
        },
        'S1_VH': {
            'collection': 'COPERNICUS/S1_GRD',
            'band': 'VH',
            'reducer': 'mean',
            'scale_native_m': 10,
            'notes': 'Sentinel-1 VH backscatter (dB), annual mean. Cross-pol — biomass/vegetation signal.',
            'extra_filter': lambda c: c.filter(ee.Filter.eq('instrumentMode', 'IW'))
                                       .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                                       .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')),
        },
    },

    # ============================================================
    # Sentinel-2 surface reflectance (2015+, gaps 2002-2014)
    # ============================================================
    's2': {
        'S2_B4_Red': {
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'band': 'B4',
            'reducer': 'median',
            'scale_native_m': 10,
            'notes': 'Red, ~665 nm. ONLY 2015+. For BSI / NDVI.',
            'extra_filter': lambda c: c.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)),
        },
        'S2_B8_NIR': {
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'band': 'B8',
            'reducer': 'median',
            'scale_native_m': 10,
            'notes': 'NIR, ~842 nm. ONLY 2015+. For NDVI.',
            'extra_filter': lambda c: c.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)),
        },
        'S2_B11_SWIR1': {
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'band': 'B11',
            'reducer': 'median',
            'scale_native_m': 20,
            'notes': 'SWIR1, ~1610 nm. ONLY 2015+. Sensitive to soil mineralogy & moisture.',
            'extra_filter': lambda c: c.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)),
        },
        'S2_B12_SWIR2': {
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'band': 'B12',
            'reducer': 'median',
            'scale_native_m': 20,
            'notes': 'SWIR2, ~2190 nm. ONLY 2015+. Strongest bare-soil SOC signal.',
            'extra_filter': lambda c: c.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)),
        },
    },
}


EXPORT_SCALE_M = 250  # default export pixel size, must match --scale


# ---------------------------------------------------------------------------
# Category groups.
#   'curated' = the 11 SOC-relevant bands kept after the audit.
#   'all'     = every category EXCEPT s1/s2 (gap years) and soil_extra
#               (leakage + categorical). Use explicit `--category s1`
#               etc. to opt those in.
# ---------------------------------------------------------------------------
# The curated set names a specific (category, band) subset per category.
# CEC was originally on OpenLandMap (deprecated); now sourced from ISRIC
# SoilGrids 2.0 — see BANDS['soil']['CEC_0_10cm'] above.
#
# All 20 channels are listed here so a fresh end-to-end pipeline run
# produces pixel-perfect alignment across every band — the existing 6
# (LAI, LST, MODIS_NPP, SoilEvaporation, TotalEvapotranspiration, Elevation)
# are redownloaded onto the same 250 m / EPSG:4326 grid as the 14 new bands.
_CURATED_BANDS = {
    'modis':  ['NDVI', 'EVI', 'LAI', 'LST', 'MODIS_NPP', 'TotalEvapotranspiration'],
    # CHIRPS Precipitation moved to ERA5-Land — CHIRPS doesn't cover >50°N
    # and ~14% of our Bavaria sample points fall in that strip. ERA5-Land
    # has global coverage. The CHIRPS asset entry stays in BANDS for any
    # user who wants it via explicit `--category chirps`.
    'era5':   ['AirTemperature', 'SoilMoisture_layer1', 'SnowDepth', 'SoilEvaporation', 'Precipitation'],
    'soil':   ['ClayContent_0_10cm', 'SandContent_0_10cm', 'pH_H2O_0_10cm',
               'BulkDensity_0_10cm', 'CEC_0_10cm'],
    'topo':   ['Elevation', 'Slope', 'Aspect', 'TWI'],
}
# Categories included in `--category all` (the safe, full set):
_ALL_CATEGORIES = ['modis', 'chirps', 'era5', 'topo', 'soil']


def _maybe_resample(img, band_cfg):
    """Apply bilinear resampling for sources coarser than the export scale.

    Without this, GEE falls back to nearest-neighbor at export time and
    coarse sources (ERA5 ~11 km, CHIRPS ~5.5 km) produce blocky output
    at 250 m. Bilinear must be called on the *source* image before
    `.clip()`/export so the pyramid policy is set correctly.
    """
    if band_cfg.get('scale_native_m', EXPORT_SCALE_M) > EXPORT_SCALE_M:
        return img.resample('bilinear')
    return img


def _annual_image(band_cfg, year, aoi):
    """Build the annual composite ee.Image for a single year/band."""
    coll = ee.ImageCollection(band_cfg['collection'])
    coll = coll.filterDate(f'{year}-01-01', f'{year+1}-01-01').filterBounds(aoi)
    if 'extra_filter' in band_cfg:
        coll = band_cfg['extra_filter'](coll)
    coll = coll.select(band_cfg['band'])

    reducer = band_cfg['reducer']
    if reducer == 'mean':
        img = coll.mean()
    elif reducer == 'sum':
        img = coll.sum()
    elif reducer == 'median':
        img = coll.median()
    elif reducer == 'max':
        img = coll.max()
    elif reducer is None:
        img = ee.Image(coll.first())   # static — just take any image
    else:
        raise ValueError(f"Unknown reducer {reducer}")

    img = _maybe_resample(img, band_cfg)
    return img.toFloat().clip(aoi).set({'year': year, 'band': band_cfg['band']})


def _static_image(band_cfg, aoi):
    """Static (non-time-varying) variables: SRTM, SoilGrids, terrain derivatives."""
    if 'image_fn' in band_cfg:
        # Custom-built derivation (e.g. server-side slope/aspect/TWI).
        img = band_cfg['image_fn']()
    else:
        # OpenLandMap returns ImageCollection of one — use .first()
        try:
            img = ee.Image(band_cfg['collection']).select(band_cfg['band'])
        except Exception:
            img = ee.Image(ee.ImageCollection(band_cfg['collection']).first()).select(band_cfg['band'])
    img = _maybe_resample(img, band_cfg)
    return img.toFloat().clip(aoi).set({'band': band_cfg['band']})


# ---------------------------------------------------------------------------
# Terrain-derivative builders. Pure server-side ee.Image expressions —
# no .getInfo() calls, no Python-side compute. The 'topo' Slope/Aspect/TWI
# entries above use these as their image_fn.
# ---------------------------------------------------------------------------
_SRTM_ASSET = 'USGS/SRTMGL1_003'
_MERIT_HYDRO_ASSET = 'MERIT/Hydro/v1_0_1'


def _terrain_slope():
    """Slope in degrees, computed server-side from SRTM 30 m."""
    srtm = ee.Image(_SRTM_ASSET)
    return ee.Terrain.slope(srtm).rename('slope')


def _terrain_aspect():
    """Aspect in degrees [0, 360), computed server-side from SRTM 30 m."""
    srtm = ee.Image(_SRTM_ASSET)
    return ee.Terrain.aspect(srtm).rename('aspect')


def _terrain_twi():
    """Topographic Wetness Index = ln(SCA / tan(slope)).

    SCA (specific catchment area) ≈ upstream drainage area from MERIT
    Hydro `upa` band (km², ~90 m native). Slope from SRTM. The slope is
    floored at 0.05° (≈ 0.00087 rad) before tan() to keep the logarithm
    finite on flat pixels — standard practice in TWI implementations.
    """
    srtm = ee.Image(_SRTM_ASSET)
    slope_deg = ee.Terrain.slope(srtm)
    slope_rad_floored = slope_deg.max(0.05).multiply(ee.Number(3.14159265358979).divide(180.0))
    # MERIT Hydro upa is in km². Convert to m² so the ratio SCA/tan(slope)
    # is dimensionally a length, log of which is the TWI.
    upa_m2 = ee.Image(_MERIT_HYDRO_ASSET).select('upa').multiply(1e6)
    twi = upa_m2.divide(slope_rad_floored.tan()).log().rename('twi')
    return twi


def submit_export(image, description, drive_folder, scale, region):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=drive_folder,
        fileNamePrefix=description,
        region=region.getInfo()['coordinates'],
        scale=scale,
        crs='EPSG:4326',
        maxPixels=int(1e10),
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True},
    )
    task.start()
    return task


def _resolve_categories_and_bands(category_arg):
    """Return list of (category, [band_names]) tuples to process."""
    if category_arg == 'curated':
        return [(c, list(_CURATED_BANDS[c])) for c in _CURATED_BANDS]
    if category_arg == 'all':
        return [(c, list(BANDS[c])) for c in _ALL_CATEGORIES]
    if category_arg in BANDS:
        return [(category_arg, list(BANDS[category_arg]))]
    raise SystemExit(f'unknown --category {category_arg!r}; '
                     f'choices: curated, all, {", ".join(BANDS)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='curated',
                    help='curated (default — 11-band SOC set) | all '
                         '(safe full set, no s1/s2/soil_extra) | one of '
                         + ', '.join(BANDS))
    ap.add_argument('--years', nargs=2, type=int, default=[2002, 2023], metavar=('START', 'END'))
    ap.add_argument('--scale', type=int, default=EXPORT_SCALE_M,
                    help='Output pixel size in meters (default 250)')
    ap.add_argument('--drive-folder', default='bavaria_bands_2002_2023')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print what would be exported without submitting tasks')
    ap.add_argument('--no-skip-done', action='store_true',
                    help='Re-submit tasks even if pipeline_state.json marks them done.')
    ap.add_argument('--no-state', action='store_true',
                    help='Do not read/write Data/pipeline_state.json.')
    ap.add_argument('--validate-assets', action='store_true',
                    help='Probe every asset in the selected category and report missing ones. '
                         'Submits NO export tasks. Catches deprecated / access-restricted assets '
                         'before you commit to a 140-task batch.')
    args = ap.parse_args()

    # Pipeline state — resumable submission. Skips tasks already submitted in a previous run.
    state = None
    if not args.no_state:
        try:
            from pipeline_state import State
            state = State()
            if not args.dry_run:
                state.start_phase('gee')
        except Exception as ex:
            print(f'[warn] could not load pipeline_state: {ex}; continuing without state tracking')

    # Only authenticate / build server-side images when actually submitting
    # or validating. The dry-run path enumerates the task list locally.
    if args.dry_run and not args.validate_assets:
        aoi = None
    else:
        _require_ee()
        ee.Initialize()
        aoi = ee.Geometry.Rectangle(BAVARIA_BBOX)

    # --validate-assets short-circuits the rest: probe each asset and report.
    # Mirrors the actual submission code paths: image_fn → call it; static
    # → try ee.Image then fall back to ImageCollection.first(); yearly →
    # try ee.ImageCollection.
    if args.validate_assets:
        cat_band_lists = _resolve_categories_and_bands(args.category)
        print(f'\nValidating assets for --category {args.category}\n')
        n_ok = n_bad = 0
        bad: list[tuple[str, str, str]] = []

        def _probe_one(cfg, expects_yearly):
            if 'image_fn' in cfg:
                img = cfg['image_fn']()
                return img.bandNames().getInfo()
            # Try as Image (static asset); then as ImageCollection (yearly source).
            errors = []
            try:
                img = ee.Image(cfg['collection']).select(cfg['band'])
                return img.bandNames().getInfo()
            except Exception as e:
                errors.append(f'Image: {str(e).splitlines()[0][:140]}')
            try:
                coll = ee.ImageCollection(cfg['collection'])
                img = ee.Image(coll.first()).select(cfg['band'])
                return img.bandNames().getInfo()
            except Exception as e:
                errors.append(f'ImageCollection: {str(e).splitlines()[0][:140]}')
            raise RuntimeError(' / '.join(errors))

        for cat, band_names in cat_band_lists:
            is_static = cat in ('topo', 'soil', 'soil_extra')
            for band_name in band_names:
                cfg = BANDS[cat][band_name]
                label = f'{cat}/{band_name}'
                try:
                    _probe_one(cfg, expects_yearly=not is_static)
                    print(f'  ✓ {label:35} ({cfg.get("collection", "image_fn")})')
                    n_ok += 1
                except Exception as ex:
                    msg = str(ex).splitlines()[0][:200]
                    print(f'  ✗ {label:35} {type(ex).__name__}: {msg}')
                    bad.append((label, cfg.get('collection', 'image_fn'), msg))
                    n_bad += 1
        print(f'\nValidation summary: {n_ok} OK, {n_bad} broken.')
        if bad:
            print('\nBroken assets — fix or remove from the curated set before resubmitting:')
            for label, coll, msg in bad:
                print(f'  - {label}  (asset: {coll})')
                print(f'      reason: {msg}')
        sys.exit(0 if n_bad == 0 else 2)

    cat_band_lists = _resolve_categories_and_bands(args.category)
    y0, y1 = args.years
    tasks = []
    n_planned = 0

    print(f'\nMode: --category {args.category}    scale={args.scale}m    years={y0}-{y1}')
    print(f'Drive folder: {args.drive_folder}\n')

    skip_done = state is not None and not args.no_skip_done and not args.dry_run
    n_skipped = 0
    failed: list[tuple[str, str]] = []
    for cat, band_names in cat_band_lists:
        is_static = cat in ('topo', 'soil', 'soil_extra')
        years_for_cat = [None] if is_static else range(y0, y1 + 1)
        for band_name in band_names:
            cfg = BANDS[cat][band_name]
            for yr in years_for_cat:
                if yr is None:
                    desc = f'{cat}_{band_name}_static'
                else:
                    # Skip if year is before the source's launch
                    if cat == 's1' and yr < 2014: continue
                    if cat == 's2' and yr < 2015: continue
                    desc = f'{cat}_{band_name}_{yr}'

                n_planned += 1
                if args.dry_run:
                    print(f'  [plan] {desc:50}  scale={args.scale}m  → Drive/{args.drive_folder}/')
                    continue
                if skip_done and state.is_done('gee', desc):
                    print(f'  · skip (already submitted): {desc}')
                    n_skipped += 1
                    continue
                try:
                    img = _static_image(cfg, aoi) if yr is None else _annual_image(cfg, yr, aoi)
                    task = submit_export(img, desc, args.drive_folder, args.scale, aoi)
                except Exception as ex:
                    # One bad asset shouldn't abort the whole batch — log and continue.
                    msg = str(ex).splitlines()[0][:160]
                    print(f'  ✗ FAIL {desc:50}  {type(ex).__name__}: {msg}')
                    failed.append((desc, msg))
                    continue
                tasks.append(task)
                if state is not None:
                    state.mark_done('gee', desc)
                print(f'  [submitted] {desc:50}  (task {task.id})')

    if args.dry_run:
        print(f'\n[dry-run] would submit {n_planned} export tasks')
    else:
        print(f'\nSubmitted {len(tasks)} export tasks  '
              f'({n_skipped} skipped — already submitted; {len(failed)} failed)')
        print(f'Drive folder: "{args.drive_folder}"')
        print('Monitor at: https://code.earthengine.google.com/tasks')
        print('Or list with: earthengine task list')
        if failed:
            print('\nFAILED tasks (not in pipeline_state.json; will retry on re-run unless fixed):')
            for desc, msg in failed:
                print(f'  - {desc}: {msg}')
            print('\nFix tip: most failures are due to a deprecated / inaccessible asset.')
            print('  Run with --validate-assets to probe each asset before submitting.')
        # Note: phase isn't marked 'done' here — GEE tasks run server-side.
        # The pull/cut phases will gate on actual file presence in Drive.


if __name__ == '__main__':
    main()
