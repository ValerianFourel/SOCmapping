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

The `--category extended` set adds the revision Tier-1/2/3 covariates on top
of the curated 20:
  • Tier 1 (landsat): strict per-year SCMaP-style bare-soil reflectance
    composite — SRC_{Blue,Green,Red,NIR,SWIR1,SWIR2,RCC,BCC,NBR2,BSI,
    ExposureCount}. Full Landsat C2 archive (1984+) so it spans 2002-2023.
  • Tier 2 (topo):    multi-scale terrain — TPI_{90,300,1000}, TRI, Roughness.
  • Tier 3 (era5/modis): ClimaticWaterBalance, SoilTemperature_layer1,
    FrostDays, GrowingDegreeDays, NDVI_Amplitude, NDVI_Integral, EVI_Amplitude.

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

NOTE on resampling: sources coarser than 250 m (ERA5-Land 11 km, CHIRPS
5566 m, MODIS 0.5-1 km) are bilinear-resampled on the source image so the
exported TIFFs are smoothly upsampled instead of GEE's blocky default.
Sources finer than 250 m (Landsat / SRTM 30 m) are aggregated with an
explicit area-weighted reduceResolution(mean) so the 30 m -> 250 m averaging
is reproducible rather than relying on GEE's export-time pyramid. See
_to_250m().
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
        # --- Tier 3 phenology proxies (within-year MOD13Q1 statistics) ---
        'NDVI_Amplitude': {
            'image_fn_yearly': lambda y, a: _veg_metric(y, a, 'NDVI', 'amplitude'),
            'band': 'NDVI_amp', 'reducer': None, 'scale_native_m': 250,
            'notes': 'Within-year NDVI max-min (MOD13Q1) — phenology amplitude, raw x0.0001.',
        },
        'NDVI_Integral': {
            'image_fn_yearly': lambda y, a: _veg_metric(y, a, 'NDVI', 'integral'),
            'band': 'NDVI_int', 'reducer': None, 'scale_native_m': 250,
            'notes': 'Within-year sum of 16-day NDVI (MOD13Q1) — growing-season integral proxy.',
        },
        'EVI_Amplitude': {
            'image_fn_yearly': lambda y, a: _veg_metric(y, a, 'EVI', 'amplitude'),
            'band': 'EVI_amp', 'reducer': None, 'scale_native_m': 250,
            'notes': 'Within-year EVI max-min (MOD13Q1) — phenology amplitude proxy.',
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
        # --- Tier 3 climate derivations (ERA5-Land, full 2002-2023 coverage) ---
        'SoilTemperature_layer1': {
            'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
            'band': 'soil_temperature_level_1',
            'reducer': 'mean',
            'scale_native_m': 11132,
            'notes': 'Soil temperature 0-7 cm (Kelvin), annual mean — decomposition driver.',
        },
        'ClimaticWaterBalance': {
            'image_fn_yearly': lambda y, a: _climatic_water_balance(y, a),
            'band': 'cwb', 'reducer': None, 'scale_native_m': 11132,
            'notes': 'P - PET (ERA5-Land, m/yr) — aridity / water balance, strong SOC covariate.',
        },
        'FrostDays': {
            'image_fn_yearly': lambda y, a: _frost_days(y, a),
            'band': 'frost_days', 'reducer': None, 'scale_native_m': 11132,
            'notes': 'Count of days with 2 m Tmin < 0 C (ERA5-Land daily).',
        },
        'GrowingDegreeDays': {
            'image_fn_yearly': lambda y, a: _growing_degree_days(y, a),
            'band': 'gdd', 'reducer': None, 'scale_native_m': 11132,
            'notes': 'GDD5: annual sum of mean-temp degrees above 5 C (ERA5-Land daily).',
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
        # --- Tier 2 multi-scale terrain (server-side from SRTM 30 m) ---
        'TPI_90': {
            'image_fn': lambda: _tpi(90), 'band': 'tpi_90', 'reducer': None,
            'scale_native_m': 30, 'notes': 'Topographic Position Index, 90 m radius (SRTM).',
        },
        'TPI_300': {
            'image_fn': lambda: _tpi(300), 'band': 'tpi_300', 'reducer': None,
            'scale_native_m': 30, 'notes': 'Topographic Position Index, 300 m radius (SRTM).',
        },
        'TPI_1000': {
            'image_fn': lambda: _tpi(1000), 'band': 'tpi_1000', 'reducer': None,
            'scale_native_m': 30, 'notes': 'Topographic Position Index, 1000 m radius (SRTM).',
        },
        'TRI': {
            'image_fn': lambda: _terrain_tri(), 'band': 'tri', 'reducer': None,
            'scale_native_m': 30, 'notes': 'Terrain Ruggedness Index (local elevation stddev, 90 m).',
        },
        'Roughness': {
            'image_fn': lambda: _terrain_roughness(), 'band': 'roughness', 'reducer': None,
            'scale_native_m': 30, 'notes': 'Local elevation range (max-min, 90 m window).',
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

    # ============================================================
    # Tier 1 — Landsat bare-soil reflectance composite (SCMaP-style).
    # Strict per-year; Landsat C2 archive (1984+) spans 2002-2023, unlike
    # Sentinel-2 (2015+). Each band selects one layer of the per-year
    # _bare_soil_composite(year); reflectance is native 30 m → 250 m by
    # area-weighted mean (_to_250m). Wall-to-wall: pixels with no bare-soil
    # observation are NoData, and SRC_ExposureCount records the obs count.
    # ============================================================
    'landsat': {
        'SRC_Blue':  {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('Blue'),  'band': 'Blue',  'reducer': None, 'scale_native_m': 30, 'notes': 'Bare-soil composite, Blue surface reflectance.'},
        'SRC_Green': {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('Green'), 'band': 'Green', 'reducer': None, 'scale_native_m': 30, 'notes': 'Bare-soil composite, Green surface reflectance.'},
        'SRC_Red':   {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('Red'),   'band': 'Red',   'reducer': None, 'scale_native_m': 30, 'notes': 'Bare-soil composite, Red surface reflectance.'},
        'SRC_NIR':   {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('NIR'),   'band': 'NIR',   'reducer': None, 'scale_native_m': 30, 'notes': 'Bare-soil composite, NIR surface reflectance.'},
        'SRC_SWIR1': {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('SWIR1'), 'band': 'SWIR1', 'reducer': None, 'scale_native_m': 30, 'notes': 'Bare-soil composite, SWIR1 (~1.6 um) reflectance.'},
        'SRC_SWIR2': {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('SWIR2'), 'band': 'SWIR2', 'reducer': None, 'scale_native_m': 30, 'notes': 'Bare-soil composite, SWIR2 (~2.2 um) — strongest bare-soil SOC signal.'},
        'SRC_RCC':   {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('RCC'),   'band': 'RCC',   'reducer': None, 'scale_native_m': 30, 'notes': 'Red chromatic coordinate R/(R+G+B) of the composite.'},
        'SRC_BCC':   {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('BCC'),   'band': 'BCC',   'reducer': None, 'scale_native_m': 30, 'notes': 'Blue chromatic coordinate B/(R+G+B) of the composite.'},
        'SRC_NBR2':  {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('NBR2'),  'band': 'NBR2',  'reducer': None, 'scale_native_m': 30, 'notes': 'NBR2 (SWIR1-SWIR2)/(SWIR1+SWIR2) of the composite.'},
        'SRC_BSI':   {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('BSI'),   'band': 'BSI',   'reducer': None, 'scale_native_m': 30, 'notes': 'Bare Soil Index of the composite.'},
        'SRC_ExposureCount': {'image_fn_yearly': lambda y, a: _bare_soil_composite(y, a).select('ExposureCount'), 'band': 'ExposureCount', 'reducer': None, 'scale_native_m': 30, 'notes': 'Number of valid bare-soil observations in the year — validity mask for SRC_* bands.'},
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
# Categories included in `--category all` (the safe, full set). Landsat SRC is
# intentionally NOT here — it is heavy (11 bands x 22 years) and pulled via
# `--category extended` or `--category landsat` to avoid a surprise batch.
_ALL_CATEGORIES = ['modis', 'chirps', 'era5', 'topo', 'soil']

# `--category extended` = curated 20 + Tier 1/2/3 revision covariates.
# (Terrain/Tier-3 bands live in the existing topo/era5/modis categories, so
# `--category all`, `topo`, `era5`, `modis` already include them; only the
# Landsat SRC bands need the dedicated `landsat` category.)
_EXTENDED_BANDS = {
    'modis':   _CURATED_BANDS['modis'] + ['NDVI_Amplitude', 'NDVI_Integral', 'EVI_Amplitude'],
    'era5':    _CURATED_BANDS['era5'] + ['SoilTemperature_layer1', 'ClimaticWaterBalance',
                                         'FrostDays', 'GrowingDegreeDays'],
    'soil':    list(_CURATED_BANDS['soil']),
    'topo':    _CURATED_BANDS['topo'] + ['TPI_90', 'TPI_300', 'TPI_1000', 'TRI', 'Roughness'],
    'landsat': list(BANDS['landsat']),
}

# `--category new` = ONLY the 23 Tier 1/2/3 bands added this revision (no
# re-render of the existing 20). Use this to append to a dataset that already
# holds the curated 20 — same grid, so the new bands align pixel-for-pixel.
_NEW_BANDS = {
    'landsat': list(BANDS['landsat']),
    'era5':    ['SoilTemperature_layer1', 'ClimaticWaterBalance', 'FrostDays', 'GrowingDegreeDays'],
    'modis':   ['NDVI_Amplitude', 'NDVI_Integral', 'EVI_Amplitude'],
    'topo':    ['TPI_90', 'TPI_300', 'TPI_1000', 'TRI', 'Roughness'],
}


def _to_250m(img, native_m, categorical=False):
    """Normalize a source image onto the 250 m export grid.

    - Coarser than 250 m (ERA5 11 km, CHIRPS 5.5 km, MODIS 0.5-1 km):
      bilinear-resample the *source* so GEE doesn't fall back to blocky
      nearest-neighbor at export time. Must be set on the source before clip.
    - Finer than 250 m (Landsat / SRTM 30 m): area-weighted aggregate via
      reduceResolution(mean) so the 30 m -> 250 m averaging is explicit and
      reproducible instead of relying on GEE's export-time pyramid. Categorical
      inputs use mode() (mean is meaningless on class indices).
    - ~250 m (MODIS NDVI/EVI, SoilGrids): leave as-is.
    """
    if native_m is None:
        native_m = EXPORT_SCALE_M
    if native_m > EXPORT_SCALE_M:
        return img.resample('bilinear')
    if native_m < EXPORT_SCALE_M - 1:
        reducer = ee.Reducer.mode() if categorical else ee.Reducer.mean()
        return (img.reduceResolution(reducer=reducer, maxPixels=1024)
                   .reproject(crs='EPSG:4326', scale=EXPORT_SCALE_M))
    return img


def _annual_image(band_cfg, year, aoi):
    """Build the annual composite ee.Image for a single year/band."""
    if 'image_fn_yearly' in band_cfg:
        # Custom per-year derivation (Landsat SRC, ERA5/MODIS derivations).
        img = band_cfg['image_fn_yearly'](year, aoi)
    else:
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

    img = _to_250m(img, band_cfg.get('scale_native_m', EXPORT_SCALE_M),
                   band_cfg.get('categorical', False))
    return img.toFloat().clip(aoi).set({'year': year, 'band': band_cfg.get('band', '')})


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
    img = _to_250m(img, band_cfg.get('scale_native_m', EXPORT_SCALE_M),
                   band_cfg.get('categorical', False))
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


# ---------------------------------------------------------------------------
# Tier 2 — multi-scale terrain derivatives (server-side from SRTM 30 m).
# TPI = elevation minus the local mean over a circular window: captures
# landscape position (ridge vs valley) at the chosen scale. TRI / Roughness
# summarize local relief.
# ---------------------------------------------------------------------------
def _tpi(radius_m):
    srtm = ee.Image(_SRTM_ASSET)
    local_mean = srtm.focal_mean(radius=radius_m, kernelType='circle', units='meters')
    return srtm.subtract(local_mean).rename(f'tpi_{radius_m}')


def _terrain_tri():
    """Terrain Ruggedness Index ~ local stddev of elevation over a 90 m window."""
    srtm = ee.Image(_SRTM_ASSET)
    return srtm.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.circle(radius=90, units='meters'),
    ).rename('tri')


def _terrain_roughness():
    """Local elevation range (max - min) over a 90 m window."""
    srtm = ee.Image(_SRTM_ASSET)
    mm = srtm.reduceNeighborhood(
        reducer=ee.Reducer.minMax(),
        kernel=ee.Kernel.circle(radius=90, units='meters'),
    )
    return mm.select('elevation_max').subtract(mm.select('elevation_min')).rename('roughness')


# ---------------------------------------------------------------------------
# Tier 1 — Landsat bare-soil reflectance composite (SCMaP-style).
# Surface reflectance from Landsat Collection-2 Level-2 (full 1984+ archive,
# so it spans 2002-2023, unlike Sentinel-2 from 2015). Per-sensor SR band
# numbers differ, so each is renamed to a common 6-band set.
# ---------------------------------------------------------------------------
_LANDSAT_C2 = {
    'LANDSAT/LT05/C02/T1_L2': {'SR_B1': 'Blue', 'SR_B2': 'Green', 'SR_B3': 'Red',
                               'SR_B4': 'NIR', 'SR_B5': 'SWIR1', 'SR_B7': 'SWIR2'},
    'LANDSAT/LE07/C02/T1_L2': {'SR_B1': 'Blue', 'SR_B2': 'Green', 'SR_B3': 'Red',
                               'SR_B4': 'NIR', 'SR_B5': 'SWIR1', 'SR_B7': 'SWIR2'},
    'LANDSAT/LC08/C02/T1_L2': {'SR_B2': 'Blue', 'SR_B3': 'Green', 'SR_B4': 'Red',
                               'SR_B5': 'NIR', 'SR_B6': 'SWIR1', 'SR_B7': 'SWIR2'},
    'LANDSAT/LC09/C02/T1_L2': {'SR_B2': 'Blue', 'SR_B3': 'Green', 'SR_B4': 'Red',
                               'SR_B5': 'NIR', 'SR_B6': 'SWIR1', 'SR_B7': 'SWIR2'},
}
_SRC_REFLECTANCE = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
# SCMaP-style bare-soil selection (tunable): NDVI lower bound drops
# water/shadow/snow, upper bound drops vegetation; NBR2 upper bound drops
# crop-residue / non-photosynthetic vegetation.
_SRC_NDVI_MIN, _SRC_NDVI_MAX, _SRC_NBR2_MAX = 0.15, 0.25, 0.075
# Cap each sensor to its N least-cloudy scenes/year. Bounds the median-stack
# depth -> bounds EE export memory AND speeds the composite up ~10x. Trade-off:
# median over the best ~N observations instead of ALL bare-soil observations
# (negligible at 250 m; a deviation from strict "all-observations" SCMaP).
_SRC_MAX_PER_SENSOR = 20


def _prep_landsat(img, band_map):
    """Scale C2-L2 SR to reflectance, rename to common bands, mask cloud/shadow/snow."""
    qa = img.select('QA_PIXEL')
    # QA_PIXEL bits (Collection 2): 1 dilated cloud, 3 cloud, 4 cloud shadow, 5 snow.
    bad = (qa.bitwiseAnd(1 << 1).neq(0)
           .Or(qa.bitwiseAnd(1 << 3).neq(0))
           .Or(qa.bitwiseAnd(1 << 4).neq(0))
           .Or(qa.bitwiseAnd(1 << 5).neq(0)))
    sr = (img.select(list(band_map.keys()))
             .rename(list(band_map.values()))
             .multiply(0.0000275).add(-0.2))           # C2-L2 SR scale/offset
    plausible = sr.gt(0.0).And(sr.lt(1.0)).reduce(ee.Reducer.min())
    return sr.updateMask(bad.Not()).updateMask(plausible)


def _landsat_harmonized(year, aoi):
    """Merge all Landsat sensors for one calendar year into a common-band collection."""
    merged = None
    for asset, band_map in _LANDSAT_C2.items():
        coll = (ee.ImageCollection(asset)
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                .filterBounds(aoi)
                .sort('CLOUD_COVER').limit(_SRC_MAX_PER_SENSOR)
                .map(lambda im, bm=band_map: _prep_landsat(im, bm)))
        merged = coll if merged is None else merged.merge(coll)
    return merged


def _bare_soil_composite(year, aoi):
    """Strict per-year SCMaP-style bare-soil composite plus derived indices.

    Returns an 11-band image: the 6 reflectance bands (median of bare-soil
    observations), RCC, BCC, NBR2, BSI, and ExposureCount (count of valid
    bare-soil observations — the validity mask for everything else).
    """
    coll = _landsat_harmonized(year, aoi)
    # median() over a merged multi-sensor collection drops the projection, which
    # breaks the downstream reduceResolution (30 m -> 250 m). Capture a Landsat
    # native 30 m projection and pin it on the output so reduceResolution has a
    # valid input grid.
    ref_proj = ee.Image(coll.first()).select('Red').projection()

    def mask_bare(img):
        ndvi = img.normalizedDifference(['NIR', 'Red'])
        nbr2 = img.normalizedDifference(['SWIR1', 'SWIR2'])
        bare = (ndvi.gt(_SRC_NDVI_MIN).And(ndvi.lt(_SRC_NDVI_MAX))
                    .And(nbr2.lt(_SRC_NBR2_MAX)))
        return img.updateMask(bare)

    bare = coll.map(mask_bare).select(_SRC_REFLECTANCE)
    # parallelScale lowers peak memory — the median over a deep multi-sensor stack
    # is what hit EE's "out of memory" on full-Bavaria exports. rename() restores
    # the plain band names that reduce() would otherwise suffix with "_median".
    comp = bare.reduce(ee.Reducer.median(), parallelScale=8).rename(_SRC_REFLECTANCE)
    count = bare.select('Red').reduce(ee.Reducer.count(), parallelScale=8).rename('ExposureCount')

    b, g, r = comp.select('Blue'), comp.select('Green'), comp.select('Red')
    nir, sw1, sw2 = comp.select('NIR'), comp.select('SWIR1'), comp.select('SWIR2')
    rgb = r.add(g).add(b)
    rcc = r.divide(rgb).rename('RCC')
    bcc = b.divide(rgb).rename('BCC')
    nbr2 = sw1.subtract(sw2).divide(sw1.add(sw2)).rename('NBR2')
    bsi = (sw1.add(r).subtract(nir.add(b))).divide(
        sw1.add(r).add(nir.add(b))).rename('BSI')
    out = comp.addBands([rcc, bcc, nbr2, bsi, count])
    return out.setDefaultProjection(ref_proj)


# ---------------------------------------------------------------------------
# Tier 3 — cheap climate / phenology derivations (full 2002-2023 coverage).
# ---------------------------------------------------------------------------
_ERA5_MONTHLY = 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
_ERA5_DAILY = 'ECMWF/ERA5_LAND/DAILY_AGGR'
_MOD13Q1 = 'MODIS/061/MOD13Q1'


def _climatic_water_balance(year, aoi):
    """Annual precipitation minus potential evaporation (ERA5-Land, m/yr).

    ERA5-Land potential_evaporation_sum is a downward-negative flux, so its
    annual magnitude is abs(sum); CWB = P - PET.
    """
    m = ee.ImageCollection(_ERA5_MONTHLY).filterDate(f'{year}-01-01', f'{year + 1}-01-01')
    precip = m.select('total_precipitation_sum').sum()
    pet = m.select('potential_evaporation_sum').sum().abs()
    return precip.subtract(pet).rename('cwb')


def _frost_days(year, aoi):
    """Count of days with 2 m minimum temperature below freezing (ERA5-Land daily)."""
    d = ee.ImageCollection(_ERA5_DAILY).filterDate(f'{year}-01-01', f'{year + 1}-01-01')
    return d.select('temperature_2m_min').map(lambda im: im.lt(273.15)).sum().rename('frost_days')


def _growing_degree_days(year, aoi):
    """GDD5: annual sum of mean-temp degrees above a 5 degC base (ERA5-Land daily)."""
    d = ee.ImageCollection(_ERA5_DAILY).filterDate(f'{year}-01-01', f'{year + 1}-01-01')
    return d.select('temperature_2m').map(
        lambda im: im.subtract(278.15).max(0)).sum().rename('gdd')


def _veg_metric(year, aoi, band, stat):
    """Within-year vegetation-index statistic from MOD13Q1 (phenology proxy).

    stat='amplitude' -> max-min across the 16-day composites;
    stat='integral'  -> sum of the 16-day composites (growing-season area).
    """
    coll = ee.ImageCollection(_MOD13Q1).filterDate(
        f'{year}-01-01', f'{year + 1}-01-01').select(band)
    if stat == 'amplitude':
        return coll.max().subtract(coll.min()).rename(f'{band}_amp')
    return coll.sum().rename(f'{band}_int')


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
    if category_arg == 'extended':
        return [(c, list(_EXTENDED_BANDS[c])) for c in _EXTENDED_BANDS]
    if category_arg == 'new':
        return [(c, list(_NEW_BANDS[c])) for c in _NEW_BANDS]
    if category_arg == 'all':
        return [(c, list(BANDS[c])) for c in _ALL_CATEGORIES]
    if category_arg in BANDS:
        return [(category_arg, list(BANDS[category_arg]))]
    raise SystemExit(f'unknown --category {category_arg!r}; '
                     f'choices: curated, extended, all, {", ".join(BANDS)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='curated',
                    help='curated (default — 20-band SOC set) | extended '
                         '(curated + Tier 1/2/3: landsat SRC, multi-scale terrain, '
                         'climate/phenology derivations) | new (ONLY the 23 Tier 1/2/3 '
                         'bands, for appending to an existing 20-band dataset) | all '
                         '(safe full set, no s1/s2/landsat/soil_extra) | one of '
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

    # Make the source-resampling target follow --scale. _to_250m() (and the
    # scale_native_m defaults) read this module global, so without this a
    # sentinel-mode `--scale 20` would still aggregate fine bands to 250 m and
    # THEN export at 20 m. Setting it here keeps source grid == export grid, so
    # `--scale 20` produces a true 20 m ("sentinel size") raster for every band.
    global EXPORT_SCALE_M
    EXPORT_SCALE_M = args.scale

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
            if 'image_fn_yearly' in cfg:
                img = cfg['image_fn_yearly'](2015, aoi)
                return img.bandNames().getInfo()
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
