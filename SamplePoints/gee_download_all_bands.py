"""Download every relevant SOC-prediction band from Google Earth Engine, 2002–2023.

Pulls 17 variables across 5 categories. All bands are exported as
annual composites (or static) GeoTIFFs at 250 m / EPSG:4326 covering
Bavaria, written to your Google Drive in a folder
`bavaria_bands_2002_2023/<category>/<band>/`.

Categories and provider notes:
  • MODIS  (existing 6 bands + NDVI + EVI; full 2002–2023 coverage)
  • CHIRPS precipitation (daily 5 km, 1981+)
  • ERA5-Land (climate; 9 km, 1950+ — air temp, soil moisture)
  • Topography (SRTM 30 m DEM, static)
  • Static soils (OpenLandMap / SoilGrids — clay, sand, silt, pH,
    bulk density, CEC; single snapshot)
  • Sentinel-1 SAR (only 2014+, temporal gap 2002–2013)
  • Sentinel-2 (only 2015+, temporal gap 2002–2014)

Run requirements:
    pip install earthengine-api
    earthengine authenticate                  # one-time, browser-based

Then:
    python SamplePoints/gee_download_all_bands.py
        [--category modis|chirps|era5|topo|soil|s1|s2|all]
        [--years 2002 2023]
        [--scale 250]                         # meters per pixel
        [--max-pixels 1e10]
        [--drive-folder bavaria_bands_2002_2023]

The script submits Export.image.toDrive() tasks (one per variable per
year). They run server-side; monitor at https://code.earthengine.google.com/tasks
or via `earthengine task list`. Downloads land in your Drive root by
default — drag them into rclone / `gdown` to bring them local.

After the GeoTIFFs are local, run `samplePoints.py` to project LUCAS
coordinates onto each new band, then update `bands_list_order` in
`config.py`.
"""
import argparse
import sys
from pathlib import Path

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed. Run:")
    print("    pip install earthengine-api")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bavaria area of interest (degrees, WGS84) — slightly buffered
# ---------------------------------------------------------------------------
BAVARIA_BBOX = [8.8, 47.2, 14.0, 50.6]   # [W, S, E, N]


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
        'LST_Day': {
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
            'notes': 'Land Surface Temperature, night (Kelvin × 0.02) — new vs your existing LST',
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
        'NPP': {
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
        'ET': {
            'collection': 'MODIS/061/MOD16A2',
            'band': 'ET',
            'reducer': 'sum',
            'scale_native_m': 500,
            'notes': 'Total Evapotranspiration, 8-day, mm × 0.1',
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
    },

    # ============================================================
    # Topography (SRTM 30m, static)
    # ============================================================
    'topo': {
        'Elevation_SRTM': {
            'collection': 'USGS/SRTMGL1_003',
            'band': 'elevation',
            'reducer': None,
            'scale_native_m': 30,
            'notes': '30m SRTM DEM — for re-derivation of slope/aspect/TWI if needed',
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
            'collection': 'OpenLandMap/SOL/SOL_CEC_USDA-4B1A_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'Cation Exchange Capacity at 10 cm in mmol/100g',
        },
        'OrganicCarbon_Reference': {
            'collection': 'OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'OpenLandMap reference SOC at 10 cm — useful as a strong prior; DO NOT include if testing zero-leakage prediction.',
        },
        'TextureUSDA': {
            'collection': 'OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02',
            'band': 'b10',
            'reducer': None,
            'scale_native_m': 250,
            'notes': 'USDA texture class (integer) at 10 cm — categorical, may need one-hot',
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

    return img.toFloat().clip(aoi).set({'year': year, 'band': band_cfg['band']})


def _static_image(band_cfg, aoi):
    """Static (non-time-varying) variables: SRTM, SoilGrids, etc."""
    # OpenLandMap returns ImageCollection of one — use .first()
    try:
        img = ee.Image(band_cfg['collection']).select(band_cfg['band'])
    except Exception:
        img = ee.Image(ee.ImageCollection(band_cfg['collection']).first()).select(band_cfg['band'])
    return img.toFloat().clip(aoi).set({'band': band_cfg['band']})


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='all',
                    choices=['all', 'modis', 'chirps', 'era5', 'topo', 'soil', 's1', 's2'])
    ap.add_argument('--years', nargs=2, type=int, default=[2002, 2023], metavar=('START', 'END'))
    ap.add_argument('--scale', type=int, default=250, help='Output pixel size in meters')
    ap.add_argument('--drive-folder', default='bavaria_bands_2002_2023')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print what would be exported without submitting tasks')
    args = ap.parse_args()

    ee.Initialize()

    aoi = ee.Geometry.Rectangle(BAVARIA_BBOX)
    cats = list(BANDS) if args.category == 'all' else [args.category]
    y0, y1 = args.years
    tasks = []
    n_planned = 0

    for cat in cats:
        is_static = cat in ('topo', 'soil')
        years_for_cat = [None] if is_static else range(y0, y1 + 1)
        for band_name, cfg in BANDS[cat].items():
            for yr in years_for_cat:
                if yr is None:
                    img = _static_image(cfg, aoi)
                    desc = f'{cat}_{band_name}_static'
                else:
                    # Skip if year is before the source's launch
                    if cat == 's1' and yr < 2014: continue
                    if cat == 's2' and yr < 2015: continue
                    img = _annual_image(cfg, yr, aoi)
                    desc = f'{cat}_{band_name}_{yr}'

                n_planned += 1
                if args.dry_run:
                    print(f'  [plan] {desc:50}  scale={args.scale}m  → Drive/{args.drive_folder}/')
                else:
                    task = submit_export(img, desc, args.drive_folder, args.scale, aoi)
                    tasks.append(task)
                    print(f'  [submitted] {desc:50}  (task {task.id})')

    if args.dry_run:
        print(f'\n[dry-run] would submit {n_planned} export tasks')
    else:
        print(f'\nSubmitted {len(tasks)} export tasks to Drive folder "{args.drive_folder}".')
        print('Monitor at: https://code.earthengine.google.com/tasks')
        print('Or list with: earthengine task list')


if __name__ == '__main__':
    main()
