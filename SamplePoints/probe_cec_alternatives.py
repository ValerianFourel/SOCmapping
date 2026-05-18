"""Probe candidate CEC asset IDs on GEE to find a working replacement for the
deprecated `OpenLandMap/SOL/SOL_CEC_USDA-4B1A_M/v02`.

Usage:
    python SamplePoints/probe_cec_alternatives.py

Prints, for each candidate, whether it loads and what its band names are.
The first one that works can be wired back into BANDS['soil']['CEC_0_10cm']
in gee_download_all_bands.py.
"""
import sys

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed. Run: pip install earthengine-api")
    sys.exit(1)

ee.Initialize()

CANDIDATES = [
    # Original (known broken — for confirmation):
    ('OpenLandMap/SOL/SOL_CEC_USDA-4B1A_M/v02',                None),
    # OpenLandMap alternative naming with CHEMSOL prefix:
    ('OpenLandMap/SOL/SOL_CHEMSOL-CEC_USDA-4B1A_M/v02',        None),
    # ISRIC SoilGrids 2.0 — community-uploaded mirrors. Try a few paths:
    ('projects/soilgrids-isric/cec_mean',                       None),
    ('projects/sat-io/open-datasets/ISRIC/SoilGrids250/cec',    None),
    ('projects/soil-modeling/soilgrids/cec_0-5cm_mean',         None),
    # Alternative: nitrogen at 0-5cm — strong SOC covariate, not the target itself.
    # Use as a substitute predictor if no CEC asset works.
    ('projects/soilgrids-isric/nitrogen_mean',                  None),
    # Another alternative: Sat-IO SoilGrids namespace:
    ('projects/sat-io/open-datasets/SoilGrids/cec_mean',        None),
]


def probe(asset_id: str, band: str | None) -> str:
    """Try to load an asset and return either its bands or the error."""
    # Try as Image first; if that fails, as ImageCollection.first().
    try:
        img = ee.Image(asset_id)
        names = img.bandNames().getInfo()
        return f'IMAGE ✓  bands={names}'
    except Exception as e1:
        try:
            img = ee.Image(ee.ImageCollection(asset_id).first())
            names = img.bandNames().getInfo()
            return f'COLLECTION ✓  bands={names}'
        except Exception as e2:
            return f'BOTH FAIL\n    Image error:      {str(e1).splitlines()[0][:160]}\n    Collection error: {str(e2).splitlines()[0][:160]}'


def main():
    print('Probing candidate CEC asset IDs on GEE…\n')
    working = []
    for asset, band in CANDIDATES:
        print(f'  {asset}')
        result = probe(asset, band)
        for line in result.splitlines():
            print(f'      {line}')
        if 'FAIL' not in result:
            working.append((asset, result))
        print()

    if working:
        print('=' * 60)
        print('Working candidates — pick one for BANDS["soil"]["CEC_0_10cm"]:')
        for asset, info in working:
            print(f'  {asset}')
            print(f'    {info}')
        print()
        print('To wire it back in: edit gee_download_all_bands.py')
        print('  - In BANDS[\'soil\']: set CEC_0_10cm collection = <working asset id>')
        print('  - Pick the right band name from the list above (e.g. cec_0-5cm_mean)')
        print('  - Re-add \'CEC_0_10cm\' to _CURATED_BANDS[\'soil\']')
        print('  - Re-add the entry to each model\'s config.py bands_list_order')
        print('  - Re-run: bash SamplePoints/run_full_pipeline.sh gee')
    else:
        print('=' * 60)
        print('No CEC alternative worked. Stay on 19 channels.')


if __name__ == '__main__':
    main()
