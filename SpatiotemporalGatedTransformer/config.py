

# --------------------------------------------------------------------------
# Path resolution — every absolute path below resolves from environment
# variables (SOC_PROJECT_ROOT / SOC_DATA_DIR / SOC_WEIGHTS_DIR) with a
# walk-up fallback, then a legacy hardcoded default. See SOCmapping/_paths.py
# for the full resolution order.
# --------------------------------------------------------------------------
import os as _os
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _paths import SOC_DATA_DIR_STR as _SOC_DATA_DIR_STR  # noqa: E402

base_path_data = _SOC_DATA_DIR_STR

file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"


time_before  = 5 # 5 works best
window_size = 5 # 5 works best
TIME_BEGINNING = '2007'
LOADING_TIME_BEGINNING = str(int(TIME_BEGINNING)-time_before)
TIME_END = '2023'
INFERENCE_TIME = '2023'
bands_list_order = ['Elevation','LAI','LST','MODIS_NPP','SoilEvaporation','TotalEvapotranspiration']
MAX_OC = 150
num_epochs = 270   # 200 works as
NUM_EPOCHS_RUN = 320 # 250 for when we have a training on all the data validation and training for the mapping Deep Neural Network
LOADING_TIME_BEGINNING_INFERENCE = str(int(INFERENCE_TIME)-time_before)
NUM_LAYERS = 2
NUM_HEADS = 8

# Now derived from SOC_DATA_DIR instead of a foreign-user hardcoded path.
# Both can still be overridden via the environment if needed:
#   SOC_PREDICTIONS_PLOTS_DIR=...   (output dir for inference plots)
#   SOC_COORDS_1MIL_CSV=...         (full path to the 1.3 M reference grid CSV)
save_path_predictions_plots = _os.environ.get(
    'SOC_PREDICTIONS_PLOTS_DIR',
    f"{base_path_data}/../predictions_plots/simpleTFT_plots",
)
file_path_coordinates_Bavaria_1mil = _os.environ.get(
    'SOC_COORDS_1MIL_CSV',
    f"{base_path_data}/Coordinates1Mil/coordinates_Bavaria_1mil.csv",
)
PICTURE_VERSION = f"{str(num_epochs)}_{str(MAX_OC)}_{INFERENCE_TIME}_version"
hidden_size = 128

def generate_seasonal_list():
    seasons = ['winter', 'spring', 'summer', 'autumn']  # Reordered with winter first
    years = range(2000, 2025)  # 2000 to 2024

    seasonal_list = []

    for year in years:
        for season in seasons:
            # Skip winter 2000 as it belongs to 1999
            if year == 2000 and season == 'winter':
                continue
            # For 2024, only include winter
            if year == 2024 and season != 'winter':
                continue
            seasonal_list.append(f"{year}_{season}")

    return seasonal_list

# Generate the list
seasons = generate_seasonal_list()

years_padded = [f"{year:04d}" for year in range(2000, 2024)]


#######################################################################
# YEARLY PATH

elevationBandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation'
LAIBandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/LAI'
LSTBandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/LST'
MODIS_NPPBandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/TotalEvapotranspiration'

SamplesCoordinates_Yearly = [elevationBandMatrixCoordinates_Yearly,  LAIBandMatrixCoordinates_Yearly,  LSTBandMatrixCoordinates_Yearly, MODIS_NPPBandMatrixCoordinates_Yearly , SoilEvaporationBandMatrixCoordinates_Yearly, TotalEvapotranspirationBandMatrixCoordinates_Yearly ]

elevationBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/StaticValue/Elevation'
LAIBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/YearlyValue/LAI'
LSTBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/YearlyValue/LST'
MODIS_NPPBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/YearlyValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/YearlyValue/TotalEvapotranspiration'

MatrixCoordinates_1mil_Yearly = [elevationBandMatrixCoordinates_1milPoints_Yearly, LAIBandMatrixCoordinates_1milPoints_Yearly, LSTBandMatrixCoordinates_1milPoints_Yearly, MODIS_NPPBandMatrixCoordinates_1milPoints_Yearly, SoilEvaporationBandMatrixCoordinates_1milPoints_Yearly, TotalEvapotranspirationBandMatrixCoordinates_1milPoints_Yearly ]

elevationTensorData = f'{base_path_data}/RasterTensorData/StaticValue/Elevation'
LAITensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/LAI'
LSTTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/LST'
MODIS_NPPTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/MODIS_NPP'
SoilEvaporationTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/SoilEvaporation'
TotalEvapotranspirationTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/TotalEvapotranspiration'

DataYearly = [elevationTensorData , LAITensorDataYearly , LSTTensorDataYearly, MODIS_NPPTensorDataYearly, SoilEvaporationTensorDataYearly, TotalEvapotranspirationTensorDataYearly ]

#######################################################################
#SEASON PATHS

elevationBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation'
LAIBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/SeasonalValue/LAI'
LSTBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/SeasonalValue/LST'
MODIS_NPPBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/SeasonalValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/SeasonalValue/TotalEvapotranspiration'

SamplesCoordinates_Seasonally = [elevationBandMatrixCoordinates_Seasonally , LAIBandMatrixCoordinates_Seasonally, LSTBandMatrixCoordinates_Seasonally, MODIS_NPPBandMatrixCoordinates_Seasonally, SoilEvaporationBandMatrixCoordinates_Seasonally, TotalEvapotranspirationBandMatrixCoordinates_Seasonally ]

elevationBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/StaticValue/Elevation'
LAIBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/SeasonalValue/LAI'
LSTBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/SeasonalValue/LST'
MODIS_NPPBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/SeasonalValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/SeasonalValue/TotalEvapotranspiration'

MatrixCoordinates_1mil_Seasonally = [elevationBandMatrixCoordinates_1milPoints_Seasonally , LAIBandMatrixCoordinates_1milPoints_Seasonally, LSTBandMatrixCoordinates_1milPoints_Seasonally, MODIS_NPPBandMatrixCoordinates_1milPoints_Seasonally, SoilEvaporationBandMatrixCoordinates_1milPoints_Seasonally, TotalEvapotranspirationBandMatrixCoordinates_1milPoints_Seasonally ]

elevationTensorData = f'{base_path_data}/RasterTensorData/StaticValue/Elevation'
LAITensorDataSeasonally = f'{base_path_data}/RasterTensorData/SeasonalValue/LAI'
LSTTensorDataSeasonally = f'{base_path_data}/RasterTensorData/SeasonalValue/LST'
MODIS_NPPTensorSeasonally = f'{base_path_data}/RasterTensorData/YearlyValue/MODIS_NPP'
SoilEvaporationTensorSeasonally = f'{base_path_data}/RasterTensorData/SeasonalValue/SoilEvaporation'
TotalEvapotranspirationTensorSeasonally = f'{base_path_data}/RasterTensorData/SeasonalValue/TotalEvapotranspiration'

DataSeasonally = [elevationTensorData, LAITensorDataSeasonally, LSTTensorDataSeasonally, MODIS_NPPTensorSeasonally, SoilEvaporationTensorSeasonally, TotalEvapotranspirationTensorSeasonally ]

#######################################################################

