

base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'

file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"


time_before  = 5
window_size = 47 # 55 worked with all just not LAI # 33 worked with LAI
window_size_LAI = 33 # we use it max, with a bigget latent space 41 works but with 10 we will just use 33 as window size
window_size_Elevation = 55 # we use it max
window_size_SoilEvaporation = 49 # works with 49 need to check above, may work with 55
window_size_MODIS_NPP = 55  # works with 41 need to check above
window_size_LST = 47 # works with it but not 49!
window_size_TotalEvapotranspiration = 55


TIME_BEGINNING = '2007'
LOADING_TIME_BEGINNING = str(int(TIME_BEGINNING)-time_before)
TIME_END = '2023'
INFERENCE_TIME = '2015'
bands_list_order = ['Elevation','LAI','LST','MODIS_NPP','SoilEvaporation','TotalEvapotranspiration']
MAX_OC = 150

LOADING_TIME_BEGINNING_INFERENCE = str(int(INFERENCE_TIME)-time_before)

NUM_EPOCH_VAE_TRAINING = 3
NUM_EPOCH_MLP_TRAINING = 5

file_path_coordinates_Bavaria_1mil = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"

mean_LST = 14378.319661458333
mean_MODIS_NPP = 3932.2029622395835
mean_totalEvapotranspiration = 120.51464589436849
std_LST = 136.12765502929688
std_MODIS_NPP = 10021.859049479166
std_totalEvapotranspiration = 18.146349589029948




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

