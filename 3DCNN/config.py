

base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'

file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"

bands = ['LAI','LST','SoilEvaporation','MODIS_NPP','Elevation','TotalEvapotranspiration']

TIME_BEGINNING = '2014'
TIME_END = '2016 '
YEARS_BACK = 1


MAX_OC = 100
window_size = 6

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

