

base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'

file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"


time_before  = 5
window_size = 5 #  41
TIME_BEGINNING = '2007'
LOADING_TIME_BEGINNING = str(int(TIME_BEGINNING)-time_before)
TIME_END = '2023'
INFERENCE_TIME = '2023'
LOADING_TIME_BEGINNING_INFERENCE = str(int(INFERENCE_TIME)-time_before)

bands_list_order = [
    # Original 6 — DO NOT REORDER (saved 6-channel checkpoints rely on this order).
    'Elevation', 'LAI', 'LST', 'MODIS_NPP', 'SoilEvaporation', 'TotalEvapotranspiration',
    # Bavaria 2002-2023 expansion (14 appended; statics symlinked-as-yearly).
    'NDVI', 'EVI', 'Precipitation', 'AirTemperature', 'SoilMoisture_layer1', 'SnowDepth',
    'ClayContent_0_10cm', 'SandContent_0_10cm', 'pH_H2O_0_10cm',
    'BulkDensity_0_10cm',
    'CEC_0_10cm',
    'Slope', 'Aspect', 'TWI',
]
MAX_OC = 150
num_epochs = 200
save_path_predictions_plots = '/home/vfourel/SOCProject/SOCmapping/predictions_plots/cnnlstm_plots'
file_path_coordinates_Bavaria_1mil = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"


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

# ---------------------------------------------------------------------------
# Bavaria 2002-2023 expansion — 14 bands appended to bands_list_order.
# All under YearlyValue/<band>/<year>/ on disk (statics symlinked).
# ---------------------------------------------------------------------------
NDVIBandMatrixCoordinates_Yearly                = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/NDVI'
EVIBandMatrixCoordinates_Yearly                 = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/EVI'
PrecipitationBandMatrixCoordinates_Yearly       = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/Precipitation'
AirTemperatureBandMatrixCoordinates_Yearly      = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/AirTemperature'
SoilMoisture_layer1BandMatrixCoordinates_Yearly = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/SoilMoisture_layer1'
SnowDepthBandMatrixCoordinates_Yearly           = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/SnowDepth'
ClayContent_0_10cmBandMatrixCoordinates_Yearly  = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/ClayContent_0_10cm'
SandContent_0_10cmBandMatrixCoordinates_Yearly  = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/SandContent_0_10cm'
pH_H2O_0_10cmBandMatrixCoordinates_Yearly       = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/pH_H2O_0_10cm'
BulkDensity_0_10cmBandMatrixCoordinates_Yearly  = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/BulkDensity_0_10cm'
CEC_0_10cmBandMatrixCoordinates_Yearly          = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/CEC_0_10cm'
SlopeBandMatrixCoordinates_Yearly               = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/Slope'
AspectBandMatrixCoordinates_Yearly              = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/Aspect'
TWIBandMatrixCoordinates_Yearly                 = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/YearlyValue/TWI'

NDVITensorDataYearly                = f'{base_path_data}/RasterTensorData/YearlyValue/NDVI'
EVITensorDataYearly                 = f'{base_path_data}/RasterTensorData/YearlyValue/EVI'
PrecipitationTensorDataYearly       = f'{base_path_data}/RasterTensorData/YearlyValue/Precipitation'
AirTemperatureTensorDataYearly      = f'{base_path_data}/RasterTensorData/YearlyValue/AirTemperature'
SoilMoisture_layer1TensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/SoilMoisture_layer1'
SnowDepthTensorDataYearly           = f'{base_path_data}/RasterTensorData/YearlyValue/SnowDepth'
ClayContent_0_10cmTensorDataYearly  = f'{base_path_data}/RasterTensorData/YearlyValue/ClayContent_0_10cm'
SandContent_0_10cmTensorDataYearly  = f'{base_path_data}/RasterTensorData/YearlyValue/SandContent_0_10cm'
pH_H2O_0_10cmTensorDataYearly       = f'{base_path_data}/RasterTensorData/YearlyValue/pH_H2O_0_10cm'
BulkDensity_0_10cmTensorDataYearly  = f'{base_path_data}/RasterTensorData/YearlyValue/BulkDensity_0_10cm'
CEC_0_10cmTensorDataYearly          = f'{base_path_data}/RasterTensorData/YearlyValue/CEC_0_10cm'
SlopeTensorDataYearly               = f'{base_path_data}/RasterTensorData/YearlyValue/Slope'
AspectTensorDataYearly              = f'{base_path_data}/RasterTensorData/YearlyValue/Aspect'
TWITensorDataYearly                 = f'{base_path_data}/RasterTensorData/YearlyValue/TWI'

SamplesCoordinates_Yearly = SamplesCoordinates_Yearly + [
    NDVIBandMatrixCoordinates_Yearly, EVIBandMatrixCoordinates_Yearly,
    PrecipitationBandMatrixCoordinates_Yearly, AirTemperatureBandMatrixCoordinates_Yearly,
    SoilMoisture_layer1BandMatrixCoordinates_Yearly, SnowDepthBandMatrixCoordinates_Yearly,
    ClayContent_0_10cmBandMatrixCoordinates_Yearly, SandContent_0_10cmBandMatrixCoordinates_Yearly,
    pH_H2O_0_10cmBandMatrixCoordinates_Yearly, BulkDensity_0_10cmBandMatrixCoordinates_Yearly,
    CEC_0_10cmBandMatrixCoordinates_Yearly,
    SlopeBandMatrixCoordinates_Yearly, AspectBandMatrixCoordinates_Yearly,
    TWIBandMatrixCoordinates_Yearly,
]
DataYearly = DataYearly + [
    NDVITensorDataYearly, EVITensorDataYearly,
    PrecipitationTensorDataYearly, AirTemperatureTensorDataYearly,
    SoilMoisture_layer1TensorDataYearly, SnowDepthTensorDataYearly,
    ClayContent_0_10cmTensorDataYearly, SandContent_0_10cmTensorDataYearly,
    pH_H2O_0_10cmTensorDataYearly, BulkDensity_0_10cmTensorDataYearly,
    CEC_0_10cmTensorDataYearly,
    SlopeTensorDataYearly, AspectTensorDataYearly, TWITensorDataYearly,
]

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

