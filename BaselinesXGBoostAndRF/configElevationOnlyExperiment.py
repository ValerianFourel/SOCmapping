



base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'

file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"


time_before  = 1 # 5
window_size = 21 # 41
TIME_BEGINNING = '2007'
LOADING_TIME_BEGINNING = str(int(TIME_BEGINNING)-time_before)
TIME_END = '2023'
INFERENCE_TIME = '2023'
bands_list_order = ['Elevation']
MAX_OC = 150
num_epochs = 200
NUM_EPOCHS_RUN = 250
LOADING_TIME_BEGINNING_INFERENCE = str(int(INFERENCE_TIME)-time_before)

save_path_predictions_plots = '/home/vfourel/SOCProject/SOCmapping/predictions_plots/trees_plots'
file_path_coordinates_Bavaria_1mil = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"
PICTURE_VERSION = f"{str(num_epochs)}_{str(MAX_OC)}_{INFERENCE_TIME}_version"

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

SamplesCoordinates_Yearly = [elevationBandMatrixCoordinates_Yearly]

elevationBandMatrixCoordinates_1milPoints_Yearly = f'{base_path_data}/Coordinates1Mil/StaticValue/Elevation'


MatrixCoordinates_1mil_Yearly = [elevationBandMatrixCoordinates_1milPoints_Yearly]

elevationTensorData = f'{base_path_data}/RasterTensorData/StaticValue/Elevation'


DataYearly = [elevationTensorData]

#######################################################################
#SEASON PATHS

elevationBandMatrixCoordinates_Seasonally = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation'


SamplesCoordinates_Seasonally = [elevationBandMatrixCoordinates_Seasonally ]

elevationBandMatrixCoordinates_1milPoints_Seasonally = f'{base_path_data}/Coordinates1Mil/StaticValue/Elevation'


MatrixCoordinates_1mil_Seasonally = [elevationBandMatrixCoordinates_1milPoints_Seasonally ]

elevationTensorData = f'{base_path_data}/RasterTensorData/StaticValue/Elevation'


DataSeasonally = [elevationTensorData ]

#######################################################################


