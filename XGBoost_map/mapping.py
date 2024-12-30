import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd

##################################################################

# Loading the Data



base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'



file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"


elevationBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/StaticValue/Elevation'
LAIBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/LAI'
LSTBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/LST'
MODIS_NPPBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/TotalEvapotranspiration'


elevationBandMatrixCoordinates_1milPoints = f'{base_path_data}/Coordinates1Mil/StaticValue/Elevation'
LAIBandMatrixCoordinates_1milPoints = f'{base_path_data}/Coordinates1Mil/YearlyValue/LAI'
LSTBandMatrixCoordinates_1milPoints = f'{base_path_data}/Coordinates1Mil/YearlyValue/LST'
MODIS_NPPBandMatrixCoordinates_1milPoints = f'{base_path_data}/Coordinates1Mil/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates_1milPoints = f'{base_path_data}/Coordinates1Mil/YearlyValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates_1milPoints = f'{base_path_data}/Coordinates1Mil/YearlyValue/TotalEvapotranspiration'



elevationTensorData = f'{base_path_data}/RasterTensorData/StaticValue/Elevation'
LAITensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/LAI'
LSTTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/LST'
MODIS_NPPTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/MODIS_NPP'
SoilEvaporationTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/SoilEvaporation'
TotalEvapotranspirationTensorDataYearly = f'{base_path_data}/RasterTensorData/YearlyValue/TotalEvapotranspiration'


# List of variable names as strings
variable_names = [
    "LAIBandMatrixCoordinates",
    "LSTBandMatrixCoordinates",
    "MODIS_NPPBandMatrixCoordinates",
    "SoilEvaporationBandMatrixCoordinates",
    "TotalEvapotranspirationBandMatrixCoordinates",
]



# Print updated variables to verify
print(f"LAIBandMatrixCoordinates: {LAIBandMatrixCoordinates}")
print(f"LSTBandMatrixCoordinates: {LSTBandMatrixCoordinates}")
print(f"MODIS_NPPBandMatrixCoordinates: {MODIS_NPPBandMatrixCoordinates}")
print(f"SoilEvaporationBandMatrixCoordinates: {SoilEvaporationBandMatrixCoordinates}")
print(f"TotalEvapotranspirationBandMatrixCoordinates: {TotalEvapotranspirationBandMatrixCoordinates}")

BandsYearly = [ LAIBandMatrixCoordinates , LSTBandMatrixCoordinates , MODIS_NPPBandMatrixCoordinates , SoilEvaporationBandMatrixCoordinates , TotalEvapotranspirationBandMatrixCoordinates  ]

BandsYearly_1milPoints = [ LAIBandMatrixCoordinates_1milPoints , LSTBandMatrixCoordinates_1milPoints , MODIS_NPPBandMatrixCoordinates_1milPoints , SoilEvaporationBandMatrixCoordinates_1milPoints , TotalEvapotranspirationBandMatrixCoordinates_1milPoints  ]




##################################################################

# Fitting the XGBoost

def filter_dataframe( year, max_oc=100):
    """
    Filter DataFrame by year and maximum OC level.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    year (int): Year to filter
    max_oc (float): Maximum OC level (default: 100)

    Returns:
    pandas.DataFrame: Filtered DataFrame
    """
    df = pd.read_excel(file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)

    filtered_df = df[
        (df['year'] == year) & 
        (df['OC'] <= max_oc)
    ]
    return filtered_df

# Example usage:
# df_filtered = filter_dataframe(df_original, year=2015, max_oc=100)

def create_prediction_visualizations(year,coordinates, predictions, save_path):
    """
    Create and save three separate map visualizations of predictions in Bavaria plus a triptych,
    with timestamps in filenames.

    Parameters:
    coordinates (numpy.array): Array of coordinates (longitude, latitude)
    predictions (numpy.array): Array of prediction values
    save_path (str): Directory path where the images should be saved
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import geopandas as gpd
    from datetime import datetime

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    individual_path = os.path.join(save_path, 'individual')
    os.makedirs(individual_path, exist_ok=True)

    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']

    # Create interpolation grid with higher resolution
    grid_x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 300)
    grid_y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 300)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate values with cubic interpolation
    grid_z = griddata(coordinates, predictions, (grid_x, grid_y), method='linear')

    # Common plotting parameters
    plot_params = {
        'figsize': (12, 10),
        'dpi': 300
    }

    # Function to set common elements for all plots
    def set_common_elements(ax, title):
        bavaria.boundary.plot(ax=ax, color='black', linewidth=1)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

    # Function to generate filename with timestamp
    def get_filename(base_name):
        return f"{base_name}_{timestamp}.png"

    # 1. Interpolated surface
    fig_interp, ax_interp = plt.subplots(**plot_params)
    contour = ax_interp.contourf(grid_x, grid_y, grid_z,
                                levels=50,
                                cmap='viridis',
                                alpha=0.8)
    set_common_elements(ax_interp, 'Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax_interp, label='Predicted Values')
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_interpolated')), 
                bbox_inches='tight')
    plt.close()

    # 2. Scatter plot
    fig_scatter, ax_scatter = plt.subplots(**plot_params)
    scatter = ax_scatter.scatter(coordinates[:, 0], coordinates[:, 1],
                               c=predictions,
                               cmap='viridis',
                               alpha=0.6,
                               s=50)
    set_common_elements(ax_scatter, 'Scatter Plot of Predicted Values')
    plt.colorbar(scatter, ax=ax_scatter, label='Predicted Values')
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_scatter')), 
                bbox_inches='tight')
    plt.close()

    # 3. Discrete points
    fig_discrete, ax_discrete = plt.subplots(**plot_params)
    discrete = ax_discrete.scatter(coordinates[:, 0], coordinates[:, 1],
                                 c=predictions,
                                 cmap='viridis',
                                 alpha=1.0,
                                 s=20)
    set_common_elements(ax_discrete, 'Discrete Points of Predicted Values')
    plt.colorbar(discrete, ax=ax_discrete, label='Predicted Values')
    plt.savefig(os.path.join(individual_path, get_filename(f'{year}_bavaria_discrete')), 
                bbox_inches='tight')
    plt.close()

    # Create triptych
    fig_triptych = plt.figure(figsize=(30, 10))

    # Interpolated plot
    ax1 = plt.subplot(131)
    contour = ax1.contourf(grid_x, grid_y, grid_z,
                          levels=50,
                          cmap='viridis',
                          alpha=0.8)
    set_common_elements(ax1, 'Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax1, label='Predicted Values')

    # Scatter plot
    ax2 = plt.subplot(132)
    scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=predictions,
                         cmap='viridis',
                         alpha=0.6,
                         s=50)
    set_common_elements(ax2, 'Scatter Plot of Predicted Values')
    plt.colorbar(scatter, ax=ax2, label='Predicted Values')

    # Discrete points
    ax3 = plt.subplot(133)
    discrete = ax3.scatter(coordinates[:, 0], coordinates[:, 1],
                          c=predictions,
                          cmap='viridis',
                          alpha=1.0,
                          s=20)
    set_common_elements(ax3, 'Discrete Points of Predicted Values')
    plt.colorbar(discrete, ax=ax3, label='Predicted Values')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, get_filename(f'{year}_bavaria_triptych')), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Example usage:
# create_prediction_map(bavaria, coordinates, predictions, save_path='output/maps')
