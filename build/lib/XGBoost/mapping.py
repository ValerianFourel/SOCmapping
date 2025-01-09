import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd

##################################################################

# Loading the Data



base_path_data = '/home/vfourel/SOCProject/SOCmapping/Data'



file_path = f"{base_path_data}/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx"


elevationBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/StaticValue/Elevation'
LAIBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/LAI'
LSTBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/LST'
MODIS_NPPBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/MODIS_NPP'
SoilEvaporationBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/SoilEvaporation'
TotalEvapotranspirationBandMatrixCoordinates = f'{base_path_data}/OC_LUCAS_LFU_LfL_Coordinates/YearlyValue/TotalEvapotranspiration'

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



# Loop to update variables dynamically
for var in variable_names:
    exec(f"{var} = {var} + '/{year}'")

# Print updated variables to verify
print(f"LAIBandMatrixCoordinates: {LAIBandMatrixCoordinates}")
print(f"LSTBandMatrixCoordinates: {LSTBandMatrixCoordinates}")
print(f"MODIS_NPPBandMatrixCoordinates: {MODIS_NPPBandMatrixCoordinates}")
print(f"SoilEvaporationBandMatrixCoordinates: {SoilEvaporationBandMatrixCoordinates}")
print(f"TotalEvapotranspirationBandMatrixCoordinates: {TotalEvapotranspirationBandMatrixCoordinates}")

BandsYearly = [ LAIBandMatrixCoordinates , LSTBandMatrixCoordinates , MODIS_NPPBandMatrixCoordinates , SoilEvaporationBandMatrixCoordinates , TotalEvapotranspirationBandMatrixCoordinates  ]




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
    df = pd.read_excel(file_path)

    filtered_df = df[
        (df['year'] == year) & 
        (df['OC'] <= max_oc)
    ]
    return filtered_df

# Example usage:
# df_filtered = filter_dataframe(df_original, year=2015, max_oc=100)

def create_prediction_map(coordinates, predictions, save_path, filename='bavaria_predictions.png'):
    """
    Create and save a map visualization of predictions in Bavaria with both discrete and interpolated views.

    Parameters:
    bavaria (GeoDataFrame): GeoDataFrame containing Bavaria's boundary
    coordinates (numpy.array): Array of coordinates (longitude, latitude)
    predictions (numpy.array): Array of prediction values
    save_path (str): Directory path where the image should be saved
    filename (str): Name of the output file (default: 'bavaria_predictions.png')
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']


    # Create interpolation grid
    grid_x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 200)
    grid_y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 200)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate values
    grid_z = griddata(coordinates, predictions, (grid_x, grid_y), method='cubic')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Discrete points
    bavaria.boundary.plot(ax=ax1, color='black', linewidth=1)
    scatter = ax1.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=predictions,
                         cmap='viridis',
                         alpha=0.6)
    ax1.set_title('Discrete Predicted Values')
    plt.colorbar(scatter, ax=ax1, label='Predicted Values')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True)

    # Plot 2: Interpolated surface
    bavaria.boundary.plot(ax=ax2, color='black', linewidth=1)
    contour = ax2.contourf(grid_x, grid_y, grid_z,
                          levels=50,
                          cmap='viridis',
                          alpha=0.8)
    ax2.scatter(coordinates[:, 0], coordinates[:, 1],
               c='red',
               s=1,
               alpha=0.1)
    ax2.set_title('Interpolated Predicted Values')
    plt.colorbar(contour, ax=ax2, label='Predicted Values')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Create full path
    full_path = os.path.join(save_path, filename)

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save figure
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Example usage:
# create_prediction_map(bavaria, coordinates, predictions, save_path='output/maps')
