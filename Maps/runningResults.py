import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from datetime import datetime

# Constants for filenames
PICTURE_VERSION = "AllModelsComparison"
MAX_OC = 150
TIME_BEGINNING = 2007
TIME_END = 2023
INFERENCE_TIME = 2015

# Create output directory
output_dir = "/home/vfourel/SOCProject/SOCmapping/AllResultsMappingTogether"
os.makedirs(output_dir, exist_ok=True)

# Define model paths dictionary
model_data = {
    "RandomForest": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/SimpleTimeModel/XGBoost/RandomForestFinalResults2023/coordinates_1mil_rf.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/SimpleTimeModel/XGBoost/RandomForestFinalResults2023/predictions_1mil_rf.npy"
    },
    "XGBoost": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/SimpleTimeModel/XGBoost/XGBoostFinalResults2023/coordinates_1mil_xgboost.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/SimpleTimeModel/XGBoost/XGBoostFinalResults2023/predictions_1mil_xgboost.npy"
    },
    "2DCNN": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/2DCNN/finalResults2023/coordinates_1mil.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/2DCNN/finalResults2023/predictions_1mil.npy"
    },
    "3DCNN": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/3DCNN/finalResults2023/coordinates_1mil.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/3DCNN/finalResults2023/predictions_1mil.npy"
    },
    "CNNLSTM": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/CNNLSTM/finalResults2023/coordinates_1mil.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/CNNLSTM/finalResults2023/predictions_1mil.npy"
    },
    "TemporalFusionTransformer": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023/concatenatedResults/coordinates.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023/concatenatedResults/predictions.npy"
    },
    "2milTransformer": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1_0000_TRANSFORM_none_LOSS_l1_2milParameters_RESULTS_2023/coordinates_1mil_2ndThird.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1_0000_TRANSFORM_none_LOSS_l1_2milParameters_RESULTS_2023/predictions_1mil_2ndThird.npy"
    },
    "20kTransformer": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_none_LOSS_l1_20kParameters_RESULTS_2023/concatenatedResults/coordinates.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_none_LOSS_l1_20kParameters_RESULTS_2023/concatenatedResults/predictions.npy"
    },
    "FoundationModel": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/results_prediction_41by41WindowSize/FoundationalModels/L1Loss_ZeroCenteredTarget/run2015_MAXOC150_v1/coordinates_1mil_2ndQuarter.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/results_prediction_41by41WindowSize/FoundationalModels/L1Loss_ZeroCenteredTarget/run2015_MAXOC150_v1/predictions_1mil_2ndQuarter.npy"
    }
}

def create_model_comparison_maps():
    """Generate interpolated maps for all models with terrain colormap and fixed 0-150 scale"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']

    # Set fixed scale as requested (0-150 SOC g/kg)
    global_min = 0
    global_max = 150
    print(f"Using fixed scale: {global_min} to {global_max} SOC g/kg")

    # Create interpolated map for each model
    for model_name, paths in model_data.items():
        try:
            # Load data and print shapes
            coordinates = np.load(paths["coordinates"])
            predictions = np.load(paths["predictions"])

            print(f"\n{model_name} data shapes:")
            print(f"  Coordinates shape: {coordinates.shape}")
            print(f"  Predictions shape: {predictions.shape}")

            # Specifically handle predictions with shape (n_samples, 1)
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                print(f"  Flattening predictions from shape {predictions.shape} to 1D array")
                predictions = predictions.flatten()
                print(f"  New predictions shape: {predictions.shape}")

            # Clip predictions to ensure they're within our display range
            predictions = np.clip(predictions, global_min, global_max)

            # Create interpolation grid
            grid_x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 300)
            grid_y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 300)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # Interpolate values
            grid_z = griddata(coordinates, predictions, (grid_x, grid_y), method='linear')

            # Create figure with adjusted layout to make room for the colorbar
            fig = plt.figure(figsize=(14, 10), dpi=300)

            # Create subplot for the map with adjusted size to make room for colorbar
            ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]

            # Create the contour plot
            contour = ax.contourf(grid_x, grid_y, grid_z,
                                levels=50,
                                cmap='terrain',  # Using original terrain colormap as requested
                                alpha=0.8,
                                vmin=global_min,
                                vmax=global_max)

            # Plot Bavaria boundaries
            bavaria.boundary.plot(ax=ax, color='black', linewidth=1)

            # Set title and labels
            ax.set_title(f'Soil Organic Carbon: {model_name}', fontsize=14, pad=20)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)

            # Add a more prominent colorbar on the right side with fixed full range
            cax = fig.add_axes([0.8, 0.1, 0.05, 0.8])  # [left, bottom, width, height]
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap='terrain'), 
                               cax=cax)
            cbar.set_label('SOC (g/kg)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

            # Add ticks at regular intervals to ensure full scale visibility
            cbar.set_ticks(np.linspace(global_min, global_max, 11))

            # Save figure
            filename = f"{PICTURE_VERSION}_{model_name}_MAX_OC_{MAX_OC}_Beginning_{TIME_BEGINNING}_End_{TIME_END}_InferenceTime{INFERENCE_TIME}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

            print(f"Created map for {model_name}")

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for better debugging

    # Create a reference colorbar
    fig, ax = plt.subplots(figsize=(8, 1))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap='terrain'), 
                cax=ax,  # Specify the axes to avoid deprecation warning
                orientation='horizontal', 
                label='SOC (g/kg)')
    plt.title('Reference Color Scale for All Models (0-150 g/kg)')
    plt.savefig(os.path.join(output_dir, f"reference_colorbar_{timestamp}.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

# Execute the mapping function
if __name__ == "__main__":
    create_model_comparison_maps()
    print(f"All maps saved to {output_dir}")

