import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from datetime import datetime
from matplotlib import cm
import matplotlib.patches as mpatches

# Constants for filenames
PICTURE_VERSION = "AllModelsComparison"
MAX_OC = 150
TIME_BEGINNING = 2007
TIME_END = 2023
INFERENCE_TIME = 2023

# Create output directory
output_dir = "/home/vfourel/SOCProject/SOCmapping/AllResultsMappingTogether_June20th_2025"
os.makedirs(output_dir, exist_ok=True)

# Load landcover data
landcover_path = "/lustre/home/vfourel/SOCProject/SOCmapping/Maps/landcover_results/landcover_values_merged.npy"
landcover_coordinates_path = "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy"

print("Loading landcover data...")
landcover_values = np.load(landcover_path, allow_pickle=True)
landcover_coordinates = np.load(landcover_coordinates_path)

# Define model paths dictionary
model_data = {
    "Random Forest": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/predictions_1mil_rf.npy"
    },
    "XGBoost": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/XGBoostFinalResults2023/coordinates_1mil_xgboost.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/XGBoostFinalResults2023/predictions_1mil_xgboost.npy"
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
    "Temporal Fusion Transformer (360k Params)": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023_360kParameters/concatenatedResults/coordinates.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023_360kParameters/concatenatedResults/predictions.npy"
    },
    "Temporal Fusion Transformer (1.1mil Params)": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023_OC150_2007to2023_transform_normalize_loss_l1_runs_1_lr_0.0002_heads_8_layers_2/coordinates_1mil.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023_OC150_2007to2023_transform_normalize_loss_l1_runs_1_lr_0.0002_heads_8_layers_2/predictions_1mil.npy"
    },
    "Large Transformer": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1_0000_TRANSFORM_none_LOSS_l1_2milParameters_RESULTS_2023/coordinates_1mil_2ndThird.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1_0000_TRANSFORM_none_LOSS_l1_2milParameters_RESULTS_2023/predictions_1mil_2ndThird.npy"
    },
    "Small Transformer": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_none_LOSS_l1_20kParameters_RESULTS_2023/concatenatedResults/coordinates.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_none_LOSS_l1_20kParameters_RESULTS_2023/concatenatedResults/predictions.npy"
    },
    "Foundation Model": {
        "coordinates": "/home/vfourel/SOCProject/SOCmapping/results_prediction_41by41WindowSize/FoundationalModels/L1Loss_ZeroCenteredTarget/run2015_MAXOC150_v1/coordinates_1mil_2ndQuarter.npy",
        "predictions": "/home/vfourel/SOCProject/SOCmapping/results_prediction_41by41WindowSize/FoundationalModels/L1Loss_ZeroCenteredTarget/run2015_MAXOC150_v1/predictions_1mil_2ndQuarter.npy"
    }
}

# Get the full terrain colormap
terrain = cm.get_cmap('terrain', 256)

# Buffer distance in degrees (approx. 20 km)
buffer_distance = 0.2

def interpolate_landcover(landcover_coords, landcover_vals, grid_x, grid_y):
    """Interpolate landcover values to the same grid as SOC predictions"""
    # Filter out None values
    valid_mask = np.array([v is not None for v in landcover_vals])
    valid_coords = landcover_coords[valid_mask]
    valid_values = np.array([float(v) for v in landcover_vals[valid_mask]])

    # Interpolate using nearest neighbor to preserve discrete classes
    landcover_grid = griddata(valid_coords, valid_values, (grid_x, grid_y), method='nearest')

    return landcover_grid

def create_model_comparison_maps():
    """Generate interpolated maps for all models with landcover overlays"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load Bavaria boundaries
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']

    # Create buffer around Bavaria
    buffered_bavaria = bavaria.geometry.buffer(buffer_distance).unary_union

    # Set fixed scale (0-150 SOC g/kg)
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

            # Handle predictions with shape (n_samples, 1)
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                print(f"  Flattening predictions from shape {predictions.shape} to 1D array")
                predictions = predictions.flatten()
                print(f"  New predictions shape: {predictions.shape}")

            # Clip predictions to the display range
            predictions = np.clip(predictions, global_min, global_max)

            # Create interpolation grid based on coordinates (within Bavaria)
            grid_x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 300)
            grid_y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 300)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # Interpolate SOC values
            grid_z = griddata(coordinates, predictions, (grid_x, grid_y), method='linear')

            # Interpolate landcover values to the same grid
            landcover_grid = interpolate_landcover(landcover_coordinates, landcover_values, grid_x, grid_y)

            # Create figure with adjusted layout for the colorbar
            fig = plt.figure(figsize=(14, 10), dpi=300)

            # Create subplot for the map
            ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]

            # Create the base contour plot with the full terrain colormap
            contour = ax.contourf(grid_x, grid_y, grid_z,
                                  levels=50,
                                  cmap=terrain,
                                  alpha=0.8,
                                  vmin=global_min,
                                  vmax=global_max)

            # Overlay built-up areas (class 50) in black
            built_up_mask = landcover_grid == 50
            if np.any(built_up_mask):
                ax.contourf(grid_x, grid_y, built_up_mask.astype(float), 
                           levels=[0.5, 1.5], colors=['black'], alpha=0.9)

            # Overlay permanent water bodies (class 80) in deep blue
            water_mask = landcover_grid == 80
            if np.any(water_mask):
                ax.contourf(grid_x, grid_y, water_mask.astype(float), 
                           levels=[0.5, 1.5], colors=['#000080'], alpha=0.9)  # Navy blue

            # Plot Bavaria boundaries
            bavaria.boundary.plot(ax=ax, color='black', linewidth=1)

            # Set title and labels with a better font and bold weight
            ax.set_title(f'{model_name} SOC map', fontsize=14, pad=20, weight='bold', fontfamily='sans-serif')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)

            # Add colorbar on the right with the full terrain colormap
            cax = fig.add_axes([0.8, 0.1, 0.05, 0.8])  # [left, bottom, width, height]
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap=terrain),
                                cax=cax)
            cbar.set_label('SOC (g/kg)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

            # Add ticks at regular intervals
            cbar.set_ticks(np.linspace(global_min, global_max, 11))

            # Add legend for landcover overlays
            legend_elements = [
                mpatches.Patch(color='black', label='Built-up areas'),
                mpatches.Patch(color='#000080', label='Permanent water bodies')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

            # Save figure
            filename = f"{PICTURE_VERSION}_{model_name}_MAX_OC_{MAX_OC}_Beginning_{TIME_BEGINNING}_End_{TIME_END}_InferenceTime{INFERENCE_TIME}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()

            print(f"Created map for {model_name}")

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create a reference colorbar with the full terrain colormap
    fig, ax = plt.subplots(figsize=(8, 1))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap=terrain),
                        cax=ax,
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
