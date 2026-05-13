import numpy as np
from mapping import create_prediction_visualizations, parallel_predict
from accelerate import Accelerator
from config import (
    bands_list_order, time_before, LOADING_TIME_BEGINNING, window_size,
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded,
    save_path_predictions_plots, SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
    DataYearly, file_path_coordinates_Bavaria_1mil, SamplesCoordinates_Seasonally,
    MatrixCoordinates_1mil_Seasonally, DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC
)

# Load coordinates and predictions
coordinates = np.load("coordinates_1mil.npy")
predictions = np.load("predictions_1mil.npy")

# Modify predictions: set values below 0 to 0
predictions = np.where(predictions < 0, 0, predictions)

# Filter out predictions above 150 and corresponding coordinates
mask = predictions <= 150
predictions = predictions[mask]
coordinates = coordinates[mask]

# Print shapes
print(f"Inference completed. Coordinates shape: {coordinates.shape}, Predictions shape: {predictions.shape}")

# Create and save visualizations
create_prediction_visualizations(
    INFERENCE_TIME,
    coordinates,
    predictions,
    save_path_predictions_plots
)
print(f"Visualizations saved to {save_path_predictions_plots}")
