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

# Initialize Accelerator
#accelerator = Accelerator()

# Assuming coordinates and predictions are obtained from parallel_predict
#coordinates, predictions, INFERENCE_TIME = parallel_predict()  # Adjust based on your actual parallel_predict implementation

# Modify predictions: set values below 1.0 to 1.0
# predictions = np.where(predictions < 1.0, 1.0, predictions)

# Save the modified coordinates and predictions
coordinates = np.load("coordinates_1mil.npy")
predictions = np.load("predictions_1mil.npy")
predictions = np.where(predictions < 1.0, 1.0, predictions)
# Only the main process handles printing and visualization
# if accelerator.is_local_main_process:
print(f"Inference completed. Coordinates shape: {coordinates.shape}, Predictions shape: {predictions.shape}")
# save_path_predictions_plots = "prediction_visualizations"  # Adjust path as needed
create_prediction_visualizations(
        INFERENCE_TIME,
        coordinates,
        predictions,
        save_path_predictions_plots
    )
print(f"Visualizations saved to {save_path_predictions_plots}")
