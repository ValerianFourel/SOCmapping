import os
import numpy as np
import re

# Set the path to your folder
folder_path = "/lustre/home/vfourel/SOCProject/SOCmapping/TemporalFusionTransformer/finalResults2023_OC150_2007to2023_transform_normalize_loss_l1_runs_1_lr_0.0002_heads_8_layers_2/runsPredictions"

# Helper function to extract numeric start index
def extract_start_index(filename):
    match = re.search(r'_(\d+)_to_\d+\.npy', filename)
    return int(match.group(1)) if match else float('inf')

# List and sort coordinate files
coordinate_files = sorted(
    [f for f in os.listdir(folder_path) if f.startswith("coordinates_")],
    key=extract_start_index
)

# List and sort prediction files
prediction_files = sorted(
    [f for f in os.listdir(folder_path) if f.startswith("predictions_")],
    key=extract_start_index
)

# Load and concatenate coordinate arrays
coordinates = [np.load(os.path.join(folder_path, f)) for f in coordinate_files]
coordinates_concat = np.concatenate(coordinates, axis=0)

# Load and concatenate prediction arrays
predictions = [np.load(os.path.join(folder_path, f)) for f in prediction_files]
predictions_concat = np.concatenate(predictions, axis=0)

# Save the final arrays
np.save(os.path.join(folder_path, "coordinates_1mil.npy"), coordinates_concat)
np.save(os.path.join(folder_path, "predictions_1mil.npy"), predictions_concat)

print("Saved coordinates_1mil.npy and predictions_1mil.npy successfully.")
