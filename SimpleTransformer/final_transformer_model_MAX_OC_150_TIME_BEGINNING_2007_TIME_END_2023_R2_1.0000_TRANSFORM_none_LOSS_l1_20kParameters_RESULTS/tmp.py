import numpy as np
import os

# Set working directory
base_dir = "/lustre/home/vfourel/SOCProject/SOCmapping/SimpleTransformer/final_transformer_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_R2_1.0000_TRANSFORM_none_LOSS_l1_20kParameters_RESULTS"
output_dir = os.path.join(base_dir, "concatenatedResults")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Ordered file lists
coord_files = [
    "coordinates_0_to_150k.npy",
    "coordinates_150k_to_300k.npy"
]

pred_files = [
    "predictions_0_to_150k.npy",
    "predictions_150k_to_300k.npy"
]

# Load and concatenate coordinates
coordinates = np.vstack([np.load(os.path.join(base_dir, f)) for f in coord_files])

# Load, reshape, and concatenate predictions
predictions = np.vstack([np.load(os.path.join(base_dir, f)).reshape(-1, 1) for f in pred_files])

# Save concatenated arrays
np.save(os.path.join(output_dir, "coordinates.npy"), coordinates)
np.save(os.path.join(output_dir, "predictions.npy"), predictions)

print("✅ Stacked files saved to 'concatenatedResults/'")
print(f"coordinates shape: {coordinates.shape}")
print(f"predictions shape: {predictions.shape}")
