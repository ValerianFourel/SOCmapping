import numpy as np

# Load the numpy files
coordinates = np.load('coordinates_1mil.npy')
predictions = np.load('predictions_1mil.npy')

# Get and print the size (shape) of each array
print("Coordinates array shape:", coordinates.shape)
print("Predictions array shape:", predictions.shape)

# Optionally, print the total number of elements
print("Total elements in coordinates:", coordinates.size)
print("Total elements in predictions:", predictions.size)
