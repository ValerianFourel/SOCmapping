#!/usr/bin/env python3
import numpy as np
import os

def get_array_info(filename):
    if not os.path.exists(filename):
        return f"File {filename} does not exist"

    try:
        # Load the .npy file
        data = np.load(filename)

        # Get array shape, size and data type
        shape = data.shape
        size = data.size  # Total number of elements
        memory_size = data.nbytes / (1024 * 1024)  # Size in MB
        dtype = data.dtype

        return f"{filename}:\n  Shape: {shape}\n  Elements: {size}\n  Memory: {memory_size:.2f} MB\n  Data type: {dtype}"

    except Exception as e:
        return f"Error loading {filename}: {str(e)}"

if __name__ == "__main__":
    files = ["coordinates_1mil.npy", "predictions_1mil.npy"]

    for file in files:
        print(get_array_info(file))
        print("-" * 50)
