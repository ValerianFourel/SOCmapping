import numpy as np

def check_shapes():
    try:
        # Load the saved numpy arrays
        predictions = np.load("predictions_1mil.npy")
        coordinates = np.load("coordinates_1mil.npy")
        
        # Print their shapes
        print("Predictions shape:", predictions.shape)
        print("Coordinates shape:", coordinates.shape)
        
        # Print total number of elements for comparison
        print("Number of predictions:", predictions.size)
        print("Number of coordinate pairs:", coordinates.shape[0])
        
    except FileNotFoundError as e:
        print(f"Error: One or both files not found - {e}")
    except Exception as e:
        print(f"Error loading files: {e}")

if __name__ == "__main__":
    check_shapes()
