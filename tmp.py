import numpy as np

# Load the .npy file
file_path = '/home/vfourel/SOCmapping/Data/Coordinates/YearlyValue/LST/2015/coordinates.npy'
coordinates = np.load(file_path)

# Display the first 5 entries
print(coordinates[:5])
