import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from socmapping.dataloader.dataloader import MultiRasterDataset 
from socmapping.xgboost.mapping import filter_dataframe , BandsYearly , create_prediction_map

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
##################################################################

YEAR = 2015
# Drawing the mapping


def get_top_sampling_years(file_path, top_n=3):
    """
    Read the Excel file and return the top n years with the most samples

    Parameters:
    file_path: str, path to the Excel file
    top_n: int, number of top years to return (default=3)
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Count samples per year and sort in descending order
        year_counts = df['year'].value_counts()
        top_years = year_counts.head(top_n)

        print(f"\nTop {top_n} years with the most samples:")
        for year, count in top_years.items():
            print(f"Year {year}: {count} samples")

        return df, top_years

    except Exception as e:
        print(f"Error reading file: {str(e)}")

# Use the function
df_original, top_years = get_top_sampling_years(file_path)

# Read the Excel file
df = pd.read_excel(file_path)

        # Get and display column names
columns = df.columns.tolist()

print("\nColumns in the dataset:")
for i, col in enumerate(columns, 1):
  print(f"{i}. {col}")

        # Print total number of columns
print(f"\nTotal number of columns: {len(columns)}")

df = filter_dataframe(YEAR)

# Create dataset and dataloader
dataset = MultiRasterDataset(BandsYearly, df)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#############################


# Prepare data for XGBoost
X_train, y_train = [], []
coordinates = []  # To store longitude and latitude

for longitudes, latitudes, batch_features, batch_targets in dataloader:
    # Store coordinates for plotting
    coordinates.append(np.column_stack((longitudes.numpy(), latitudes.numpy())))

    # Concatenate all values in the batch_features dictionary
    concatenated_features = np.concatenate([value.numpy() for value in batch_features.values()], axis=1)
    # Flatten the features
    flattened_features = concatenated_features.reshape(concatenated_features.shape[0], -1)
    X_train.extend(flattened_features)
    y_train.extend(batch_targets.numpy())

X_train = np.array(X_train)
y_train = np.array(y_train)
coordinates = np.vstack(coordinates)  # Stack all coordinates

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
print("XGBoost model trained successfully!")

# Make predictions
predictions = xgb_model.predict(X_train)

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1],
                     c=predictions,
                     cmap='viridis',  # You can change the colormap
                     alpha=0.6)
plt.colorbar(scatter, label='Predicted Values')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Values on Geographic Coordinates')
plt.grid(True)
plt.show()

create_prediction_map(coordinates, predictions, save_path, filename='bavaria_predictions.png')