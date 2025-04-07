import pandas as pd
import numpy as np

# Load the Parquet file
df = pd.read_parquet('EmbeddingsVAEs2007to2023OCsamples_smaller.parquet')
print(f"DataFrame loaded. Number of rows: {len(df)}")

# Define latent_z columns (based on your provided list)
latent_z_columns = [
    'elevation_latent_z',
    'LAI_latent_z_1', 'LAI_latent_z_2', 'LAI_latent_z_3', 'LAI_latent_z_4', 'LAI_latent_z_5',
    'LST_latent_z_1', 'LST_latent_z_2', 'LST_latent_z_3', 'LST_latent_z_4', 'LST_latent_z_5',
    'MODIS_NPP_latent_z_1', 'MODIS_NPP_latent_z_2', 'MODIS_NPP_latent_z_3', 'MODIS_NPP_latent_z_4', 'MODIS_NPP_latent_z_5',
    'SoilEvaporation_latent_z_1', 'SoilEvaporation_latent_z_2', 'SoilEvaporation_latent_z_3', 
    'SoilEvaporation_latent_z_4', 'SoilEvaporation_latent_z_5',
    'TotalEvapotranspiration_latent_z_1', 'TotalEvapotranspiration_latent_z_2', 
    'TotalEvapotranspiration_latent_z_3', 'TotalEvapotranspiration_latent_z_4', 'TotalEvapotranspiration_latent_z_5'
]

# --- Analyze latent_z columns ---
print("\n=== Analysis of latent_z Columns ===")
for col in latent_z_columns:
    print(f"\nAnalyzing {col}:")

    # Verify all entries are NumPy arrays
    is_array = df[col].apply(lambda x: isinstance(x, np.ndarray))
    if not is_array.all():
        non_array_count = (~is_array).sum()
        print(f"  Warning: {col} has non-array values in {non_array_count} rows")
        non_array_types = df[col][~is_array].apply(type).value_counts()
        print(f"  Types of non-array values:\n{non_array_types}")
        continue  # Skip further analysis if not all are arrays

    print("  All entries are NumPy arrays")

    # Check array data types (dtypes)
    dtypes = df[col].apply(lambda x: x.dtype)
    unique_dtypes = dtypes.unique()
    if len(unique_dtypes) > 1:
        print(f"  Warning: Inconsistent dtypes: {unique_dtypes}")
    else:
        print(f"  Array dtype: {unique_dtypes[0]}")

    # Check array shapes
    shapes = df[col].apply(lambda x: x.shape)
    unique_shapes = shapes.unique()
    if len(unique_shapes) > 1:
        print(f"  Warning: Inconsistent shapes: {unique_shapes}")
    else:
        print(f"  Array shape: {unique_shapes[0]}")

    # Check for missing arrays (None or NaN)
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        print(f"  Rows with missing arrays: {missing_count}")

    # Check for NaN within arrays
    has_nan = df[col].apply(lambda x: np.isnan(x).any() if x is not None else False)
    rows_with_nan = has_nan.sum()
    if rows_with_nan > 0:
        print(f"  Rows with NaN values in array: {rows_with_nan}")
        total_nan = df[col].apply(lambda x: np.isnan(x).sum() if x is not None else 0).sum()
        print(f"  Total NaN values within arrays: {total_nan}")

    # Compute summary statistics of array values
    all_values = np.concatenate(df[col].dropna().values)
    if all_values.size > 0:
        non_nan_values = all_values[~np.isnan(all_values)]
        if non_nan_values.size > 0:
            stats = {
                'min': np.min(non_nan_values),
                'max': np.max(non_nan_values),
                'mean': np.mean(non_nan_values),
                'std': np.std(non_nan_values)
            }
            print(f"  Value stats - min: {stats['min']:.2f}, max: {stats['max']:.2f}, "
                  f"mean: {stats['mean']:.2f}, std: {stats['std']:.2f}")
        else:
            print("  All values are NaN")
    else:
        print("  No arrays present")

# --- Metadata analysis ---
print("\n=== Metadata Columns Analysis ===")
metadata_cols = ['longitude', 'latitude', 'year_of_sample', 'oc']
for col in metadata_cols:
    if col in df.columns:
        print(f"\nAnalyzing {col}:")
        nan_count = df[col].isna().sum()
        print(f"  Missing values: {nan_count}")
        if nan_count < len(df):
            print(f"  Range: min = {df[col].min()}, max = {df[col].max()}")

print("\nAnalysis complete. Check the output for details.")
