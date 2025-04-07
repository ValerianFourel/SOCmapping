import pandas as pd

# Load your Parquet file
df = pd.read_parquet('EmbeddingsVAEs2007to2023OCsamples_smaller.parquet')

# Pick one latent_z column to inspect (e.g., elevation_latent_z)
col = 'elevation_latent_z'

# Print the type of the first few entries
for i in range(min(5, len(df))):  # Check first 5 rows or fewer if dataset is smaller
    value = df[col].iloc[i]
    print(f"Row {i}, type: {type(value)}, value: {value}")
