import pandas as pd

# Path to the Parquet file
file_path = "/fast/vfourel/SOCProject/run_20250209_163939/batch_0001_size_256.parquet"

# Load the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Display basic information about the DataFrame
print("DataFrame Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# If you want to see all columns and a sample row fully
pd.set_option('display.max_columns', None)
print("\nSample row (first row):")
print(df.iloc[0])
