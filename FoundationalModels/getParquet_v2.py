import pandas as pd
import numpy as np

# Path to the Parquet file
file_path = "/fast/vfourel/SOCProject/run_20250209_163939/batch_0001_size_256.parquet"

# Load the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Function to get embedding size
def get_embedding_size(embedding):
    """Returns the size of an embedding, handling nested lists."""
    if embedding is None:
        return 0
    # Convert to numpy array and flatten to count total elements
    arr = np.array(embedding)
    return arr.size

# Inspect elevation_embedding size (should be consistent across rows)
elevation_size = get_embedding_size(df['elevation_embedding'].iloc[0])
print(f"Elevation Embedding Size: {elevation_size}")

# Inspect channel_time_embeddings sizes
# We'll check the first row and a few keys to confirm consistency
cte = df['channel_time_embeddings'].iloc[0]
print("\nChannel Time Embeddings Sizes:")
for key, embedding in cte.items():
    size = get_embedding_size(embedding)
    print(f"  {key}: {size} elements")

# Count total number of channel-time embeddings per row (non-None entries)
cte_counts = df['channel_time_embeddings'].apply(lambda x: sum(1 for v in x.values() if v is not None))
print(f"\nNumber of non-None channel_time_embeddings per row (min/max/mean):")
print(f"  Min: {cte_counts.min()}")
print(f"  Max: {cte_counts.max()}")
print(f"  Mean: {cte_counts.mean():.2f}")
