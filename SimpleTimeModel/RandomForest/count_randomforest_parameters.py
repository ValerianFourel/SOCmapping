from sklearn.ensemble import RandomForestRegressor
from config import bands_list_order, window_size  # For feature size estimation
import numpy as np

def count_rf_nodes(model):
    """Count the total number of nodes across all trees in the RandomForestRegressor."""
    total_nodes = 0
    for estimator in model.estimators_:
        total_nodes += estimator.tree_.node_count
    return total_nodes


def main():
    # Model configuration matching your training script
    model = RandomForestRegressor(
        n_estimators=1000,   # Number of trees
        max_depth=10,        # Maximum depth of each tree
        n_jobs=-1,           # Use all available cores
        random_state=42      # For reproducibility
    )

    # Since RF requires training data to build trees, we'll simulate a small fit
    # to get a realistic node count. In practice, this depends on the data.
    # Here, we use a placeholder dataset based on your config.
    n_samples = 100  # Small sample size for demonstration
    n_features = len(bands_list_order) * window_size * window_size  # Flattened feature size
    X_dummy = np.random.rand(n_samples, n_features)
    y_dummy = np.random.rand(n_samples)
    model.fit(X_dummy, y_dummy)

    # Count total nodes across all trees
    total_nodes = count_rf_nodes(model)
    
    # Hyperparameters from your training script
    hyperparameters = {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "random_state": model.random_state
    }

    # Print hyperparameter summary
    print("RandomForestRegressor Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    
    # Print feature size (input dimensionality)
    print(f"\nInput Features (Flattened): {n_features} (bands={len(bands_list_order)}, window_size={window_size})")
    
    # Print total number of nodes
    print(f"\nTotal Number of Nodes Across All Trees: {total_nodes:,}")
    
    # Note on "parameters"
    print("\nNote: RandomForest doesn't have 'trainable parameters' like neural networks. "
          "Complexity is reflected in the number of trees, nodes, and feature size.")


if __name__ == "__main__":
    main()
