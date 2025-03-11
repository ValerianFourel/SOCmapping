import numpy as np
import xgboost as xgb
from config import bands_list_order, window_size  # For feature size estimation


def count_xgb_nodes(model):
    """Count the total number of nodes across all trees in the XGBRegressor."""
    # Get booster dump as a list of tree strings
    tree_dump = model.get_booster().get_dump()
    total_nodes = 0
    for tree in tree_dump:
        # Each line in the dump represents a node; count lines excluding empty ones
        nodes_in_tree = len([line for line in tree.split('\n') if line.strip()])
        total_nodes += nodes_in_tree
    return total_nodes


def main():
    # Model configuration matching your training script
    model = xgb.XGBRegressor(
        objective="reg:squarederror",  # Regression with squared error
        n_estimators=1000,             # Number of trees
        max_depth=10,                  # Maximum depth of each tree
        learning_rate=0.1,             # Learning rate
        random_state=42                # Implicitly set via Booster's seed
    )

    # Simulate a small fit to estimate node count (requires training data)
    n_samples = 100  # Small sample size for demonstration
    n_features = len(bands_list_order) * window_size * window_size  # Flattened feature size
    X_dummy = np.random.rand(n_samples, n_features)
    y_dummy = np.random.rand(n_samples)
    model.fit(X_dummy, y_dummy)

    # Count total nodes across all trees
    total_nodes = count_xgb_nodes(model)
    
    # Hyperparameters from your training script
    hyperparameters = {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "objective": model.objective,
        "min_child_weight": model.min_child_weight,  # Default value
        "subsample": model.subsample,                # Default value
        "colsample_bytree": model.colsample_bytree,  # Default value
    }

    # Print hyperparameter summary
    print("XGBRegressor Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    
    # Print feature size (input dimensionality)
    print(f"\nInput Features (Flattened): {n_features} (bands={len(bands_list_order)}, window_size={window_size})")
    
    # Print total number of nodes
    print(f"\nTotal Number of Nodes Across All Trees: {total_nodes:,}")
    
    # Note on "parameters"
    print("\nNote: XGBoost doesn't have 'trainable parameters' like neural networks. "
          "Complexity is reflected in the number of trees, nodes, and feature size.")


if __name__ == "__main__":
    main()
