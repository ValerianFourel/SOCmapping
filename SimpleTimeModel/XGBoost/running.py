import argparse
import concurrent.futures
import copy
import multiprocessing
from pathlib import Path
# Import Union and Optional for older Python compatibility
from typing import List, Tuple, Dict, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened


# Assuming these imports point to correctly defined modules and functions
from dataloader.dataloaderMulti import (
    MultiRasterDatasetMultiYears,
    filter_dataframe,
    separate_and_add_data,
)
from dataloader.dataloaderMultiMapping import MultiRasterDataset1MilMultiYears
from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
from balancedDataset import resample_training_df
from mapping import create_prediction_visualizations


# Assuming config.py contains necessary configurations
from config import (
    TIME_BEGINNING,
    TIME_END,
    INFERENCE_TIME,
    MAX_OC,
    MatrixCoordinates_1mil_Yearly,
    MatrixCoordinates_1mil_Seasonally,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC,
)

# --- Constants ---
DEFAULT_BATCH_SIZE_PREDICT = 8
DEFAULT_SAVE_DIR_PLOTS = Path("./plots")
DEFAULT_SAVE_DIR_PREDICTIONS = Path("./predictions")

# Ensure output directories exist
DEFAULT_SAVE_DIR_PLOTS.mkdir(parents=True, exist_ok=True)
DEFAULT_SAVE_DIR_PREDICTIONS.mkdir(parents=True, exist_ok=True)

# --- Type Hinting (using Union for compatibility) ---
ArrayLike = Union[np.ndarray, List[float], pd.Series]

# --- Helper Functions ---

def flatten_and_deduplicate(path_list: List[Any]) -> List[str]:
    """Flattens a list of lists/strings and removes duplicates."""
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_and_deduplicate(item)) # Recurse for nested lists
        elif isinstance(item, str):
            flattened.append(item)
    return list(dict.fromkeys(flattened)) # Deduplicate while preserving order

def get_top_sampling_years(file_path: Union[Path, str], top_n: int = 3) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Reads an Excel file, reports the top N years with the most samples.

    Args:
        file_path: Path to the Excel file.
        top_n: Number of top years to return.

    Returns:
        A tuple containing the DataFrame and a Series of top years' counts,
        or (None, None) if an error occurs.
    """
    try:
        df = pd.read_excel(file_path)
        year_counts = df['year'].value_counts()
        top_years = year_counts.head(top_n)

        print(f"\nTop {top_n} years with the most samples in {file_path}:")
        for year, count in top_years.items():
            print(f"Year {year}: {count} samples")

        return df, top_years

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

def modify_coordinate_paths(
    yearly_paths: List[str],
    seasonally_paths: List[str],
    inference_time: str
) -> Tuple[List[str], List[str]]:
    """
    Modifies coordinate file paths based on inference time.

    Args:
        yearly_paths: List of yearly coordinate paths.
        seasonally_paths: List of seasonally coordinate paths.
        inference_time: The inference time string (e.g., 'YYYYMMDD').

    Returns:
        A tuple containing the modified yearly and seasonally paths.
    """
    modified_yearly = []
    modified_seasonally = []
    inference_year = inference_time[:4]

    # Process seasonally paths
    for path_str in seasonally_paths:
        path = Path(path_str)
        if path.name == 'Elevation':
            modified_seasonally.append(path_str) # Keep original
        elif path.name == 'MODIS_NPP':
            modified_seasonally.append(str(path / inference_year))
        else:
            modified_seasonally.append(str(path / inference_time))

    # Process yearly paths
    for path_str in yearly_paths:
        path = Path(path_str)
        if path.name == 'Elevation':
            modified_yearly.append(path_str) # Keep original
        else:
            modified_yearly.append(str(path / inference_year))

    return modified_yearly, modified_seasonally


# --- Core Logic Functions ---

def load_and_prepare_training_data(
    samples_coords_paths: List[str],
    data_paths: List[str],
    training_df: pd.DataFrame,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads data using DataLoader and prepares it for model training."""
    print(f"Creating training dataset with {len(training_df)} samples...")
    dataset = MultiRasterDatasetMultiYears(samples_coords_paths, data_paths, training_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max(1, multiprocessing.cpu_count() // 2))

    X_train_list, y_train_list, coordinates_list = [], [], []

    print("Loading training data...")
    for longitudes, latitudes, batch_features, batch_targets in tqdm(dataloader):
        # Convert tensors to numpy
        longs_np = longitudes.numpy()
        lats_np = latitudes.numpy()
        features_np = batch_features.numpy() # Shape: [batch, features, h, w]
        targets_np = batch_targets.numpy()

        # Create mask for valid coordinates (not NaN)
        valid_mask = ~(np.isnan(longs_np) | np.isnan(lats_np))

        if not np.any(valid_mask):
            continue # Skip batch if all coordinates are invalid

        # Filter coordinates
        valid_coords = np.column_stack((longs_np[valid_mask], lats_np[valid_mask]))
        coordinates_list.append(valid_coords)

        # Flatten features: reshape from [batch, features, h, w] to [batch, features*h*w]
        flattened_features = features_np.reshape(features_np.shape[0], -1)

        # Filter features and targets using the mask
        X_train_list.extend(flattened_features[valid_mask])
        y_train_list.extend(targets_np[valid_mask])

    # Convert lists to numpy arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    coordinates = np.vstack(coordinates_list)

    print(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    return X_train, y_train, coordinates

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'xgboost'
) -> Any:
    """Trains either a RandomForest or XGBoost model."""
    print(f"Training {model_type} model...")
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
            oob_score=True
        )
        model.fit(X_train, y_train)
        print(f"RandomForest model trained successfully! OOB Score: {model.oob_score_:.4f}")
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("XGBoost model trained successfully!")
    else:
        raise ValueError("Unsupported model type. Choose 'rf' or 'xgboost'.")

    return model

def process_prediction_batch(
    df_chunk: pd.DataFrame,
    model: Any, # Changed from model_copy, assuming original model is passed
    samples_coords_paths: List[str],
    data_paths: List[str],
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a chunk of data for prediction using the provided model.

    Args:
        df_chunk: DataFrame containing the coordinates for this chunk.
        model: The trained machine learning model (e.g., RandomForest, XGBoost).
               Assumed to be thread-safe for prediction.
        samples_coords_paths: List of paths to sample coordinate sources.
        data_paths: List of paths to raster data sources.
        batch_size: The batch size for the DataLoader.

    Returns:
        A tuple containing:
        - np.ndarray: An array of coordinates (longitude, latitude) for the predictions.
        - np.ndarray: An array of the corresponding predictions.
        Returns empty arrays if no features are processed.
    """
    # Create dataset for the current chunk
    chunk_dataset = MultiRasterDataset1MilMultiYears(samples_coords_paths, data_paths, df_chunk)

    # *** FIX: Set num_workers=0 ***
    # This prevents creating extra processes within each thread, reducing memory overhead.
    # Data loading will happen sequentially within the calling thread.
    chunk_dataloader = DataLoader(
        chunk_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle for prediction
        num_workers=0   # Use 0 workers when running inside a ThreadPoolExecutor
    )

    chunk_coordinates_list = []
    chunk_features_list = []

    # Iterate through the dataloader to get features and coordinates
    for longitudes, latitudes, batch_features in chunk_dataloader:
        # Store coordinates
        coords_np = np.column_stack((longitudes.numpy(), latitudes.numpy()))
        chunk_coordinates_list.append(coords_np)

        # Process features
        features_np = batch_features.numpy() # Shape: [batch, features, h, w]
        # Flatten features: reshape from [batch, features, h, w] to [batch, features*h*w]
        flattened_features = features_np.reshape(features_np.shape[0], -1)
        chunk_features_list.extend(flattened_features)

    # Handle cases where a chunk might yield no valid data
    if not chunk_features_list:
        return np.empty((0, 2)), np.empty((0,)) # Return empty arrays

    # Convert lists to numpy arrays
    chunk_features = np.array(chunk_features_list)
    chunk_coordinates = np.vstack(chunk_coordinates_list)

    # Perform prediction using the model
    # Using 'model' directly, assuming it's thread-safe for .predict()
    chunk_predictions = model.predict(chunk_features)

    return chunk_coordinates, chunk_predictions

def parallel_predict(
    df_full: pd.DataFrame,
    model: Any,
    samples_coords_paths: List[str],
    data_paths: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE_PREDICT,
    num_threads: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs parallel prediction on the full dataset."""
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    print(f"Starting parallel prediction with {num_threads} threads...")

    df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

    chunk_size = max(1, len(df_shuffled) // num_threads)
    df_chunks = [df_shuffled[i:min(i + chunk_size, len(df_shuffled))] for i in range(0, len(df_shuffled), chunk_size)]
    actual_num_threads = len(df_chunks)
    if actual_num_threads < num_threads:
        print(f"Warning: Reducing number of threads to {actual_num_threads} due to small data size.")
        num_threads = actual_num_threads


    all_coordinates_list = []
    all_predictions_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        model_copies = [copy.deepcopy(model) for _ in range(num_threads)]

        futures = {
            executor.submit(
                process_prediction_batch,
                chunk,
                model_copies[i % num_threads],
                samples_coords_paths,
                data_paths,
                batch_size
            ): i for i, chunk in enumerate(df_chunks)
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Predicting Chunks"):
            try:
                coordinates, predictions = future.result()
                if coordinates.size > 0:
                    all_coordinates_list.append(coordinates)
                    all_predictions_list.append(predictions)
            except Exception as e:
                chunk_index = futures[future]
                print(f"Error processing chunk {chunk_index}: {e}")

    if not all_predictions_list:
        print("Warning: No predictions were generated.")
        return np.empty((0, 2)), np.empty((0,))

    all_coordinates = np.vstack(all_coordinates_list)
    all_predictions = np.concatenate(all_predictions_list)

    print(f"Parallel prediction finished. Total predictions: {len(all_predictions)}")
    return all_coordinates, all_predictions

def plot_predictions(coordinates: np.ndarray, predictions: np.ndarray, title: str, save_path: Optional[Union[Path, str]] = None):
    """Creates a scatter plot of predictions on geographic coordinates."""
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=predictions,
        cmap='viridis',
        alpha=0.6,
        s=5
    )
    plt.colorbar(scatter, label='Predicted Values')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


# --- Main Execution ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model and predict SOC values.")
    parser.add_argument(
        '--model', type=str, choices=['rf', 'xgboost'], default='rf',
        help='Model type: rf (Random Forest) or xgboost (XGBoost)'
    )
    parser.add_argument(
        '--target-fraction', type=float, default=0.75,
        help='Fraction of max bin count for resampling target variable'
    )
    parser.add_argument(
        '--num-bins', type=int, default=128,
        help='Number of bins for target variable resampling'
    )
    parser.add_argument(
        '--prediction-batch-size', type=int, default=DEFAULT_BATCH_SIZE_PREDICT,
        help='Batch size for parallel prediction'
    )
    parser.add_argument(
        '--num-threads', type=int, default=None,
        help='Number of threads for parallel prediction (default: all cores)'
    )
    parser.add_argument(
        '--coords-1mil-csv', type=str,
        default="/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv",
        help='Path to the 1 million coordinates CSV file for prediction'
    )
    parser.add_argument(
        '--plot-dir', type=str, default=str(DEFAULT_SAVE_DIR_PLOTS),
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--prediction-dir', type=str, default=str(DEFAULT_SAVE_DIR_PREDICTIONS),
        help='Directory to save prediction arrays'
    )

    args = parser.parse_args()
    return args

def main():
    """Main function to execute the workflow."""
    args = parse_arguments()
    plot_dir = Path(args.plot_dir)
    prediction_dir = Path(args.prediction_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print("--- Configuration ---")
    print(f"Model Type: {args.model}")
    print(f"Resampling Bins: {args.num_bins}")
    print(f"Resampling Target Fraction: {args.target_fraction}")
    print(f"Prediction Batch Size: {args.prediction_batch_size}")
    print(f"Prediction Threads: {args.num_threads or 'Default (All Cores)'}")
    print(f"Inference Time: {INFERENCE_TIME}")
    print(f"Training Time Range: {TIME_BEGINNING} - {TIME_END}")
    print(f"Max OC: {MAX_OC}")
    print("-" * 20)

    # --- Data Preparation ---
    print("--- Preparing Data ---")
    df_train_filtered = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    if df_train_filtered is None or df_train_filtered.empty:
        print("Error: Filtering resulted in an empty DataFrame. Exiting.")
        return

    df_train_resampled = resample_training_df(
        df_train_filtered,
        num_bins=args.num_bins,
        target_fraction=args.target_fraction
    )
    print(f"Training data resampled: {len(df_train_resampled)} samples")

    samples_coordinates_paths_nested, data_paths_nested = separate_and_add_data()
    samples_coordinates_paths = flatten_and_deduplicate(samples_coordinates_paths_nested)
    data_paths = flatten_and_deduplicate(data_paths_nested)
    print(f"Found {len(samples_coordinates_paths)} unique sample coordinate sources.")
    print(f"Found {len(data_paths)} unique data sources.")

    X_train, y_train, train_coordinates = load_and_prepare_training_data(
        samples_coordinates_paths,
        data_paths,
        df_train_resampled
    )

    if X_train.size == 0 or y_train.size == 0:
        print("Error: No valid training data loaded. Exiting.")
        return

    # --- Model Training ---
    print("\n--- Training Model ---")
    model = train_model(X_train, y_train, args.model)

    train_predictions = model.predict(X_train)
    plot_predictions(
        train_coordinates,
        train_predictions,
        f'Training Data Predictions ({args.model})',
        plot_dir / f'training_predictions_{args.model}.png'
    )

    # --- Prediction on Full Dataset ---
    print("\n--- Predicting on Full Dataset ---")
    coords_1mil_path = Path(args.coords_1mil_csv)
    try:
        df_full = pd.read_csv(coords_1mil_path)
        print(f"Loaded prediction coordinates: {len(df_full)} points from {coords_1mil_path}")
        print("Prediction Coordinates Head:\n", df_full.head())
    except FileNotFoundError:
        print(f"Error: Prediction coordinates file not found at {coords_1mil_path}. Exiting.")
        return
    except Exception as e:
        print(f"Error reading prediction coordinates file {coords_1mil_path}: {e}. Exiting.")
        return

    # Modify paths if needed (ensure this logic is correct for prediction dataset)
    matrix_yearly_modified, matrix_seasonally_modified = modify_coordinate_paths(
        copy.deepcopy(MatrixCoordinates_1mil_Yearly), # Use copies to avoid modifying originals
        copy.deepcopy(MatrixCoordinates_1mil_Seasonally),
        INFERENCE_TIME
    )
    samples_coords_1mil, data_1mil = separate_and_add_data_1mil_inference()
    samples_coords_1mil = list(dict.fromkeys(flatten_paths(samples_coords_1mil)))
    data_1mil = list(dict.fromkeys(flatten_paths(data_1mil)))
    # IMPORTANT: Check if parallel_predict needs these modified paths or the original ones
    # Assuming original paths are sufficient for the dataloader used in prediction for now:
    pred_coordinates, predictions = parallel_predict(
        df_full=df_full[:300000],
        model=model,
        samples_coords_paths=samples_coords_1mil, # Or prediction-specific paths if needed
        data_paths=data_1mil,                       # Or prediction-specific paths if needed
        batch_size=args.prediction_batch_size,
        num_threads=args.num_threads
    )

    if predictions.size > 0:
        save_path_coords = prediction_dir / f"coordinates_1mil_{args.model}.npy"
        save_path_preds = prediction_dir / f"predictions_1mil_{args.model}.npy"
        np.save(save_path_coords, pred_coordinates)
        np.save(save_path_preds, predictions)
        print(f"Predictions saved to {save_path_coords} and {save_path_preds}")

        plot_predictions(
            pred_coordinates,
            predictions,
            f'Predicted Values on 1 Million Coordinates ({args.model})',
            plot_dir / f'bavaria_predictions_1mil_{args.model}.png'
        )

        print("\n--- Creating Additional Visualizations ---")
        if args.model == 'rf':
            viz_save_path = plot_dir / 'randomForest_plots'
        else:
            viz_save_path = plot_dir / 'xgboost_plots'
        viz_save_path.mkdir(parents=True, exist_ok=True)

        create_prediction_visualizations(
            INFERENCE_TIME,
            pred_coordinates,
            predictions,
            str(viz_save_path)
        )
        print(f"Additional visualizations saved in {viz_save_path}")
    else:
        print("Skipping saving and plotting of predictions due to empty results.")

    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    main()
