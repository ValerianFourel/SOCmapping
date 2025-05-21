import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataloader.dataloaderMulti import MultiRasterDatasetMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, seasons,
    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly,
    SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, years_padded
)
from mapping import create_prediction_visualizations, parallel_predict
from torch.utils.data import DataLoader
import multiprocessing
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=False):
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    training_dfs = []
    
    if use_validation:
        validation_indices = []
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) >= 4:
                val_samples = bin_data.sample(n=min(8, len(bin_data)))
                validation_indices.extend(val_samples.index)
                train_samples = bin_data.drop(val_samples.index)
                if len(train_samples) > 0:
                    if len(train_samples) < min_samples:
                        resampled = train_samples.sample(n=min_samples, replace=True)
                        training_dfs.append(resampled)
                    else:
                        training_dfs.append(train_samples)
        
        if not training_dfs or not validation_indices:
            raise ValueError("No training or validation data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        validation_df = df.loc[validation_indices].drop('bin', axis=1)
        return training_df, validation_df
    
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        
        if not training_dfs:
            raise ValueError("No training data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        return training_df

def parse_arguments():
    """
    Parse command-line arguments for model selection and training parameters.
    """
    parser = argparse.ArgumentParser(description='Train a model and generate predictions for mapping.')
    parser.add_argument('--model', type=str, choices=['rf', 'xgboost'], 
                        default='xgboost',
                        help='Model type: rf (Random Forest) or xgboost (XGBoost)')



    return parser.parse_args()

def calculate_metrics(y_true, y_pred, dataset_name, model_name):
    """
    Calculate R² (correlation squared), MAE, RMSE, and RPIQ metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        print(f"\n{dataset_name} Metrics ({model_name}): No valid data after filtering NaN/inf")
        return {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
    
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = corr ** 2 if np.isfinite(corr) else 0.0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    q1, q3 = np.percentile(y_true, [25, 75])
    iqr = q3 - q1
    rpiq = iqr / rmse if rmse != 0 else float('inf')
    
    print(f"\n{dataset_name} Metrics ({model_name}):")
    print(f"R² (correlation squared): {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RPIQ: {rpiq:.4f}")
    
    return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'RPIQ': rpiq}

def modify_matrix_coordinates(
    MatrixCoordinates_1mil_Yearly=MatrixCoordinates_1mil_Yearly,
    MatrixCoordinates_1mil_Seasonally=MatrixCoordinates_1mil_Seasonally,
    INFERENCE_TIME=INFERENCE_TIME
):
    """
    Modify paths in MatrixCoordinates based on INFERENCE_TIME.
    """
    for path in MatrixCoordinates_1mil_Seasonally:
        folders = path.split('/')
        last_folder = folders[-1]
        if last_folder == 'Elevation':
            continue
        elif last_folder == 'MODIS_NPP':
            new_path = f"{path}/{INFERENCE_TIME[:4]}"
        else:
            new_path = f"{path}/{INFERENCE_TIME}"
        folders.append(INFERENCE_TIME if last_folder != 'MODIS_NPP' else INFERENCE_TIME[:4])
        new_path = '/'.join(folders)
        MatrixCoordinates_1mil_Seasonally[MatrixCoordinates_1mil_Seasonally.index(path)] = new_path

    for path in MatrixCoordinates_1mil_Yearly:
        folders = path.split('/')
        last_folder = folders[-1]
        if last_folder == 'Elevation':
            continue
        else:
            folders.append(INFERENCE_TIME[:4])
            new_path = '/'.join(folders)
            MatrixCoordinates_1mil_Yearly[MatrixCoordinates_1mil_Yearly.index(path)] = new_path

    return MatrixCoordinates_1mil_Yearly, MatrixCoordinates_1mil_Seasonally

def flatten_paths(path_list):
    """
    Flatten a nested list of paths.
    """
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def main():
    args = parse_arguments()
    
    # Load and filter initial dataframe
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    
    # Prepare data paths
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))
    
    
    # Resample training dataframe
    training_df = create_balanced_dataset(df)
    
    # Create normalized datasets
    train_dataset = MultiRasterDatasetMultiYears(
        samples_coordinates_array_path, data_array_path, training_df
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Prepare training and validation data
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    for longitudes, latitudes, batch_features, batch_targets in train_dataloader:
        longs = longitudes.numpy()
        lats = latitudes.numpy()
        valid_mask = ~(np.isnan(longs) | np.isnan(lats))
        if not np.any(valid_mask):
            continue
        features_np = batch_features.numpy()
        flattened_features = features_np.reshape(features_np.shape[0], -1)
        filtered_features = flattened_features[valid_mask]
        filtered_targets = batch_targets.numpy()[valid_mask]
        X_train.extend(filtered_features)
        y_train.extend(filtered_targets)
    

    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    
    # Check for NaN or inf values
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("Warning: NaN or inf values found in X_train")
    if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
        print("Warning: NaN or inf values found in X_val")
    
    # Train the model
    if args.model == 'rf':
        model = RandomForestRegressor(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42)
    else:
        model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=1000, max_depth=10, 
            learning_rate=0.1, random_state=42
        )
    model.fit(X_train, y_train)
    print(f"{args.model} model trained successfully!")
    
    # Evaluate the model
    y_pred_train = model.predict(X_train)
    calculate_metrics(y_train, y_pred_train, "Training", args.model)
    
    # Load coordinates for 1 million points prediction
    file_path = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"
    try:
        df_full = pd.read_csv(file_path)
        print(df_full.head())
    except Exception as e:
        print(f"Error loading CSV: {e}")
    
    # Modify paths for prediction
    BandsYearly_1milPoints, _ = modify_matrix_coordinates()
    num_cpus = multiprocessing.cpu_count()
    
    # Parallel prediction on 1 million points
    coordinates, predictions = parallel_predict(
        df_full=df_full,
        model=model,
        batch_size=8,
        num_threads=num_cpus
    )
    
    # Save predictions
    save_path_coords = f"coordinates_1mil_{args.model}.npy"
    save_path_preds = f"predictions_1mil_{args.model}.npy"
    np.save(save_path_coords, coordinates)
    np.save(save_path_preds, predictions)
    
    # Visualize predictions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=predictions, cmap='viridis', alpha=0.6
    )
    plt.colorbar(scatter, label='Predicted Values')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Predicted Values on Geographic Coordinates (1M Points)')
    plt.grid(True)
    plt.show()
    
    # Save visualization
    save_path = (
        '/home/vfourel/SOCProject/SOCmapping/predictions_plots/randomForest_plots'
        if args.model == 'rf'
        else '/home/vfourel/SOCProject/SOCmapping/predictions_plots/xgboost_plots'
    )
    create_prediction_visualizations(INFERENCE_TIME, coordinates, predictions, save_path)

if __name__ == "__main__":
    main()