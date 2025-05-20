import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataloader.dataloaderMulti import MultiRasterDatasetMultiYears, filter_dataframe, separate_and_add_data
#from dataloader.dataframe_loader import 

from torch.utils.data import DataLoader
import argparse
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, seasons,
    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly,
    SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, years_padded
)

from balancedDataset import resample_training_df


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost and Random Forest models with correlation-based R², MAE, RMSE, and RPIQ evaluation')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    return parser.parse_args()

def load_data():
    """
    Load the training and validation DataFrames from Parquet files.
    Returns:
        validation_df: DataFrame containing validation data
        training_df: DataFrame containing training data
    """
    validation_path = '/lustre/home/vfourel/SOCProject/SOCmapping/balancedDataset/output/final_validation_df.parquet'
    training_path = '/lustre/home/vfourel/SOCProject/SOCmapping/balancedDataset/output/final_training_df.parquet'

    validation_file = Path(validation_path)
    training_file = Path(training_path)

    if not validation_file.exists():
        raise FileNotFoundError(f"Validation file not found at {validation_file}")
    if not training_file.exists():
        raise FileNotFoundError(f"Training file not found at {training_file}")

    validation_df = pd.read_parquet(validation_file)
    training_df = pd.read_parquet(training_file)

    return validation_df, training_df


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

def main():
    args = parse_arguments()
    
    try:
        # Load DataFrames from Parquet files
        validation_df, training_df = load_data()
        
        print("Validation DataFrame:")
        print(validation_df.head())
        print(f"Validation size: {len(validation_df)}")
        
        print("\nTraining DataFrame (before resampling):")
        print(training_df.head())
        print(f"Training size (before resampling): {len(training_df)}")
        
        # Resample training DataFrame
        training_df = resample_training_df(training_df, num_bins=args.num_bins, target_fraction=args.target_fraction)
        
        print("\nResampled Training DataFrame:")
        print(training_df.head())
        print(f"Resampled training size: {len(training_df)}")
        
        # Prepare data using MultiRasterDatasetMultiYears
        samples_coordinates_array_path, data_array_path = separate_and_add_data()
        
        samples_coordinates_array_path = list(dict.fromkeys([item for sublist in samples_coordinates_array_path for item in (sublist if isinstance(sublist, list) else [sublist])]))
        data_array_path = list(dict.fromkeys([item for sublist in data_array_path for item in (sublist if isinstance(sublist, list) else [sublist])]))
        
        # Create datasets
        train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, training_df)
        val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, validation_df)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Prepare data for models
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        print("\nProcessing training data...")
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
        
        print("Processing validation data...")
        for longitudes, latitudes, batch_features, batch_targets in val_dataloader:
            longs = longitudes.numpy()
            lats = latitudes.numpy()
            valid_mask = ~(np.isnan(longs) | np.isnan(lats))
            
            if not np.any(valid_mask):
                continue
            
            features_np = batch_features.numpy()
            flattened_features = features_np.reshape(features_np.shape[0], -1)
            filtered_features = flattened_features[valid_mask]
            filtered_targets = batch_targets.numpy()[valid_mask]
            
            X_val.extend(filtered_features)
            y_val.extend(filtered_targets)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        
        # Check for NaN or inf in features
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Warning: NaN or inf values found in X_train")
        if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
            print("Warning: NaN or inf values found in X_val")
        
        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        print("\nXGBoost model trained successfully!")
        
        # Evaluate XGBoost on training set
        xgb_train_pred = xgb_model.predict(X_train)
        calculate_metrics(y_train, xgb_train_pred, "Training", "XGBoost")
        
        # Evaluate XGBoost on validation set
        xgb_val_pred = xgb_model.predict(X_val)
        calculate_metrics(y_val, xgb_val_pred, "Validation", "XGBoost")
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        rf_model.fit(X_train, y_train)
        print("\nRandom Forest model trained successfully!")
        
        # Evaluate Random Forest on training set
        rf_train_pred = rf_model.predict(X_train)
        calculate_metrics(y_train, rf_train_pred, "Training", "Random Forest")
        
        # Evaluate Random Forest on validation set
        rf_val_pred = rf_model.predict(X_val)
        calculate_metrics(y_val, rf_val_pred, "Validation", "Random Forest")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()