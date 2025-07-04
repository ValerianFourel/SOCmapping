
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataloader.dataloaderMulti import MultiRasterDatasetMultiYears, filter_dataframe, separate_and_add_data
# from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from torch.utils.data import DataLoader
import argparse
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, seasons,
    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly,
    SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, years_padded
)
from balancedDataset import create_validation_train_sets,resample_training_df

#from configElevationOnlyExperiment import  (
#    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, seasons,
#    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly,
#    SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally,
#    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, years_padded
#)

# Function to create balanced dataset (unchanged)
def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True):
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
                val_samples = bin_data.sample(n=min(13, len(bin_data)))
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
        print('Size of the training set:   ' ,len(training_df))
        print('Size of the validation set:   ' ,len(validation_df))
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
        return training_df, None
    


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost and Random Forest models with correlation-based R², MAE, RMSE, and RPIQ evaluation')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of times to run the process')
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

def main():
    args = parse_arguments()
    
    num_runs = args.num_runs
    results = []
    
    # Load and process data paths once (assuming they are fixed)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    samples_coordinates_array_path = list(dict.fromkeys([item for sublist in samples_coordinates_array_path for item in (sublist if isinstance(sublist, list) else [sublist])]))
    data_array_path = list(dict.fromkeys([item for sublist in data_array_path for item in (sublist if isinstance(sublist, list) else [sublist])]))
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        try:
            # Create validation and training sets
            validation_df, training_df = create_validation_train_sets(
                df=None,
                output_dir=args.output_dir,
                target_val_ratio=args.target_val_ratio,
                use_gpu=args.use_gpu,
                distance_threshold=args.distance_threshold
            )
            df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
            training_df, validation_df=  create_balanced_dataset(df=df)
            
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
            
            # Evaluate XGBoost
            xgb_train_pred = xgb_model.predict(X_train)
            xgb_val_pred = xgb_model.predict(X_val)
            xgb_train_metrics = calculate_metrics(y_train, xgb_train_pred, "Training", "XGBoost")
            xgb_val_metrics = calculate_metrics(y_val, xgb_val_pred, "Validation", "XGBoost")
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=1000,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            print("\nRandom Forest model trained successfully!")
            
            # Evaluate Random Forest
            rf_train_pred = rf_model.predict(X_train)
            rf_val_pred = rf_model.predict(X_val)
            rf_train_metrics = calculate_metrics(y_train, rf_train_pred, "Training", "Random Forest")
            rf_val_metrics = calculate_metrics(y_val, rf_val_pred, "Validation", "Random Forest")
            
            # Store metrics for this run
            run_metrics = {
                "XGBoost_Train": xgb_train_metrics,
                "XGBoost_Val": xgb_val_metrics,
                "RandomForest_Train": rf_train_metrics,
                "RandomForest_Val": rf_val_metrics
            }
            results.append(run_metrics)
            
        except Exception as e:
            print(f"Error in run {run+1}: {e}")
    
    # After all runs, compute and print average metrics
    print("\nSummary of all runs:")
    metrics_keys = ["R²", "MAE", "RMSE", "RPIQ"]
    for model_set in ["XGBoost_Train", "XGBoost_Val", "RandomForest_Train", "RandomForest_Val"]:
        for metric in metrics_keys:
            values = [run[model_set][metric] for run in results if metric in run[model_set]]
            if values:
                avg = np.mean(values)
                std = np.std(values)
                print(f"{model_set} {metric}: {avg:.4f} ± {std:.4f}")
            else:
                print(f"{model_set} {metric}: No data")

if __name__ == "__main__":
    main()
