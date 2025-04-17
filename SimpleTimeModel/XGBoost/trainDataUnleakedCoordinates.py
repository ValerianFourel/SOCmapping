import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from pathlib import Path
from typing import List, Tuple

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost and Random Forest models with pre-generated 5-fold datasets')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--data-dir', type=str, default='/home/vfourel/SOCProject/SOCmapping/balancedDataset/outputCrossFold08km8Percent', help='Directory containing parquet files')
    return parser.parse_args()

def load_fold_datasets(directory: str, num_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load training and validation parquet files for each fold from the specified directory.
    
    Args:
        directory (str): Path to the directory containing parquet files.
        num_folds (int): Number of folds to load (default is 5).
    
    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (training_df, validation_df) tuples for each fold.
    
    Raises:
        FileNotFoundError: If a required parquet file is missing.
        ValueError: If the directory is invalid or no valid fold data is found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory {directory} does not exist or is not a directory.")
    
    fold_dfs = []
    
    for fold in range(1, num_folds + 1):
        train_file = directory / f'final_training_df_fold{fold}.parquet'
        val_file = directory / f'final_validation_df_fold{fold}.parquet'
        
        try:
            if not train_file.exists():
                raise FileNotFoundError(f"Training file {train_file} not found.")
            if not val_file.exists():
                raise FileNotFoundError(f"Validation file {val_file} not found.")
            
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(val_file)
            
            print(f"Fold {fold}: Loaded {len(train_df)} training rows and {len(val_df)} validation rows.")
            fold_dfs.append((train_df, val_df))
        
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping fold {fold}.")
        except Exception as e:
            print(f"Error loading fold {fold}: {e}. Skipping fold {fold}.")
    
    if not fold_dfs:
        raise ValueError("No valid fold datasets were loaded from the directory.")
    
    return fold_dfs

def resample_training_df(training_df, num_bins=128, target_fraction=0.75):
    oc_values = training_df['OC'].dropna()
    bins = pd.qcut(oc_values, q=num_bins, duplicates='drop')
    bin_counts = bins.value_counts().sort_index()
    target_count = int(bin_counts.max() * target_fraction)
    
    resampled_dfs = []
    for bin_label in bin_counts.index:
        bin_mask = pd.cut(training_df['OC'], bins=bins.cat.categories) == bin_label
        bin_df = training_df[bin_mask]
        if len(bin_df) < target_count:
            additional_samples = target_count - len(bin_df)
            sampled_df = bin_df.sample(n=additional_samples, replace=True, random_state=42)
            resampled_dfs.append(pd.concat([bin_df, sampled_df], ignore_index=True))
        else:
            resampled_dfs.append(bin_df)
    
    return pd.concat(resampled_dfs, ignore_index=True)

def calculate_metrics(y_true, y_pred, dataset_name, model_name, fold_num):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
    
    if len(y_true) == 0:
        print(f"{dataset_name} ({model_name}, Fold {fold_num}): No valid data")
        return {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
    
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = corr ** 2 if np.isfinite(corr) else 0.0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    rpiq = iqr / rmse if rmse != 0 else float('inf')
    
    print(f"{dataset_name} ({model_name}, Fold {fold_num}): R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, RPIQ={rpiq:.4f}")
    return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'RPIQ': rpiq}

def average_metrics(metrics_list):
    valid_metrics = [m for m in metrics_list if not np.any(np.isnan(list(m.values())))]
    if not valid_metrics:
        print("No valid metrics to average")
        return {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
    
    avg_metrics = {
        key: np.mean([m[key] for m in valid_metrics])
        for key in ['R²', 'MAE', 'RMSE', 'RPIQ']
    }
    print(f"Averaged Metrics: R²={avg_metrics['R²']:.4f}, MAE={avg_metrics['MAE']:.4f}, RMSE={avg_metrics['RMSE']:.4f}, RPIQ={avg_metrics['RPIQ']:.4f}")
    return avg_metrics

def main():
    args = parse_arguments()
    
    # Load pre-generated fold datasets
    fold_dfs = load_fold_datasets(args.data_dir)
    
    required_columns = ['GPS_LONG', 'GPS_LAT', 'OC']
    k = len(fold_dfs)  # Number of folds based on loaded data
    
    # Initialize metrics storage
    metrics = {
        'xgb_train': [], 'xgb_val': [],
        'rf_train': [], 'rf_val': []
    }
    
    for fold, (train_df, val_df) in enumerate(fold_dfs, 1):
        print(f"\n=== Fold {fold}/{k} ===")
        try:
            # Validate DataFrame columns
            if not all(col in train_df.columns for col in required_columns):
                raise ValueError(f"Training DataFrame for fold {fold} missing required columns: {required_columns}")
            if not all(col in val_df.columns for col in required_columns):
                raise ValueError(f"Validation DataFrame for fold {fold} missing required columns: {required_columns}")
            
            # Resample training data
            train_df = resample_training_df(train_df, args.num_bins, args.target_fraction)
            
            # Prepare features and targets
            X_train = train_df[['GPS_LONG', 'GPS_LAT']].values
            y_train = train_df['OC'].values
            X_val = val_df[['GPS_LONG', 'GPS_LAT']].values
            y_val = val_df['OC'].values
            
            print(f"Training size: {len(train_df)}, Validation size: {len(val_df)}")
            
            # Validate data integrity
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                print("Warning: NaN or inf values in X_train")
            if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
                print("Warning: NaN or inf values in X_val")
            
            # Train and evaluate XGBoost
            xgb_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            
            metrics['xgb_train'].append(calculate_metrics(
                y_train, xgb_model.predict(X_train), "Training", "XGBoost", fold
            ))
            metrics['xgb_val'].append(calculate_metrics(
                y_val, xgb_model.predict(X_val), "Validation", "XGBoost", fold
            ))
            
            # Train and evaluate Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=1000,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            metrics['rf_train'].append(calculate_metrics(
                y_train, rf_model.predict(X_train), "Training", "Random Forest", fold
            ))
            metrics['rf_val'].append(calculate_metrics(
                y_val, rf_model.predict(X_val), "Validation", "Random Forest", fold
            ))
            
        except Exception as e:
            print(f"Error in Fold {fold}: {e}")
            nan_metrics = {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
            for key in metrics:
                metrics[key].append(nan_metrics)
    
    # Print averaged metrics
    print("\n=== Final Averaged Results ===")
    for key in metrics:
        print(f"\n{key.replace('_', ' ').title()}:")
        average_metrics(metrics[key])

if __name__ == "__main__":
    main()