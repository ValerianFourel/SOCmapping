import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config_v2_time_multi import TIME_BEGINNING, TIME_END, MAX_OC
from dataloader.dataloaderMulti import filter_dataframe
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost and Random Forest models using GPS_LONG and GPS_LAT with 10-fold averaged metrics')
    parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--use_validation', action='store_true', default=True, help='Whether to use validation set')
    return parser.parse_args()

def resample_training_df(training_df, num_bins=128, target_fraction=0.75):
    oc_values = training_df['OC'].dropna()
    bins = pd.qcut(oc_values, q=num_bins, duplicates='drop')
    
    bin_counts = bins.value_counts().sort_index()
    max_count = bin_counts.max()
    target_count = int(max_count * target_fraction)
    
    print(f"Max bin count: {max_count}")
    print(f"Target count per bin (at least): {target_count}")
    
    resampled_dfs = []
    
    for bin_label in bin_counts.index:
        bin_mask = pd.cut(training_df['OC'], bins=bins.cat.categories) == bin_label
        bin_df = training_df[bin_mask]
        
        if len(bin_df) < target_count:
            additional_samples = target_count - len(bin_df)
            sampled_df = bin_df.sample(n=additional_samples, replace=True, random_state=42)
            resampled_dfs.append(pd.concat([bin_df, sampled_df]))
        else:
            resampled_dfs.append(bin_df)
    
    resampled_df = pd.concat(resampled_dfs, ignore_index=True)
    
    new_bins = pd.qcut(resampled_df['OC'], q=num_bins, duplicates='drop')
    new_bin_counts = new_bins.value_counts().sort_index()
    
    print("\nBin counts before resampling:")
    print(bin_counts)
    print("\nBin counts after resampling:")
    print(new_bin_counts)
    
    return resampled_df

def calculate_metrics(y_true, y_pred, dataset_name, model_name, simulation_num=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        print(f"\n{dataset_name} Metrics ({model_name}, Simulation {simulation_num}): No valid data after filtering NaN/inf")
        return {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
    
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = corr ** 2 if np.isfinite(corr) else 0.0
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    q1, q3 = np.percentile(y_true, [25, 75])
    iqr = q3 - q1
    rpiq = iqr / rmse if rmse != 0 else float('inf')
    
    if simulation_num is not None:
        print(f"\n{dataset_name} Metrics ({model_name}, Simulation {simulation_num}):")
    else:
        print(f"\n{dataset_name} Metrics ({model_name}):")
    print(f"R² (correlation squared): {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RPIQ: {rpiq:.4f}")
    
    return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'RPIQ': rpiq}

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4, use_validation=True, random_state=None):
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
                val_samples = bin_data.sample(n=min(13, len(bin_data)), random_state=random_state)
                validation_indices.extend(val_samples.index)
                train_samples = bin_data.drop(val_samples.index)
                if len(train_samples) > 0:
                    if len(train_samples) < min_samples:
                        resampled = train_samples.sample(n=min_samples, replace=True, random_state=random_state)
                        training_dfs.append(resampled)
                    else:
                        training_dfs.append(train_samples)
        if not training_dfs or not validation_indices:
            raise ValueError("No training or validation data available after binning")
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        validation_df = df.loc[validation_indices].drop('bin', axis=1)
        print('Size of the training set:   ', len(training_df))
        print('Size of the validation set: ', len(validation_df))
        return training_df, validation_df
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True, random_state=random_state)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        if not training_dfs:
            raise ValueError("No training data available after binning")
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        return training_df

def average_metrics(metrics_list):
    valid_metrics = [m for m in metrics_list if not np.any(np.isnan(list(m.values())))]
    if not valid_metrics:
        return {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
    
    avg_metrics = {
        'R²': np.mean([m['R²'] for m in valid_metrics]),
        'MAE': np.mean([m['MAE'] for m in valid_metrics]),
        'RMSE': np.mean([m['RMSE'] for m in valid_metrics]),
        'RPIQ': np.mean([m['RPIQ'] for m in valid_metrics])
    }
    
    print("\nAveraged Metrics:")
    print(f"R² (correlation squared): {avg_metrics['R²']:.4f}")
    print(f"MAE: {avg_metrics['MAE']:.4f}")
    print(f"RMSE: {avg_metrics['RMSE']:.4f}")
    print(f"RPIQ: {avg_metrics['RPIQ']:.4f}")
    
    return avg_metrics

def main():
    args = parse_arguments()
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)

    # Ensure required columns exist
    required_columns = ['GPS_LONG', 'GPS_LAT', 'OC']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Lists to store metrics for each simulation
    xgb_train_metrics_all = []
    xgb_val_metrics_all = []
    rf_train_metrics_all = []
    rf_val_metrics_all = []

    # Run 10 simulations
    for sim in range(10):
        print(f"\n=== Simulation {sim + 1} ===")
        random_state = np.random.randint(0, 10000)  # Different seed for each simulation

        try:
            # Create train-validation split with different random state
            if args.use_validation:
                train_df, val_df = create_balanced_dataset(df, use_validation=True, random_state=random_state)
            else:
                train_df = create_balanced_dataset(df, use_validation=False, random_state=random_state)
                val_df = pd.DataFrame()  # Empty validation set

            # Resample training DataFrame
            train_df = resample_training_df(train_df, num_bins=args.num_bins, target_fraction=args.target_fraction)
            
            print("\nTraining DataFrame:")
            print(train_df[['GPS_LONG', 'GPS_LAT', 'OC']].head())
            print(f"Training size: {len(train_df)}")
            
            print("\nValidation DataFrame:")
            print(val_df[['GPS_LONG', 'GPS_LAT', 'OC']].head() if not val_df.empty else "Empty")
            print(f"Validation size: {len(val_df)}")
            
            # Prepare features (GPS_LONG, GPS_LAT) and target (OC)
            X_train = train_df[['GPS_LONG', 'GPS_LAT']].values
            y_train = train_df['OC'].values
            X_val = val_df[['GPS_LONG', 'GPS_LAT']].values if not val_df.empty else np.array([])
            y_val = val_df['OC'].values if not val_df.empty else np.array([])
            
            print(f"Training features shape: {X_train.shape}")
            print(f"Validation features shape: {X_val.shape if len(X_val) > 0 else 'Empty'}")
            
            # Check for NaN or inf in features
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                print("Warning: NaN or inf values found in X_train")
            if len(X_val) > 0 and (np.any(np.isnan(X_val)) or np.any(np.isinf(X_val))):
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
            xgb_train_metrics = calculate_metrics(y_train, xgb_train_pred, "Training", "XGBoost", sim + 1)
            xgb_train_metrics_all.append(xgb_train_metrics)
            
            if len(X_val) > 0:
                xgb_val_pred = xgb_model.predict(X_val)
                xgb_val_metrics = calculate_metrics(y_val, xgb_val_pred, "Validation", "XGBoost", sim + 1)
                xgb_val_metrics_all.append(xgb_val_metrics)
            else:
                xgb_val_metrics_all.append({'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan})
            
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
            rf_train_metrics = calculate_metrics(y_train, rf_train_pred, "Training", "Random Forest", sim + 1)
            rf_train_metrics_all.append(rf_train_metrics)
            
            if len(X_val) > 0:
                rf_val_pred = rf_model.predict(X_val)
                rf_val_metrics = calculate_metrics(y_val, rf_val_pred, "Validation", "Random Forest", sim + 1)
                rf_val_metrics_all.append(rf_val_metrics)
            else:
                rf_val_metrics_all.append({'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan})
            
        except Exception as e:
            print(f"Error in simulation {sim + 1}: {e}")
            # Append NaN metrics to keep lists aligned
            nan_metrics = {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'RPIQ': np.nan}
            xgb_train_metrics_all.append(nan_metrics)
            xgb_val_metrics_all.append(nan_metrics)
            rf_train_metrics_all.append(nan_metrics)
            rf_val_metrics_all.append(nan_metrics)

    # Calculate and print averaged metrics
    print("\n=== Final Averaged Results ===")
    print("\nXGBoost Training Metrics (Averaged):")
    average_metrics(xgb_train_metrics_all)
    
    print("\nXGBoost Validation Metrics (Averaged):")
    average_metrics(xgb_val_metrics_all)
    
    print("\nRandom Forest Training Metrics (Averaged):")
    average_metrics(rf_train_metrics_all)
    
    print("\nRandom Forest Validation Metrics (Averaged):")
    average_metrics(rf_val_metrics_all)

if __name__ == "__main__":
    main()
