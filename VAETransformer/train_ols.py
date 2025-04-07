import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
import argparse
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom R² function
def custom_r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_mean) ** 2)
    return 1 - (ssr / sst) if sst != 0 else 0

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train OLS and Random Forest with Lasso feature selection on VAE latent embeddings')
    parser.add_argument('--data_path', type=str, default='/home/vfourel/SOCProject/SOCmapping/VAETransformer/EmbeddingsVAEs2007to2023OCsamples.parquet', help='Path to Parquet file')
    parser.add_argument('--bands', type=str, nargs='+', default=['Elevation', 'LAI', 'LST', 'MODIS_NPP', 'TotalEvapotranspiration'], help='List of bands')
    return parser.parse_args()

# Data preprocessing
def preprocess_data(df, selected_bands):
    latent_cols = [col for col in df.columns if 'latent_z' in col]
    time_varying_selected = [band for band in selected_bands if band != 'Elevation']
    time_steps = max([int(col.split('_')[-1]) for col in latent_cols if col.startswith(time_varying_selected[0] + '_latent_z_')] if time_varying_selected else [0])

    X = []
    for _, row in df.iterrows():
        sample_features = []
        if 'Elevation' in selected_bands and 'elevation_latent_z' in df.columns:
            sample_features.extend(row['elevation_latent_z'])
        for band in time_varying_selected:
            for t in range(1, time_steps + 1):
                col_name = f'{band}_latent_z_{t}'
                if col_name in df.columns:
                    sample_features.extend(row[col_name])
        X.append(sample_features)
    
    X = np.array(X)
    y = df['oc'].values  # Target: organic carbon
    return X, y

# Double selection with Lasso
def double_selection_lasso(X_train, X_test, y_train):
    # First stage: Feature selection with LassoCV
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    
    # Second stage: Select features using optimal alpha
    selector = SelectFromModel(lasso_cv, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature indices
    selected_features = selector.get_support()
    logger.info(f"Number of features selected: {X_train_selected.shape[1]} out of {X_train.shape[1]}")
    
    return X_train_selected, X_test_selected, selected_features

# Main execution
if __name__ == "__main__":
    args = parse_args()

    # Load and preprocess data
    df = pd.read_parquet(args.data_path)
    X, y = preprocess_data(df, args.bands)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Double selection with Lasso
    X_train_selected, X_test_selected, selected_features = double_selection_lasso(X_train, X_test, y_train)

    # OLS Regression with selected features
    X_train_selected_with_const = sm.add_constant(X_train_selected)
    X_test_selected_with_const = sm.add_constant(X_test_selected)
    ols_model = sm.OLS(y_train, X_train_selected_with_const).fit()
    y_pred_ols = ols_model.predict(X_test_selected_with_const)
    ols_mse = mean_squared_error(y_test, y_pred_ols)
    ols_r2 = custom_r2_score(y_test, y_pred_ols)

    logger.info("OLS Regression Results (with Lasso-selected features):")
    print(ols_model.summary())
    logger.info(f"OLS Test MSE: {ols_mse:.4f}, R²: {ols_r2:.4f}")

    # Random Forest with selected features
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_selected, y_train)
    y_pred_rf = rf_model.predict(X_test_selected)
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_r2 = custom_r2_score(y_test, y_pred_rf)

    logger.info("Random Forest Results (with Lasso-selected features):")
    logger.info(f"Random Forest Test MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'feature': np.arange(X_train_selected.shape[1]),
        'importance': rf_model.feature_importances_
    })
    logger.info("Top 10 most important features:")
    logger.info(feature_importance.sort_values('importance', ascending=False).head(10))