import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import argparse
import logging
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom R² Implementation ---
def custom_r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_mean = np.mean(y_true)
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ssr / sst) if sst != 0 else 0
    return r2

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP and Random Forest on VAE latent embeddings with L1 double selection')
    parser.add_argument('--data_path', type=str, default='/home/vfourel/SOCProject/SOCmapping/VAETransformer/EmbeddingsVAEs2007to2023OCsamples_smaller.parquet', help='Path to Parquet file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for MLP training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension for MLP')
    parser.add_argument('--weights_dir', type=str, default='./weights', help='Directory to save MLP weights')
    parser.add_argument('--bands', type=str, nargs='+', default=['Elevation', 'LAI', 'LST', 'MODIS_NPP', 'TotalEvapotranspiration'], help='List of bands to include')
    return parser.parse_args()
# --- MLP Model Definition ---
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer
        self.relu1 = nn.ReLU()                       # Activation
        self.fc2 = nn.Linear(hidden_dim, 1)          # Hidden to output

    def forward(self, x):
        x = self.fc1(x)    # Pass through hidden layer
        x = self.relu1(x)  # Apply ReLU
        x = self.fc2(x)    # Pass to output layer
        return x
# --- Data Preprocessing ---
def preprocess_data(df, selected_bands):
    """
    Preprocess the dataframe to extract features from the specified bands' latent embeddings.
    
    Args:
        df (pd.DataFrame): Input dataframe with latent_z columns.
        selected_bands (list): List of band names to include (e.g., ['Elevation', 'LAI', 'LST']).
    
    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target array (organic carbon values).
        input_dim (int): Number of input features.
        feature_names (list): Names of the features.
    """
    # Identify all latent_z columns
    latent_cols = [col for col in df.columns if 'latent_z' in col]

    # Separate time-varying bands (exclude Elevation, which is static)
    time_varying_selected = [band for band in selected_bands if band != 'Elevation']

    # Determine the number of time steps from the first time-varying band
    if time_varying_selected:
        first_band = time_varying_selected[0]
        time_steps = max([int(col.split('_')[-1]) for col in latent_cols if col.startswith(first_band + '_latent_z_')], default=0)
    else:
        time_steps = 0

    # Generate feature names based on selected bands
    feature_names = []
    if 'Elevation' in selected_bands and 'elevation_latent_z' in df.columns:
        elevation_dim = len(df.iloc[0]['elevation_latent_z'])
        feature_names.extend([f'elevation_latent_z_{i}' for i in range(elevation_dim)])
    for band in time_varying_selected:
        for t in range(1, time_steps + 1):
            col_name = f'{band}_latent_z_{t}'
            if col_name in df.columns:
                latent_dim = len(df.iloc[0][col_name])
                feature_names.extend([f'{band}_latent_z_{t}_{i}' for i in range(latent_dim)])

    # Extract features for each sample
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
    y = df['oc'].values  # Target variable (organic carbon)
    
    return X, y, X.shape[1], feature_names

# --- L1 Double Selection ---
def l1_double_selection(X_train, y_train, X_test, feature_names, alpha=0.1):
    lasso_y = LassoLars(alpha=alpha)
    lasso_y.fit(X_train, y_train)
    selected_y = lasso_y.coef_ != 0

    selected_features = np.zeros(X_train.shape[1], dtype=bool)
    for i in range(X_train.shape[1]):
        if selected_y[i]:
            selected_features[i] = True
        else:
            lasso_x = LassoLars(alpha=alpha)
            X_i = X_train[:, i]
            X_rest = np.delete(X_train, i, axis=1)
            lasso_x.fit(X_rest, X_i)
            if np.any(lasso_x.coef_ != 0):
                selected_features[i] = True

    selected_indices = np.where(selected_features)[0]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]

    return X_train_selected, X_test_selected, selected_names

# --- Training Function for MLP ---
def train_mlp(X_train, y_train, X_test, y_test, input_dim, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPRegressor(input_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_epochs = 1000
    best_r2 = float('-inf')
    patience = 500
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        test_preds = []
        test_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                test_preds.extend(outputs.cpu().numpy().flatten())
                test_true.extend(y_batch.cpu().numpy().flatten())
        test_r2 = custom_r2_score(test_true, test_preds)
        test_mse = mean_squared_error(test_true, test_preds)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_mse': test_mse, 'test_r2': test_r2})

        if test_r2 > best_r2:
            best_r2 = test_r2
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.weights_dir}/best_mlp_weights.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} with best R²: {best_r2:.4f}")
                break

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch} - Test R²: {test_r2:.4f}, Sample Predictions: {test_preds[:5]}, True: {test_true[:5]}")

    model.load_state_dict(torch.load(f'{args.weights_dir}/best_mlp_weights.pth'))
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    test_true = y_test
    
    return model, test_preds, test_true

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # Load data
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

    # Configuration (assuming these are defined in config)
    from config import TIME_BEGINNING, TIME_END, window_size, time_before

    # Initialize Weights & Biases
    wandb.init(project="socmapping-VAETransformer-MLPRegressor", config={
        "time_beginning": TIME_BEGINNING,
        "time_end": TIME_END,
        "window_size": window_size,
        "time_before": time_before,
        "bands": args.bands,
        "mlp_epochs": 1000,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "weights_dir": args.weights_dir,
        "normalized_bands": ['LST', 'MODIS_NPP', 'TotalEvapotranspiration']
    })

    # Preprocess data with selected bands
    selected_bands = args.bands
    X, y, input_dim, feature_names = preprocess_data(df, selected_bands)
    logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    # Split data: 10% test, 90% train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=42)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train MLP on selected features
    logger.info("Training MLP for up to 1000 epochs on selected features with L1 loss...")
    mlp_model, mlp_test_preds, mlp_test_true = train_mlp(X_train, y_train, X_test, y_test, input_dim, args)
    mlp_mse = mean_squared_error(mlp_test_true, mlp_test_preds)
    mlp_r2 = custom_r2_score(mlp_test_true, mlp_test_preds)
    logger.info(f"MLP (Test) - MSE: {mlp_mse:.4f}, R2: {mlp_r2:.4f}")
    wandb.log({'mlp_mse': mlp_mse, 'mlp_r2': mlp_r2})

    # Train Random Forest on selected features
    logger.info("Training Random Forest on selected features...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_r2 = custom_r2_score(y_test, y_pred_rf)
    logger.info(f"Random Forest (Test) - MSE: {rf_mse:.4f}, R2: {rf_r2:.4f}")
    wandb.log({'rf_mse': rf_mse, 'rf_r2': rf_r2})

    # L1 Double Selection
    logger.info("Performing L1 double selection...")
    X_train_selected, X_test_selected, selected_names = l1_double_selection(X_train, y_train, X_test, feature_names)
    logger.info(f"Selected {len(selected_names)} features: {selected_names}")

    # OLS Regression with all variables
    logger.info("Performing OLS regression with all variables...")
    X_train_with_const = sm.add_constant(X_train)
    ols_model_all = sm.OLS(y_train, X_train_with_const).fit()
    logger.info("OLS (All Variables) Summary:")
    print(ols_model_all.summary())

    # Predict with OLS on test set
    X_test_with_const = sm.add_constant(X_test)
    y_pred_ols = ols_model_all.predict(X_test_with_const)
    ols_mse = mean_squared_error(y_test, y_pred_ols)
    ols_r2 = custom_r2_score(y_test, y_pred_ols)
    logger.info(f"OLS (All Variables, Test) - MSE: {ols_mse:.4f}, R2: {ols_r2:.4f}")
    wandb.log({'ols_mse': ols_mse, 'ols_r2': ols_r2})

    # Finish W&B run
    wandb.finish()

    # Log file size
    file_size_mb = Path(args.data_path).stat().st_size / (1024 * 1024)
    logger.info(f"Parquet file size: {file_size_mb:.2f} MB")