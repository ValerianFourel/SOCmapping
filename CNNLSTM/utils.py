# Import required libraries
import numpy as np  # Numerical computing library for array operations and mathematical functions
import pandas as pd  # Data manipulation and analysis library
import pickle  # Python object serialization module for saving/loading data
import matplotlib.pyplot as plt  # Plotting library for data visualization
# Import various machine learning tools from scikit-learn
from sklearn import preprocessing  # Data preprocessing tools (scaling, normalization)
from sklearn import metrics  # Performance measurement tools
from sklearn import linear_model  # Linear regression models
from sklearn import ensemble  # Ensemble learning methods
from sklearn import semi_supervised  # Semi-supervised learning algorithms
from sklearn import datasets  # Sample datasets
from sklearn import model_selection  # Model selection tools (cross-validation)
import config as cfg  # Custom configuration settings


def save_pickle(filename, data):
    """Save data to a pickle file
    Args:
        filename: Path where the pickle file will be saved
        data: Python object to be serialized and saved
    """
    with open(filename, 'wb') as f:  # Open file in binary write mode
        pickle.dump(data, f)  # Serialize and save the data


def load_pickle(filename):
    """Load data from a pickle file
    Args:
        filename: Path to the pickle file to be loaded
    Returns:
        data: Deserialized Python object from the pickle file
    """
    with open(filename, 'rb') as f:  # Open file in binary read mode
        data = pickle.load(f)  # Deserialize the data
    return data


def calc_dist(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
    Returns:
        dist: Euclidean distance between the points
    """
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # Euclidean distance formula
    return dist


def generate_xy():
    """Generate feature matrices and target variable for model training
    Returns:
        x_cnn_common: Common CNN input features
        x_ts_evi: EVI time series features
        x_ts_lsp: LSP time series features
        x_ts_evi_lsp: Concatenated EVI and LSP features
        y: Target variable array
    """
    # Load sample data from CSV file
    df_samples = pd.read_csv(cfg.f_df_samples)
    # Extract target variable
    y = np.array(df_samples[cfg.target_var_name])

    # Load different feature sets from pickle files
    x_cnn_common = load_pickle(filename=cfg.f_data_DL_common)
    x_ts_evi = load_pickle(filename=cfg.f_data_DL_evi)
    x_ts_lsp = load_pickle(filename=cfg.f_data_DL_lsp)

    # Combine EVI and LSP features along the feature axis (axis=2)
    x_ts_evi_lsp = np.concatenate([x_ts_evi, x_ts_lsp], axis=2)

    # Optional preprocessing steps (commented out):
    # Scale EVI features to range [0, 100]
    # x_ts_evi = preprocessing.minmax_scale(x_ts_evi.reshape(-1, 1), feature_range=(0, 100), axis=0).reshape(x_ts_evi.shape)
    # Randomly shuffle target variable
    # y = y[np.random.permutation(len(y))]

    return x_cnn_common, x_ts_evi, x_ts_lsp, x_ts_evi_lsp, y
