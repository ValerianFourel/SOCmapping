import numpy as np
import matplotlib.pyplot as plt
from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, seasons,
                    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly,
                    SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, 
                    DataSeasonally, file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, years_padded)
from mapping import create_prediction_visualizations, parallel_predict
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import argparse

def modify_matrix_coordinates(MatrixCoordinates_1mil_Yearly=MatrixCoordinates_1mil_Yearly, 
                            MatrixCoordinates_1mil_Seasonally=MatrixCoordinates_1mil_Seasonally, 
                            INFERENCE_TIME=INFERENCE_TIME):
    # Update MatrixCoordinates_1mil_Seasonally
    for i, path in enumerate(MatrixCoordinates_1mil_Seasonally):
        folders = path.split('/')
        last_folder = folders[-1]
        if last_folder == 'Elevation':
            continue
        elif last_folder == 'MODIS_NPP':
            new_path = f"{path}/{INFERENCE_TIME[:4]}"
        else:
            new_path = f"{path}/{INFERENCE_TIME}"
        MatrixCoordinates_1mil_Seasonally[i] = new_path

    # Update MatrixCoordinates_1mil_Yearly
    for i, path in enumerate(MatrixCoordinates_1mil_Yearly):
        if 'Elevation' in path:
            continue
        new_path = f"{path}/{INFERENCE_TIME[:4]}"
        MatrixCoordinates_1mil_Yearly[i] = new_path

    return MatrixCoordinates_1mil_Yearly, MatrixCoordinates_1mil_Seasonally

def parse_arguments():
    parser = argparse.ArgumentParser(description='Random Forest Regression for SOC Mapping')
    parser.add_argument('--model', type=str, choices=['rf'], 
                       default='rf',
                       help='Model type: rf (Random Forest)')
    return parser.parse_args()

def get_top_sampling_years(file_path, top_n=3):
    """Get the top n years with most samples from Excel file"""
    try:
        df = pd.read_excel(file_path)
        year_counts = df['year'].value_counts()
        top_years = year_counts.head(top_n)
        print(f"\nTop {top_n} years with the most samples:")
        for year, count in top_years.items():
            print(f"Year {year}: {count} samples")
        return df, top_years
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, None

def flatten_paths(path_list):
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened

def main():
    args = parse_arguments()
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    
    # Prepare data paths
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    # Create dataset and dataloader
    dataset = MultiRasterDataset(samples_coordinates_array_path, data_array_path, df)
    print("Dataset length:", len(df))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Prepare training data
    X_train, y_train = [], []
    coordinates = []

    for longitudes, latitudes, batch_features, batch_targets in dataloader:
        longs = longitudes.numpy()
        lats = latitudes.numpy()
        valid_mask = ~(np.isnan(longs) | np.isnan(lats))
        
        if not np.any(valid_mask):
            continue
            
        coordinates.append(np.column_stack((longs[valid_mask], lats[valid_mask])))
        features_np = batch_features.numpy()
        flattened_features = features_np.reshape(features_np.shape[0], -1)
        filtered_features = flattened_features[valid_mask]
        filtered_targets = batch_targets.numpy()[valid_mask]
        
        X_train.extend(filtered_features)
        y_train.extend(filtered_targets)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    coordinates = np.vstack(coordinates)

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    print("RandomForest model trained successfully!")

    # Make predictions
    predictions = model.predict(X_train)

    # Training set visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=predictions, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Predicted Values')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Training Set Predictions')
    plt.grid(True)
    plt.show()

    # Load full prediction coordinates
    file_path_coords = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"
    try:
        df_full = pd.read_csv(file_path_coords)
        print(df_full.head())
    except Exception as e:
        print(f"Error loading coordinates file: {e}")
        return

    # Modify paths for inference
    BandsYearly_1milPoints, _ = modify_matrix_coordinates()
    num_cpus = multiprocessing.cpu_count()

    # Parallel prediction
    coordinates, predictions = parallel_predict(
        df_full=df_full,
        model=model,
        bands_yearly=BandsYearly_1milPoints,
        batch_size=8,
        num_threads=num_cpus
    )
    save_path_coords = "coordinates_1mil.npy"
    save_path_preds = "predictions_1mil.npy"

    np.save(save_path_coords, coordinates)
    np.save(save_path_preds, predictions)
    # Final visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=predictions, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Predicted Values')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Full Map Predictions')
    plt.grid(True)
    plt.show()

    # Save predictions
    save_path = '/home/vfourel/SOCProject/SOCmapping/predictions_plots/randomForest_plots'
    create_prediction_visualizations(INFERENCE_TIME, coordinates, predictions, save_path)

if __name__ == "__main__":
    main()
