import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from pathlib import Path
import scipy  # Added for QhullError

def load_inference_data():
    """
    Load .npy files with explicit paths, print shapes, and ensure correct dimensions.
    Trims arrays, aligns coordinates and predictions, and picks max prediction for duplicate coords.
    """
    root_dir = "/home/vfourel/SOCProject/SOCmapping/results_predictions"
    method_data = {}

    method_configs = {
        "3DCNN": {
            "subfolder": "2015InferenceRun_v1",
            "files": ["coordinates_1mil.npy", "predictions_1mil.npy"]
        },
        "CNNLSTM": {
            "subfolder": "2015Inference",
            "files": ["coordinates_1mil.npy", "predictions_1mil.npy"]
        },
        "FoundationalModels": {
            "subfolder": "run2015_MAXOC150_v1",
            "files": [
                "coordinates_1mil_1stQuarter.npy", # "coordinates_1mil_2ndQuarter.npy"
                # "coordinates_1mil_3rdQuarter.npy", "coordinates_1mil_4thQuarter.npy",
              "predictions_1mil_1stQuarter.npy"# "predictions_1mil_2ndQuarter.npy",
                # "predictions_1mil_3rdQuarter.npy", "predictions_1mil_4thQuarter.npy"
            ]
        },
        "RandomForest": {
            "subfolder": "2015Inference",
            "files": ["coordinates_1mil.npy", "predictions_1mil.npy"]
        },
        "SimpleCNN": {
            "subfolder": "2015Inference",
            "files": ["coordinates_1mil.npy", "predictions_1mil.npy"]
        },
        "SimpleTransformer": {
            "subfolder": "2015Inference",
            "files": [
                "coordinates_1mil_1stThird.npy", "coordinates_1mil_2ndThird.npy",
                "coordinates_1mil_3rdThird.npy", "predictions_1mil_1stThird.npy",
                "predictions_1mil_2ndThird.npy", "predictions_1mil_3rdThird.npy"
            ]
        },
        "XGBoost": {
            "subfolder": "2015Inference",
            "files": ["coordinates_1mil.npy", "predictions_1mil.npy"]
        }
    }

    for method, config in method_configs.items():
        subfolder_path = f"{root_dir}/{method}/{config['subfolder']}"
        coords_list = []
        preds_list = []

        print(f"\nLoading files for {method}:")
        for file_name in config["files"]:
            file_path = f"{subfolder_path}/{file_name}"
            data = np.load(file_path)
            print(f"  {file_name}: shape={data.shape}")
            
            if "coordinates" in file_name:
                # Ensure coordinates are (N, 2)
                if data.shape[1] != 2:
                    data = data.T  # Transpose if loaded as (2, N)
                coords_list.append(data)
            elif "predictions" in file_name:
                # Ensure predictions are 1D
                if data.ndim > 1:
                    data = data.flatten()
                preds_list.append(data)

        if len(coords_list) > 1:
            # Trim coordinates to smallest N
            min_coords_n = min(arr.shape[0] for arr in coords_list)
            coords_list = [arr[:min_coords_n, :] for arr in coords_list]
            coordinates = np.concatenate(coords_list, axis=0)

            # Trim predictions to smallest N
            min_preds_n = min(arr.shape[0] for arr in preds_list)
            preds_list = [arr[:min_preds_n] for arr in preds_list]
            predictions = np.concatenate(preds_list, axis=0)
        else:
            coordinates = coords_list[0]
            predictions = preds_list[0]

        # Ensure coordinates and predictions have the same number of points
        min_points = min(coordinates.shape[0], predictions.shape[0])
        coordinates = coordinates[:min_points, :]
        predictions = predictions[:min_points]

        # Handle duplicate coordinates by picking the max prediction
        unique_coords, indices, counts = np.unique(coordinates, axis=0, return_inverse=True, return_counts=1)
        if np.any(counts > 1):
            print(f"{method}: Found {np.sum(counts > 1)} duplicate coordinate pairs, aggregating by max prediction")
            unique_predictions = np.zeros(unique_coords.shape[0])
            for i in range(unique_coords.shape[0]):
                # Find all predictions for this unique coordinate
                pred_indices = np.where(indices == i)[0]
                unique_predictions[i] = np.max(predictions[pred_indices])
            coordinates = unique_coords
            predictions = unique_predictions
        else:
            print(f"{method}: No duplicate coordinates found")

        method_data[method] = {
            'coordinates': coordinates,  # (N, 2)
            'predictions': predictions   # (N,)
        }
        print(f"{method} final: coords shape={coordinates.shape}, preds shape={predictions.shape}")

    return method_data

def plot_interpolated_maps(method_data, 
                          save_path="/home/vfourel/SOCProject/SOCmapping/results_predictions/output/maps", 
                          year='2015'):
    """
    Create and save interpolated maps with Bavaria contours.
    Falls back to 'nearest' if 'linear' fails. Maxxed out contours with rainbow vibes.
    """
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']
    
    n_methods = len(method_data)
    fig, axes = plt.subplots(nrows=(n_methods + 1) // 2, ncols=2, 
                           figsize=(12, 6 * ((n_methods + 1) // 2)))
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    all_predictions = [data['predictions'] for data in method_data.values()]
    vmin = np.min(np.concatenate(all_predictions))
    vmax = np.max(np.concatenate(all_predictions))
    
    for idx, (method, data) in enumerate(method_data.items()):
        coords = data['coordinates']
        preds = data['predictions']
        
        unique_coords = np.unique(coords, axis=0)
        print(f"{method}: unique coords={len(unique_coords)}, preds={len(preds)}")
        
        if len(unique_coords) < 3:
            print(f"{method}: Too few unique points, skipping plot")
            axes[idx].text(0.5, 0.5, f"{method}: Insufficient unique points ({len(unique_coords)})",
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        grid_x = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 300)
        grid_y = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 300)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        try:
            grid_z = griddata(coords, preds, (grid_x, grid_y), method='linear')
            if np.all(np.isnan(grid_z)):
                raise ValueError("Linear interpolation produced all NaNs")
        except (ValueError, scipy.spatial.qhull.QhullError) as e:
            print(f"{method}: Linear interpolation failed ({str(e)}), using nearest")
            grid_z = griddata(coords, preds, (grid_x, grid_y), method='nearest')
        
        ax = axes[idx]
        contour = ax.contourf(grid_x, grid_y, grid_z,
                              levels=100,  # Maxxing contours with finer transitions
                              cmap='hsv',  # Rainbowed max contrast
                              vmin=vmin, vmax=vmax,
                              alpha=0.9)  # Upped alpha for that GG glow
        
        bavaria.boundary.plot(ax=ax, color='black', linewidth=1.5)  # Thicker boundary for contrast
        ax.set_title(f'{method} - Interpolated Predictions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', alpha=0.5)  # Subtle grid to keep focus on colors
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, label='Predicted Values')
    cbar.ax.tick_params(labelsize=10)  # Readable colorbar ticks
    
    for idx in range(len(method_data), len(axes)):
        fig.delaxes(axes[idx])
    
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'{year}_bavaria_interpolated_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved interpolated maps to: {output_file}")

if __name__ == "__main__":
    method_data = load_inference_data()
    plot_interpolated_maps(method_data)