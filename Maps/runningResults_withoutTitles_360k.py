import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from datetime import datetime
from matplotlib import cm
import matplotlib.patches as mpatches
import pickle

# Set publication-quality font sizes and styling globally
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 28,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.linewidth': 2.0,
    'grid.linewidth': 1.2,
    'lines.linewidth': 2.5,
})

# Constants for filenames
PICTURE_VERSION = "SGT_Rescaled_NoTitles"
MAX_OC = 150
TIME_BEGINNING = 2007
TIME_END = 2023
INFERENCE_TIME = 2023

# Create output directory
output_dir = "/home/valerian/SGTPublication/SOCmapping/SGT_Maps_Rescaled_NoTitles"
os.makedirs(output_dir, exist_ok=True)

# Load landcover data
landcover_path = "/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/Archive/landcover_bavaria_Visual_results_npy_s/landcover_values_merged.npy"
landcover_coordinates_path = "/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy"

print("Loading landcover data...")
landcover_values = np.load(landcover_path, allow_pickle=True)
landcover_coordinates = np.load(landcover_coordinates_path)

# Define SGT model paths with normalization info
model_data = {
    "SGT_360k": {
        "display_name": "Spatiotemporal Gated Transformer (360k Params)",
        "type": "split",
        "base_dir": "/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_360kParameters",
        "file_pattern": [
            ("coordinates_0k_to_120k.npy", "predictions_0k_to_120k.npy"),
            ("coordinates_120k_to_240k.npy", "predictions_120k_to_240k.npy"),
            ("coordinates_240k_to_300k.npy", "predictions_240k_to_300k.npy"),
            ("coordinates_300k_to_360k.npy", "predictions_300k_to_360k.npy")
        ],
        "normalization_pkl": None  # Add path if available
    },
    "SGT_1mil_v1": {
        "display_name": "Spatiotemporal Gated Transformer (1mil Params - v1)",
        "type": "split",
        "base_dir": "/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/Archive/finalResults2023_OC150_2007to2023_transform_normalize_loss_l1_runs_1_lr_0.0002_heads_8_layers_2_TemporalFusionTransformer_Inference/runsPredictions",
        "file_pattern": [
            ("coordinates_0_to_100000.npy", "predictions_0_to_100000.npy"),
            ("coordinates_100000_to_200000.npy", "predictions_100000_to_200000.npy"),
            ("coordinates_200000_to_300000.npy", "predictions_200000_to_300000.npy"),
            ("coordinates_300000_to_400000.npy", "predictions_300000_to_400000.npy"),
            ("coordinates_400000_to_420000.npy", "predictions_400000_to_420000.npy")
        ],
        "normalization_pkl": "/home/valerian/SGTPublication/residual_Maps_Bavaria_1milTFT/residual_Maps_Bavaria_1milTFT/analysis_results.pkl"
    },
    "SGT_1.1mil_v2": {
        "display_name": "Spatiotemporal Gated Transformer (1.1mil Params - v2)",
        "type": "split",
        "base_dir": "/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_1milVersion_v2",
        "file_pattern": [
            ("coordinates_0k_to_120k.npy", "predictions_0k_to_120k.npy"),
            ("coordinates_120k_to_240k.npy", "predictions_120k_to_240k.npy"),
            ("coordinates_240k_to_360k.npy", "predictions_240k_to_360k.npy"),
            ("coordinates_360k_to_480k.npy", "predictions_360k_to_480k.npy")
        ],
        "normalization_pkl": "/home/valerian/SGTPublication/residual_Maps_Bavaria_1milTFT/residual_Maps_Bavaria_1milTFT/analysis_results.pkl"
    }
}

# Get the full terrain colormap
terrain = cm.get_cmap('terrain', 256)

# Buffer distance in degrees (approx. 20 km)
buffer_distance = 0.2

def extract_normalization_params(pkl_path):
    """Extract mean and std from pickle file for denormalization"""
    print(f"\n  Attempting to load normalization parameters from pickle file...")
    print(f"  Path: {pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  ✓ Pickle file loaded successfully")
        print(f"  Keys found: {list(data.keys())}")
        
        # Method 1: Check for stats dict with stored normalization params
        if 'stats' in data and isinstance(data['stats'], dict):
            if 'target_mean' in data['stats'] and 'target_std' in data['stats']:
                stored_mean = data['stats']['target_mean']
                stored_std = data['stats']['target_std']
                print(f"  ✓ Found stored normalization parameters in 'stats' dict:")
                print(f"    - Mean: {stored_mean:.6f}")
                print(f"    - Std:  {stored_std:.6f}")
        
        # Method 2: Extract actual targets from nested train_results and val_results
        train_targets = None
        val_targets = None
        
        if 'train_results' in data and isinstance(data['train_results'], dict):
            if 'targets' in data['train_results']:
                train_targets = data['train_results']['targets']
                print(f"  ✓ Found training targets in 'train_results': shape {train_targets.shape}")
        
        if 'val_results' in data and isinstance(data['val_results'], dict):
            if 'targets' in data['val_results']:
                val_targets = data['val_results']['targets']
                print(f"  ✓ Found validation targets in 'val_results': shape {val_targets.shape}")
        
        # Compute statistics from actual train+val targets
        if train_targets is not None or val_targets is not None:
            all_targets = []
            if train_targets is not None:
                all_targets.append(train_targets.flatten())
            if val_targets is not None:
                all_targets.append(val_targets.flatten())
            
            all_targets = np.concatenate(all_targets)
            mean_target = np.mean(all_targets)
            std_target = np.std(all_targets)
            
            print(f"\n  ✓✓✓ COMPUTED Normalization parameters from train+val targets:")
            print(f"    - Mean: {mean_target:.6f}")
            print(f"    - Std:  {std_target:.6f}")
            print(f"    - N samples: {len(all_targets):,}")
            
            # Compare with stored stats if available
            if 'stats' in data and 'target_mean' in data['stats']:
                stored_mean = data['stats']['target_mean']
                stored_std = data['stats']['target_std']
                print(f"\n  Comparison with stored stats:")
                print(f"    - Stored mean: {stored_mean:.6f} (diff: {abs(mean_target - stored_mean):.6f})")
                print(f"    - Stored std:  {stored_std:.6f} (diff: {abs(std_target - stored_std):.6f})")
            
            return mean_target, std_target
        
        print(f"\n  ⚠ WARNING: Could not extract normalization parameters from pickle file")
        return None, None
        
    except FileNotFoundError:
        print(f"  ⚠ WARNING: Pickle file not found at {pkl_path}")
        return None, None
    except Exception as e:
        print(f"  ⚠ WARNING: Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def rescale_predictions_to_target_distribution(predictions, target_mean, target_std, clip_min=2.5, clip_max=150):
    """
    Rescale predictions to match the target distribution.
    
    The predictions are not in standard normalized form - they have their own distribution.
    We need to:
    1. Standardize predictions (z-score: zero mean, unit variance)
    2. Scale to match target distribution
    3. Clip to [clip_min, clip_max] range
    """
    if target_mean is None or target_std is None:
        print(f"  ⚠ WARNING: No target statistics available, using predictions as-is")
        return np.clip(predictions, clip_min, clip_max)
    
    # Get prediction statistics
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    print(f"\n  ✓ Rescaling predictions to match target distribution:")
    print(f"    Prediction stats:  mean={pred_mean:.6f}, std={pred_std:.6f}")
    print(f"    Target stats:      mean={target_mean:.6f}, std={target_std:.6f}")
    
    # Step 1: Standardize predictions (z-score)
    standardized = (predictions - pred_mean) / pred_std
    print(f"    After standardization: mean={standardized.mean():.6f}, std={standardized.std():.6f}")
    
    # Step 2: Scale to match target distribution
    rescaled = standardized * target_std + target_mean
    print(f"    After rescaling:       mean={rescaled.mean():.6f}, std={rescaled.std():.6f}")
    print(f"    Range before clipping: [{rescaled.min():.2f}, {rescaled.max():.2f}]")
    
    # Step 3: Clip to [clip_min, clip_max]
    clipped = np.clip(rescaled, clip_min, clip_max)
    n_clipped_low = np.sum(rescaled < clip_min)
    n_clipped_high = np.sum(rescaled > clip_max)
    pct_clipped_low = 100.0 * n_clipped_low / len(rescaled)
    pct_clipped_high = 100.0 * n_clipped_high / len(rescaled)
    
    print(f"    After clipping to [{clip_min}, {clip_max}]:")
    print(f"      - Values clipped to {clip_min}: {n_clipped_low:,} ({pct_clipped_low:.2f}%)")
    print(f"      - Values clipped to {clip_max}: {n_clipped_high:,} ({pct_clipped_high:.2f}%)")
    print(f"      - Final range: [{clipped.min():.2f}, {clipped.max():.2f}]")
    print(f"      - Final mean: {clipped.mean():.2f}, std: {clipped.std():.2f}")
    
    return clipped

def load_and_validate_split_data(base_dir, file_pattern):
    """Load and concatenate split coordinate and prediction files with validation"""
    coordinates_list = []
    predictions_list = []
    
    print(f"\n  Loading split files from: {base_dir}")
    print(f"  {'='*70}")
    
    for coord_file, pred_file in file_pattern:
        coord_path = os.path.join(base_dir, coord_file)
        pred_path = os.path.join(base_dir, pred_file)
        
        if not os.path.exists(coord_path):
            print(f"  ⚠ WARNING: {coord_file} not found - SKIPPING")
            continue
        if not os.path.exists(pred_path):
            print(f"  ⚠ WARNING: {pred_file} not found - SKIPPING")
            continue
        
        coords = np.load(coord_path)
        preds = np.load(pred_path)
        
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            preds = preds.flatten()
        
        print(f"  ✓ {coord_file}")
        print(f"    - Coordinates: {coords.shape}")
        print(f"    - Predictions: {preds.shape}")
        
        if coords.shape[0] != preds.shape[0]:
            print(f"  ✗ ERROR: Shape mismatch! {coords.shape[0]} coords vs {preds.shape[0]} preds")
            print(f"    SKIPPING this pair")
            continue
        
        coordinates_list.append(coords)
        predictions_list.append(preds)
    
    if not coordinates_list or not predictions_list:
        raise ValueError("No valid coordinate-prediction pairs found!")
    
    coordinates = np.concatenate(coordinates_list, axis=0)
    predictions = np.concatenate(predictions_list, axis=0)
    
    print(f"\n  {'='*70}")
    print(f"  ✓ FINAL CONCATENATED SHAPES:")
    print(f"    - Coordinates: {coordinates.shape}")
    print(f"    - Predictions: {predictions.shape}")
    print(f"  {'='*70}")
    
    if coordinates.shape[0] != predictions.shape[0]:
        raise ValueError(f"Final shape mismatch! {coordinates.shape[0]} coords vs {predictions.shape[0]} preds")
    
    return coordinates, predictions

def interpolate_landcover(landcover_coords, landcover_vals, grid_x, grid_y):
    """Interpolate landcover values to the same grid as SOC predictions"""
    valid_mask = np.array([v is not None for v in landcover_vals])
    valid_coords = landcover_coords[valid_mask]
    valid_values = np.array([float(v) for v in landcover_vals[valid_mask]])
    landcover_grid = griddata(valid_coords, valid_values, (grid_x, grid_y), method='nearest')
    return landcover_grid

def create_sgt_maps():
    """Generate interpolated maps for SGT models with rescaled predictions (NO TITLES)
    
    Rescaling process:
    1. Standardize predictions: z = (pred - pred_mean) / pred_std
    2. Scale to target distribution: rescaled = z * target_std + target_mean
    3. Clip to [2.5, 150] g/kg range
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nLoading Bavaria boundaries...")
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']
    buffered_bavaria = bavaria.geometry.buffer(buffer_distance).unary_union

    # Set fixed scale (2.5-150 SOC g/kg) - ENFORCED FOR ALL MODELS
    global_min = 2.5
    global_max = 150
    print(f"\n{'='*80}")
    print(f"FIXED COLOR SCALE FOR ALL MODELS: [{global_min}, {global_max}] SOC g/kg")
    print(f"{'='*80}")

    # Process each SGT model
    for model_key, model_info in model_data.items():
        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {model_info['display_name']}")
            print(f"{'='*80}")
            
            # Load normalization parameters if available
            mean_target, std_target = None, None
            if model_info.get('normalization_pkl'):
                mean_target, std_target = extract_normalization_params(model_info['normalization_pkl'])
            
            # Load data based on type
            if model_info.get('type') == 'single':
                # Load single coordinate and prediction files
                print(f"\n  Loading single files...")
                coordinates = np.load(model_info['coordinates'])
                predictions = np.load(model_info['predictions'])
                
                # Flatten predictions if needed
                if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                    predictions = predictions.flatten()
                
                print(f"  ✓ Loaded single files")
                print(f"    - Coordinates: {coordinates.shape}")
                print(f"    - Predictions: {predictions.shape}")
                
                if coordinates.shape[0] != predictions.shape[0]:
                    raise ValueError(f"Shape mismatch! {coordinates.shape[0]} coords vs {predictions.shape[0]} preds")
            else:
                # Load split files (default/backward compatibility)
                coordinates, predictions = load_and_validate_split_data(
                    model_info['base_dir'],
                    model_info['file_pattern']
                )
            
            print(f"\n  Original prediction statistics (before rescaling):")
            print(f"    - Min: {predictions.min():.4f}")
            print(f"    - Max: {predictions.max():.4f}")
            print(f"    - Mean: {predictions.mean():.4f}")
            print(f"    - Std: {predictions.std():.4f}")
            
            # Rescale predictions to match target distribution and clip to [2.5, 150]
            predictions = rescale_predictions_to_target_distribution(
                predictions, mean_target, std_target, clip_min=2.5, clip_max=150
            )

            # Create interpolation grid
            print(f"\n  Creating interpolation grid...")
            grid_x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 300)
            grid_y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 300)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # Interpolate
            print(f"  Interpolating SOC values...")
            grid_z = griddata(coordinates, predictions, (grid_x, grid_y), method='linear')
            
            print(f"  Interpolating landcover values...")
            landcover_grid = interpolate_landcover(landcover_coordinates, landcover_values, grid_x, grid_y)

            # Create figure
            print(f"  Creating figure...")
            fig = plt.figure(figsize=(16, 12), dpi=300)
            ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])

            # Base contour plot - ENFORCED SCALE: 2.5-150 SOC g/kg
            contour = ax.contourf(grid_x, grid_y, grid_z,
                                  levels=50,
                                  cmap=terrain,
                                  alpha=0.8,
                                  vmin=global_min,
                                  vmax=global_max)
            print(f"  ✓ Contour plot created with vmin={global_min}, vmax={global_max}")

            # Overlay built-up areas (class 50) in black
            built_up_mask = landcover_grid == 50
            if np.any(built_up_mask):
                ax.contourf(grid_x, grid_y, built_up_mask.astype(float), 
                           levels=[0.5, 1.5], colors=['black'], alpha=0.9)

            # Overlay permanent water bodies (class 80) in deep blue
            water_mask = landcover_grid == 80
            if np.any(water_mask):
                ax.contourf(grid_x, grid_y, water_mask.astype(float), 
                           levels=[0.5, 1.5], colors=['#000080'], alpha=0.9)

            # Plot Bavaria boundaries
            bavaria.boundary.plot(ax=ax, color='black', linewidth=2.5)

            # NO TITLE - labels only
            ax.set_xlabel('Longitude', labelpad=15)
            ax.set_ylabel('Latitude', labelpad=15)
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.tick_params(axis='both', which='major', labelsize=20, pad=10, width=2.0, length=8)

            # Add colorbar - Scale: 2.5-150 SOC g/kg (ENFORCED)
            cax = fig.add_axes([0.78, 0.1, 0.04, 0.8])
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap=terrain),
                                cax=cax)
            cbar.set_label('SOC (g/kg)', fontsize=24, labelpad=18)
            cbar.ax.tick_params(labelsize=20, width=2.0, length=8, pad=10)
            cbar.set_ticks(np.linspace(global_min, global_max, 11))
            print(f"  ✓ Colorbar created with scale [{global_min}, {global_max}] SOC g/kg")

            # Add legend
            legend_elements = [
                mpatches.Patch(color='black', label='Built-up areas'),
                mpatches.Patch(color='#000080', label='Permanent water bodies')
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=18,
                     framealpha=0.9, edgecolor='black', fancybox=False)

            # Save figure
            filename = f"{PICTURE_VERSION}_{model_key}_MAX_OC_{MAX_OC}_Beginning_{TIME_BEGINNING}_End_{TIME_END}_InferenceTime{INFERENCE_TIME}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

            print(f"\n  ✓✓✓ Successfully created map: {filename}")

        except Exception as e:
            print(f"\n  ✗✗✗ ERROR processing {model_info['display_name']}: {e}")
            import traceback
            traceback.print_exc()

    # Create reference colorbar - SCALE: 2.5-150 SOC g/kg
    print(f"\n{'='*80}")
    print("Creating reference colorbar...")
    print(f"Scale: [{global_min}, {global_max}] SOC g/kg")
    fig, ax = plt.subplots(figsize=(12, 2), dpi=300)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(global_min, global_max), cmap=terrain),
                        cax=ax,
                        orientation='horizontal')
    cbar.set_label('SOC (g/kg)', fontsize=26, labelpad=18)
    cbar.ax.tick_params(labelsize=24, width=2.0, length=8, pad=10)
    cbar.set_ticks(np.linspace(global_min, global_max, 11))
    plt.title(f'Reference Color Scale ({global_min}-{global_max} g/kg SOC)', 
              fontsize=28, pad=20, weight='bold')
    plt.savefig(os.path.join(output_dir, f"reference_colorbar_{timestamp}.png"),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Reference colorbar saved")

    
if __name__ == "__main__":
    print("="*80)
    print("SPATIOTEMPORAL GATED TRANSFORMER MAPPING SCRIPT - RESCALED")
    print("Rescales predictions to match target distribution (2.5-150 g/kg)")
    print("="*80)
    create_sgt_maps()
    print(f"\n{'='*80}")
    print(f"✓✓✓ ALL MAPS SAVED TO: {output_dir}")
    print(f"{'='*80}")
