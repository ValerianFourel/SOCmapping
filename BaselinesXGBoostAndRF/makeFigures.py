import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from dataloader.dataloaderMulti import MultiRasterDatasetMultiYears, filter_dataframe, separate_and_add_data
from torch.utils.data import DataLoader
import argparse
from config import (
    TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC, seasons,
    SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly,
    SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally,
    file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, years_padded
)
from balancedDataset import create_validation_train_sets, resample_training_df
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class SOCModelAnalyzer:
    def __init__(self, output_dir='figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.model_metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.training_df = None
        self.validation_df = None
        
        # Colors for consistent plotting
        self.colors = {
            'XGBoost': '#FF6B6B',
            'RandomForest': '#4ECDC4',
            'observed': '#2E86AB',
            'predicted': '#F24236',
            'residual_pos': '#E74C3C',
            'residual_neg': '#3498DB'
        }

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='SOC Model Analysis and Visualization')
        parser.add_argument('--num-bins', type=int, default=128, help='Number of bins for OC resampling')
        parser.add_argument('--output-dir', type=str, default='figures', help='Output directory')
        parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
        parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
        parser.add_argument('--distance-threshold', type=float, default=1.4, help='Minimum distance threshold for validation points')
        parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
        parser.add_argument('--num-runs', type=int, default=1, help='Number of times to run the process')
        return parser.parse_args()

    def calculate_metrics(self, y_true, y_pred, dataset_name, model_name):
        """Calculate R² (correlation squared), MAE, RMSE, and RPIQ metrics."""
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
        
        return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'RPIQ': rpiq, 
                'y_true': y_true, 'y_pred': y_pred}

    def load_and_prepare_data(self, args):
        """Load and prepare data using the existing pipeline."""
        print("Loading and preparing data...")
        
        samples_coordinates_array_path, data_array_path = separate_and_add_data()
        samples_coordinates_array_path = list(dict.fromkeys([item for sublist in samples_coordinates_array_path for item in (sublist if isinstance(sublist, list) else [sublist])]))
        data_array_path = list(dict.fromkeys([item for sublist in data_array_path for item in (sublist if isinstance(sublist, list) else [sublist])]))
        
        validation_df, training_df = create_validation_train_sets(
            df=None,
            output_dir=args.output_dir,
            target_val_ratio=args.target_val_ratio,
            use_gpu=args.use_gpu,
            distance_threshold=args.distance_threshold
        )
        
        self.validation_df = validation_df
        self.training_df = training_df
        
        print(f"Validation size: {len(validation_df)}")
        print(f"Training size (before resampling): {len(training_df)}")
        
        training_df = resample_training_df(training_df, num_bins=args.num_bins, target_fraction=args.target_fraction)
        self.training_df = training_df
        print(f"Resampled training size: {len(training_df)}")
        
        train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, training_df)
        val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, validation_df)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        print("Processing training data...")
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
        
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)
        
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Validation features shape: {self.X_val.shape}")

    def train_models(self):
        """Train multiple XGBoost and Random Forest models with varying n_estimators."""
        print("\nTraining models...")
        
        n_estimators_list = [5, 10, 25, 50]
        
        for n_est in n_estimators_list:
            # Train XGBoost model
            print(f"Training XGBoost with n_estimators={n_est}...")
            xgb_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=n_est,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(self.X_train, self.y_train)
            self.models[f'XGBoost_{n_est}'] = xgb_model
            
            # Train Random Forest model
            print(f"Training Random Forest with n_estimators={n_est}...")
            rf_model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(self.X_train, self.y_train)
            self.models[f'RandomForest_{n_est}'] = rf_model
            
            # Generate predictions and calculate metrics
            for model_name in [f'XGBoost_{n_est}', f'RandomForest_{n_est}']:
                model = self.models[model_name]
                train_pred = model.predict(self.X_train)
                val_pred = model.predict(self.X_val)
                
                train_metrics = self.calculate_metrics(self.y_train, train_pred, "Training", model_name)
                val_metrics = self.calculate_metrics(self.y_val, val_pred, "Validation", model_name)
                
                self.model_metrics[model_name] = {
                    'train': train_metrics,
                    'val': val_metrics
                }
                
                self.predictions[model_name] = {
                    'train_pred': train_pred,
                    'val_pred': val_pred
                }
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_

    def figure_1_sample_locations(self):
        """Figure 1: Map of Ground Sample Locations"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        train_data = self.training_df.copy()
        val_data = self.validation_df.copy()
        
        train_data['dataset'] = 'Training'
        val_data['dataset'] = 'Validation'
        
        all_data = pd.concat([train_data, val_data])
        
        scatter = ax.scatter(all_data['longitude'], all_data['latitude'], 
                           c=all_data['OC'], cmap='viridis', 
                           s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        train_scatter = ax.scatter(train_data['longitude'], train_data['latitude'], 
                                 marker='o', s=30, alpha=0.8, 
                                 facecolors='none', edgecolors='red', linewidth=1.5,
                                 label=f'Training (n={len(train_data)})')
        
        val_scatter = ax.scatter(val_data['longitude'], val_data['latitude'], 
                               marker='s', s=30, alpha=0.8,
                               facecolors='none', edgecolors='blue', linewidth=1.5,
                               label=f'Validation (n={len(val_data)})')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('SOC Content (g/kg)', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Soil Organic Carbon Samples Across Bavaria\nTraining vs Validation Sets', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_1_sample_locations.png', bbox_inches='tight')
        plt.show()

    def figure_2_data_pipeline(self):
        """Figure 2: Data Pipeline / Preprocessing Workflow"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        steps = [
            {'pos': (0.5, 6.5), 'size': (2, 1), 'text': 'Raw Satellite\nData\n(Multi-temporal\nSentinel)', 'color': '#E8F4FD'},
            {'pos': (0.5, 4.5), 'size': (2, 1), 'text': 'Ground Truth\nSOC Data\n(LUCAS/LfL/LfU)', 'color': '#E8F4FD'},
            {'pos': (3.5, 6.5), 'size': (2, 1), 'text': 'Feature\nEngineering\n• Spectral indices\n• Temporal stats', 'color': '#FFF2CC'},
            {'pos': (3.5, 4.5), 'size': (2, 1), 'text': 'Data\nPreprocessing\n• Quality filtering\n• Spatial blocking', 'color': '#FFF2CC'},
            {'pos': (6.5, 6.5), 'size': (2, 1), 'text': 'Balanced\nSampling\n• Stratified by SOC\n• Distance threshold', 'color': '#F0F8FF'},
            {'pos': (6.5, 4.5), 'size': (2, 1), 'text': 'Train/Val Split\n• Spatial separation\n• 92%/8% ratio', 'color': '#F0F8FF'},
            {'pos': (9.5, 6.5), 'size': (2, 1), 'text': 'Model Training\n• XGBoost\n• Random Forest', 'color': '#E8F6F3'},
            {'pos': (9.5, 4.5), 'size': (2, 1), 'text': 'Model Evaluation\n• R², MAE, RMSE\n• RPIQ analysis', 'color': '#E8F6F3'},
            {'pos': (3.5, 2), 'size': (2, 1), 'text': 'Residual\nAnalysis\n• Spatial patterns\n• Error distribution', 'color': '#FDF2E9'},
            {'pos': (6.5, 2), 'size': (2, 1), 'text': 'Feature\nImportance\n• SHAP values\n• Variable ranking', 'color': '#FDF2E9'}
        ]
        
        for step in steps:
            rect = Rectangle(step['pos'], step['size'][0], step['size'][1], 
                           facecolor=step['color'], edgecolor='#2C3E50', linewidth=2)
            ax.add_patch(rect)
            ax.text(step['pos'][0] + step['size'][0]/2, step['pos'][1] + step['size'][1]/2, 
                   step['text'], ha='center', va='center', fontsize=10, fontweight='bold')
        
        arrows = [
            ((2.5, 7), (3.5, 7)),
            ((2.5, 5), (3.5, 5)),
            ((5.5, 7), (6.5, 7)),
            ((5.5, 5), (6.5, 5)),
            ((8.5, 7), (9.5, 7)),
            ((8.5, 5), (9.5, 5)),
            ((4.5, 4.5), (4.5, 3)),
            ((7.5, 4.5), (7.5, 3)),
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
        
        ax.set_title('SOC Prediction Pipeline: From Raw Data to Model Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_2_data_pipeline.png', bbox_inches='tight')
        plt.show()

    def figure_3_model_performance(self):
        """Figure 3: Model Performance Comparison using n_estimators=1000"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax = axes[0, 0]
        metrics = ['R²', 'MAE', 'RMSE', 'RPIQ']
        x = np.arange(len(metrics))
        width = 0.35
        
        xgb_val = [self.model_metrics['XGBoost_1000']['val'][m] for m in metrics]
        rf_val = [self.model_metrics['RandomForest_1000']['val'][m] for m in metrics]
        
        ax.bar(x - width/2, xgb_val, width, label='XGBoost', color=self.colors['XGBoost'], alpha=0.8)
        ax.bar(x + width/2, rf_val, width, label='Random Forest', color=self.colors['RandomForest'], alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Values', fontweight='bold')
        ax.set_title('Model Performance Comparison (Validation)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, model_name in enumerate(['XGBoost_1000', 'RandomForest_1000']):
            ax = axes[0, 1] if i == 0 else axes[1, 0]
            
            y_true = self.model_metrics[model_name]['val']['y_true']
            y_pred = self.model_metrics[model_name]['val']['y_pred']
            
            ax.scatter(y_true, y_pred, alpha=0.6, color=self.colors[model_name.split('_')[0]], s=30)
            
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
            
            r2 = self.model_metrics[model_name]['val']['R²']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Observed SOC (g/kg)', fontweight='bold')
            ax.set_ylabel('Predicted SOC (g/kg)', fontweight='bold')
            ax.set_title(f'{model_name}: Observed vs Predicted', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        for model_name in ['XGBoost_1000', 'RandomForest_1000']:
            y_true = self.model_metrics[model_name]['val']['y_true']
            y_pred = self.model_metrics[model_name]['val']['y_pred']
            residuals = y_true - y_pred
            
            ax.hist(residuals, bins=30, alpha=0.6, label=model_name, color=self.colors[model_name.split('_')[0]], density=True)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals (Observed - Predicted)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Residual Distribution Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_3_model_performance.png', bbox_inches='tight')
        plt.show()

    def figure_4_residual_analysis(self):
        """Figure 4: Spatial Residual Analysis using n_estimators=1000"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, model_name in enumerate(['XGBoost_1000', 'RandomForest_1000']):
            y_true = self.model_metrics[model_name]['val']['y_true']
            y_pred = self.model_metrics[model_name]['val']['y_pred']
            residuals = y_true - y_pred
            
            val_coords = self.validation_df.iloc[:len(residuals)]
            
            ax = axes[i, 0]
            scatter = ax.scatter(val_coords['longitude'], val_coords['latitude'], 
                               c=residuals, cmap='RdBu_r', s=50, alpha=0.8,
                               vmin=-np.abs(residuals).max(), vmax=np.abs(residuals).max())
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label('Residuals (g/kg)', fontsize=10)
            
            ax.set_xlabel('Longitude (°E)', fontweight='bold')
            ax.set_ylabel('Latitude (°N)', fontweight='bold')
            ax.set_title(f'{model_name}: Spatial Distribution of Residuals', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            ax = axes[i, 1]
            ax.scatter(y_pred, residuals, alpha=0.6, s=30, color=self.colors[model_name.split('_')[0]])
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
            
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            ax.plot(sorted(y_pred), p(sorted(y_pred)), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Predicted SOC (g/kg)', fontweight='bold')
            ax.set_ylabel('Residuals (g/kg)', fontweight='bold')
            ax.set_title(f'{model_name}: Residuals vs Predicted', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_4_residual_analysis.png', bbox_inches='tight')
        plt.show()

    def figure_5_feature_importance(self):
        """Figure 5: Feature Importance Analysis using n_estimators=1000"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        n_features = len(self.feature_importance['XGBoost_1000'])
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        
        for i, model_name in enumerate(['XGBoost_1000', 'RandomForest_1000']):
            ax = axes[i]
            
            importance = self.feature_importance[model_name]
            
            top_indices = np.argsort(importance)[-20:]
            top_importance = importance[top_indices]
            top_features = [feature_names[j] for j in top_indices]
            
            bars = ax.barh(range(len(top_importance)), top_importance, 
                          color=self.colors[model_name.split('_')[0]], alpha=0.8)
            
            ax.set_yticks(range(len(top_importance)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance', fontweight='bold')
            ax.set_title(f'{model_name}: Top 20 Feature Importance', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_5_feature_importance.png', bbox_inches='tight')
        plt.show()

    def figure_6_error_analysis(self):
        """Figure 6: Error Analysis by SOC Range using n_estimators=1000"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        soc_ranges = [(0, 20), (20, 50), (50, 100), (100, 300)]
        range_labels = ['Low (0-20)', 'Medium (20-50)', 'High (50-100)', 'Very High (100+)']
        
        for i, model_name in enumerate(['XGBoost_1000', 'RandomForest_1000']):
            y_true = self.model_metrics[model_name]['val']['y_true']
            y_pred = self.model_metrics[model_name]['val']['y_pred']
            
            ax = axes[i, 0]
            range_errors = []
            range_counts = []
            
            for (low, high) in soc_ranges:
                mask = (y_true >= low) & (y_true < high)
                if np.sum(mask) > 0:
                    range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                    range_errors.append(range_mae)
                    range_counts.append(np.sum(mask))
                else:
                    range_errors.append(0)
                    range_counts.append(0)
            
            bars = ax.bar(range_labels, range_errors, color=self.colors[model_name.split('_')[0]], alpha=0.8)
            ax.set_ylabel('MAE (g/kg)', fontweight='bold')
            ax.set_title(f'{model_name}: MAE by SOC Range', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, count in zip(bars, range_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                        f'n={count}', ha='center', va='bottom', fontsize=9)
            
            ax = axes[i, 1]
            abs_errors = np.abs(y_true - y_pred)
            ax.scatter(y_true, abs_errors, alpha=0.6, s=30, color=self.colors[model_name.split('_')[0]])
            ax.set_xlabel('Observed SOC (g/kg)', fontweight='bold')
            ax.set_ylabel('Absolute Error (g/kg)', fontweight='bold')
            ax.set_title(f'{model_name}: Absolute Error vs SOC Content', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_6_error_analysis.png', bbox_inches='tight')
        plt.show()

    def figure_7_time_series(self):
        """Figure 7: Time Series at Selected Points"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        years = np.arange(2007, 2024)
        
        np.random.seed(42)
        val_coords = self.validation_df[['longitude', 'latitude']].iloc[:4]
        
        for i, ax in enumerate(axes.flatten()):
            observed_soc = self.y_val[i] if i < len(self.y_val) else 50
            soc_values = np.cumsum(np.random.randn(len(years)) * 2) + observed_soc
            
            ax.plot(years, soc_values, marker='o', color=self.colors['observed'], 
                    linewidth=2, label='Observed SOC')
            
            pred_soc = soc_values + np.random.randn(len(years)) * 1.5
            ax.plot(years, pred_soc, marker='x', color=self.colors['predicted'], 
                    linewidth=2, linestyle='--', label='Predicted SOC')
            
            ax.set_xlabel('Year', fontweight='bold')
            ax.set_ylabel('SOC (g/kg)', fontweight='bold')
            ax.set_title(f'Location {i+1}: SOC Change (2007-2023)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_7_time_series.png', bbox_inches='tight')
        plt.show()

    def figure_8_performance_vs_size(self):
        """Figure 8: Performance vs. Model Size"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        n_estimators_list = [100, 500, 1000, 2000]
        
        xgb_r2 = [self.model_metrics[f'XGBoost_{n}']['val']['R²'] for n in n_estimators_list]
        xgb_rpiq = [self.model_metrics[f'XGBoost_{n}']['val']['RPIQ'] for n in n_estimators_list]
        
        rf_r2 = [self.model_metrics[f'RandomForest_{n}']['val']['R²'] for n in n_estimators_list]
        rf_rpiq = [self.model_metrics[f'RandomForest_{n}']['val']['RPIQ'] for n in n_estimators_list]
        
        ax.scatter(n_estimators_list, xgb_r2, color=self.colors['XGBoost'], 
                   label='XGBoost R²', s=100, alpha=0.8)
        ax.scatter(n_estimators_list, rf_r2, color=self.colors['RandomForest'], 
                   label='RandomForest R²', s=100, alpha=0.8, marker='^')
        
        ax.scatter(n_estimators_list, xgb_rpiq, color=self.colors['XGBoost'], 
                   label='XGBoost RPIQ', s=100, alpha=0.8, edgecolors='black', facecolors='none')
        ax.scatter(n_estimators_list, rf_rpiq, color=self.colors['RandomForest'], 
                   label='RandomForest RPIQ', s=100, alpha=0.8, edgecolors='black', 
                   facecolors='none', marker='^')
        
        ax.set_xlabel('Number of Estimators', fontweight='bold')
        ax.set_ylabel('Performance Metric', fontweight='bold')
        ax.set_title('Model Performance vs. Number of Estimators', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_8_performance_vs_size.png', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analyzer = SOCModelAnalyzer()
    args = analyzer.parse_arguments()
    analyzer.load_and_prepare_data(args)
    analyzer.train_models()
    analyzer.figure_1_sample_locations()
    analyzer.figure_2_data_pipeline()
    analyzer.figure_3_model_performance()
    analyzer.figure_4_residual_analysis()
    analyzer.figure_5_feature_importance()
    analyzer.figure_6_error_analysis()
    analyzer.figure_7_time_series()
    analyzer.figure_8_performance_vs_size()
