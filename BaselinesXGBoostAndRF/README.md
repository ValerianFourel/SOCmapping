<details><summary>Thinking</summary>

The user wants me to remove several sections from the README.md I created:
1. Installation section
2. Citations section
3. License section
4. Contact section
5. The specific installation code blocks with git clone and pip install
6. The Required Dependencies section

They also want me to change the training file name from "train_baselines.py" to "trainSimulations.py"

Let me create a cleaned up version of the README.md with these changes.

</details>

# Baseline Models for Soil Organic Carbon (SOC) Prediction in Bavaria

This repository contains baseline machine learning models for predicting and mapping soil organic carbon (SOC) content across Bavaria, Germany, using multi-temporal satellite and environmental data from 2007-2023.

## Overview

We implement two baseline regression models:
- **XGBoost Regressor**: Gradient boosting ensemble method
- **Random Forest Regressor**: Bagging ensemble method

Both models are trained on multi-year raster datasets with balanced sampling strategies to predict SOC content at various spatial resolutions.

## Models Performance

### Performance Summary Across Different Raster Sizes

| Raster Size | Resolution | XGBoost Val R² | XGBoost Val RMSE | RF Val R² | RF Val RMSE |
|-------------|------------|----------------|------------------|-----------|-------------|
| 5×5         | 1.25km     | 0.277 ± 0.041  | 13.58 ± 0.43     | 0.272 ± 0.051 | 11.45 ± 0.36 |
| 11×11       | 2.75km     | 0.258 ± 0.027  | 13.68 ± 0.68     | 0.258 ± 0.030 | 11.52 ± 0.32 |
| 21×21       | 5.25km     | 0.281 ± 0.025  | 12.83 ± 0.41     | 0.302 ± 0.027 | 10.64 ± 0.44 |
| 31×31       | 7.75km     | 0.264 ± 0.048  | 13.24 ± 0.38     | 0.265 ± 0.045 | 11.23 ± 0.40 |

### Key Findings

- **Training Performance**: Both models show excellent training performance (R² > 0.8) but suffer from overfitting
- **Validation Performance**: Random Forest generally outperforms XGBoost on validation data
- **Optimal Resolution**: 21×21 raster size (5.25km resolution) shows the best validation performance
- **Robustness**: Random Forest demonstrates better generalization with lower validation RMSE

## Dataset

- **Temporal Coverage**: 2007-2023 (17 years)
- **Spatial Coverage**: Bavaria, Germany
- **Data Source**: LUCAS (Land Use and Coverage Area frame Survey) soil data
- **Target Variable**: Soil Organic Carbon (OC) content
- **Features**: Multi-temporal satellite imagery and environmental variables

## Usage

### Basic Training

```bash
python trainSimulations.py
```

### Advanced Configuration

```bash
python trainSimulations.py \
    --num-bins 128 \
    --output-dir results \
    --target-val-ratio 0.08 \
    --distance-threshold 1.4 \
    --target-fraction 0.75 \
    --num-runs 5
```

### Parameters

- ```--num-bins```: Number of bins for OC value balancing (default: 128)
- ```--output-dir```: Directory for saving results (default: 'output')
- ```--target-val-ratio```: Validation set ratio (default: 0.08)
- ```--distance-threshold```: Minimum distance between training/validation points (default: 1.4)
- ```--target-fraction```: Fraction of max bin count for resampling (default: 0.75)
- ```--num-runs```: Number of training runs for statistical significance (default: 5)

## Model Architecture

### XGBoost Configuration
```python
xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
```

### Random Forest Configuration
```python
RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

## Evaluation Metrics

We evaluate models using four key metrics:

1. **R² (Correlation Squared)**: Measures linear relationship strength
2. **MAE (Mean Absolute Error)**: Average absolute prediction error
3. **RMSE (Root Mean Square Error)**: Standard deviation of prediction errors
4. **RPIQ (Ratio of Performance to Interquartile Range)**: Model precision relative to data variability

## Data Processing Pipeline

1. **Data Loading**: Multi-year raster data loading using custom DataLoader
2. **Spatial Filtering**: Geographic and temporal filtering for Bavaria region
3. **Balanced Sampling**: Quantile-based binning for balanced OC distribution
4. **Validation Split**: Spatially-aware train/validation splitting
5. **Feature Engineering**: Raster data flattening and preprocessing

## File Structure

```
├── trainSimulations.py         # Main training script
├── config.py                   # Configuration parameters
├── balancedDataset.py          # Dataset balancing utilities
├── dataloader/
│   ├── dataloaderMulti.py      # Multi-year raster data loader
│   └── dataframe_loader.py     # DataFrame processing utilities
├── results/                    # Training results and metrics
└── README.md                   # This file
```

## Results Interpretation

### Training vs Validation Performance Gap

The significant gap between training and validation performance indicates:
- **Overfitting**: Models memorize training patterns rather than generalizing
- **Spatial Autocorrelation**: SOC values are spatially correlated, making validation harder
- **Need for Regularization**: More sophisticated regularization techniques may be needed

### Model Comparison

- **XGBoost**: Higher training performance but worse generalization
- **Random Forest**: Better balance between training and validation performance
- **Recommendation**: Random Forest is preferred for operational SOC mapping

## Future Improvements

1. **Advanced Regularization**: Implement spatial regularization techniques
2. **Feature Engineering**: Add soil type, climate, and topographic features
3. **Deep Learning**: Compare with CNN-based approaches
4. **Ensemble Methods**: Combine predictions from multiple models
5. **Cross-Validation**: Implement spatial cross-validation strategies

