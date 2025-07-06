# CNN-LSTM Model for Soil Organic Carbon (SOC) Prediction in Bavaria

This repository contains a deep learning model that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for predicting and mapping soil organic carbon (SOC) content across Bavaria, Germany, using multi-temporal satellite and environmental data from 2007-2023.

## Overview

The **RefittedCovLSTM** model integrates:
- **CNN layers**: For spatial feature extraction from satellite imagery
- **LSTM layers**: For temporal sequence processing across multiple years
- **Fully connected layers**: For final SOC prediction

This architecture leverages both spatial patterns in satellite data and temporal dynamics to improve SOC prediction accuracy.

## Model Architecture

### RefittedCovLSTM Components

```python
RefittedCovLSTM(
    num_channels=6,          # Number of satellite bands
    lstm_input_size=128,     # LSTM input size (fixed by CNN output)
    lstm_hidden_size=128,    # LSTM hidden state size
    num_layers=2,            # Number of LSTM layers
    dropout=0.25             # Dropout rate
)
```

### Architecture Flow

1. **Input**: ```[batch, bands, height, width, time]```
2. **CNN Processing**: 
   - Conv2D layers with ReLU activation
   - MaxPool2D for spatial downsampling
   - Feature extraction per time step
3. **LSTM Processing**:
   - Temporal sequence modeling
   - Takes last time step output
4. **Feature Fusion**:
   - Combines CNN and LSTM features
   - Final prediction through fully connected layers
5. **Output**: Single SOC value per sample

### Technical Specifications

- **CUDA Compatibility**: Optimized for CUDA 11.8
- **Precision**: Float32 for optimal GPU performance
- **Spatial Reduction**: 4x downsampling through MaxPool operations
- **Temporal Modeling**: Bidirectional information flow through LSTM

## Dataset

- **Temporal Coverage**: 2007-2023 (17 years)
- **Spatial Coverage**: Bavaria, Germany
- **Data Source**: LUCAS soil survey data + satellite imagery
- **Input Dimensions**: Multi-band satellite imagery with temporal stacks
- **Target Variable**: Soil Organic Carbon (OC) content

## Key Features

### Advanced Training Pipeline

- **Balanced Sampling**: Quantile-based OC distribution balancing
- **Spatial Validation**: Distance-based train/validation splitting
- **Multi-run Experiments**: Statistical significance through multiple runs
- **Distributed Training**: Accelerate framework for multi-GPU support

### Loss Functions

- **Composite L1**: Combines L1 loss with scaled chi-squared loss
- **Composite L2**: Combines L2 loss with scaled chi-squared loss
- **Standard**: L1 Loss, Mean Squared Error (MSE)

### Target Transformations

- **Log Transform**: For handling skewed OC distributions
- **Normalization**: Z-score standardization
- **None**: Raw target values

## Usage

### Basic Training

```bash
python trainSimulations.py
```

### Advanced Configuration

```bash
python trainSimulations.py \
    --lr 0.0002 \
    --loss_type composite_l2 \
    --loss_alpha 0.5 \
    --target_transform normalize \
    --use_validation \
    --target-val-ratio 0.08 \
    --distance-threshold 1.2 \
    --num-runs 5
```

### Key Parameters

- ```--lr```: Learning rate (default: 0.0002)
- ```--loss_type```: Loss function type (```composite_l1```, ```composite_l2```, ```l1```, ```mse```)
- ```--loss_alpha```: Weight for L1/L2 loss in composite loss
- ```--target_transform```: Target transformation (```none```, ```log```, ```normalize```)
- ```--use_validation```: Enable validation split
- ```--target-val-ratio```: Validation set ratio
- ```--distance-threshold```: Minimum distance between train/val points
- ```--num-runs```: Number of training runs for statistical significance

## Model Performance

### Evaluation Metrics

- **R² (Correlation Squared)**: Measures linear relationship strength
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Standard deviation of prediction errors
- **RPIQ (Ratio of Performance to Interquartile Range)**: Model precision relative to data variability

### Experiment Tracking

The model uses **Weights & Biases (wandb)** for comprehensive experiment tracking:
- Real-time training metrics
- Model architecture logging
- Hyperparameter tracking
- Multi-run comparison
- Best model selection

We have obtained the following mapping:

![Mapping of SOC prediction using the CNNLSTM for the year 2023](../assets/AllModelsComparison/AllModelsComparison_CNNLSTM_MAX_OC_150_Beginning_2007_End_2023_InferenceTime2023_20250527_155525.png)

## Data Processing Pipeline

### Spatial-Temporal Data Handling

1. **Multi-Year Raster Loading**: Efficiently loads satellite imagery across years
2. **Spatial Filtering**: Geographic boundaries for Bavaria region
3. **Temporal Stacking**: Creates time series for each spatial location
4. **Balanced Sampling**: Ensures representative OC distribution
5. **Normalization**: Feature standardization across bands and time
6. **Spatial Validation**: Prevents data leakage through distance constraints

### Input Preprocessing

- **Feature Normalization**: Z-score standardization per band
- **Spatial Windowing**: Configurable window sizes around sample points
- **Temporal Alignment**: Consistent time series across all samples
- **Missing Data Handling**: Robust to incomplete temporal sequences

## File Structure

```
├── models.py                    # CNN-LSTM model architecture
├── trainSimulations.py          # Main training script
├── config.py                    # Configuration parameters
├── balancedDataset.py           # Dataset balancing utilities
├── dataloader/
│   ├── dataloaderMultiYears.py  # Multi-year raster data loader
│   └── dataframe_loader.py      # DataFrame processing utilities
├── results/                     # Training results and metrics
└── README.md                    # This file
```

## Advanced Features

### Composite Loss Functions

```python
def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Combines L2 loss with scaled chi-squared loss for robust training"""
    errors = targets - outputs
    l2_loss = torch.mean(errors ** 2)
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))
    return alpha * l2_loss + (1 - alpha) * chi2_scaled
```

### Spatial Validation Strategy

- **Distance-based splitting**: Ensures spatial independence between train/validation
- **Configurable thresholds**: Adjustable minimum distance requirements
- **Balanced representation**: Maintains OC distribution across splits

### Multi-Run Statistical Analysis

- **Automatic model selection**: Saves best performing model across runs
- **Comprehensive metrics**: Aggregated statistics across multiple experiments
- **Reproducibility**: Consistent random seeding and configuration tracking

## Model Outputs

### Saved Artifacts

1. **Best Model**: PyTorch state dict with highest validation R²
2. **Training Metrics**: Comprehensive performance logs
3. **Normalization Stats**: Feature scaling parameters for inference
4. **Validation Data**: Train/validation splits for reproducibility

### Performance Monitoring

- **Real-time visualization**: Live training curves via wandb
- **Multi-metric evaluation**: R², MAE, RMSE, RPIQ tracking
- **Early stopping**: Automatic termination on performance plateaus
- **Best model checkpointing**: Saves optimal weights during training

## Future Improvements

1. **Attention Mechanisms**: Incorporate spatial and temporal attention
2. **Multi-scale Features**: Hierarchical feature extraction
3. **Ensemble Methods**: Combine multiple model predictions
4. **Transfer Learning**: Pre-trained feature extractors
5. **Uncertainty Quantification**: Bayesian neural networks for prediction intervals
