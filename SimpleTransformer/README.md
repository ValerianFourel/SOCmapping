
# Transformer Model for Soil Organic Carbon (SOC) Prediction in Bavaria

This repository contains transformer-based deep learning models for predicting and mapping soil organic carbon (SOC) content across Bavaria, Germany, using multi-temporal satellite and environmental data from 2007-2023.

## Overview

We implement two transformer architectures for SOC prediction:
- **SimpleTransformer**: Lightweight transformer with basic encoder layers
- **SimpleTransformerV2**: Advanced transformer with positional encoding, pre-normalization, and enhanced capacity

Both models leverage the self-attention mechanism to capture complex temporal relationships in multi-year satellite imagery for improved SOC prediction accuracy.

We obtained the 2 mappings of SOC predictions over Bavaria for the year 2023:

![Mapping of SOC prediction using the 20k Transformer for the year 2023](../assets/AllModelsComparison/AllModelsComparison_20kTransformer_MAX_OC_150_Beginning_2007_End_2023_InferenceTime2023_20250527_155525.png)

*Mapping of SOC prediction using the 20k Transformer (2007-2023 dataset, inference time: 2023)*


![Mapping of SOC prediction using the 2.4mil Transformer for the year 2023](../assets/AllModelsComparison/AllModelsComparison_2milTransformer_MAX_OC_150_Beginning_2007_End_2023_InferenceTime2023_20250527_155525.png)

*Mapping of SOC prediction using the 2.4mil Transformer (2007-2023 dataset, inference time: 2023)*

## Model Architectures

### SimpleTransformer (Lightweight)

```python
SimpleTransformer(
    input_channels=6,         # Number of satellite bands
    input_height=33,          # Spatial height dimension
    input_width=33,           # Spatial width dimension  
    input_time=4,             # Temporal sequence length
    num_heads=2,              # Number of attention heads
    num_layers=1,             # Number of transformer layers
    dropout_rate=0.3          # Dropout rate
)
```

**Key Features:**
- Basic transformer encoder with reduced complexity
- Linear input projection to match d_model requirements
- Minimal feedforward dimensions (32)
- Suitable for quick experiments and baseline comparisons

### SimpleTransformerV2 (Advanced)

```python
SimpleTransformerV2(
    input_channels=6,         # Number of satellite bands
    input_height=33,          # Spatial height dimension
    input_width=33,           # Spatial width dimension
    input_time=4,             # Temporal sequence length
    num_heads=16,             # Number of attention heads
    num_layers=6,             # Number of transformer layers
    dropout_rate=0.3          # Dropout rate
)
```

**Advanced Features:**
- **Positional Encoding**: Learnable positional embeddings for temporal awareness
- **Pre-Normalization**: Layer normalization before attention for training stability
- **Large Feedforward Networks**: 4x expansion ratio for increased capacity
- **Multi-Head Attention**: Up to 16 attention heads for complex pattern recognition
- **Deep Architecture**: Up to 6 transformer layers for hierarchical feature learning

## Architecture Flow

### Data Processing Pipeline

1. **Input Reshaping**: ```[batch, bands, height, width, time]``` → ```[batch, time, features]```
2. **Linear Projection**: Map flattened spatial features to d_model dimension
3. **Positional Encoding** (V2 only): Add learnable temporal position embeddings
4. **Transformer Processing**: Multi-head self-attention and feedforward layers
5. **Temporal Aggregation**: Process entire sequence and extract final representation
6. **Prediction Head**: Multi-layer perceptron for SOC value regression

### Self-Attention Mechanism

The transformer models capture temporal dependencies through:
- **Multi-head attention**: Parallel attention mechanisms for different representation subspaces
- **Temporal modeling**: Direct relationships between different time steps
- **Global context**: Each time step can attend to all other time steps
- **Parameter sharing**: Consistent temporal processing across sequences

## Dataset

- **Temporal Coverage**: 2007-2023 (17 years)
- **Spatial Coverage**: Bavaria, Germany
- **Data Source**: LUCAS soil survey data + multi-temporal satellite imagery
- **Input Format**: ```[batch_size, bands, height, width, time_steps]```
- **Target Variable**: Soil Organic Carbon (OC) content

## Optimal Configuration

Based on extensive experiments, the best performing configuration is:
- **Model**: SimpleTransformerV2
- **Loss Function**: ```composite_l2``` (combines L2 and scaled chi-squared loss)
- **Target Transform**: ```log``` (log transformation of OC values)
- **Learning Rate**: 0.0002
- **Loss Alpha**: 0.5 (equal weighting of L2 and chi-squared components)

## Usage

### Basic Training

```bash
python trainSimulations.py
```

### Optimal Configuration Training

```bash
python trainSimulations.py \
    --lr 0.0002 \
    --num_heads 16 \
    --num_layers 6 \
    --loss_type composite_l2 \
    --loss_alpha 0.5 \
    --target_transform log \
    --use_validation \
    --num-runs 4
```

### Advanced Configuration

```bash
python trainSimulations.py \
    --lr 0.0002 \
    --num_heads 8 \
    --num_layers 4 \
    --loss_type composite_l1 \
    --loss_alpha 0.3 \
    --target_transform normalize \
    --target-val-ratio 0.08 \
    --distance-threshold 1.2 \
    --num-runs 5 \
    --save_train_and_val True
```

### Key Parameters

- ```--lr```: Learning rate (default: 0.0002)
- ```--num_heads```: Number of attention heads (default: from config)
- ```--num_layers```: Number of transformer layers (default: from config)
- ```--loss_type```: Loss function (```composite_l1```, ```composite_l2```, ```l1```, ```mse```)
- ```--loss_alpha```: Weight for primary loss in composite functions
- ```--target_transform```: Target preprocessing (```none```, ```log```, ```normalize```)
- ```--use_validation```: Enable spatial validation split
- ```--distance-threshold```: Minimum distance between train/validation points
- ```--num-runs```: Number of independent training runs
- ```--save_train_and_val```: Save datasets for residual analysis

## Advanced Features

### Composite Loss Functions

#### Composite L2-Chi² Loss (Recommended)
```python
def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Combines L2 loss with scaled chi-squared loss"""
    errors = targets - outputs
    l2_loss = torch.mean(errors ** 2)
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))
    scale_factor = l2_loss / chi2_loss
    return alpha * l2_loss + (1 - alpha) * scale_factor * chi2_loss
```

**Benefits:**
- Robust to outliers through chi-squared component
- Maintains MSE properties through L2 component
- Adaptive scaling for balanced loss contribution

### Target Transformations

#### Log Transformation (Recommended)
- Handles skewed OC distributions effectively
- Stabilizes variance across OC ranges
- Improves model convergence for soil data

#### Normalization
- Z-score standardization of targets
- Consistent scaling for optimization
- Preserves relative relationships

### Spatial Validation Strategy

- **Distance-based splitting**: Ensures spatial independence
- **Configurable thresholds**: Adjustable minimum distances
- **Balanced sampling**: Maintains OC distribution across splits
- **Statistical validation**: Multiple runs for robust evaluation

## Model Comparison

### Parameter Counts

| Model | Parameters | Capacity | Use Case |
|-------|------------|----------|----------|
| SimpleTransformer | ~10K-100K | Low | Quick experiments, baselines |
| SimpleTransformerV2 | ~1M-10M | High | Production, complex patterns |

### Attention Patterns

- **Lightweight**: Simple temporal relationships, basic feature interactions
- **Advanced**: Complex multi-scale temporal dependencies, rich feature representations

## Performance Monitoring

### Evaluation Metrics

- **R² (Correlation Squared)**: Measures predictive accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **RPIQ (Ratio of Performance to Interquartile Range)**: Precision relative to data spread

### Experiment Tracking

- **Weights & Biases Integration**: Real-time metrics and visualization
- **Multi-run Statistics**: Aggregated performance across experiments
- **Best Model Selection**: Automatic identification of optimal weights
- **Comprehensive Logging**: Architecture, hyperparameters, and results

## File Structure

```
├── modelSimpleTransformer.py        # SimpleTransformer architecture
├── modelSimpleTransformerNew.py     # SimpleTransformerV2 architecture  
├── trainSimulations.py              # Main training script
├── config.py                        # Model and training configuration
├── balancedDataset.py               # Dataset balancing utilities
├── dataloader/
│   ├── dataloaderMultiYears.py      # Multi-temporal data loading
│   └── dataframe_loader.py          # DataFrame processing
├── results/                         # Training outputs and metrics
└── README.md                        # This file
```

## Transformer Advantages for SOC Prediction

### Self-Attention Benefits

1. **Long-range Dependencies**: Direct connections between distant time steps
2. **Parallel Processing**: Efficient computation compared to RNNs
3. **Interpretability**: Attention weights reveal temporal importance
4. **Scalability**: Consistent performance with longer sequences

### Spatial-Temporal Modeling

1. **Global Context**: Each spatial location can attend to its full temporal history
2. **Dynamic Relationships**: Adaptive temporal weighting based on content
3. **Multi-scale Features**: Hierarchical attention across multiple layers
4. **Robust Representations**: Less sensitive to missing or noisy time steps

## Best Practices

### Training Recommendations

1. **Use SimpleTransformerV2** for production applications
2. **Apply log transformation** for skewed OC distributions  
3. **Use composite_l2 loss** for robust training
4. **Set learning rate to 0.0002** for stable convergence
5. **Run multiple experiments** for statistical significance

### Hyperparameter Tuning

- **Start with 4-6 layers** for most applications
- **Use 8-16 attention heads** for complex patterns
- **Apply 0.3 dropout rate** to prevent overfitting
- **Balance loss components** with alpha=0.5

## Future Improvements

1. **Vision Transformer Integration**: Spatial attention mechanisms
2. **Multi-scale Temporal Modeling**: Hierarchical time representations  
3. **Uncertainty Quantification**: Bayesian transformer variants
4. **Transfer Learning**: Pre-trained models for soil applications
5. **Ensemble Methods**: Combine multiple transformer predictions

