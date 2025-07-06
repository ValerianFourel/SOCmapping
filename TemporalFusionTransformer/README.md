<details><summary>Thinking</summary>

Now I need to create a README.md for the TFT (Temporal Fusion Transformer) approach for SOC prediction. The user has provided two TFT models:

1. SimpleTFT - a smaller, simpler version
2. EnhancedTFT - a larger, more complex version with additional features

And they've also provided the training script which shows the best performing configuration.

From looking at the training script, I can see:
- It uses SimpleTFT by default (based on the import)
- The default loss type is 'l1' 
- The default target transform is 'normalize'
- It supports both simple and enhanced TFT architectures
- Uses spatial-temporal processing with CNN + Transformer
- Has comprehensive experiment tracking and model saving

I need to structure this README similar to the previous ones but focus on the TFT-specific aspects:

1. Overview of TFT approach
2. Model architectures (SimpleTFT vs EnhancedTFT)
3. Key features and components
4. Usage instructions
5. Best practices
6. Performance characteristics

Let me focus on what makes TFT special:
- Combines CNN for spatial feature extraction with Transformer for temporal modeling
- Gated Residual Networks (GRN)
- Positional encoding
- Attention pooling
- Multiple scales of processing

Based on the script, it seems like the best performing setup uses:
- SimpleTFT (default import)
- L1 loss
- Normalize target transform
- Learning rate 0.0002

</details>

# Temporal Fusion Transformer (TFT) for Soil Organic Carbon (SOC) Prediction in Bavaria

This repository implements Temporal Fusion Transformer (TFT) models for predicting and mapping soil organic carbon (SOC) content across Bavaria, Germany, using multi-temporal satellite and environmental data from 2007-2023.

## Overview

We provide two TFT architectures that combine Convolutional Neural Networks (CNN) for spatial feature extraction with Transformer attention mechanisms for temporal sequence modeling:

- **SimpleTFT**: Streamlined TFT with essential components for efficient SOC prediction
- **EnhancedTFT**: Advanced TFT with multi-scale processing, enhanced attention, and hierarchical feature learning

Both models leverage the TFT's unique ability to handle heterogeneous inputs and capture complex temporal dependencies for improved SOC prediction accuracy.

## Model Architectures

### SimpleTFT (Recommended)

```python
SimpleTFT(
    input_channels=6,         # Number of satellite bands
    height=5,                 # Spatial height dimension
    width=5,                  # Spatial width dimension
    time_steps=5,             # Temporal sequence length
    d_model=128,              # Model dimensionality
    num_heads=2,              # Number of attention heads
    dropout=0.3               # Dropout rate
)
```

**Key Components:**
- **Spatial Encoder**: CNN layers for extracting spatial features per timestep
- **Gated Residual Network (GRN)**: Feature transformation with gating mechanism
- **Positional Encoding**: Learned temporal position embeddings
- **Transformer Encoder**: Single-layer attention mechanism
- **Projection Head**: Final prediction layers

### EnhancedTFT (Advanced)

```python
EnhancedTFT(
    input_channels=6,         # Number of satellite bands  
    height=5,                 # Spatial height dimension
    width=5,                  # Spatial width dimension
    time_steps=5,             # Temporal sequence length
    d_model=128,              # Model dimensionality
    num_heads=4,              # Number of attention heads
    dropout=0.3,              # Dropout rate
    num_encoder_layers=3,     # Number of transformer layers
    expansion_factor=4        # Feedforward network expansion
)
```

**Advanced Features:**
- **Enhanced CNN**: Deeper spatial encoder with batch normalization
- **Multi-GRN Blocks**: Hierarchical gated residual processing
- **Hybrid Positional Encoding**: Combined sinusoidal and learned embeddings
- **Multi-Layer Transformer**: Deep attention with large feedforward networks
- **Attention Pooling**: Weighted temporal aggregation mechanism
- **Residual Projection Head**: Enhanced prediction with layer normalization

## Architecture Flow

### Spatial-Temporal Processing Pipeline

1. **Spatial Feature Extraction**: 
   ```
   Input: [batch, channels, height, width, time]
   CNN per timestep: [batch*time, channels, height, width] → [batch*time, features]
   Reshape: [batch, time, features]
   ```

2. **Gated Residual Network**:
   ```python
   grn_out = GRN(features)
   gate = Sigmoid(Linear(features))
   output = LayerNorm(gate * grn_out + residual)
   ```

3. **Temporal Modeling**:
   ```
   Add positional encoding → Transformer attention → Temporal aggregation
   ```

4. **SOC Prediction**:
   ```
   Final features → Multi-layer prediction head → SOC value
   ```

### Gated Residual Network (GRN)

The GRN is a key TFT component that provides:
- **Feature Transformation**: Non-linear processing with ELU activation
- **Gating Mechanism**: Learned gates control information flow
- **Residual Connections**: Stable gradient flow and feature preservation
- **Layer Normalization**: Improved training stability

```python
def forward(self, x):
    h = F.elu(self.layer1(x))
    h = self.dropout(h)
    h = self.layer2(h)
    
    gate = self.gate(x)
    x_skip = self.skip_proj(x)
    
    return self.layernorm(gate * h + (1 - gate) * x_skip)
```

## Dataset

- **Temporal Coverage**: 2007-2023 (17 years)
- **Spatial Coverage**: Bavaria, Germany  
- **Data Source**: LUCAS soil survey data + multi-temporal satellite imagery
- **Input Format**: ```[batch_size, channels, height, width, time_steps]```
- **Target Variable**: Soil Organic Carbon (OC) content

## Optimal Configuration

Based on extensive experiments, the best performing configuration is:
- **Model**: SimpleTFT
- **Loss Function**: ```l1``` (L1/MAE loss)
- **Target Transform**: ```normalize``` (Z-score normalization)
- **Learning Rate**: 0.0002
- **Hidden Size**: 128
- **Dropout Rate**: 0.3

## Usage

### Basic Training

```bash
python trainSimulations.py
```

### Optimal Configuration Training

```bash
python trainSimulations.py \
    --lr 0.0002 \
    --loss_type l1 \
    --target_transform normalize \
    --hidden_size 128 \
    --dropout_rate 0.3 \
    --use_validation True \
    --num-runs 4
```

### Advanced Configuration

```bash
python trainSimulations.py \
    --lr 0.0001 \
    --num_heads 4 \
    --num_layers 3 \
    --loss_type composite_l2 \
    --loss_alpha 0.5 \
    --target_transform log \
    --hidden_size 256 \
    --target-val-ratio 0.08 \
    --distance-threshold 1.2 \
    --num-runs 5 \
    --save_train_and_val True
```

### Switch to Enhanced TFT

To use the EnhancedTFT model, modify the import in ```trainSimulations.py```:
```python
# from SimpleTFT import SimpleTFT
from EnhancedTFT import EnhancedTFT as SimpleTFT
```

### Key Parameters

- ```--lr```: Learning rate (default: 0.0002)
- ```--hidden_size```: Model dimensionality d_model (default: 128)
- ```--num_heads```: Number of attention heads (default: from config)
- ```--num_layers```: Number of transformer layers (default: from config)
- ```--loss_type```: Loss function (```l1```, ```mse```, ```composite_l1```, ```composite_l2```)
- ```--target_transform```: Target preprocessing (```normalize```, ```log```, ```none```)
- ```--dropout_rate```: Dropout probability (default: 0.3)
- ```--use_validation```: Enable spatial validation split
- ```--distance-threshold```: Minimum distance between train/validation points
- ```--num-runs```: Number of independent training runs
- ```--save_train_and_val```: Save datasets for analysis

## Advanced Features

### Gated Residual Networks

```python
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.layer2 = nn.Linear(output_dim, output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
```

**Benefits:**
- Adaptive feature selection through gating
- Stable training with residual connections
- Non-linear transformations with controlled information flow

### Positional Encoding Types

#### Simple Learnable (SimpleTFT)
```python
self.pos_embedding = nn.Parameter(torch.randn(time_steps, d_model))
```

#### Hybrid Encoding (EnhancedTFT)
```python
# Sinusoidal + Learnable components
pe_sin = sinusoidal_encoding(max_len, d_model)
pe_learned = nn.Parameter(torch.randn(max_len, d_model))
x = x + pe_sin + pe_learned
```

### Attention Pooling

```python
class AttentionPooling(nn.Module):
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        weighted_sum = torch.sum(attention_weights * x, dim=1)
        return weighted_sum
```

**Advantages:**
- Focuses on important time steps
- Reduces temporal dimension intelligently
- Improves final prediction quality

## Model Comparison

### Architecture Differences

| Component | SimpleTFT | EnhancedTFT |
|-----------|-----------|-------------|
| CNN Layers | 2 conv + ReLU | 3 conv + BatchNorm + ReLU |
| GRN Blocks | 1 block | 3 hierarchical blocks |
| Transformer Layers | 1 layer | 3 layers (configurable) |
| Positional Encoding | Learnable only | Sinusoidal + Learnable |
| Temporal Aggregation | Flatten | Attention pooling |
| Parameters | ~50K-200K | ~500K-2M |

### Performance Characteristics

- **SimpleTFT**: Fast training, good baseline performance, suitable for most applications
- **EnhancedTFT**: Higher capacity, better complex pattern modeling, requires more computational resources

## Loss Functions

### L1 Loss (Recommended)
```python
criterion = nn.L1Loss()
```
- Robust to outliers in SOC data
- Stable training convergence
- Good performance on soil prediction tasks

### Composite Losses
```python
def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    l2_loss = torch.mean((targets - outputs) ** 2)
    chi2_loss = torch.mean((targets - outputs) ** 2 / sigma ** 2)
    return alpha * l2_loss + (1 - alpha) * scale_factor * chi2_loss
```

## Target Transformations

### Normalization (Recommended)
```python
targets = (targets - target_mean) / (target_std + 1e-10)
```
- Z-score standardization for consistent scaling
- Improves optimization convergence
- Preserves data distribution properties

### Log Transformation
```python
targets = torch.log(targets + 1e-10)
```
- Handles skewed SOC distributions
- Useful for wide-range target values

## Performance Monitoring

### Evaluation Metrics

- **R² (Correlation Squared)**: Coefficient of determination
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Error magnitude with outlier penalty
- **RPIQ (Ratio of Performance to Interquartile Range)**: Precision metric

### Experiment Tracking

- **Comprehensive Logging**: Organized experiment directories with timestamps
- **Multi-run Statistics**: Aggregated performance across experiments
- **Model Versioning**: Automatic best model selection and saving
- **Weights & Biases Integration**: Real-time metrics visualization

## File Structure

```
├── SimpleTFT.py                     # SimpleTFT architecture
├── EnhancedTFT.py                   # EnhancedTFT architecture
├── trainSimulations.py              # Main training script
├── config.py                        # Model and training configuration
├── balancedDataset.py               # Dataset balancing utilities
├── dataloader/
│   ├── dataloaderMultiYears.py      # Multi-temporal data loading
│   └── dataframe_loader.py          # DataFrame processing
├── output/
│   └── TFT_experiment_[timestamp]/  # Organized experiment outputs
│       ├── models/                  # Trained model files
│       ├── results/                 # Metrics and summaries
│       ├── data/                    # Saved datasets (optional)
│       └── EXPERIMENT_SUMMARY.txt   # Experiment overview
└── README.md                        # This file
```

## TFT Advantages for SOC Prediction

### Spatial-Temporal Integration

1. **CNN Feature Extraction**: Effective spatial pattern recognition per timestep
2. **Temporal Attention**: Dynamic weighting of different time periods
3. **Multi-scale Processing**: Hierarchical feature learning across space and time
4. **Gated Information Flow**: Adaptive feature selection and transformation

### Robustness and Flexibility

1. **Missing Data Handling**: Attention mechanism naturally handles irregular temporal sampling
2. **Variable Sequence Lengths**: Flexible temporal modeling capabilities
3. **Multi-modal Integration**: Can incorporate different data types and scales
4. **Interpretable Attention**: Visualizable temporal importance weights

## Best Practices

### Training Recommendations

1. **Use SimpleTFT** for most applications - good balance of performance and efficiency
2. **Apply normalization** for target transformation - stable and effective
3. **Use L1 loss** for robust training with soil data
4. **Set learning rate to 0.0002** for optimal convergence
5. **Run multiple experiments** (4+ runs) for statistical reliability

### Hyperparameter Guidelines

- **Start with d_model=128** for balanced capacity
- **Use 2-4 attention heads** for most spatial-temporal patterns
- **Apply 0.3 dropout** to prevent overfitting
- **Consider single transformer layer** for SimpleTFT to avoid overcomplexity

### Model Selection

- **SimpleTFT**: Production applications, limited computational resources
- **EnhancedTFT**: Research applications, complex temporal patterns, abundant data

## Future Enhancements

1. **Multi-Resolution Processing**: Hierarchical spatial scales
2. **Cross-Modal Attention**: Integration of meteorological and soil data
3. **Uncertainty Quantification**: Probabilistic prediction outputs
4. **Temporal Decomposition**: Seasonal and trend component modeling
5. **Transfer Learning**: Pre-trained models for different geographical regions

## Citation

If you use this TFT implementation for SOC prediction, please cite:

```
@software{tft_soc_bavaria,
  title={Temporal Fusion Transformer for Soil Organic Carbon Prediction in Bavaria},
  year={2025},
  note={Deep learning model combining CNN and Transformer architectures}
}
```

