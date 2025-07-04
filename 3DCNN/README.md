# 3D CNN for Soil Organic Carbon Regression

A PyTorch implementation of a 3D Convolutional Neural Network for predicting Soil Organic Carbon (SOC) content using spatiotemporal environmental data.

### Overview
This project implements a deep learning approach to predict soil organic carbon content by analyzing spatiotemporal patterns in environmental variables. The 3D CNN architecture processes multi-dimensional data including spatial coordinates, temporal sequences, and multiple environmental features to estimate SOC levels across different landscapes and time periods.

![Performance comparison of different models for SOC prediction](assets/AllModelsComparison/AllModelsComparison_3DCNN_MAX_OC_150_Beginning_2007_End_2023_InferenceTime2023_20250527_155525.png)

*Performance comparison of different models for SOC prediction (2007-2023 dataset, inference time: 2023)*

Key Features
3D Convolutional Architecture: Processes spatiotemporal data with dimensions for location (height × width), time series, and multiple environmental channels
Multi-GPU Support: Distributed training using Hugging Face Accelerate
Regularization: Dropout layers to prevent overfitting on limited SOC datasets
Flexible Input Dimensions: Adaptable to different study areas and temporal resolutions
SOC-Specific Design: Optimized for soil carbon prediction tasks
Model Architecture
The Small3DCNN model consists of:

Input: 6 environmental channels × 10×10 spatial grid × 4 time steps
3D Convolutional Layers: Three progressive layers (16→32→64 filters) with ReLU activation
Pooling: MaxPool3d for dimensionality reduction
Regularization: 3D and standard dropout (30% rate)
Output: Single regression value (SOC content)
Environmental Input Channels
Typical SOC-relevant variables include:

Normalized Difference Vegetation Index (NDVI)
Temperature data
Precipitation patterns
Soil moisture content
Elevation/topography
Land cover classifications


```bash
accelerate launch --multi_gpu train.py
```

@software{3dcnn_soc_regression,
  title={3D CNN for Soil Organic Carbon Regression},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
