

# SOC Mapping Neural Networks Used

This project explores various techniques to map Soil Organic Carbon (SOC) content using ground samples and satellite remote sensing data (bands: Elevation, LAI, LST, MODIS NPP, Soil Evaporation, Total Evapotranspiration). Below is a summary of the methods implemented, from simple non-neural network approaches to advanced neural network models, including foundational ones.

## A. Non-Neural Network Techniques

### XGBoost
- **What it does**: Uses gradient-boosted trees to predict SOC from satellite data.  
- **Why it’s here**: Fast, effective, and a strong baseline for comparison.

### Random Forest
- **What it does**: Builds multiple decision trees and averages their predictions for SOC.  
- **Why it’s here**: Robust and good at finding patterns without overfitting.

## B. Simple Neural Network Techniques

### Simple CNN Techniques
- **What it does**: Applies a basic Convolutional Neural Network to extract spatial features from single-time satellite images.  
- **Why it’s here**: Tests if spatial patterns alone can predict SOC.

## C. Multiyear Techniques

### 3D CNN
- **What it does**: Uses 3D convolutions to process spatial and temporal satellite data together.  
- **Why it’s here**: Captures how SOC changes over space and time.

### CNN-LSTM
- **What it does**: Combines CNNs for spatial features with LSTMs for time-series modeling.  
- **Why it’s here**: Handles seasonal and yearly trends in SOC data.

### Simple Transformer
- **What it does**: Uses a lightweight Transformer to process satellite data as a sequence.  
- **Why it’s here**: Explores attention mechanisms for SOC prediction.

### VAE-Transformer
- **What it does**: Pairs a Variational Autoencoder with a Transformer to compress and predict SOC.  
- **Why it’s here**: Tests if latent features improve SOC mapping.

## D. Foundational Models

### Foundational Models (e.g., SpectralGPT)
- **What it does**: Fine-tunes a pre-trained Vision Transformer (SpectralGPT) with a Transformer regressor for SOC prediction.  
- **Why it’s here**: Leverages advanced pre-training for top accuracy.

to use:
````bash
condor_submit_bid 1000 -i -append request_memory=481920 -append request_cpus=100 -append request_disk=200G -append request_gpus=3 -append 'requirements = CUDADeviceName == "NVIDIA A100-SXM4-80GB"'
```
