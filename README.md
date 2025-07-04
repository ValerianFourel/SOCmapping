
# SOC Mapping Neural Networks Used

This project, based on the master's thesis *"Machine Learning-Based Spatial Modeling of Soil Organic Carbon in Bavaria Using Multi-Source Remote Sensing and Ground Truth Data"* by Valerian Raphael Fourel, explores various machine learning techniques to map Soil Organic Carbon (SOC) content in Bavaria's topsoil (0-20 cm). It leverages ground samples and multi-temporal satellite remote sensing data, including the following bands: **Elevation**, **LAI (Leaf Area Index)**, **LST (Land Surface Temperature)**, **MODIS NPP (Net Primary Productivity)**, **Soil Evaporation**, and **Total Evapotranspiration**. Below is a summary of the methods implemented, ranging from non-neural network approaches to advanced neural network models, including foundational ones.

---

## Project Overview

The goal of this project is to achieve state-of-the-art performance in Digital Soil Mapping (DSM) for SOC prediction across Bavaria, utilizing a comprehensive dataset spanning 2007 to 2023. The thesis evaluates multiple machine learning architectures, with a focus on their ability to handle spatiotemporal data and avoid data leakage through spatial separation of training and validation sets. The best-performing model, the **Temporal Fusion Transformer (TFT)**, achieved an RPIQ of 1.102 and an R² of 0.64, making it suitable for operational use.

---

## A. Non-Neural Network Techniques

### XGBoost
- **What it does**: Utilizes gradient-boosted decision trees to predict SOC from satellite data.
- **Why it’s here**: Provides a fast, effective baseline for comparison with neural network models.
- **Performance**: Achieved an R² of 0.58 and an RPIQ of 1.40 under biased sampling conditions.

### Random Forest
- **What it does**: Constructs multiple decision trees and averages their predictions for SOC estimation.
- **Why it’s here**: Offers robustness and excels at identifying patterns without overfitting.
- **Performance**: Recorded an R² of 0.56 and an RPIQ of 1.32 under biased sampling.

---

## B. Simple Neural Network Techniques

### Simple CNN Techniques
- **What it does**: Applies a basic Convolutional Neural Network (CNN) to extract spatial features from single-time satellite images.
- **Why it’s here**: Tests the capability of spatial patterns alone to predict SOC.
- **Performance**: Not explicitly detailed in the thesis, but generally underperformed compared to more complex models.

---

## C. Multiyear Techniques

### 3D CNN
- **What it does**: Employs 3D convolutions to process both spatial and temporal satellite data simultaneously.
- **Why it’s here**: Captures SOC variations over space and time, leveraging multi-year data.
- **Performance**: Achieved an R² of 0.22, RMSE of 9.2 g/kg, and MAE of 5.3 g/kg.

### CNN-LSTM
- **What it does**: Combines CNNs for spatial feature extraction with Long Short-Term Memory (LSTM) units for time-series modeling.
- **Why it’s here**: Addresses seasonal and yearly trends in SOC data.
- **Performance**: Recorded an R² of 0.52, RMSE of 7.2 g/kg, and MAE of 4.3 g/kg.

### Simple Transformer
- **What it does**: Uses a lightweight Transformer architecture to process satellite data as a sequence.
- **Why it’s here**: Explores the effectiveness of attention mechanisms in SOC prediction.
- **Performance**: Varied from R² of 0.33 (small version, 20.8k parameters) to 0.51 (large version, 2M parameters).

---

## D. Foundational Models

### Foundational Models (e.g., SpectralGPT)
- **What it does**: Fine-tunes a pre-trained Vision Transformer (SpectralGPT) with a Transformer regressor for SOC prediction.
- **Why it’s here**: Leverages advanced pre-training on large datasets to potentially improve accuracy.
- **Performance**: Achieved competitive R² values (0.61-0.64), but underperformed compared to the TFT in this study.

---

## Best Performing Model: Temporal Fusion Transformer (TFT)
- **What it does**: A specialized Transformer architecture designed to handle spatiotemporal data, integrating spatial and temporal features effectively.
- **Why it’s here**: Achieves breakthrough performance by capturing complex dependencies in multi-year satellite data.
- **Performance**: Recorded the highest metrics with an R² of 0.64, RPIQ of 1.102, RMSE of 4.7 g/kg, and MAE of 2.7 g/kg (1.1M parameters version).
- **Key Features**: 
  - Utilizes 1.1 million parameters.
  - Excels in balancing bias and variance, particularly for medium to high SOC values.
  - Benefits from spatial separation in dataset splitting to ensure reliable performance metrics.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch
- Accelerate (for multi-GPU support)
- WandB (for experiment tracking)
- Condor (for job submission on a cluster)

### Environment Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd SOCmapping
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Training Script
The main training script (`train.py`) is configured to use the parameters specified in `config.py`. To execute the training on a cluster with multiple GPUs, use the following commands:

#### Cluster Submission with Condor
```bash
condor_submit_bid 1000 -i \
  -append request_memory=481920 \
  -append request_cpus=100 \
  -append request_disk=200G \
  -append request_gpus=3 \
  -append 'requirements = CUDADeviceName == "NVIDIA A100-SXM4-80GB"' \
  train.sub
```
Ensure `train.sub` is configured to call `accelerate launch` as below.

#### Multi-GPU Training
```bash
accelerate launch --multi_gpu train.py --save_train_and_val True
```
- `--save_train_and_val True`: Saves training and validation datasets for reproducibility.

### Configuration Parameters
The `config.py` file defines key parameters used in the main run:
```python
bands_list_order = ['Elevation', 'LAI', 'LST', 'MODIS_NPP', 'SoilEvaporation', 'TotalEvapotranspiration']
MAX_OC = 160  # Maximum SOC value (g/kg)
time_before = 5  # Years of historical data before prediction
window_size = 33  # Spatial window size
TIME_BEGINNING = '2007'  # Start of data period
TIME_END = '2023'  # End of data period
INFERENCE_TIME = '2015'  # Year for inference
num_epochs = 20  # Number of training epochs
```

---

## Key Findings
- **Spatial Separation**: Critical for preventing data leakage, with biased sampling inflating R² by up to 67%. The TFT’s performance was validated with unbiased spatial splits.
- **High SOC Prediction**: All models struggled with SOC values >120 g/kg due to limited samples in carbon-rich areas (e.g., mountains).
- **Operational Use**: The TFT’s RPIQ > 1 indicates reliability for practical soil mapping in Bavaria.

---

## Data
- **Ground Truth**: Sourced from LUCAS, LFU, and LfL datasets, covering 16,341 samples across Bavaria.
- **Remote Sensing**: Multi-year satellite data processed at 500m resolution (250m for Elevation), normalized and harmonized to ETRS89/UTM zone 32N.

---

## Future Work
- Collect more samples from carbon-rich environments to improve high SOC prediction.
- Extend the methodology to other regions and soil properties for broader applicability.

---

## Acknowledgments
This work builds on the thesis submitted to the University of Tübingen in 2025, supervised by Dr. Nafiseh Kakhani and Prof. Dr. Holger Brandt. Special thanks to the Methods Center team and collaborators for their support.

