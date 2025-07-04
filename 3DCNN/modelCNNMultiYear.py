import torch
import torch.nn as nn


class Small3DCNN(nn.Module):
    """
    A 3D Convolutional Neural Network for Soil Organic Carbon (SOC) prediction.

    This model is designed to process spatiotemporal data for SOC estimation, where:
    - Spatial dimensions (height, width) represent geographic coordinates or grid cells
    - Temporal dimension represents time series data (e.g., seasonal measurements)
    - Multiple input channels represent different environmental variables affecting SOC

    SOC Relevance:
    - SOC varies spatially across landscapes due to topography, land use, and soil properties
    - SOC changes temporally due to seasonal cycles, management practices, and climate
    - Multiple environmental factors (temperature, precipitation, vegetation indices, etc.) 
      influence SOC dynamics and need to be processed simultaneously
    """

    def __init__(self, input_channels=6, input_height=10, input_width=10, input_time=4, dropout_rate=0.3):
        """
        Initialize the 3D CNN for SOC prediction.

        Args:
            input_channels (int): Number of environmental variables/features
                                 Examples for SOC: NDVI, temperature, precipitation, 
                                 soil moisture, elevation, land cover type
            input_height (int): Spatial dimension - latitude grid cells
            input_width (int): Spatial dimension - longitude grid cells  
            input_time (int): Temporal dimension - time steps (e.g., seasons, years)
            dropout_rate (float): Regularization to prevent overfitting on limited SOC datasets
        """
        super(Small3DCNN, self).__init__()

        # 3D Convolutional layers - extract spatiotemporal patterns in SOC-related data
        # Conv3d processes (channels, depth/time, height, width) tensors
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=2, padding=1)
        # First layer: Extract basic spatiotemporal features from environmental variables
        # Small kernel (2x2x2) captures local SOC patterns and short-term temporal changes

        self.conv2 = nn.Conv3d(16, 32, kernel_size=2, padding=1)
        # Second layer: Combine basic features into more complex SOC-relevant patterns
        # Increased channels (32) allow detection of more diverse SOC relationships

        self.conv3 = nn.Conv3d(32, 64, kernel_size=2, padding=1)
        # Third layer: High-level feature extraction for complex SOC dynamics
        # Captures interactions between multiple environmental factors over space and time

        # Activation function - introduces non-linearity for complex SOC relationships
        self.relu = nn.ReLU()
        # ReLU is suitable for SOC modeling as it handles positive correlations well
        # (e.g., higher vegetation → higher SOC)

        # Pooling layer - reduces spatial/temporal resolution while preserving important features
        self.pool = nn.MaxPool3d(2, 2)
        # MaxPool3d reduces computational complexity and focuses on dominant SOC patterns
        # Helps model generalize across different scales of SOC variation

        # Dropout layers - prevent overfitting (critical for limited SOC field data)
        self.dropout_conv = nn.Dropout3d(p=dropout_rate)  # For 3D convolutional layers
        self.dropout_fc = nn.Dropout(p=dropout_rate)      # For fully connected layers
        # SOC datasets are often small and expensive to collect, making regularization essential

        # Dynamically compute the size after convolution operations
        # This ensures compatibility with different input dimensions for various study areas
        self.flatten_size = self._get_conv_output_size(input_channels, input_height, input_width, input_time)

        # Fully connected layers - final SOC prediction from extracted features
        self.fc1 = nn.Linear(self.flatten_size, 64)
        # Dense layer combines all spatiotemporal features for SOC estimation

        self.fc2 = nn.Linear(64, 1)
        # Output layer: Single value representing predicted SOC content (e.g., g C/kg soil)

    def _get_conv_output_size(self, channels, h, w, t):
        """
        Calculate the flattened tensor size after convolution operations.

        This is crucial for SOC applications where input dimensions may vary:
        - Different study areas have different spatial extents
        - Time series length may vary (monthly, seasonal, annual data)
        - Ensures model architecture adapts to available SOC datasets
        """
        # Create dummy input tensor with same dimensions as actual SOC data
        dummy_input = torch.zeros(1, channels, h, w, t)  # (batch, channels, height, width, time)

        # Forward pass through convolutional layers to determine output size
        x = self.conv1(dummy_input)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Return total number of elements for flattening
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        """
        Forward pass for SOC prediction.

        Input tensor x shape: (batch_size, channels, height, width, time)
        - batch_size: Number of SOC samples/locations in the batch
        - channels: Environmental variables (NDVI, temperature, etc.)
        - height, width: Spatial coordinates (latitude, longitude grid)
        - time: Temporal sequence (seasonal or annual measurements)
        """

        # First convolutional block - detect basic SOC-environment relationships
        x = self.conv1(x)           # Extract local spatiotemporal patterns
        x = self.relu(x)            # Apply non-linear activation
        x = self.dropout_conv(x)    # Regularization to prevent overfitting
        x = self.pool(x)            # Reduce spatial/temporal resolution

        # Second convolutional block - identify intermediate SOC patterns
        x = self.conv2(x)           # Combine basic features into complex patterns
        x = self.relu(x)            # Non-linear activation
        x = self.dropout_conv(x)    # Regularization
        x = self.pool(x)            # Further dimensionality reduction

        # Third convolutional block - capture high-level SOC dynamics
        x = self.conv3(x)           # Extract complex spatiotemporal SOC relationships
        x = self.relu(x)            # Non-linear activation
        x = self.dropout_conv(x)    # Regularization
        x = self.pool(x)            # Final spatial/temporal pooling

        # Flatten spatiotemporal features for final SOC prediction
        x = x.view(x.size(0), -1)   # Convert 3D features to 1D vector

        # Fully connected layers for final SOC estimation
        x = self.fc1(x)             # Dense layer to combine all features
        x = self.relu(x)            # Non-linear activation
        x = self.dropout_fc(x)      # Final regularization
        x = self.fc2(x)             # Output layer: predicted SOC value

        return x.squeeze()          # Remove extra dimensions, return SOC prediction

    def count_parameters(self):
        """
        Count trainable parameters in the model.

        Important for SOC applications to:
        - Assess model complexity relative to available training data
        - Ensure model size is appropriate for SOC dataset size
        - Compare different architectures for SOC prediction tasks
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
