import torch
import torch.nn as nn

# SOC3DResidualBlock class remains the same
class SOC3DResidualBlock(nn.Module):
    """
    3D Residual Block for Soil Organic Carbon prediction
    Processes 3D spatial features while maintaining spatial information through skip connections
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels (int): Number of input channels (e.g., spectral bands, soil properties)
            out_channels (int): Number of output channels
            stride (int): Stride for convolution operation
        """
        super(SOC3DResidualBlock, self).__init__()

        # Main convolutional path
        self.conv_block = nn.Sequential(
            # First 3D convolution layer
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),  # Normalize the features
            nn.ELU(inplace=True),         # Non-linear activation

            # Second 3D convolution layer
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)   # Final normalization before skip connection
        )

        # Skip connection path
        self.shortcut = nn.Sequential()
        # If dimensions change, adjust skip connection with 1x1x1 convolution
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        """Forward pass combining main path with skip connection"""
        residual = x                    # Save input for skip connection
        out = self.conv_block(x)        # Main convolutional path
        out += self.shortcut(residual)  # Add skip connection
        out = self.elu(out)           # Final activation
        return out

class SOCPredictor3DCNN(nn.Module):
    """
    7-layer deep 3D CNN with ResNet architecture for Soil Organic Carbon content prediction
    """
    def __init__(self, input_channels, input_depth, input_height, input_width):
        super(SOCPredictor3DCNN, self).__init__()

        # Initial feature extraction block remains the same
        self.init_conv = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Seven residual layers with increasing feature channels
        #self.layer1 = self._make_layer(64, 64, 2, stride=1)      # Layer 1: Fine features
        self.layer2 = self._make_layer(64, 128, 2, stride=1)     # Layer 2: Fine-mid features
        self.layer3 = self._make_layer(128, 256, 2, stride=1)    # Layer 3: Mid features
        self.layer4 = self._make_layer(256, 512, 2, stride=2)    # Layer 4: Mid-high features
        #self.layer5 = self._make_layer(512, 512, 2, stride=1)    # Layer 5: High features
        self.layer6 = self._make_layer(512, 512, 2, stride=2)   # Layer 6: Complex features
        #self.layer7 = self._make_layer(1024, 1024, 2, stride=1)  # Layer 7: Final features

        # Calculate flattened size for dense layer
        self.flatten_size = self._get_conv_output_size(input_channels, 
                                                      input_depth, 
                                                      input_height, 
                                                      input_width)

        # Dense layer for feature compression
        self.dense = nn.Sequential(
            nn.Linear(self.flatten_size, 256),  # Increased intermediate features
            #nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.0)
        )

        # Final MLP for SOC prediction
        # Replace final ELU with linear activation for regression
        self.mlp = nn.Sequential(
            nn.Linear(256, 128,bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(128, 64,bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(64, 1, bias=False) # Remove final ELU for regression
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Creates a layer of residual blocks"""
        layers = []
        layers.append(SOC3DResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(SOC3DResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, channels, d, h, w):
        """Calculate the size of the flattened features after convolutions"""
        dummy_input = torch.zeros(1, channels, d, h, w)
        x = self.init_conv(dummy_input)
        #x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.layer6(x)
        #x = self.layer7(x)
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        """Forward pass through the network"""
        x = self.init_conv(x)

        # Process through all 7 residual layers
        #x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.layer6(x)
        #x = self.layer7(x)
        
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        soc_prediction = self.mlp(x)

        return soc_prediction

def get_trainable_params(model):
    """
    Returns the number of trainable parameters in the model.

    Args:
        model: PyTorch model instance

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Example parameters for soil data
    input_channels = 6    # Number of soil properties or spectral bands
    input_depth = 1     # Soil depth layers
    input_height = 17    # Spatial dimension height
    input_width = 17     # Spatial dimension width
    batch_size = 2       # Number of samples to process at once

    # Initialize model
    model = SOCPredictor3DCNN(input_channels, input_depth, input_height, input_width)
    print(model._get_conv_output_size)
    # Example input tensor
    soil_data = torch.randn(batch_size, input_channels, input_depth, input_height, input_width)
    print( get_trainable_params(model))
    # Get prediction
    soc_prediction = model(soil_data)
    print(f"SOC Prediction shape: {soc_prediction.shape}")  # Should be [batch_size, 1]
    print(soc_prediction)