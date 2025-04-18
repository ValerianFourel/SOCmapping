import torch
import torch.nn as nn


class ResNet2DCNN(nn.Module):
    def __init__(self, input_channels=26, input_height=5, input_width=5, dropout_rate=0.3):
        super(ResNet2DCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        # 1x1 convolutions for skip connections to match dimensions
        self.skip1 = nn.Conv2d(input_channels, 16, kernel_size=1) if input_channels != 16 else nn.Identity()
        self.skip2 = nn.Conv2d(16, 32, kernel_size=1) if 16 != 32 else nn.Identity()

        # Activation
        self.relu = nn.ReLU()

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.dropout_conv = nn.Dropout2d(p=dropout_rate)  # For convolutional layers
        self.dropout_fc = nn.Dropout(p=dropout_rate)      # For fully connected layers

        # Compute the flattened size dynamically
        self.flatten_size = self._get_conv_output_size(input_channels, input_height, input_width)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _get_conv_output_size(self, channels, h, w):
        dummy_input = torch.zeros(1, channels, h, w)  # Adjusted for 2D input
        x = self.conv1(dummy_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        # First convolutional block with skip connection
        identity = self.skip1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x + identity  # Skip connection
        x = self.dropout_conv(x)
        x = self.pool(x)

        # Second convolutional block with skip connection
        identity = self.skip2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x + identity  # Skip connection
        x = self.dropout_conv(x)
        x = self.pool(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)