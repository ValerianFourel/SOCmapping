import numpy as np
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, input_channels=6, dropout_rate=0.3):
        super(SmallCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Additional convolutional layers
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Skip connection layers (1x1 convolutions for dimension matching)
        self.skip1 = nn.Conv2d(16, 32, kernel_size=1)
        self.skip2 = nn.Conv2d(32, 64, kernel_size=1)
        self.skip3 = nn.Conv2d(64, 32, kernel_size=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.dropout_conv = nn.Dropout2d(p=dropout_rate)
        self.dropout_fc = nn.Dropout(p=dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolutional block
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.dropout_conv(x1)
        x1_pooled = self.pool(x1)

        # Second convolutional block with skip connection
        x2 = self.conv2(x1_pooled)
        x2 = self.relu(x2)
        x2 = self.dropout_conv(x2)
        # Skip connection from first block (with dimension matching)
        skip_x1 = self.skip1(self.pool(x1))
        x2 = x2 + skip_x1
        x2_pooled = self.pool(x2)

        # Third convolutional block with skip connection
        x3 = self.conv3(x2_pooled)
        x3 = self.relu(x3)
        x3 = self.dropout_conv(x3)
        # Skip connection from second block
        skip_x2 = self.skip2(self.pool(x2))
        x3 = x3 + skip_x2
        x3_pooled = self.pool(x3)

        # Fourth convolutional block (additional depth)
        x4 = self.conv4(x3_pooled)
        x4 = self.relu(x4)
        x4 = self.dropout_conv(x4)
        x4 = x4 + x3_pooled  # Residual connection

        # Fifth convolutional block (additional depth)
        x5 = self.conv5(x4)
        x5 = self.relu(x5)
        x5 = self.dropout_conv(x5)
        # Skip connection from third block
        skip_x3 = self.skip3(x3_pooled)
        x5 = x5 + skip_x3

        # Flatten and fully connected layers
        x = x5.view(x5.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
