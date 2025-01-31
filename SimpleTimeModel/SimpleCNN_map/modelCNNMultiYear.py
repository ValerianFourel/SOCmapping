import torch
import torch.nn as nn

class Small3DCNN(nn.Module):
    def __init__(self, input_channels=6, dropout_rate=0.3):
        super(Small3DCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=2, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool3d(2, 2)

        # Dropout layers
        self.dropout_conv = nn.Dropout3d(p=dropout_rate)  # For convolutional layers
        self.dropout_fc = nn.Dropout(p=dropout_rate)      # For fully connected layers

        # Fully connected layers
        self.fc1 = nn.Linear(576, 64)  # Adjusted based on the output size after pooling
        self.fc2 = nn.Linear(64, 1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = self.pool(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = self.pool(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.relu(x)
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
