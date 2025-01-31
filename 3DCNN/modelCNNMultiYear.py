import torch
import torch.nn as nn


class Small3DCNN(nn.Module):
    def __init__(self, input_channels=6, input_height=10, input_width=10, input_time = 4, dropout_rate=0.3):
        super(Small3DCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=2, padding=1)
        # Activation
        self.relu = nn.ReLU()

        # Pooling layer
        self.pool = nn.MaxPool3d(2, 2)

        # Dropout layers
        self.dropout_conv = nn.Dropout3d(p=dropout_rate)  # For convolutional layers
        self.dropout_fc = nn.Dropout(p=dropout_rate)      # For fully connected layers

        # Compute the flattened size dynamically
        self.flatten_size = self._get_conv_output_size(input_channels, input_height, input_width,input_time)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)

        

    def _get_conv_output_size(self, channels, h, w,t):
        dummy_input = torch.zeros(1, channels, h, w, t)  # Adjusted for 3D input
        x = self.conv1(dummy_input)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        return int(torch.prod(torch.tensor(x.shape[1:])))

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
