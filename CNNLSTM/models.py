# coding=utf-8  # Specifies UTF-8 encoding for the Python source code file

# Import required PyTorch libraries and modules
import torch  # Main PyTorch library
import torchvision  # PyTorch computer vision library
from torchvision import models, datasets  # Pre-trained models and dataset utilities
import torchvision.transforms as transforms  # Image transformation tools
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functional operations
import torch.optim as optim  # Optimization algorithms
# import config as cfg  # Commented out configuration import


class ConvNet(nn.Module):
    """Convolutional Neural Network implementation
    A simple CNN with 2 convolutional layers followed by 2 fully connected layers"""
    def __init__(self, num_channels):
        super(ConvNet, self).__init__()
        self.num_channels = num_channels
        # First conv layer: input_channels -> 16 output channels, 2x2 kernel
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=16, kernel_size=(2, 2), stride=1, padding=1)
        # Second conv layer: 16 -> 16 channels, 2x2 kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2), stride=1, padding=1)
        # First fully connected layer: 64 -> 16 neurons
        self.fc1 = nn.Linear(64, 16)
        # Output layer: 16 -> 1 neuron
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        """Forward pass of the network"""
        x = x[:, :, :, :]  # Ensure proper input dimensions
        # Apply convolutions with ReLU activation and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Flatten the output for fully connected layers
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU
        x = self.fc2(x).reshape(-1)  # Output layer
        return x

    def num_flat_features(self, x):
        """Calculate the total number of features after flattening"""
        sizes = x.size()[1:]  # Get all dimensions except batch size
        num_features = 1
        for s in sizes:
            num_features *= s
        return num_features


class SimpleLSTM(nn.Module):
    """Simple LSTM network implementation
    Single LSTM layer followed by two fully connected layers"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.25):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        # LSTM layer configuration
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # Fully connected layers for output processing
        self.fc_1 = nn.Linear(hidden_size, 16)
        self.fc_2 = nn.Linear(16, 1)

    def forward(self, x):
        """Forward pass of the network"""
        x, (h_n, c_n) = self.lstm(x)  # Process sequence through LSTM
        x = torch.tanh(self.fc_1(x[:, -1, :]))  # Apply tanh activation to first FC layer
        x = self.fc_2(x).reshape(-1)  # Output layer
        return x

    def init_hidden(self):
        """Initialize hidden state"""
        return torch.randn(1, 24, self.hidden_size)


class CovLSTM(nn.Module):
    """Combined CNN-LSTM network implementation
    Processes both image data (CNN) and sequential data (LSTM) in parallel"""
    def __init__(self, cnn_num_channels,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers=1, lstm_dropout=0):
        super(CovLSTM, self).__init__()
        # CNN components
        self.conv1 = nn.Conv2d(in_channels=cnn_num_channels, out_channels=6, kernel_size=(2, 2), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(2, 2), stride=1, padding=1)
        self.fc_cnn_1 = nn.Linear(64, 16)

        # LSTM components
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        self.fc_lstm_1 = nn.Linear(lstm_hidden_size, 16)

        # Final combined fully connected layer
        self.fc_final = nn.Linear(16+16, 1)

    def forward(self, x_cnn, x_lstm):
        """Forward pass combining CNN and LSTM outputs"""
        # Process CNN input
        x_cnn = x_cnn[:, :, :, :]
        x_cnn = F.max_pool2d(F.relu(self.conv1(x_cnn)), (2, 2))
        x_cnn = F.max_pool2d(F.relu(self.conv2(x_cnn)), (2, 2))
        x_cnn = x_cnn.view(-1, self.num_flat_features(x_cnn))
        x_cnn = F.relu(self.fc_cnn_1(x_cnn))

        # Process LSTM input
        x_lstm, (h_n, c_n) = self.lstm(x_lstm)
        x_lstm = torch.tanh(self.fc_lstm_1(x_lstm[:, -1, :]))

        # Combine CNN and LSTM outputs
        v_combined = torch.cat((x_cnn, x_lstm), 1)
        pred = self.fc_final(v_combined).reshape(-1)
        return pred

    def num_flat_features(self, x):
        """Calculate the total number of features after flattening"""
        sizes = x.size()[1:]
        num_features = 1
        for s in sizes:
            num_features *= s
        return num_features
