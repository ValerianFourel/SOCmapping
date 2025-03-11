# coding=utf-8  # Specifies UTF-8 encoding for the Python source code file

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import time_before, window_size  # Import required config variables

class RefittedCovLSTM(nn.Module):
    """Refitted Convolutional LSTM Network
    Combines CNN for spatial feature extraction and LSTM for temporal sequence processing
    Optimized for CUDA 11.8 compatibility"""
    
    def __init__(self, num_channels, lstm_input_size, lstm_hidden_size, 
                 num_layers=1, dropout=0.25):
        """
        Initialize the RefittedCovLSTM model
        
        Args:
            num_channels (int): Number of input channels (bands)
            lstm_input_size (int): Size of LSTM input (flattened spatial features after CNN)
            lstm_hidden_size (int): Size of LSTM hidden state
            num_layers (int): Number of LSTM layers (default: 1)
            dropout (float): Dropout rate between layers (default: 0.25)
        """
        super(RefittedCovLSTM, self).__init__()
        
        # CNN Components
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True,
        ).to(dtype=torch.float32)  # Explicitly set float32 for CUDA compatibility
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True
        ).to(dtype=torch.float32)
        
        # Calculate CNN output size per time step
        # Two maxpool2d operations with kernel=2 reduce spatial dims by 4
        self.cnn_output_height = window_size // 4
        self.cnn_output_width = window_size // 4
        self.cnn_output_size = 64 * self.cnn_output_height * self.cnn_output_width
        self.fc_cnn = nn.Linear(self.cnn_output_size, 128).to(dtype=torch.float32)

        # LSTM Component
        self.lstm = nn.LSTM(
            input_size=128,  # Size after fc_cnn
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bias=True
        ).to(dtype=torch.float32)

        # Final fully connected layers
        self.fc_lstm = nn.Linear(lstm_hidden_size, 64).to(dtype=torch.float32)
        self.fc_final = nn.Linear(128 + 64, 32).to(dtype=torch.float32)
        self.output = nn.Linear(32, 1).to(dtype=torch.float32)

        # Ensure all parameters are on the correct device
        self.cuda_compatible = torch.cuda.is_available()
        if self.cuda_compatible:
            self.cuda()

    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, bands, height, width, time]
            
        Returns:
            torch.Tensor: Output predictions of shape [batch]
        """
        # Ensure input is in correct dtype
        x = x.to(dtype=torch.float32)
        
        batch_size, num_channels, height, width, time_steps = x.size()
        
        # Process CNN for each time step
        # Reshape to combine batch and time for CNN processing
        x_cnn = x.permute(0, 4, 1, 2, 3)  # [batch, time, bands, height, width]
        x_cnn = x_cnn.reshape(batch_size * time_steps, num_channels, height, width)
        
        # Apply CNN layers
        x_cnn = F.max_pool2d(F.relu(self.conv1(x_cnn)), 2)
        x_cnn = F.max_pool2d(F.relu(self.conv2(x_cnn)), 2)
        x_cnn = x_cnn.view(batch_size * time_steps, -1)
        x_cnn = F.relu(self.fc_cnn(x_cnn))
        
        # Reshape back to include time dimension for LSTM
        x_cnn = x_cnn.view(batch_size, time_steps, 128)  # [batch, time, features]

        # LSTM Processing
        x_lstm, _ = self.lstm(x_cnn)
        x_lstm = x_lstm[:, -1, :]  # Take last time step
        x_lstm = F.relu(self.fc_lstm(x_lstm))

        # CNN features from last time step
        x_cnn_last = x_cnn[:, -1, :]  # Use the last time step's CNN features
        
        # Combine CNN and LSTM outputs
        combined = torch.cat((x_cnn_last, x_lstm), dim=1)
        x = F.relu(self.fc_final(combined))
        output = self.output(x).reshape(-1)
        
        return output

    def to(self, *args, **kwargs):
        """Override to method to ensure CUDA compatibility"""
        super().to(*args, **kwargs)
        for param in self.parameters():
            param.data = param.data.to(dtype=torch.float32)
        return self

    def cuda(self, device=None):
        """Override cuda method for compatibility"""
        if self.cuda_compatible:
            return super().cuda(device)
        return self

# Example usage
if __name__ == "__main__":
    # Example initialization
    model = RefittedCovLSTM(
        num_channels=6,  # Match your dataset's number of bands
        lstm_input_size=128,  # This is now fixed by CNN output
        lstm_hidden_size=128,
        num_layers=2,
        dropout=0.25
    )
    
    # Example input tensor matching your batch size
    batch_size = 256
    sample_input = torch.randn(batch_size, 6, 33, 33, 4)  # [batch, bands, height, width, time]
    
    if torch.cuda.is_available():
        model = model.cuda()
        sample_input = sample_input.cuda()
    
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # Should be [batch_size]