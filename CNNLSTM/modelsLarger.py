# coding=utf-8  # Specifies UTF-8 encoding for the Python source code file

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import time_before, window_size  # Import required config variables

class EnhancedRefittedCovLSTM(nn.Module):
    """Enhanced Refitted Convolutional LSTM Network
    Combines deeper CNN with residual connections and LSTM for temporal sequence processing
    Optimized for CUDA 11.8 compatibility"""
    
    def __init__(self, num_channels, lstm_input_size, lstm_hidden_size, 
                 num_layers=3, dropout=0.3):
        """
        Initialize the EnhancedRefittedCovLSTM model
        
        Args:
            num_channels (int): Number of input channels (bands)
            lstm_input_size (int): Size of LSTM input (flattened spatial features after CNN)
            lstm_hidden_size (int): Size of LSTM hidden state
            num_layers (int): Number of LSTM layers (default: 3)
            dropout (float): Dropout rate between layers (default: 0.3)
        """
        super(EnhancedRefittedCovLSTM, self).__init__()
        
        # CNN Components with Residual Blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64)
        ).to(dtype=torch.float32)
        self.residual_conv1 = nn.Conv2d(num_channels, 64, kernel_size=1, stride=1, padding=0, bias=True).to(dtype=torch.float32)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128)
        ).to(dtype=torch.float32)
        self.residual_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True).to(dtype=torch.float32)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256)
        ).to(dtype=torch.float32)
        
        # Calculate CNN output size per time step
        # One maxpool2d operation with kernel=2 reduces spatial dims by 2 (5x5 -> 2x2)
        self.cnn_output_height = 5 // 2
        self.cnn_output_width = 5 // 2
        self.cnn_output_size = 256 * self.cnn_output_height * self.cnn_output_width
        self.fc_cnn = nn.Linear(self.cnn_output_size, 256).to(dtype=torch.float32)

        # LSTM Component
        self.lstm = nn.LSTM(
            input_size=256,  # Size after fc_cnn
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bias=True
        ).to(dtype=torch.float32)

        # Final fully connected layers
        self.fc_lstm = nn.Linear(lstm_hidden_size, 128).to(dtype=torch.float32)
        self.fc_final = nn.Linear(256 + 128, 64).to(dtype=torch.float32)
        self.output = nn.Linear(64, 1).to(dtype=torch.float32)

        # Dropout
        self.dropout = nn.Dropout(dropout)

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
        x = x.to(dtype=torch.float32)
        
        batch_size, num_channels, height, width, time_steps = x.size()
        
        # Process CNN for each time step
        x_cnn = x.permute(0, 4, 1, 2, 3)  # [batch, time, bands, height, width]
        x_cnn = x_cnn.reshape(batch_size * time_steps, num_channels, height, width)
        
        # Apply CNN layers with residual connections
        residual = self.residual_conv1(x_cnn)
        x_cnn = self.conv1(x_cnn)
        x_cnn = F.relu(x_cnn + residual)
        
        residual = self.residual_conv2(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = F.relu(x_cnn + residual)
        x_cnn = F.max_pool2d(x_cnn, 2)  # Only one max-pool
        
        x_cnn = self.conv3(x_cnn)
        
        x_cnn = x_cnn.view(batch_size * time_steps, -1)
        x_cnn = F.relu(self.fc_cnn(x_cnn))
        x_cnn = self.dropout(x_cnn)
        
        # Reshape back to include time dimension for LSTM
        x_cnn = x_cnn.view(batch_size, time_steps, 256)  # [batch, time, features]

        # LSTM Processing
        x_lstm, _ = self.lstm(x_cnn)
        x_lstm = x_lstm[:, -1, :]  # Take last time step
        x_lstm = F.relu(self.fc_lstm(x_lstm))
        x_lstm = self.dropout(x_lstm)

        # CNN features from last time step
        x_cnn_last = x_cnn[:, -1, :]  # Use the last time step's CNN features
        
        # Combine CNN and LSTM outputs
        combined = torch.cat((x_cnn_last, x_lstm), dim=1)
        x = F.relu(self.fc_final(combined))
        x = self.dropout(x)
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

    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    # Example initialization
    model = EnhancedRefittedCovLSTM(
        num_channels=6,  # Match your dataset's number of bands
        lstm_input_size=256,  # Fixed by CNN output
        lstm_hidden_size=256,  # Increased for more parameters
        num_layers=3,  # Increased for more complexity
        dropout=0.3
    )
    
    # Print number of parameters
    num_params = model.count_parameters()
    print(f"Number of trainable parameters: {num_params}")
    
    # Example input tensor matching your batch size
    batch_size = 256
    sample_input = torch.randn(batch_size, 6, 5, 5, 5)  # [batch, bands, height, width, time]
    
    if torch.cuda.is_available():
        model = model.cuda()
        sample_input = sample_input.cuda()
    
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # Should be [batch_size]
