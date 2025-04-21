import torch
import torch.nn as nn

class SimpleTransformerV2(nn.Module):
    def __init__(self, input_channels=6, input_height=33, input_width=33, input_time=4, num_heads=16, num_layers=6, dropout_rate=0.3):
        super(SimpleTransformerV2, self).__init__()

        # Calculate d_model ensuring it is divisible by num_heads
        self.d_model = input_channels * input_height * input_width
        if self.d_model % num_heads != 0:
            self.d_model = (self.d_model // num_heads + 1) * num_heads

        # Linear projection to match d_model dimension (increased dimensionality)
        self.input_projection = nn.Linear(input_channels * input_height * input_width, self.d_model * 2)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, input_time, self.d_model * 2))

        # Transformer encoder layers with increased feedforward dimension
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model * 2, 
            nhead=num_heads, 
            dropout=dropout_rate, 
            dim_feedforward=self.d_model * 4,  # Significantly larger feedforward layer
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Additional normalization layer after transformer
        self.norm = nn.LayerNorm(self.d_model * 2)

        # Fully connected layers with increased capacity
        self.fc1 = nn.Linear(self.d_model * 2 * input_time, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, channels, height, width, time = x.size()

        # Reshape input to (batch_size, time, features)
        x = x.permute(0, 4, 1, 2, 3)  # Move time dimension to position 1
        x = x.reshape(batch_size, time, -1)  # Flatten spatial dimensions

        # Project to higher d_model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer expects input shape (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Apply normalization
        x = self.norm(x)

        # Reshape back to (batch_size, sequence_length * d_model)
        x = x.transpose(0, 1)
        x = x.reshape(batch_size, -1)

        # Fully connected layers with dropout and residual connections
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    NUM_LAYERS = 1  # Increased number of layers
    NUM_HEADS = 8  # Increased number of attention heads
    model = SimpleTransformerV2(
        input_channels=6,
        input_height=5,
        input_width=5,
        input_time=5,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=0.3
    )

    print('The larger model has this many parameters: ', model.count_parameters())
