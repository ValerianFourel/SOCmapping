import torch
import torch.nn as nn
from config import NUM_LAYERS, NUM_HEADS

class SimpleTransformer(nn.Module):
    def __init__(self, input_channels=6, input_height=33, input_width=33, input_time=4, num_heads=2, num_layers=1, dropout_rate=0.3):
        super(SimpleTransformer, self).__init__()

        # Calculate d_model ensuring it is divisible by num_heads
        self.d_model = input_channels * input_height # * input_width
        if self.d_model % num_heads != 0:
            self.d_model = (self.d_model // num_heads + 1) * num_heads

        # Linear projection to match d_model dimension
        self.input_projection = nn.Linear(input_channels * input_height * input_width, self.d_model)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dropout=dropout_rate, dim_feedforward=32)  # Reduced feedforward dimension)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(self.d_model * input_time, 16)
        self.fc2 = nn.Linear(16, 1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, channels, height, width, time = x.size()

        # Reshape input to (batch_size, time, features)
        x = x.permute(0, 4, 1, 2, 3)  # Move time dimension to position 1
        x = x.reshape(batch_size, time, -1)  # Flatten spatial dimensions

        # Project to d_model dimension
        x = self.input_projection(x)

        # Transformer expects input shape (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Reshape back to (batch_size, sequence_length * d_model)
        x = x.transpose(0, 1)
        x = x.reshape(batch_size, -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":

    model = SimpleTransformer(
                input_channels=6,
                input_height=5,
                input_width=5,
                input_time=5,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
                dropout_rate=0.3
            )

    print('The current model has that many parameters:  ',model.count_parameters())