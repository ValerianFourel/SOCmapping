import torch
import torch.nn as nn



class TransformerForRegression(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, hidden_dim, seq_len, spatial_dim, output_dim):
        super(TransformerForRegression, self).__init__()
        
        # Flatten spatial dimensions into the sequence
        self.spatial_dim = spatial_dim
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding for temporal + spatial dimension
        total_seq_len = seq_len * spatial_dim * spatial_dim
        self.positional_encoding = nn.Parameter(torch.zeros(1, total_seq_len, hidden_dim))
        
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Fully connected layer for regression output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten spatial dimensions
        batch_size, seq_len, h, w, feature_dim = x.shape
        x = x.view(batch_size, seq_len * h * w, feature_dim)  # Shape: (batch, total_seq_len, feature_dim)
        
        # Embedding and positional encoding
        x = self.embedding(x) + self.positional_encoding
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling across the sequence
        x = x.mean(dim=1)  # Shape: (batch, hidden_dim)
        
        # Regression output
        return self.fc(x)
        s
def get_trainable_params(model):
    """
    Returns the number of trainable parameters in the model.

    Args:
        model: PyTorch model instance

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example instantiation
feature_dim = 6          # Number of features
num_heads = 4            # Number of attention heads
num_layers = 2           # Number of transformer layers
hidden_dim = 128         # Hidden dimension of transformer
seq_len = 10             # Temporal sequence length
spatial_dim = 32         # Spatial dimensions (32x32 grid)
output_dim = 1           # Single regression output

model = TransformerForRegression(feature_dim, num_heads, num_layers, hidden_dim, seq_len, spatial_dim, output_dim)
