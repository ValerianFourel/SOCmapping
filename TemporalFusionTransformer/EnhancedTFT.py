import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedTFT(nn.Module):
    def __init__(self, input_channels=6, height=5, width=5, time_steps=5, d_model=128, num_heads=4, 
                 dropout=0.3, num_encoder_layers=3, expansion_factor=4):
        super(EnhancedTFT, self).__init__()

        self.time_steps = time_steps

        # Enhanced CNN to extract more complex spatial features per timestep
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Downsample to fixed spatial size
        )

        # Calculate flattened feature dimension
        self.feature_dim = 128 * 4 * 4  # 2048

        # Feature projection to d_model
        self.feature_projection = nn.Linear(self.feature_dim, d_model)

        # Multiple GRN blocks for hierarchical feature processing
        self.grn_blocks = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout) 
            for _ in range(3)
        ])

        # Enhanced positional encoding with learned and sinusoidal components
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=time_steps)

        # Multi-layer transformer encoder with more heads and larger feed-forward network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout, 
            dim_feedforward=d_model * expansion_factor,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Attention pooling to focus on important time steps
        self.attention_pooling = AttentionPooling(d_model)

        # Enhanced projection head with residual connections
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W, T]
        B, C, H, W, T = x.shape
        assert T == self.time_steps

        # Process each timestep through CNN: (B*T, C, H, W)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x = self.spatial_encoder(x)  # (B*T, 128, 4, 4)
        x = x.view(B, T, self.feature_dim)  # (B, T, feature_dim)

        # Project features to d_model dimension
        x = self.feature_projection(x)  # (B, T, d_model)

        # Apply multiple GRN blocks
        for grn in self.grn_blocks:
            x = grn(x)  # (B, T, d_model)

        # Add positional encodings
        x = self.pos_encoding(x)  # (B, T, d_model)

        # Apply transformer encoder (already batch_first=True)
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # Apply attention pooling to focus on important timesteps
        x = self.attention_pooling(x)  # (B, d_model)

        # Final prediction
        x = self.head(x)  # (B, 1)

        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(GatedResidualNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dim, output_dim)
        self.layer2 = nn.Linear(output_dim, output_dim)

        # Skip connection with dimension matching if needed
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

        # Normalization and regularization
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Main network branch
        h = F.elu(self.layer1(x))
        h = self.dropout(h)
        h = self.layer2(h)

        # Skip connection
        x_skip = self.skip_proj(x)

        # Gating mechanism
        g = self.gate(x)

        # Combine with residual connection
        output = self.layernorm(g * h + (1 - g) * x_skip)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Learnable position embedding component
        self.learned_pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # Add positional encoding
        x = x + self.pe[:x.size(1), :] + self.learned_pe[:x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        attention_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        weighted_sum = torch.sum(attention_weights * x, dim=1)  # (batch_size, d_model)
        return weighted_sum


if __name__ == "__main__":
    model = EnhancedTFT()
    print("Number of trainable parameters:", model.count_parameters())

    # Test with dummy input
    batch_size = 8
    dummy_input = torch.randn(batch_size, 6, 5, 5, 5)  # [B, C, H, W, T]
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
