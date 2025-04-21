import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTFT(nn.Module):
    def __init__(self, input_channels=6, height=5, width=5, time_steps=5, d_model=128, num_heads=2, dropout=0.3):
        super(SimpleTFT, self).__init__()

        self.time_steps = time_steps

        # CNN to extract spatial features per timestep
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Downsample to fixed spatial size
        )

        # Calculate flattened feature dimension
        self.feature_dim = 32 * 4 * 4  # 512

        # Gated residual network (simplified GRN block)
        self.grn = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.Sigmoid()
        )
        # Add a projection layer to match dimensions of x to d_model for residual connection
        self.residual_proj = nn.Linear(self.feature_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model)

        # Positional encoding for temporal dynamics
        self.pos_embedding = nn.Parameter(torch.randn(time_steps, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final projection
        self.head = nn.Sequential(
            nn.Linear(time_steps * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W, T]
        B, C, H, W, T = x.shape
        assert T == self.time_steps

        # Move time to front and reshape for CNN: (B*T, C, H, W)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x = self.spatial_encoder(x)  # (B*T, 32, 4, 4)
        x = x.view(B, T, -1)  # (B, T, feature_dim)

        # Apply Gated Residual Network
        grn_out = self.grn(x)  # (B, T, d_model)
        gate = self.gate(x)  # (B, T, d_model)
        # Project original input to match d_model dimension for residual connection
        x_proj = self.residual_proj(x)  # (B, T, d_model)
        x = self.layernorm(gate * grn_out + x_proj)  # (B, T, d_model)

        # Add positional embeddings
        x = x + self.pos_embedding  # (B, T, d_model)

        # Transformer expects: (T, B, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)  # (T, B, d_model)

        # Flatten and predict
        x = x.permute(1, 0, 2).reshape(B, -1)
        x = self.head(x)

        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SimpleTFT()
    print("Number of trainable parameters:", model.count_parameters())
