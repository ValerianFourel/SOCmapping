import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnhancedSGT(nn.Module):
    """Spatiotemporal Gated Transformer — V2 with:
      - Per-band grouped input projection so heterogeneous bands (NDVI, LST,
        pH, CEC, terrain, ...) get independent learned scaling before mixing.
      - GELU activations and GroupNorm (over BatchNorm) for stability with
        small batches and DDP.
      - Wider head with linear skip — lets the model emit a direct linear
        regression of pooled features and learn only the residual via the
        MLP. This is critical for heavy-tailed SOC targets (0.8-150 g/kg),
        where the V1 narrowing head with heavy dropout collapsed predictions
        to a narrow band around the mean.
      - Less aggressive spatial pooling (5x5 -> 3x3, not 4x4) to preserve
        more spatial structure on the small 5x5 input window.
    """

    def __init__(self, input_channels=20, height=5, width=5, time_steps=5,
                 d_model=128, num_heads=4, dropout=0.3, num_encoder_layers=3,
                 expansion_factor=4, per_band_features=4):
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels

        # === BOTTOM: per-band 1x1 projection, then mix ============
        # Grouped 1x1 conv: each of `input_channels` bands gets its own
        # independent linear projection to `per_band_features` learned
        # channels. NDVI's per-band weights are decoupled from pH's, LST's,
        # CEC's, etc. — they no longer share parameters in the first layer.
        self.per_band_proj = nn.Conv2d(
            input_channels,
            input_channels * per_band_features,
            kernel_size=1,
            groups=input_channels,   # one independent conv per band
        )
        mid_channels = 96
        # Cross-band mixing (1x1 conv with all groups combined)
        self.channel_mix = nn.Sequential(
            nn.Conv2d(input_channels * per_band_features, mid_channels, kernel_size=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
        )

        # === SPATIAL ENCODER (kept compact for 5x5 input) ============
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(mid_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)),   # 5x5 -> 3x3 (was 4x4, less aggressive)
        )
        self.feature_dim = 128 * 3 * 3       # 1152

        self.feature_projection = nn.Linear(self.feature_dim, d_model)

        # === MIDDLE: GRN + positional + transformer + attention pool ===
        self.grn_blocks = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout)
            for _ in range(3)
        ])
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=time_steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=d_model * expansion_factor,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)
        self.attention_pooling = AttentionPooling(d_model)

        # === TOP: widen-then-narrow head + linear skip ==============
        # The MLP head outputs a residual ON TOP of a direct linear projection
        # of pooled features. The linear skip gives the model a strong
        # "linear regression baseline" that can reach extreme SOC values
        # (peatlands at 100-150 g/kg) without fighting through dropout.
        head_hidden = d_model * 2            # widen
        self.head_norm = nn.LayerNorm(d_model)
        self.head_mlp = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),       # much less dropout right before output
            nn.Linear(d_model, 1),
        )
        # Direct linear skip from pooled features. Crucial for heavy-tailed
        # regression: the MLP head only learns corrections to this baseline.
        self.linear_skip = nn.Linear(d_model, 1)

    def forward(self, x):
        """x: (B, C, H, W, T) — C=input_channels, T=time_steps"""
        B, C, H, W, T = x.shape
        assert T == self.time_steps, f"expected T={self.time_steps}, got {T}"
        assert C == self.input_channels, (
            f"expected C={self.input_channels}, got {C}")

        # Process each timestep through the CNN stack independently
        # (B, C, H, W, T) -> (B*T, C, H, W)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)

        # Per-band projection + cross-band mix
        x = self.per_band_proj(x)            # (B*T, C * per_band_features, H, W)
        x = self.channel_mix(x)              # (B*T, mid_channels, H, W)

        # Spatial encoder
        x = self.spatial_encoder(x)          # (B*T, 128, 3, 3)
        x = x.reshape(B, T, self.feature_dim)
        x = self.feature_projection(x)       # (B, T, d_model)

        # GRN blocks
        for grn in self.grn_blocks:
            x = grn(x)

        # Transformer with positional encoding
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)      # (B, T, d_model)

        # Attention pool across time
        pooled = self.attention_pooling(x)   # (B, d_model)

        # Linear skip baseline + MLP residual
        baseline = self.linear_skip(pooled)              # (B, 1)
        residual = self.head_mlp(self.head_norm(pooled)) # (B, 1)
        out = baseline + residual                        # (B, 1)

        return out.squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.layer2 = nn.Linear(output_dim, output_dim)
        self.skip_proj = (nn.Linear(input_dim, output_dim)
                          if input_dim != output_dim else nn.Identity())
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.layer1(x))     # GELU > ELU here
        h = self.dropout(h)
        h = self.layer2(h)
        x_skip = self.skip_proj(x)
        g = self.gate(x)
        return self.layernorm(g * h + (1 - g) * x_skip)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.learned_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :] + self.learned_pe[:x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        attn = F.softmax(self.attention(x), dim=1)
        return (attn * x).sum(dim=1)


if __name__ == "__main__":
    model = EnhancedSGT(input_channels=20)
    print(f"Parameters: {model.count_parameters():,}")
    out = model(torch.randn(8, 20, 5, 5, 5))
    print(f"Output shape: {out.shape}")
