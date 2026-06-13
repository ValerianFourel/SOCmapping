import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSGT(nn.Module):
    """Compact gated CNN+Transformer flagship (~84k params at d=32).

    Crispness change (bestrun-bands, 2026-06): the head now emits a DIRECT
    LINEAR baseline plus an MLP residual, i.e. ``out = linear_skip(feat) +
    mlp(norm(feat))`` — the same mechanism the crisp original SGT (EnhancedSGT)
    uses ("linear skip ... crucial for heavy-tailed targets"). The plain MLP
    head regresses toward the mean, which is what made the production map look
    soft and high-biased; the linear skip restores dynamic range so high-SOC
    pockets are predicted sharply (a crisper map) without touching the
    encoder/GRN/transformer that give the good spatial-CV fit. It adds only a
    few hundred parameters, so the "smaller is better" flagship is preserved.

    Set ``use_linear_skip=False`` to recover the original pre-change head for an
    A/B comparison (gate any crispness gain on spatial-CV R^2 staying >= 0.377).
    """

    def __init__(self, input_channels=6, height=5, width=5, time_steps=5,
                 d_model=128, num_heads=2, dropout=0.3, use_linear_skip=True,
                 spatial_pool='avg', use_static_head=True, head_hidden=64,
                 use_film=False):
        super(SimpleSGT, self).__init__()

        self.time_steps = time_steps
        self.use_linear_skip = use_linear_skip
        self.spatial_pool = spatial_pool
        self.use_static_head = use_static_head
        self.use_film = use_film

        # CNN to extract spatial features per timestep
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Spatial pooling to a fixed 4x4 grid. 'avgmax' concatenates average and
        # max pooling: avg preserves the smooth signal (fit), max preserves
        # sharp local features/edges (crispness, as in the CNN-LSTM front-end).
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.max_pool = nn.AdaptiveMaxPool2d((4, 4))
        pool_mult = 2 if spatial_pool == 'avgmax' else 1
        self.feature_dim = 32 * 4 * 4 * pool_mult  # 512 (avg) or 1024 (avgmax)

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

        # === Head: linear skip (dynamic range / crispness) + MLP residual ===
        feat_dim = time_steps * d_model
        self.head_norm = nn.LayerNorm(feat_dim)
        self.head_mlp = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )
        # FiLM: static terrain/soil covariates predict a (scale, shift) that
        # MODULATES the spatiotemporal feature before the head — lets terrain
        # gate the whole prediction (mountain vs plain regime) multiplicatively,
        # far more expressive than the additive static head.
        self.film = nn.Linear(input_channels, 2 * feat_dim) if use_film else None
        # Direct linear baseline from the pooled temporal features. Lets the
        # network emit a high-dynamic-range linear regression and learn only the
        # residual via the MLP — keeps sharp high-SOC predictions instead of
        # collapsing toward the mean.
        self.linear_skip = nn.Linear(feat_dim, 1) if use_linear_skip else None

        # === SHARP static-covariate head =================================
        # The mountain-high / plain-low SOC differential is driven by the
        # static topographic + soil covariates (Elevation, Slope, Aspect, TWI,
        # TPI, TRI, Roughness, Clay/Sand/pH/BulkDensity/CEC). The CNN avg-pools
        # them (smooths) and the transformer treats them as temporal. This head
        # reads the covariate vector at the EXACT location (centre pixel,
        # un-pooled) and adds its contribution directly to the output, so the
        # terrain/soil drivers hit the prediction sharply -> a crisper map with
        # a stronger high/low differential.
        self.static_head = nn.Sequential(
            nn.Linear(input_channels, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        ) if use_static_head else None

    def _spatial(self, x):
        x = self.conv(x)
        if self.spatial_pool == 'avgmax':
            return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        if self.spatial_pool == 'max':
            return self.max_pool(x)
        return self.avg_pool(x)

    def forward(self, x):
        # x: [B, C, H, W, T]
        B, C, H, W, T = x.shape
        assert T == self.time_steps

        # centre-pixel covariate vector (un-pooled, time-averaged) for the sharp
        # static head — the exact location's terrain/soil signature.
        centre = x[:, :, H // 2, W // 2, :].mean(dim=-1)  # (B, C)

        # Move time to front and reshape for CNN: (B*T, C, H, W)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x = self._spatial(x)  # (B*T, 32*pool_mult, 4, 4)
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

        # Flatten temporal features and predict (linear skip + MLP residual)
        feat = x.permute(1, 0, 2).reshape(B, -1)  # (B, T*d_model)
        if self.film is not None:                 # FiLM terrain modulation
            g, b = self.film(centre).chunk(2, dim=-1)
            feat = feat * (1.0 + torch.tanh(g)) + b
        out = self.head_mlp(self.head_norm(feat))
        if self.linear_skip is not None:
            out = out + self.linear_skip(feat)
        if self.static_head is not None:
            out = out + self.static_head(centre)   # sharp terrain/soil term

        return out.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    for pool in ('avg', 'avgmax'):
        for skip in (False, True):
            m = SimpleSGT(input_channels=43, d_model=32, num_heads=2,
                          use_linear_skip=skip, spatial_pool=pool)
            x = torch.randn(4, 43, 5, 5, 5)
            y = m(x)
            print(f"pool={pool:6s} skip={skip!s:5s} params={m.count_parameters():,} "
                  f"out={tuple(y.shape)}")
