import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedTFT(nn.Module):
    def __init__(self, 
                 input_channels=6, 
                 height=33, 
                 width=33, 
                 time_steps=5, 
                 d_model=128, 
                 num_heads=4, 
                 dropout=0.3,
                 num_scales=3):
        super(EnhancedTFT, self).__init__()

        self.time_steps = time_steps
        self.d_model = d_model
        self.num_scales = num_scales

        # Multi-scale spatial encoder
        self.spatial_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 16 * (2**i), kernel_size=3, padding=1),
                nn.BatchNorm2d(16 * (2**i)),
                nn.ReLU(),
                nn.Conv2d(16 * (2**i), 32 * (2**i), kernel_size=3, padding=1),
                nn.BatchNorm2d(32 * (2**i)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            ) for i in range(num_scales)
        ])

        # Calculate feature dimensions for each scale
        self.feature_dims = [32 * (2**i) * 4 * 4 for i in range(num_scales)]
        self.total_feature_dim = sum(self.feature_dims)

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.total_feature_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # Enhanced Gated Linear Unit (GLU)
        self.glu = nn.ModuleDict({
            'linear': nn.Linear(d_model, d_model * 2),
            'gate': nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
        })
        
        # Variable Selection Network
        self.var_selection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softmax(dim=-1)
        )
        
        # Positional encoding with learnable parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, time_steps, d_model))
        self.scale_embedding = nn.Parameter(torch.randn(1, num_scales, d_model))

        # Multi-layer Transformer with cross-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout, 
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Static context enrichment
        self.static_context = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # Output projection for single value
        self.head = nn.Sequential(
            nn.Linear(time_steps * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, static_features=None):
        # x: [B, C, H, W, T]
        B, C, H, W, T = x.shape
        assert T == self.time_steps

        # Multi-scale spatial feature extraction
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        scale_features = []
        for encoder in self.spatial_encoders:
            feats = encoder(x)
            scale_features.append(feats.view(B, T, -1))
        
        # Concatenate multi-scale features
        x = torch.cat(scale_features, dim=-1)  # [B, T, total_feature_dim]
        
        # Scale fusion
        x = self.scale_fusion(x)  # [B, T, d_model]

        # Enhanced GLU
        glu_out = self.glu['linear'](x)
        linear, gate = glu_out.chunk(2, dim=-1)
        gate = self.glu['gate'](x)
        x = linear * gate  # [B, T, d_model]

        # Variable selection
        weights = self.var_selection(x)
        x = x * weights  # [B, T, d_model]

        # Add positional and scale embeddings
        x = x + self.pos_embedding
        if self.num_scales > 1:
            x = x + self.scale_embedding.mean(dim=1)

        # Incorporate static features if provided
        if static_features is not None:
            static_context = self.static_context(static_features).unsqueeze(1)
            x = x + static_context

        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, T, d_model]

        # Output projection
        x = x.reshape(B, -1)
        output = self.head(x)  # [B, 1]
        
        return output.squeeze(-1)  # [B]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Example usage
    model = EnhancedTFT(
        input_channels=6,
        height=33,
        width=33,
        time_steps=5,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        num_scales=3
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 6, 33, 33, 5)
    static_features = torch.randn(batch_size, 128)
    output = model(x, static_features)
    
    print("Number of trainable parameters:", model.count_parameters())
    print(f"Output shape: {output.shape}")
