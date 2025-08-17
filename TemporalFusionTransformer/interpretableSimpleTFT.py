import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpretableSimpleTFT(nn.Module):
    def __init__(self, input_channels=6, height=5, width=5, time_steps=5, d_model=128, num_heads=2, dropout=0.3):
        super(InterpretableSimpleTFT, self).__init__()

        self.time_steps = time_steps
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.d_model = d_model

        # Variable selection network for feature importance
        self.variable_selection = nn.Sequential(
            nn.Linear(input_channels, input_channels * 2),
            nn.ReLU(),
            nn.Linear(input_channels * 2, input_channels),
            nn.Softmax(dim=-1)
        )

        # Store feature importance weights
        self.feature_weights = None
        self.attention_weights = None

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

        # Transformer encoder with attention capture
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Register hooks to capture attention weights
        self.register_attention_hooks()

        # Final projection
        self.head = nn.Sequential(
            nn.Linear(time_steps * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def register_attention_hooks(self):
        """Register hooks to capture attention weights from transformer layers"""
        def attention_hook(module, input, output):
            # For TransformerEncoderLayer, we need to access the self-attention module
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, '_attention_weights'):
                self.attention_weights = module.self_attn._attention_weights.detach()

        # Register hook on transformer encoder layers
        for layer in self.transformer_encoder.layers:
            layer.register_forward_hook(attention_hook)

            # Also register a hook specifically on the self-attention module to capture weights
            def self_attn_hook(module, input, output):
                # MultiheadAttention returns (attn_output, attn_output_weights)
                if len(output) > 1 and output[1] is not None:
                    self.attention_weights = output[1].detach()
                    # Store as an attribute for access in the parent hook
                    module._attention_weights = output[1].detach()
                return output

            layer.self_attn.register_forward_hook(self_attn_hook)

    def compute_feature_importance(self, x):
        """Compute feature importance using variable selection"""
        # x: [B, C, H, W, T]
        B, C, H, W, T = x.shape

        # Average across spatial and temporal dimensions to get feature representation
        feature_representation = x.mean(dim=(2, 3, 4))  # [B, C]

        # Apply variable selection network
        feature_weights = self.variable_selection(feature_representation)  # [B, C]
        self.feature_weights = feature_weights.detach()

        return feature_weights

    def forward(self, x, return_attention=False):
        # x: [B, C, H, W, T] or [B, T, H, W, C]

        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == self.time_steps:  # [B, T, H, W, C]
                x = x.permute(0, 4, 2, 3, 1)  # Convert to [B, C, H, W, T]
            # else assume [B, C, H, W, T]
        else:
            raise ValueError(f"Expected 5D input tensor, got {x.dim()}D")

        B, C, H, W, T = x.shape
        assert T == self.time_steps
        assert C == self.input_channels

        # Compute feature importance
        feature_weights = self.compute_feature_importance(x)  # [B, C]

        # Apply feature selection weights
        # Expand weights to match input dimensions
        feature_weights_expanded = feature_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, C, 1, 1, 1]
        x_weighted = x * feature_weights_expanded  # [B, C, H, W, T]

        # Move time to front and reshape for CNN: (B*T, C, H, W)
        x_weighted = x_weighted.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x_spatial = self.spatial_encoder(x_weighted)  # (B*T, 32, 4, 4)
        x_spatial = x_spatial.view(B, T, -1)  # (B, T, feature_dim)

        # Apply Gated Residual Network
        grn_out = self.grn(x_spatial)  # (B, T, d_model)
        gate = self.gate(x_spatial)  # (B, T, d_model)
        # Project original input to match d_model dimension for residual connection
        x_proj = self.residual_proj(x_spatial)  # (B, T, d_model)
        x_processed = self.layernorm(gate * grn_out + x_proj)  # (B, T, d_model)

        # Add positional embeddings
        x_processed = x_processed + self.pos_embedding  # (B, T, d_model)

        # Transformer expects: (T, B, d_model)
        x_transformer_input = x_processed.permute(1, 0, 2)
        x_transformer_output = self.transformer_encoder(x_transformer_input)  # (T, B, d_model)

        # Flatten and predict
        x_final = x_transformer_output.permute(1, 0, 2).reshape(B, -1)
        output = self.head(x_final)

        if return_attention:
            return output.squeeze(), self.attention_weights, self.feature_weights

        return output.squeeze()

    def get_temporal_importance(self):
        """Extract temporal importance from attention weights"""
        if self.attention_weights is None:
            # Return uniform weights if no attention weights captured
            return torch.ones(self.time_steps) / self.time_steps

        # Average attention weights across heads and queries
        # attention_weights shape: [B, num_heads, seq_len, seq_len]
        temporal_importance = self.attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]

        # Sum attention received by each position (column-wise sum)
        temporal_importance = temporal_importance.sum(dim=0)  # [seq_len]

        # Normalize to sum to 1
        temporal_importance = temporal_importance / temporal_importance.sum()

        return temporal_importance.detach()

    def get_feature_importance(self):
        """Get the latest feature importance weights"""
        if self.feature_weights is None:
            return torch.ones(self.input_channels) / self.input_channels

        # Return average across batch if multiple samples
        return self.feature_weights.mean(dim=0).detach()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def interpret_prediction(self, x):
        """
        Comprehensive interpretation of a single prediction
        Returns prediction along with temporal and feature importance
        """
        with torch.no_grad():
            prediction, attention_weights, feature_weights = self.forward(x.unsqueeze(0), return_attention=True)

            temporal_importance = self.get_temporal_importance()
            feature_importance = self.get_feature_importance()

            return {
                'prediction': prediction.item() if prediction.dim() == 0 else prediction.squeeze().cpu().numpy(),
                'temporal_importance': temporal_importance.cpu().numpy(),
                'feature_importance': feature_importance.cpu().numpy(),
                'raw_attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None,
                'raw_feature_weights': feature_weights.cpu().numpy() if feature_weights is not None else None
            }

# Compatibility wrapper to maintain the same interface as your original SimpleTFT
class SimpleTFT(InterpretableSimpleTFT):
    """
    Wrapper class that maintains backward compatibility with your original SimpleTFT
    while adding interpretability features
    """
    def __init__(self, input_channels=6, height=5, width=5, time_steps=5, d_model=128, num_heads=2, dropout=0.3):
        super(SimpleTFT, self).__init__(input_channels, height, width, time_steps, d_model, num_heads, dropout)

    def forward(self, x):
        # Default behavior - don't return attention weights unless specifically requested
        return super().forward(x, return_attention=False)

if __name__ == "__main__":
    # Test both models
    print("Testing InterpretableSimpleTFT:")
    model = InterpretableSimpleTFT()
    print("Number of trainable parameters:", model.count_parameters())

    # Test input
    x = torch.randn(2, 6, 5, 5, 5)  # [batch, channels, height, width, time]

    # Regular forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Forward pass with attention
    output, attention_weights, feature_weights = model(x, return_attention=True)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape if attention_weights is not None else 'None'}")
    print(f"Feature weights shape: {feature_weights.shape}")

    # Test interpretation
    single_sample = x[0]  # Take first sample
    interpretation = model.interpret_prediction(single_sample)
    print("\nInterpretation results:")
    print(f"Prediction: {interpretation['prediction']}")
    print(f"Temporal importance: {interpretation['temporal_importance']}")
    print(f"Feature importance: {interpretation['feature_importance']}")

    print("\nTesting backward compatibility SimpleTFT wrapper:")
    model_compat = SimpleTFT()
    output_compat = model_compat(x)
    print(f"Compatible output shape: {output_compat.shape}")
