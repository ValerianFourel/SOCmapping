import torch
import torch.nn as nn

# ======== HELPER MODULES for TFT ========

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(GatedLinearUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        sig_out = self.sigmoid(self.fc1(x))
        lin_out = self.fc2(x)
        gated_out = sig_out * lin_out
        return self.dropout(gated_out)

class GatedResidualNetwork(nn.Module):
    """Simplified Gated Residual Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.glu(x)
        out = self.layer_norm(self.dropout(x) + residual)
        return out

# ======== MODIFIED TEMPORAL FUSION TRANSFORMER ========

class TemporalFusionTransformerRegressor(nn.Module):
    def __init__(self, embedding_dim=110592, num_channels=5, num_temporal_steps=5, d_model=128, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TemporalFusionTransformerRegressor, self).__init__()

        self.embedding_dim = embedding_dim  # 110592
        self.num_channels = num_channels    # 5
        self.num_temporal_steps = num_temporal_steps  # 5
        self.total_temporal_steps = num_channels * num_temporal_steps  # 25
        self.d_model = d_model

        # --- Input Projections ---
        self.static_proj = nn.Linear(embedding_dim, d_model)  # [110592] -> [d_model]
        self.temporal_proj = nn.Linear(embedding_dim, d_model)  # [110592] -> [d_model]

        # --- Static Feature Processing ---
        self.static_context_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # --- Temporal Feature Processing (Transformer Encoder) ---
        self.pos_encoder = nn.Parameter(torch.randn(1, self.total_temporal_steps, d_model))  # [1, 25, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # --- Static Enrichment ---
        self.enrichment_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.enrichment_gate = GatedLinearUnit(d_model, d_model, dropout)

        # --- Fusion ---
        self.fusion_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # --- Output Layer ---
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, embeddings_elevation_instrument, embeddings_remaining_tensor):
        """
        Args:
            embeddings_elevation_instrument: [batch_size, 1, embedding_dim], e.g., [64, 1, 110592]
            embeddings_remaining_tensor: [batch_size, num_channels, num_temporal_steps, embedding_dim], e.g., [64, 5, 5, 110592]
        Returns:
            output: [batch_size, output_dim], e.g., [64, 1]
        """
        batch_size = embeddings_elevation_instrument.size(0)

        # 1. Project Static Features
        static_features = embeddings_elevation_instrument.squeeze(1)  # [64, 110592]
        static_proj = self.static_proj(static_features)  # [64, d_model]
        static_context = self.static_context_grn(static_proj)  # [64, d_model]
        static_context_expanded = static_context.unsqueeze(1).expand(-1, self.total_temporal_steps, -1)  # [64, 25, d_model]

        # 2. Project Temporal Features
        temporal_features = embeddings_remaining_tensor.view(batch_size, self.total_temporal_steps, self.embedding_dim)  # [64, 25, 110592]
        temporal_proj = self.temporal_proj(temporal_features)  # [64, 25, d_model]

        # 3. Process Temporal Features
        temporal_encoded = temporal_proj + self.pos_encoder[:, :self.total_temporal_steps, :]  # [64, 25, d_model]
        temporal_attended = self.transformer_encoder(temporal_encoded)  # [64, 25, d_model]

        # 4. Static Enrichment
        combined_for_enrichment = temporal_attended + static_context_expanded  # [64, 25, d_model]
        enriched_temporal = self.enrichment_grn(combined_for_enrichment)  # [64, 25, d_model]
        gated_enriched_temporal = self.enrichment_gate(enriched_temporal) * temporal_attended  # [64, 25, d_model]

        # 5. Fusion
        pooled_features = gated_enriched_temporal.mean(dim=1)  # [64, d_model]
        final_representation = self.fusion_grn(pooled_features)  # [64, d_model]

        # 6. Output Prediction
        output = self.output_layer(final_representation)  # [64, output_dim]

        return output

    def count_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage and parameter counting:
if __name__ == "__main__":
    # Dummy data
    batch_size = 64
    embedding_dim = 110592
    num_channels = 5
    num_temporal_steps = 5
    d_model = 128
    output_dim = 1

    embeddings_elevation = torch.randn(batch_size, 1, embedding_dim)
    embeddings_remaining = torch.randn(batch_size, num_channels, num_temporal_steps, embedding_dim)

    # Initialize model
    model = TemporalFusionTransformerRegressor(
        embedding_dim=embedding_dim,
        num_channels=num_channels,
        num_temporal_steps=num_temporal_steps,
        d_model=d_model,
        output_dim=output_dim
    )

    # Forward pass
    output = model(embeddings_elevation, embeddings_remaining)
    print("Output shape:", output.shape)  # [64, 1]

    # Count and print total parameters
    total_params = model.count_parameters()
    print(f"Total trainable parameters: {total_params:,}")