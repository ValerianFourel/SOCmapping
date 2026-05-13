import torch
import torch.nn as nn

# Small Transformer for regression
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=2875392, num_tokens=26, d_model=512, nhead=8, num_layers=2, dim_feedforward=1024, output_dim=1):
        super(TransformerRegressor, self).__init__()
        
        # Input projection: map large input_dim per token to d_model
        self.token_dim = input_dim // num_tokens  # e.g., 110592 = 2875392 / 26
        self.input_proj = nn.Linear(self.token_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: pool transformer output and map to single value
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool across tokens
        self.fc_out = nn.Linear(d_model, output_dim)
        
        # Positional encoding (optional, can be omitted if order doesn’t matter)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.token_dim)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc_out(x)
        return x

# MLP for regression
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=2875392, hidden_dim=512, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

