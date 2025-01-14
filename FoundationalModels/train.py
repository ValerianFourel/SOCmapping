
from functools import partial

import torch
import torch.nn as nn
from IEEE_TPAMI_SpectralGPT.util import video_vit
from IEEE_TPAMI_SpectralGPT.util.logging import master_print as print
from IEEE_TPAMI_SpectralGPT.models_mae_spectral import MaskedAutoencoderViT
from config import imageSize

model = MaskedAutoencoderViT(
        img_size=imageSize,
        in_chans=1,
        patch_size=8,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        num_frames=18,
        pred_t_dim=100,
        t_patch_size=3,
        mask_ratio=0.90,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


class LatentMLP(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, dropout=0.1):
        """
        Args:
            input_dim (int): Dimension of input features (1280 from latent)
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout rate
        """
        super().__init__()

        # First we'll process each token, then aggregate
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # After aggregating tokens, final MLP layers
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)  # Final output dimension of 1
        )

    def forward(self, latent):
        """
        Args:
            latent: Tensor of shape [batch_size, num_tokens, embedding_dim]
                   In this case [2, 25, 1280]
        Returns:
            output: Tensor of shape [batch_size, 1]
                   In this case [2, 1]
        """
        # Process each token: [2, 25, 1280] -> [2, 25, hidden_dim]
        token_features = self.token_mlp(latent)

        # Average across tokens: [2, 25, hidden_dim] -> [2, hidden_dim]
        pooled_features = torch.mean(token_features, dim=1)

        # Final MLP to get output: [2, hidden_dim] -> [2, 1]
        output = self.final_mlp(pooled_features)

        return output

# Example usage:
def process_latent(latent, target=None):
    """
    Args:
        latent: Tensor from encoder [2, 25, 1280]
        target: Optional target values for training [2, 1]
    """
    # Initialize model
    model = LatentMLP(input_dim=1280)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    latent = latent.to(device)

    # Forward pass
    output = model(latent)

    if target is not None:
        # Define loss function (MSE for regression)
        criterion = nn.MSELoss()
        target = target.to(device)
        loss = criterion(output, target)

        # Example of backward pass (if training)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return output, loss

    return output

# Example of how to use it:
"""
# Your latent tensor
latent = torch.randn(2, 25, 1280)

# For training:
target = torch.randn(2, 1)  # Example target values
output, loss = process_latent(latent, target)
print(f"Output shape: {output.shape}")  # [2, 1]
print(f"Loss: {loss.item()}")

# For inference:
output = process_latent(latent)
print(f"Output shape: {output.shape}")  # [2, 1]
"""

def print_tensor_shapes(item, prefix=''):
    if isinstance(item, torch.Tensor):
        print(f"{prefix}Tensor shape:", item.shape)
    elif isinstance(item, (list, tuple)):
        print(f"{prefix}List/Tuple of length:", len(item))
        for i, subitem in enumerate(item):
            print_tensor_shapes(subitem, prefix=f"{prefix}[{i}] ")
    elif isinstance(item, dict):
        for key, value in item.items():
            print_tensor_shapes(value, prefix=f"{prefix}['{key}'] ")
    else:
        print(f"{prefix}Not a tensor, type:", type(item))


if __name__ == '__main__':
    input = torch.rand(2, 18, imageSize, imageSize)
    output = model.forward_encoder(input, 0.90)
    print('check size of the output \n \n')
    print_tensor_shapes(output)
    print('check output MLP \n \n')
    print_tensor_shapes(process_latent(output[0]))
