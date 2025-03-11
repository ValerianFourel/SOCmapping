import torch
from model_transformer import TransformerRegressor
from config import bands_list_order, time_before, window_size


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Configuration matching your training script
    num_patches = (96 // 8) * (96 // 8)  # Fixed patch size from training script (96x96 with patch_size=8)
    embedding_dim_per_patch = 768        # Default ViT embedding dimension (vit_base_patch8)
    total_embedding_dim = embedding_dim_per_patch * num_patches * (1 + (len(bands_list_order) - 1) * time_before)
    num_tokens = 1 + (len(bands_list_order) - 1) * time_before  # Elevation + other bands * time_before

    # Initialize the TransformerRegressor model
    model = TransformerRegressor(
        input_dim=total_embedding_dim,  # Total embedding dimension from ViT
        num_tokens=num_tokens,          # Number of tokens (Elevation + bands * time)
        d_model=512,                    # Transformer model dimension
        nhead=8,                        # Number of attention heads
        num_layers=2,                   # Number of transformer layers
        dim_feedforward=1024,           # Feedforward dimension
        output_dim=1                    # Regression output (OC)
    )

    # Count parameters
    total_params = count_parameters(model)
    
    # Print detailed breakdown
    print("SpectralGPTPlus TransformerRegressor Parameter Breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
