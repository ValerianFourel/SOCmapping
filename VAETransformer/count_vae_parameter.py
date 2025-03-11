import torch
from modelTransformerVAE import TransformerVAE
from config import bands_list_order, window_size, time_before


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Model configuration matching your training script
    vae_model = TransformerVAE(
        input_channels=len(bands_list_order),  # Number of bands
        input_height=window_size,              # Spatial height
        input_width=window_size,               # Spatial width
        input_time=time_before,                # Temporal dimension
        num_heads=2,                           # Number of attention heads
        latent_dim=8,                          # Latent dimension size
        dropout_rate=0.3                       # Dropout rate
    )

    # Count parameters
    total_params = count_parameters(vae_model)
    
    # Print detailed breakdown (optional)
    print("TransformerVAE Parameter Breakdown:")
    for name, param in vae_model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
