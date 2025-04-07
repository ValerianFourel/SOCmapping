import torch
import torch.nn as nn
from modelTransformerVAE import TransformerVAE  # Import your TransformerVAE class
from config import window_size, bands_list_order  # Import config variables

# Hyperparameters from your script
VAE_NUM_HEADS = 2
VAE_LATENT_DIM_ELEVATION = 16
VAE_LATENT_DIM_OTHERS = 32
VAE_DROPOUT_RATE = 0.3
window_size =33

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Dictionary to store parameter counts
    param_counts = {}

    # Iterate over each band
    for band in bands_list_order:
        # Set band-specific hyperparameters
        latent_dim = VAE_LATENT_DIM_ELEVATION if band == 'Elevation' else VAE_LATENT_DIM_OTHERS
        num_heads = VAE_NUM_HEADS if band == 'Elevation' else VAE_NUM_HEADS * 2

        # Instantiate the VAE model
        vae = TransformerVAE(
            input_channels=1,
            input_height=window_size,
            input_width=window_size,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout_rate=VAE_DROPOUT_RATE
        )

        # Count parameters
        num_params = count_parameters(vae)
        param_counts[band] = num_params

        # Print result
        print(f"Band: {band}, Number of trainable parameters: {num_params}")

    # Summary
    print("\nSummary of parameter counts:")
    for band, count in param_counts.items():
        print(f"{band}: {count}")
