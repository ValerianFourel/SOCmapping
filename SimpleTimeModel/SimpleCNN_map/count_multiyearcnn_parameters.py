import torch
from modelCNNMultiYear import Small3DCNN
from config import bands_list_order, window_size, time_before


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Model configuration matching your training script
    model = Small3DCNN(
        input_channels=len(bands_list_order),  # Number of input bands
        input_height=window_size,              # Spatial height
        input_width=window_size,               # Spatial width
        input_time=time_before                 # Temporal dimension
    )

    # Count parameters
    total_params = count_parameters(model)
    
    # Print detailed breakdown
    print("MultiYearCNN (Small3DCNN) Parameter Breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
