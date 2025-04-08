import torch
from modelCNN import SmallCNN  # Assuming this is where SimpleTimeCNN is defined
from config import bands_list_order  # Using bands_list_order for input channels


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Model configuration matching your training script
    # Assuming input_channels corresponds to the number of bands
    model = SmallCNN(input_channels=len(bands_list_order))

    # Count parameters
    total_params = count_parameters(model)
    
    # Print detailed breakdown
    print("SimpleTimeCNN (SmallCNN) Parameter Breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
