import torch
from models import RefittedCovLSTM  # Replace 'your_module' with the actual module name
from config import bands_list_order, window_size, time_before


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Model configuration matching your dataset
    model = RefittedCovLSTM(
        num_channels=len(bands_list_order),  # Number of input bands
        lstm_input_size=128,                 # Fixed by CNN output
        lstm_hidden_size=128,                # LSTM hidden state size
        num_layers=2,                        # Number of LSTM layers
        dropout=0.25                         # Dropout rate
    )

    # Count parameters
    total_params = count_parameters(model)
    
    # Print detailed breakdown
    print("RefittedCovLSTM Parameter Breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
