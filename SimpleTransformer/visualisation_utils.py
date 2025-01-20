import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_output_target_difference(all_outputs, all_targets, epoch, run, save_dir='quickViz'):
    """
    Create and save a visualization comparing binned averages of targets and outputs.

    Parameters:
    -----------
    all_outputs : array-like
        Array of output values
    all_targets : array-like
        Array of target values
    epoch : int or str
        Current epoch number for filename
    run : int or str
        Run identifier for filename
    save_dir : str
        Directory to save the plot (default: 'quickViz')
    """
    # Convert inputs to numpy arrays if they aren't already
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)

    # Create bins
    n_bins = 50
    bins = np.linspace(min(all_targets), max(all_targets), n_bins + 1)
    bin_indices = np.digitize(all_targets, bins) - 1

    # Calculate averages for each bin
    target_means = np.zeros(n_bins)
    output_means = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            target_means[i] = np.mean(all_targets[mask])
            output_means[i] = np.mean(all_outputs[mask])

    # Calculate differences
    differences = target_means - output_means

    # Create the visualization
    fig, ax = plt.subplots(figsize=(15, 6))

    # Set the width of each bar and positions
    width = 0.35
    x = np.arange(n_bins)

        # Create bars
    ax.bar(x - width/2, target_means, width, label='Target Averages', color='blue', alpha=0.6)
    ax.bar(x + width/2, differences, width, label='Difference (Target - Output)', color='red', alpha=0.6)

    # Customize the plot
    ax.set_xlabel('Target Value (Bin Lower Bound)')
    ax.set_ylabel('Values')
    ax.set_title(f'Target Averages and Differences per Bin (Epoch {epoch}, Run {run})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set x-ticks to show bin lower bounds
    # Only show every nth tick to prevent overcrowding
    n_ticks_to_show = 5
    tick_indices = np.linspace(0, n_bins-1, n_ticks_to_show, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{bins[i]:.2f}' for i in tick_indices], rotation=45)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    filename = f'diffOutputTarget3dCNN_epoch_{epoch}_{run}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    return save_path  # Return the path where the image was saved


def analyze_oc_distribution(df, bins=50, width=60):
    """
    Display distribution and descriptive statistics of df['OC'] in command line

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'OC' column
    bins : int, optional
        Number of bins for histogram (default: 50)
    width : int, optional
        Width of the ASCII plot (default: 60)
    """
    import numpy as np

    # Basic input validation
    if 'OC' not in df.columns:
        raise ValueError("Column 'OC' not found in DataFrame")

    # Calculate histogram
    hist, bin_edges = np.histogram(df['OC'].dropna(), bins=bins)
    max_height = max(hist)

    # Calculate descriptive statistics
    stats = df['OC'].describe()

    # Print distribution visualization
    print("\nDistribution of OC values:")
    print("-" * width)

    for h in hist:
        bar_length = int(width * h / max_height)
        print('█' * bar_length + ' ' * (width - bar_length) + f' ({h})')

    print("-" * width)

    # Print descriptive statistics
    print("\nDescriptive Statistics:")
    print("-" * width)
    print(f"Count:         {stats['count']:.0f}")
    print(f"Mean:          {stats['mean']:.3f}")
    print(f"Std Dev:       {stats['std']:.3f}")
    print(f"Min:           {stats['min']:.3f}")
    print(f"25th Percent:  {stats['25%']:.3f}")
    print(f"Median:        {stats['50%']:.3f}")
    print(f"75th Percent:  {stats['75%']:.3f}")
    print(f"Max:           {stats['max']:.3f}")

    # Calculate additional statistics
    print(f"Skewness:      {df['OC'].skew():.3f}")
    print(f"Kurtosis:      {df['OC'].kurtosis():.3f}")
    print("-" * width)

# Example usage:
# analyze_oc_distribution(your_dataframe)

def visualize_batch_distributions(outputs, targets, n_bins=50, width=80, max_samples=1000):
    """
    Visualize the distribution of model outputs, targets, and their differences in the command line.

    Parameters:
    -----------
    outputs : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth values
    n_bins : int, optional
        Number of bins for histogram (default: 50)
    width : int, optional
        Width of the ASCII plot (default: 80)
    max_samples : int, optional
        Maximum number of samples to use for visualization (default: 1000)
    """
    import numpy as np
    import torch

    # Convert to numpy and flatten
    outputs_np = outputs.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()
    diff_np = outputs_np - targets_np

    # Sample if too many points
    if len(outputs_np) > max_samples:
        idx = np.random.choice(len(outputs_np), max_samples, replace=False)
        outputs_np = outputs_np[idx]
        targets_np = targets_np[idx]
        diff_np = diff_np[idx]

    def print_distribution(data, title, width=width, n_bins=n_bins):
        hist, bin_edges = np.histogram(data, bins=n_bins)
        max_height = max(hist)

        print(f"\n{title}")
        print("-" * width)

        for h in hist:
            bar_length = int(width * h / max_height)
            print('█' * bar_length + ' ' * (width - bar_length) + f' ({h})')

        # Print statistics
        print("-" * width)
        print(f"Mean: {np.mean(data):.3f}")
        print(f"Std:  {np.std(data):.3f}")
        print(f"Min:  {np.min(data):.3f}")
        print(f"Max:  {np.max(data):.3f}")
        print(f"25%:  {np.percentile(data, 25):.3f}")
        print(f"50%:  {np.percentile(data, 50):.3f}")
        print(f"75%:  {np.percentile(data, 75):.3f}")

    # Print distributions
    print_distribution(outputs_np, "Model Outputs Distribution")
    print_distribution(targets_np, "Targets Distribution")
    print_distribution(diff_np, "Difference (Outputs - Targets) Distribution")

    # Print additional metrics
    print("\nAdditional Metrics:")
    print("-" * width)
    print(f"Mean Absolute Error: {np.mean(np.abs(diff_np)):.3f}")
    print(f"Root Mean Square Error: {np.sqrt(np.mean(np.square(diff_np))):.3f}")
    print(f"Output/Target Correlation: {np.corrcoef(outputs_np, targets_np)[0,1]:.3f}")

# Example usage in your training loop:
"""
for batch_idx, (longitudes, latitudes, batch_features, batch_targets) in enumerate(pbar):
    batch_features = batch_features.float()
    batch_targets = batch_targets.float()

    optimizer.zero_grad()
    outputs = model(batch_features)
    loss = criterion(outputs, batch_targets)

    # Visualize distributions every N batches or when needed
    if batch_idx % visualization_interval == 0:
        visualize_batch_distributions(outputs, batch_targets)

    accelerator.backward(loss)
    optimizer.step()
"""
