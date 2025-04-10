# balanced_dataset.py
import pandas as pd
import numpy as np
seed=42

def create_balanced_dataset(df, use_validation=False, n_bins=128, min_ratio=3/4):
    """
    Creates a balanced dataset by binning and resampling based on the 'OC' column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing 'OC' column
    use_validation : bool, optional
        Whether to create a validation set (default: False)
    n_bins : int, optional
        Number of bins for qcut (default: 128)
    min_ratio : float, optional
        Minimum ratio of samples per bin relative to max (default: 3/4)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    tuple
        (training_df, validation_df) if use_validation=True
        (training_df, None) if use_validation=False
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create bins based on 'OC' column
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)
    
    training_dfs = []
    
    if use_validation:
        validation_indices = []
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) >= 4:
                val_samples = bin_data.sample(n=min(13, len(bin_data)), random_state=seed)
                validation_indices.extend(val_samples.index)
                train_samples = bin_data.drop(val_samples.index)
                if len(train_samples) > 0:
                    if len(train_samples) < min_samples:
                        resampled = train_samples.sample(n=min_samples, replace=True, random_state=seed)
                        training_dfs.append(resampled)
                    else:
                        training_dfs.append(train_samples)
        
        if not training_dfs or not validation_indices:
            raise ValueError("No training or validation data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        validation_df = df.loc[validation_indices].drop('bin', axis=1)
        return training_df, validation_df
    
    else:
        for bin_idx in range(len(bin_counts)):
            bin_data = df[df['bin'] == bin_idx]
            if len(bin_data) > 0:
                if len(bin_data) < min_samples:
                    resampled = bin_data.sample(n=min_samples, replace=True, random_state=seed)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(bin_data)
        
        if not training_dfs:
            raise ValueError("No training data available after binning")
        
        training_df = pd.concat(training_dfs).drop('bin', axis=1)
        return training_df, None