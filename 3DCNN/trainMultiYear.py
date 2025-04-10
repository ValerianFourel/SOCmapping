import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,
                   seasons, years_padded, num_epochs,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from torch.utils.data import Dataset, DataLoader
from modelCNNMultiYear import Small3DCNN
from accelerate import Accelerator
import argparse

def create_balanced_dataset(df, n_bins=128, min_ratio=3/4):
    bins = pd.qcut(df['OC'], q=n_bins, labels=False, duplicates='drop')
    df['bin'] = bins
    bin_counts = df['bin'].value_counts()
    max_samples = bin_counts.max()
    min_samples = max(int(max_samples * min_ratio), 5)

    validation_indices = []
    training_dfs = []

    for bin_idx in range(len(bin_counts)):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) >= 4:
            val_samples = bin_data.sample(n=min(13, len(bin_data)))
            validation_indices.extend(val_samples.index)
            train_samples = bin_data.drop(val_samples.index)
            if len(train_samples) > 0:
                if len(train_samples) < min_samples:
                    resampled = train_samples.sample(n=min_samples, replace=True)
                    training_dfs.append(resampled)
                else:
                    training_dfs.append(train_samples)

    if not training_dfs:
        raise ValueError("No training data available after binning and sampling")
    if not validation_indices:
        raise ValueError("No validation data available after binning and sampling")

    training_df = pd.concat(training_dfs)
    validation_df = df.loc[validation_indices]
    training_df = training_df.drop('bin', axis=1)
    validation_df = validation_df.drop('bin', axis=1)

    print(f"Number of bins with data: {len(bin_counts)}")
    print(f"Min Number in a bins with data: {min_samples}")
    print(f"Original data size: {len(df)}")
    print(f"Training set size: {len(training_df)}")
    print(f"Validation set size: {len(validation_df)}")

    return training_df, validation_df

def get_target_transform(target_transform_type, train_targets):
    if target_transform_type == "normalize":
        mean = train_targets.mean()
        std = train_targets.std()
        if std == 0:
            std = 1.0
        def transform(x):
            return (x - mean) / std
        def inverse_transform(x):
            return x * std + mean
        return transform, inverse_transform, {"mean": mean.item(), "std": std.item()}
    elif target_transform_type == "log":
        def transform(x):
            return torch.log1p(x)  # log(1 + x) for stability
        def inverse_transform(x):
            return torch.expm1(x)
        return transform, inverse_transform, {}
    else:
        def transform(x):
            return x
        def inverse_transform(x):
            return x
        return transform, inverse_transform, {}

def train_model(model, train_loader, val_loader, num_epochs=100, target_transform_type="none", loss_type="L2"):
    accelerator = Accelerator()
    device = accelerator.device

    model = model.to(device)
    criterion = nn.MSELoss() if loss_type == "L2" else nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lowered LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Compute transformation stats
    all_train_targets = []
    for _, _, _, targets in train_loader:
        all_train_targets.append(targets)
    all_train_targets = torch.cat(all_train_targets).float()
    transform, inverse_transform, transform_params = get_target_transform(target_transform_type, all_train_targets)

    if accelerator.is_main_process:
        print(f"Target transform: {target_transform_type}, Params: {transform_params}")
        print(f"Training targets - Min: {all_train_targets.min().item()}, Max: {all_train_targets.max().item()}")

    if accelerator.is_main_process:
        wandb.config.update({"target_transform": target_transform_type, "loss_type": loss_type, "lr": 0.0001})
        wandb.config.update(transform_params)

    best_val_loss = float('inf')
    best_model = None
    patience = 10  # Early stopping patience
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_size = 0

        for longitudes, latitudes, features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(features)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                accelerator.print(f"Epoch {epoch+1}: NaN/Inf in outputs")
                continue
            transformed_targets = transform(targets.float())
            loss = criterion(outputs, transformed_targets)
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"Epoch {epoch+1}: NaN/Inf in loss")
                continue
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            gathered_loss = accelerator.gather_for_metrics(loss.detach())
            running_loss += gathered_loss.mean().item()
            train_loader_size += 1

        train_loss = running_loss / train_loader_size if train_loader_size > 0 else float('nan')

        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets = []
        val_loader_size = 0

        with torch.no_grad():
            for longitudes, latitudes, features, targets in val_loader:
                outputs = model(features)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    accelerator.print(f"Epoch {epoch+1}: NaN/Inf in val outputs")
                    continue
                transformed_targets = transform(targets.float())
                loss = criterion(outputs, transformed_targets)
                val_loss += accelerator.gather_for_metrics(loss.detach()).mean().item()
                val_loader_size += 1

                gathered_outputs = accelerator.gather_for_metrics(outputs)
                gathered_targets = accelerator.gather_for_metrics(targets.float())
                val_outputs.extend(inverse_transform(gathered_outputs).cpu().numpy())
                val_targets.extend(gathered_targets.cpu().numpy())

        if len(val_outputs) == 0 or len(val_targets) == 0:
            accelerator.print(f"Epoch {epoch+1}: Validation data empty")
            val_loss = float('nan')
            r_squared = 0.0
            rmse = float('nan')
            mae = float('nan')
        else:
            val_loss = val_loss / val_loader_size if val_loader_size > 0 else float('nan')
            val_outputs = np.array(val_outputs)
            val_targets = np.array(val_targets)

            output_std = np.std(val_outputs)
            target_std = np.std(val_targets)
            accelerator.print(f"Epoch {epoch+1}: Output std: {output_std:.4f}, Target std: {target_std:.4f}")

            if output_std < 1e-6 or target_std < 1e-6:
                correlation = 0.0
                accelerator.print(f"Epoch {epoch+1}: No variability in outputs or targets")
            else:
                corr_matrix = np.corrcoef(val_outputs.flatten(), val_targets.flatten())
                correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

            r_squared = correlation ** 2
            mse = np.mean((val_outputs - val_targets) ** 2) if not np.any(np.isnan(val_outputs)) else float('nan')
            rmse = np.sqrt(mse) if not np.isnan(mse) else float('nan')
            mae = np.mean(np.abs(val_outputs - val_targets)) if not np.any(np.isnan(val_outputs)) else float('nan')

        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'correlation': correlation,
                'r_squared': r_squared,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'output_std': output_std if len(val_outputs) > 0 else 0.0,
                'target_std': target_std if len(val_targets) > 0 else 0.0
            })

        if accelerator.is_main_process and val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            wandb.run.summary["best_val_loss"] = best_val_loss
            patience_counter = 0
        elif accelerator.is_main_process and not np.isnan(val_loss):
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step(val_loss)  # Adjust learning rate based on val_loss

        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        accelerator.print(f'Validation Loss: {val_loss:.4f}')
        accelerator.print(f'R²: {r_squared:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}\n')

    if best_model is not None:
        model.load_state_dict(best_model)
    return model, val_outputs, val_targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D CNN with target transformation and loss options")
    parser.add_argument("--target_transform", type=str, default="log", choices=["none", "normalize", "log"],
                        help="Target transformation: none, normalize (zero-centered, std-scaled), or log")
    parser.add_argument("--loss", type=str, default="L2", choices=["L1", "L2"],
                        help="Loss function: L1 (MAE) or L2 (MSE)")
    args = parser.parse_args()

    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(
            project="socmapping-3dcnn",
            config={
                "max_oc": MAX_OC,
                "time_beginning": TIME_BEGINNING,
                "time_end": TIME_END,
                "window_size": window_size,
                "time_before": time_before,
                "bands": len(bands_list_order),
                "epochs": num_epochs,
                "batch_size": 256,
                "learning_rate": 0.001,
                "target_transform": args.target_transform,
                "loss": args.loss
            }
        )

    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    samples_coordinates_array_path, data_array_path = separate_and_add_data()

    def flatten_paths(path_list):
        flattened = []
        for item in path_list:
            if isinstance(item, list):
                flattened.extend(flatten_paths(item))
            else:
                flattened.append(item)
        return flattened

    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))

    train_df, val_df = create_balanced_dataset(df)
    if len(val_df) == 0:
        raise ValueError("Validation DataFrame is empty after balancing")
    
    if accelerator.is_main_process:
        wandb.run.summary["train_size"] = len(train_df)
        wandb.run.summary["val_size"] = len(val_df)

    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
    val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    accelerator.print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    for batch in train_loader:
        _, _, first_batch, _ = batch 
        break
    accelerator.print("Size of the first batch:", first_batch.shape)

    model = Small3DCNN(
        input_channels=len(bands_list_order),
        input_height=window_size,
        input_width=window_size,
        input_time=time_before
    )
    
    if accelerator.is_main_process:
        wandb.run.summary["model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model, val_outputs, val_targets = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs,
        target_transform_type=args.target_transform, loss_type=args.loss
    )

    if accelerator.is_main_process:
        model_path = (f'cnn_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                      f'TIME_END_{TIME_END}_TRANSFORM_{args.target_transform}_LOSS_{args.loss}_best.pth')
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)
        wandb.finish()

    accelerator.print("Model trained and saved successfully!")

    # Best model with none/l2