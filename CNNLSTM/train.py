# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator

# Assuming these are imported from your existing files
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears 
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (TIME_BEGINNING, TIME_END, MAX_OC, seasons, years_padded,
                    num_epochs,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
from models import RefittedCovLSTM

def train_model(model, train_loader, val_loader, num_epochs=20, accelerator=None):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare with Accelerate
    train_loader, val_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, model, optimizer
    )

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            
            running_loss += loss.item()
            
            if accelerator.is_main_process:
                wandb.log({
                    'train_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })

        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets_list = []
        
        with torch.no_grad():
            for longitudes, latitudes, features, targets in val_loader:
                features = features.to(accelerator.device)
                targets = targets.to(accelerator.device).float()
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_outputs.extend(outputs.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        val_outputs = np.array(val_outputs)
        val_targets_list = np.array(val_targets_list)
        correlation = np.corrcoef(val_outputs, val_targets_list)[0, 1]
        r_squared = correlation ** 2
        mse = np.mean((val_outputs - val_targets_list) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(val_outputs - val_targets_list))

        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_avg': train_loss,
                'val_loss': val_loss,
                'correlation': correlation,
                'r_squared': r_squared,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            })

            if val_loss < best_val_loss and epoch % 5 ==0:
                best_val_loss = val_loss
                wandb.run.summary['best_val_loss'] = best_val_loss
                accelerator.save_state(f'model_checkpoint_epoch_{epoch+1}.pth')
                wandb.save(f'model_checkpoint_epoch_{epoch+1}.pth')
        
        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        accelerator.print(f'Validation Loss: {val_loss:.4f}\n')

    return model, val_outputs, val_targets_list

if __name__ == "__main__":
    # Initialize Accelerate
    accelerator = Accelerator()
    
    # Initialize wandb
    wandb.init(
        project="socmapping-CNNLSTM",
        config={
            "max_oc": MAX_OC,
            "time_beginning": TIME_BEGINNING,
            "time_end": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands": len(bands_list_order),
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 0.001,
            "lstm_hidden_size": 128,
            "num_layers": 2,
            "dropout_rate": 0.25
        }
    )

    # Data preparation
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

    # Create datasets
    train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, df)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)  # Using same dataset for simplicity
    
    # Get batch size info
    for batch in train_loader:
        _, _, first_batch, _ = batch
        break
    first_batch_size = first_batch.shape
    if accelerator.is_main_process:
        print("Size of the first batch:", first_batch_size)

    # Initialize model
    # model = RefittedCovLSTM(
    #     num_channels=len(bands_list_order),
    #     lstm_input_size=len(bands_list_order) * window_size * window_size,
    #     lstm_hidden_size=128,
    #     num_layers=2,
    #     dropout=0.25
    # )
    

    model = RefittedCovLSTM(
        num_channels=len(bands_list_order),  # 6 in your case
        lstm_input_size=128,  # Fixed by CNN output
        lstm_hidden_size=128,
        num_layers=2,
        dropout=0.25
    )


    
    # Train model
    model, val_outputs, val_targets = train_model(model, train_loader, val_loader, 
                                                num_epochs=num_epochs, accelerator=accelerator)

    # Save final model
    if accelerator.is_main_process:
        final_model_path = f'cnnlstm_model_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_TIME_END_{TIME_END}.pth'
        accelerator.save(model.state_dict(), final_model_path)
        wandb.save(final_model_path)
        print("Model trained and saved successfully!")

    wandb.finish()
