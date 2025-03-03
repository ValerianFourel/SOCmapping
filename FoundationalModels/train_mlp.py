from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import glob
import os
import argparse
import random
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='MLP Regression Training on SpectralGPT Embeddings')
    parser.add_argument('--parquet_dir', default='/fast/vfourel/SOCProject/run_20250209_163939', 
                        type=str, help='Directory containing Parquet files')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--hidden_size', default=512, type=int, help='MLP hidden layer size')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index (0-4)')
    parser.add_argument('--files_per_epoch', default=2, type=int, choices=[1, 2, 3, 4], 
                        help='Number of Parquet files to load per epoch (1-4)')
    args = parser.parse_args()
    return args

# Custom Dataset for dynamic loading of Parquet files
class DynamicParquetDataset(Dataset):
    def __init__(self, parquet_files, files_per_load=2):
        self.parquet_files = parquet_files
        self.files_per_load = min(files_per_load, len(parquet_files))  # Ensure we don’t exceed available files
        self.bands = ['LAI', 'LST', 'MODIS_NPP', 'SoilEvaporation', 'TotalEvapotranspiration']
        self.years = range(2007, 2012)  # Based on time_before=5
        self.cte_slots = len(self.bands) * len(self.years)
        
        # Determine input size from the first file
        df_sample = pd.read_parquet(parquet_files[0])
        first_row = df_sample.iloc[0]
        self.elevation_emb_size = np.array(first_row['elevation_embedding']).flatten().size
        cte_sample = first_row['channel_time_embeddings']
        self.cte_emb_size = np.array(next(v for v in cte_sample.values() if v is not None)).flatten().size
        self.input_size = self.elevation_emb_size + (self.cte_slots * self.cte_emb_size)
        print(f"Input size: {self.input_size} "
              f"(Elevation: {self.elevation_emb_size}, CTE: {self.cte_slots} slots * {self.cte_emb_size} each)")

        self.current_files = []
        self.current_data = None
        self.load_new_files()  # Initial load

    def load_new_files(self):
        # Randomly select files_per_load Parquet files
        self.current_files = random.sample(self.parquet_files, self.files_per_load)
        dfs = [pd.read_parquet(f) for f in self.current_files]
        self.current_data = pd.concat(dfs, ignore_index=True)
        self.X, self.y = self._preprocess_embeddings()
        print(f"Loaded {len(self.current_files)} files with {len(self.y)} samples")

    def _preprocess_embeddings(self):
        X_list = []
        y_list = []

        for idx, row in self.current_data.iterrows():
            elev_emb = np.array(row['elevation_embedding']).flatten()
            if elev_emb.size != self.elevation_emb_size:
                elev_emb = np.resize(elev_emb, self.elevation_emb_size)

            cte = row['channel_time_embeddings']
            cte_flat = []
            for band in self.bands:
                for year in self.years:
                    key = f"{band}_{year}"
                    emb = cte.get(key)
                    if emb is not None:
                        emb_flat = np.array(emb).flatten()
                        if emb_flat.size != self.cte_emb_size:
                            emb_flat = np.resize(emb_flat, self.cte_emb_size)
                        cte_flat.extend(emb_flat)
                    else:
                        cte_flat.extend(np.zeros(self.cte_emb_size))
            
            expected_cte_size = self.cte_slots * self.cte_emb_size
            if len(cte_flat) != expected_cte_size:
                cte_flat = np.resize(cte_flat, expected_cte_size)
            
            features = np.concatenate([elev_emb, cte_flat])
            X_list.append(features)
            y_list.append(row['organic_carbon'])

        return np.stack(X_list), np.array(y_list)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

# Custom DataLoader with dynamic file loading per epoch
class DynamicParquetLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Reload new files for each epoch
        self.dataset.load_new_files()
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_X = torch.stack([self.dataset[i][0] for i in batch_indices])
            batch_y = torch.stack([self.dataset[i][1] for i in batch_indices])
            yield batch_X, batch_y

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training Function
def train_model(model, train_loader, test_loader, args, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    console = Console()
    best_rmse = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress.update(task, advance=1)
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                test_preds.extend(outputs.cpu().numpy().flatten())
                test_true.extend(batch_y.cpu().numpy().flatten())
        
        test_r2 = r2_score(test_true, test_preds)
        test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), os.path.join(args.parquet_dir, f'best_mlp_model_gpu{args.gpu}.pth'))
        
        console.print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
                      f"Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")

    console.print(f"Best Test RMSE: {best_rmse:.4f}")

# Main Execution
if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu < torch.cuda.device_count() else 'cpu')
    print(f"Using device: {device}")
    
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "batch_*_size_*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {args.parquet_dir}")
    print(f"Found {len(parquet_files)} Parquet files")
    
    # Split files into train and test sets (85% train, 15% test)
    train_files, test_files = train_test_split(parquet_files, test_size=0.15, random_state=42)
    
    train_dataset = DynamicParquetDataset(train_files, files_per_load=args.files_per_epoch)
    test_dataset = DynamicParquetDataset(test_files, files_per_load=args.files_per_epoch)
    
    train_loader = DynamicParquetLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DynamicParquetLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    model = MLP(input_size=train_dataset.input_size, hidden_size=args.hidden_size).to(device)
    print(f"MLP Model initialized with input size: {train_dataset.input_size}")
    
    train_model(model, train_loader, test_loader, args, device)
