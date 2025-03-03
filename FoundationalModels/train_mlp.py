import argparse
import glob
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP Regression Training on SpectralGPT Embeddings")
    parser.add_argument(
        "--parquet_dir",
        default="/fast/vfourel/SOCProject/run_20250209_163939",
        type=str,
        help="Directory containing Parquet files",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="GPU index (0-7 typically)")
    parser.add_argument(
        "--files_per_epoch",
        default=2,
        type=int,
        choices=[1, 2, 3, 4],
        help="Number of Parquet files to load per epoch (1-4)",
    )
    return parser.parse_args()

def flatten_nested_array(arr):
    """Recursively flatten a nested array or list into a 1D list of scalars."""
    flat = []
    for item in arr:
        if isinstance(item, (list, np.ndarray)):
            flat.extend(flatten_nested_array(item))
        else:
            flat.append(item)
    return flat

# Custom Dataset for dynamic loading of Parquet files
class DynamicParquetDataset(IterableDataset):
    def __init__(self, parquet_files: List[str], files_per_load: int, device: torch.device, shuffle: bool = False) -> None:
        self.parquet_files = parquet_files
        self.files_per_load = min(files_per_load, len(parquet_files))
        self.device = device
        self.shuffle = shuffle  # Add shuffle flag
        self.embedding_size = 110592  # Size of each embedding
        self.num_embeddings = 26     # 1 elevation + 25 channel-time embeddings
        self.input_size = self.embedding_size * self.num_embeddings  # 110592 * 26

        # Validate embedding sizes from the first file
        df_sample = pd.read_parquet(parquet_files[0])
        first_row = df_sample.iloc[0]

        # Process elevation embedding
        try:
            elev_emb = flatten_nested_array(first_row["elevation_embedding"])
            elev_emb = np.asarray(elev_emb, dtype=np.float32)  # Direct conversion since already flattened
            if elev_emb.size != self.embedding_size:
                logger.warning(f"Expected elevation embedding size {self.embedding_size}, got {elev_emb.size}")
                elev_emb = np.resize(elev_emb, self.embedding_size)
        except Exception as e:
            logger.error(f"Failed to process elevation_embedding: {first_row['elevation_embedding']}, error: {e}")
            raise ValueError("elevation_embedding has an incompatible format.")

        # Process channel-time embeddings
        cte_sample = first_row["channel_time_embeddings"]
        try:
            valid_cte = next(v for v in cte_sample.values() if v is not None)
            cte_emb = flatten_nested_array(valid_cte)
            cte_emb = np.asarray(cte_emb, dtype=np.float32)  # Direct conversion since already flattened
            if cte_emb.size != self.embedding_size:
                logger.warning(f"Expected channel-time embedding size {self.embedding_size}, got {cte_emb.size}")
                cte_emb = np.resize(cte_emb, self.embedding_size)
            if len(cte_sample) != 25:
                logger.warning(f"Expected 25 channel-time embeddings, got {len(cte_sample)}")
        except Exception as e:
            logger.error(f"Failed to process channel_time_embeddings: {cte_sample}, error: {e}")
            raise ValueError("channel_time_embeddings has an incompatible format.")

        logger.info(f"Input size: {self.input_size} (26 embeddings * {self.embedding_size} each)")

    def _load_files(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a random sample of Parquet files."""
        current_files = random.sample(self.parquet_files, self.files_per_load)
        dfs = [pd.read_parquet(f) for f in current_files]
        data = pd.concat(dfs, ignore_index=True)
        X, y = self._preprocess_embeddings(data)
        logger.info(f"Loaded {len(current_files)} files with {len(y)} samples on {self.device}")
        return X, y

    def _preprocess_embeddings(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess embeddings into a single concatenated feature vector."""
        X_list = []
        y_list = []

        for idx, row in data.iterrows():
            try:
                # Process elevation embedding
                elev_emb = flatten_nested_array(row["elevation_embedding"])
                elev_emb = np.asarray(elev_emb, dtype=np.float32)
                if elev_emb.size != self.embedding_size:
                    logger.warning(
                        f"Row {idx} elevation_embedding size {elev_emb.size}, resizing to {self.embedding_size}"
                    )
                    elev_emb = np.resize(elev_emb, self.embedding_size)

                # Process all channel-time embeddings
                cte = row["channel_time_embeddings"]
                cte_flat = []
                for key, emb in cte.items():
                    if emb is not None:
                        emb_flat = flatten_nested_array(emb)
                        emb_flat = np.asarray(emb_flat, dtype=np.float32)
                        if emb_flat.size != self.embedding_size:
                            logger.warning(
                                f"Row {idx} {key} size {emb_flat.size}, resizing to {self.embedding_size}"
                            )
                            emb_flat = np.resize(emb_flat, self.embedding_size)
                        cte_flat.extend(emb_flat)
                    else:
                        cte_flat.extend(np.zeros(self.embedding_size, dtype=np.float32))

                # Ensure we have exactly 25 channel-time embeddings
                expected_cte_size = self.embedding_size * 25
                if len(cte_flat) != expected_cte_size:
                    logger.warning(
                        f"Row {idx} cte_flat size {len(cte_flat)}, resizing to {expected_cte_size}"
                    )
                    cte_flat = np.resize(cte_flat, expected_cte_size)

                # Concatenate all embeddings
                features = np.concatenate([elev_emb, cte_flat])
                if features.size != self.input_size:
                    logger.error(
                        f"Row {idx} features size {features.size}, expected {self.input_size}, skipping"
                    )
                    continue

                X_list.append(features)
                y_list.append(float(row["organic_carbon"]))

            except Exception as e:
                logger.error(f"Row {idx} preprocessing error: {e}, skipping")
                continue

        if not X_list:
            raise ValueError("No valid samples processed from the current batch of files.")

        return np.stack(X_list), np.array(y_list, dtype=np.float32)

    def __iter__(self):
        X, y = self._load_files()
        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)
        indices = list(range(len(y)))
        if self.shuffle:
            random.shuffle(indices)  # Shuffle indices if shuffle is True
        for i in indices:
            yield X_tensor[i], y_tensor[i]

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 100, output_size: int = 1, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.to(device)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def train_model(
    model: nn.Module,
    train_dataset: DynamicParquetDataset,
    test_dataset: DynamicParquetDataset,
    args: argparse.Namespace,
    device: torch.device
) -> None:
    """Train the MLP model with dynamic data loading."""
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    console = Console()
    best_rmse = float("inf")

    # Use DataLoader without shuffle for IterableDataset, shuffling handled internally
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch + 1}/{args.epochs}", total=len(train_loader))
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress.update(task, advance=1)

        train_loss /= len(train_loader)

        model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                test_preds.extend(outputs.squeeze().cpu().numpy())
                test_true.extend(batch_y.cpu().numpy())

        test_r2 = r2_score(test_true, test_preds)
        test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), os.path.join(args.parquet_dir, f"best_mlp_model_gpu{args.gpu}.pth"))
            logger.info(f"Saved model with RMSE: {best_rmse:.4f}")

        console.print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}"
        )

    console.print(f"Best Test RMSE: {best_rmse:.4f}")

def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu < torch.cuda.device_count() else "cpu")
    if device.type == "cpu":
        logger.warning(f"Falling back to CPU. GPU {args.gpu} not available. Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Using device: {device}")

    parquet_files = glob.glob(os.path.join(args.parquet_dir, "batch_*_size_*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {args.parquet_dir}")
    logger.info(f"Found {len(parquet_files)} Parquet files")

    train_files, test_files = train_test_split(parquet_files, test_size=0.15, random_state=42)
    logger.info(f"Training files: {len(train_files)}, Test files: {len(test_files)}")

    # Enable shuffling for training dataset only
    train_dataset = DynamicParquetDataset(train_files, files_per_load=args.files_per_epoch, device=device, shuffle=True)
    test_dataset = DynamicParquetDataset(test_files, files_per_load=args.files_per_epoch, device=device, shuffle=False)

    model = MLP(input_size=train_dataset.input_size, hidden_size=100, device=device)
    logger.info(f"MLP Model initialized with input size: {train_dataset.input_size}")

    train_model(model, train_dataset, test_dataset, args, device)

if __name__ == "__main__":
    main()
