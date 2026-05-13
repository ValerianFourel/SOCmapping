import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from accelerate import Accelerator
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears, NormalizedMultiRasterDatasetMultiYears
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, MAX_OC,NUM_EPOCHS_RUN,
                   seasons, years_padded, num_epochs, NUM_HEADS, NUM_LAYERS,hidden_size,
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly,
                   DataYearly, SamplesCoordinates_Seasonally, bands_list_order,
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, window_size,
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, time_before)
# SGT model variants — selected at runtime via --model-size:
#   small → SimpleSGT  at d_model=128, num_heads=2, num_encoder_layers=1
#                       → 360,593 trainable params
#   big   → EnhancedSGT at d_model=128, num_heads=4, num_encoder_layers=3,
#                       expansion_factor=4
#                       → 1,120,546 trainable params
#           ← architecture matches Model A's saved .pth (verified
#           empirically by rebuttal/verify_model_a_architecture.py:
#           state_dict load + 32-sample prediction reproduction to 2e-6
#           against the saved analysis_results.pkl).
from SimpleSGT import SimpleSGT
from EnhancedSGT import EnhancedSGT
import argparse
import contextlib
from balancedDataset import create_validation_train_sets,create_balanced_dataset
import uuid
import os
import datetime

# Increase the timeout value
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 
torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=20))


def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleSGT model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mse'], help='Type of loss function')
    parser.add_argument('--target_transform', type=str, default='normalize', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation',  type=bool, default=True, help='Whether to use validation set')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.2, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=4, help='Number of times to run the process')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size for the model')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model')
    parser.add_argument('--save_train_and_val', type=bool, default=False, help='Dropout rate for the model')
    # --- Rebuttal / audited reproduction ---------------------------------
    # When --rebuttal is set, the script overrides hyperparameters to the
    # audited Model A v2 recipe (val proper R²=0.594, Pearson r²=0.626,
    # RMSE=4.76, n=1,359 — see rebuttal/table_2_corrected.md) and routes
    # outputs to rebuttal/audited_runs/SGT_<size>_audited_<ts>_<sha>/ with
    # a lineage JSON. Use --model-size to pick the SGT variant.
    parser.add_argument('--rebuttal', action='store_true',
                        help='Lock audited Model-A-v2 hyperparameters, '
                             'route outputs to rebuttal/audited_runs/, '
                             'and write a provenance JSON for the rerun.')
    parser.add_argument('--model-size', type=str, default='big',
                        choices=['small', 'big'],
                        help='SGT variant: "small" = SimpleSGT '
                             '(d_model=128, num_heads=2, 1 layer; 360,593 params); '
                             '"big" = EnhancedSGT (d_model=128, num_heads=4, '
                             'num_encoder_layers=3, expansion_factor=4; 1,120,546 '
                             'params — matches Model A architecture).')
    # --- Gradient-accumulation knobs -------------------------------------
    # Target effective batch = num_GPUs × per_gpu_batch × accum_steps. The
    # historic 8-GPU training (wandb run rxeolg7e, R²=0.755) used
    # 8 × 256 × 1 = 2048. When fewer GPUs are available we keep the
    # effective batch by raising accum_steps (4 GPUs → 2 accum, 1 GPU → 8
    # accum, etc.). Set --effective-batch-size 0 to disable accumulation
    # (= old behaviour, batch = num_GPUs × per_gpu_batch).
    parser.add_argument('--per-gpu-batch-size', type=int, default=256,
                        help='Per-process (per-GPU) micro-batch size')
    parser.add_argument('--effective-batch-size', type=int, default=2048,
                        help='Target total batch over all GPUs × accumulation. '
                             '0 disables gradient accumulation.')
    parser.add_argument('--accum-steps', type=int, default=0,
                        help='Manual override for gradient accumulation steps. '
                             '0 = auto from --effective-batch-size and '
                             'accelerator.num_processes.')
    return parser.parse_args()


def _resolve_accum_steps(args, num_processes):
    """Compute the gradient-accumulation factor from CLI args.
    --accum-steps takes precedence; otherwise auto from
    --effective-batch-size / (num_GPUs × per_gpu_batch). Always ≥ 1."""
    if args.accum_steps > 0:
        return int(args.accum_steps)
    if args.effective_batch_size <= 0:
        return 1
    per_step = max(1, num_processes * args.per_gpu_batch_size)
    target = max(args.effective_batch_size, per_step)
    return max(1, target // per_step)

def train_model(model, train_loader, val_loader,target_mean,target_std, num_epochs=num_epochs, accelerator=None, lr=0.001,
                loss_type='l1', target_transform='none', min_r2=0.5, use_validation=True,
                accum_steps=1):
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare with Accelerator
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    if use_validation and val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # Handle target normalization if selected
    if target_transform == 'normalize':

        if accelerator.is_main_process:
            print(f"Target mean: {target_mean}, Target std: {target_std}")
    else:
        target_mean, target_std = 0.0, 1.0  # No normalization applied

    best_r2 = -float('inf')
    best_model_state = None
    epoch_metrics = []

    n_total = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

            # Apply target transformation
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)  # Add small constant to avoid log(0)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)
            # 'none' requires no transformation

            # Gradient accumulation: skip DDP all-reduce on intermediate
            # micro-batches, fire optimizer.step() only on the last one of
            # each window (or the very last micro-batch of the epoch).
            is_last_micro = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == n_total)
            sync_ctx = contextlib.nullcontext() if is_last_micro else accelerator.no_sync(model)
            with sync_ctx:
                outputs = model(features)
                loss = criterion(outputs, targets) / accum_steps
                accelerator.backward(loss)

            running_loss += loss.item() * accum_steps  # un-scale for logging
            if is_last_micro:
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                wandb.log({
                    'train_loss': loss.item() * accum_steps,
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })

        train_loss = running_loss / max(n_total, 1)

        if use_validation and val_loader is not None:
            model.eval()
            val_outputs = []
            val_targets_list = []

            with torch.no_grad():
                for longitudes, latitudes, features, targets in val_loader:
                    features = features.to(accelerator.device)
                    targets = targets.to(accelerator.device).float()

                    # Apply the same transformation to validation targets
                    if target_transform == 'log':
                        targets = torch.log(targets + 1e-10)
                    elif target_transform == 'normalize':
                        targets = (targets - target_mean) / (target_std + 1e-10)

                    outputs = model(features)
                    val_outputs.extend(outputs.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())

            # Gather validation outputs and targets across all processes
            val_outputs_tensor = torch.tensor(val_outputs).to(accelerator.device)
            val_targets_tensor = torch.tensor(val_targets_list).to(accelerator.device)
            val_outputs_all = accelerator.gather(val_outputs_tensor).cpu().numpy()
            val_targets_all = accelerator.gather(val_targets_tensor).cpu().numpy()

            if accelerator.is_main_process:
                # Apply inverse transformation to get original scale
                if target_transform == 'log':
                    original_val_outputs = np.exp(val_outputs_all)
                    original_val_targets = np.exp(val_targets_all)
                elif target_transform == 'normalize':
                    original_val_outputs = val_outputs_all * target_std + target_mean
                    original_val_targets = val_targets_all * target_std + target_mean
                else:
                    original_val_outputs = val_outputs_all
                    original_val_targets = val_targets_all

                # Assuming original_val_outputs and original_val_targets are NumPy arrays
                min_outputs = np.min(original_val_outputs)
                max_outputs = np.max(original_val_outputs)
                min_targets = np.min(original_val_targets)
                max_targets = np.max(original_val_targets)
                if use_validation:
                    accelerator.print("Min of original_val_outputs:", min_outputs)
                    accelerator.print("Max of original_val_outputs:", max_outputs)
                    accelerator.print("Min of original_val_targets:", min_targets)
                    accelerator.print("Max of original_val_targets:", max_targets)
                # Compute metrics on original scale.
                # `r_squared` is the COEFFICIENT OF DETERMINATION (1 - SS_res/SS_tot),
                # which is what scikit-learn's r2_score reports and what reviewers mean by
                # R² for regression on held-out data. Squared Pearson correlation is kept
                # alongside as `pearson_r2` for continuity with v1 reporting; the two
                # agree only when predictions are unbiased and identically scaled.
                if len(original_val_outputs) > 1 and np.std(original_val_outputs) > 1e-6 and np.std(original_val_targets) > 1e-6:
                    correlation = float(np.corrcoef(original_val_outputs, original_val_targets)[0, 1])
                    pearson_r2 = correlation ** 2
                    ss_res = float(np.sum((original_val_targets - original_val_outputs) ** 2))
                    ss_tot = float(np.sum((original_val_targets - np.mean(original_val_targets)) ** 2))
                    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
                else:
                    correlation = 0.0
                    pearson_r2 = 0.0
                    r_squared = 0.0
                mse = np.mean((original_val_outputs - original_val_targets) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(original_val_outputs - original_val_targets))

                # Compute IQR and RPIQ
                iqr = np.percentile(original_val_targets, 75) - np.percentile(original_val_targets, 25)
                rpiq = iqr / rmse if rmse > 0 else float('inf')

                # Compute validation loss on transformed scale
                val_outputs_tensor_all = torch.from_numpy(val_outputs_all).to(accelerator.device)
                val_targets_tensor_all = torch.from_numpy(val_targets_all).to(accelerator.device)
                val_loss = criterion(val_outputs_tensor_all, val_targets_tensor_all).item()
            else:
                val_loss = float('nan')
                correlation = float('nan')
                pearson_r2 = float('nan')
                r_squared = float('nan')
                mse = float('nan')
                rmse = float('nan')
                mae = float('nan')
                rpiq = float('nan')
        else:
            val_loss = float('nan')
            val_outputs = np.array([])
            val_targets_list = np.array([])
            correlation = float('nan')
            pearson_r2 = float('nan')
            r_squared = 1.0
            mse = float('nan')
            rmse = float('nan')
            mae = float('nan')
            rpiq = float('nan')

        if accelerator.is_main_process:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss_avg': train_loss,
                'val_loss': val_loss,
                'correlation': correlation,
                'pearson_r2': pearson_r2,
                'r_squared': r_squared,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'rpiq': rpiq
            }
            wandb.log(log_dict)
            epoch_metrics.append(log_dict)

            # Save model if it has the best R² and meets minimum threshold
            if use_validation and r_squared > best_r2 and r_squared >= min_r2:
                best_r2 = r_squared
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2
            elif not use_validation and epoch == num_epochs - 1:
                best_r2 = 1.0
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2

        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        if use_validation:
            accelerator.print(f'Validation Loss: {val_loss:.4f}')
            accelerator.print(f'R²: {r_squared:.4f}')
            accelerator.print(f'RMSE: {rmse:.4f}')
            accelerator.print(f'MAE: {mae:.4f}')
            accelerator.print(f'RPIQ: {rpiq:.4f}\n')
            # Print the results
            

    return model, val_outputs, val_targets_list, best_model_state, best_r2, epoch_metrics

def compute_average_metrics(all_runs_metrics):
    if not all_runs_metrics:
        return {}

    metric_sums = {}
    metric_counts = {}

    for run_metrics in all_runs_metrics:
        for epoch_metrics in run_metrics:
            for metric, value in epoch_metrics.items():
                if metric == 'epoch':
                    continue
                if not np.isnan(value):
                    if metric not in metric_sums:
                        metric_sums[metric] = 0.0
                        metric_counts[metric] = 0
                    metric_sums[metric] += value
                    metric_counts[metric] += 1

    avg_metrics = {}
    for metric in metric_sums:
        avg_metrics[metric] = metric_sums[metric] / metric_counts[metric] if metric_counts[metric] > 0 else float('nan')

    return avg_metrics

def compute_min_distance_stats(min_distance_stats_all):
    if not min_distance_stats_all:
        return {}

    stats = {'mean': [], 'median': [], 'min': [], 'max': [], 'std': []}
    for stat_dict in min_distance_stats_all:
        for key in stats:
            if not np.isnan(stat_dict[key]):
                stats[key].append(stat_dict[key])

    avg_stats = {}
    for key in stats:
        avg_stats[f'avg_{key}'] = np.mean(stats[key]) if stats[key] else float('nan')
        avg_stats[f'std_{key}'] = np.std(stats[key]) if stats[key] else float('nan')

    return avg_stats

def compute_training_statistics_oc():
    df_train = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        # Calculate target statistics from balanced dataset
    target_mean = df_train['OC'].mean()
    target_std = df_train['OC'].std()

    
    return target_mean, target_std

def save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename='training_metrics.txt'):
    with open(filename, 'w') as f:
        f.write("Training Metrics and Configuration\n")
        f.write("=" * 50 + "\n\n")

        # Write args
        f.write("Command Line Arguments:\n")
        f.write("-" * 30 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")

        # Write wandb runs info
        f.write("Wandb Runs Information:\n")
        f.write("-" * 30 + "\n")
        for run_idx, run_info in enumerate(wandb_runs_info, 1):
            f.write(f"Run {run_idx}:\n")
            f.write(f"  Project: {run_info['project']}\n")
            f.write(f"  Run Name: {run_info['name']}\n")
            f.write(f"  Run ID: {run_info['id']}\n")
            f.write("\n")

        # Write average metrics
        f.write("Average Metrics Across Runs:\n")
        f.write("-" * 30 + "\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

        # Write min distance stats
        f.write("Average Min Distance Statistics:\n")
        f.write("-" * 30 + "\n")
        for stat, value in min_distance_stats.items():
            f.write(f"{stat}: {value:.4f}\n")
        f.write("\n")

        # Write best metrics
        f.write("Best Metrics Across Runs:\n")
        f.write("-" * 30 + "\n")
        for metric, values in all_best_metrics.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {np.mean(values):.4f}\n")
            f.write(f"  Std: {np.std(values):.4f}\n")
            f.write(f"  Values: {[f'{v:.4f}' for v in values]}\n")
            f.write("\n")

import json 

def build_sgt_model(args):
    """Construct the SGT variant requested via --model-size.
    Both classes share the same forward signature (B, C, H, W, T) → (B,).
    """
    if args.model_size == 'small':
        # SimpleSGT signature: input_channels, height, width, time_steps,
        # d_model, num_heads, dropout. No num_encoder_layers / expansion_factor.
        return SimpleSGT(
            input_channels=len(bands_list_order),
            height=window_size,
            width=window_size,
            time_steps=time_before,
            d_model=args.hidden_size,
            num_heads=args.num_heads,
            dropout=args.dropout_rate,
        )
    # 'big' — EnhancedSGT (Model A architecture)
    return EnhancedSGT(
        input_channels=len(bands_list_order),
        height=window_size,
        width=window_size,
        time_steps=time_before,
        d_model=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout_rate,
        num_encoder_layers=args.num_layers,
        expansion_factor=4,
    )


def _resolve_rebuttal_preset(args):
    """When --rebuttal is set, override hyperparameters to match the audited
    Model A v2 recipe (val proper R²=0.594, Pearson r²=0.626, RMSE=4.76,
    n=1359 — see rebuttal/table_2_corrected.md). The user can still pass
    --num-runs and per-gpu-batch-size to control parallelism; everything
    else is locked.

    Architecture (verified empirically against Model A's saved .pth — see
    rebuttal/verify_model_a_architecture.py):
      big   → EnhancedSGT(d_model=128, num_heads=4, num_encoder_layers=3,
                          dropout=0.3, expansion_factor=4)  → 1,120,546 params
              ← matches Model A's saved checkpoint exactly (state_dict
                shapes + 32-sample prediction reproduction to 2e-6)
      small → SimpleSGT (d_model=128, num_heads=2, 1 transformer layer)
              → 360,593 params
    """
    if not args.rebuttal:
        return args
    args.lr = 2e-4
    args.dropout_rate = 0.3
    args.loss_type = 'mse'
    args.target_transform = 'normalize'
    args.target_val_ratio = 0.08
    args.distance_threshold = 1.2
    args.target_fraction = 0.75
    args.use_validation = True
    args.save_train_and_val = True
    if args.model_size == 'big':
        # EnhancedSGT defaults — reproduces Model A's 1,120,546-param model.
        args.num_heads = 4
        args.num_layers = 3
    else:
        # SimpleSGT at d_model=128 → 360,593 params.
        args.num_heads = 2
        args.num_layers = 1
    return args


def _git_sha_short():
    """Return short HEAD commit SHA from the current repo, or 'nogit' if
    not available (so the script never crashes on a non-git checkout)."""
    import subprocess as _sp
    try:
        out = _sp.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                               cwd=os.path.dirname(os.path.abspath(__file__)),
                               stderr=_sp.DEVNULL, timeout=5).decode().strip()
        return out or 'nogit'
    except Exception:
        return 'nogit'


def _resolve_rebuttal_paths(args, timestamp):
    """If --rebuttal is on, route output to rebuttal/audited_runs/ with a
    strict naming convention: SGT_<size>_audited_<timestamp>_<sha>. Returns
    (experiment_name, experiment_dir, git_sha)."""
    git_sha = _git_sha_short()
    if not args.rebuttal:
        return None, None, git_sha
    # Resolve rebuttal/ as a sibling of SpatiotemporalGatedTransformer/.
    socmapping_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    rebuttal_runs = socmapping_root / 'rebuttal' / 'audited_runs'
    rebuttal_runs.mkdir(parents=True, exist_ok=True)
    name = f'SGT_{args.model_size}_audited_{timestamp}_{git_sha}'
    return name, str(rebuttal_runs / name), git_sha


if __name__ == "__main__":
    args = parse_args()
    args = _resolve_rebuttal_preset(args)
    # Set num_runs to 1 if use_validation is False
    if not args.use_validation:
        args.num_runs = 1
        num_epochs = NUM_EPOCHS_RUN
    accelerator = Accelerator()

    # ---- Resolve effective batch & accumulation -----------------------
    # The historic 8-GPU run used per_gpu=256 × num_gpus=8 × accum=1 = 2048.
    # When the user launches with fewer GPUs we automatically scale up the
    # accumulation factor to keep the effective batch constant.
    NUM_PROCESSES = accelerator.num_processes
    ACCUM_STEPS = _resolve_accum_steps(args, NUM_PROCESSES)
    EFFECTIVE_BATCH = NUM_PROCESSES * args.per_gpu_batch_size * ACCUM_STEPS
    if accelerator.is_main_process:
        print(f'[gradient-accumulation]  num_gpus={NUM_PROCESSES}  '
              f'per_gpu_batch={args.per_gpu_batch_size}  '
              f'accum_steps={ACCUM_STEPS}  '
              f'effective_batch={EFFECTIVE_BATCH}')

    # Create experiment folder with descriptive naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Rebuttal mode: short, structured name + dedicated rebuttal/audited_runs/
    rebuttal_name, rebuttal_dir, GIT_SHA = _resolve_rebuttal_paths(args, timestamp)
    if args.rebuttal:
        experiment_name = rebuttal_name
        experiment_dir = rebuttal_dir
    else:
        experiment_name = (f"SGT_experiment_{timestamp}_"
                          f"OC{MAX_OC}_"
                          f"{TIME_BEGINNING}to{TIME_END}_"
                          f"transform_{args.target_transform}_"
                          f"loss_{args.loss_type}_"
                          f"runs_{args.num_runs}_"
                          f"lr_{args.lr}_"
                          f"heads_{args.num_heads}_"
                          f"layers_{args.num_layers}")
        experiment_dir = os.path.join(args.output_dir, experiment_name)

    # Create experiment configuration (available to all processes)
    experiment_config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "args": vars(args),
        "config_params": {
            "MAX_OC": MAX_OC,
            "TIME_BEGINNING": TIME_BEGINNING,
            "TIME_END": TIME_END,
            "window_size": window_size,
            "time_before": time_before,
            "bands_count": len(bands_list_order),
            "num_epochs": num_epochs
        }
    }
    # Rebuttal-only lineage block for audit trail
    if args.rebuttal:
        experiment_config["rebuttal_audited"] = True
        experiment_config["git_sha"] = GIT_SHA
        experiment_config["model_size"] = args.model_size
        experiment_config["model_class"] = (
            "EnhancedSGT" if args.model_size == 'big' else "SimpleSGT")
        experiment_config["audited_hyperparameters"] = {
            "lr": args.lr, "dropout_rate": args.dropout_rate,
            "loss_type": args.loss_type, "target_transform": args.target_transform,
            "target_val_ratio": args.target_val_ratio,
            "distance_threshold": args.distance_threshold,
            "target_fraction": args.target_fraction,
            "num_heads": args.num_heads, "num_layers": args.num_layers,
            "num_runs": args.num_runs, "num_epochs": num_epochs,
            "per_gpu_batch_size": args.per_gpu_batch_size,
            "effective_batch_size": EFFECTIVE_BATCH,
            "accum_steps": ACCUM_STEPS, "num_processes": NUM_PROCESSES,
        }
        experiment_config["lineage"] = (
            "Reproduces Model A v2 (residualModels1mil_normalize_composite_l2_v2/) "
            "which achieved val proper R²=0.594, Pearson r²=0.626, RMSE=4.76 g/kg, "
            "bias=+1.13 g/kg, n_val=1,359 at commit 8dce131 (wandb run rxeolg7e). "
            "See rebuttal/table_2_corrected.md for the corrected metric and "
            "rebuttal/proper_r2_all_projects_FULL.md for the full leaderboard.")

    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Created experiment directory: {experiment_dir}")

        # Save experiment configuration
        config_file = os.path.join(experiment_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        print(f"Experiment configuration saved to: {config_file}")

        # Rebuttal-only README pinning the lineage in human-readable form
        if args.rebuttal:
            readme_path = os.path.join(experiment_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# {experiment_name}\n\n")
                f.write(f"- **Timestamp:** {timestamp}\n")
                f.write(f"- **Git SHA:** `{GIT_SHA}`\n")
                f.write(f"- **Model size:** `{args.model_size}` "
                        f"(`{experiment_config['model_class']}`)\n")
                f.write(f"- **Hyperparameters:** locked by `--rebuttal` to the "
                        f"audited Model A v2 recipe\n")
                f.write(f"- **Num GPUs / accum / effective batch:** "
                        f"{NUM_PROCESSES} × {args.per_gpu_batch_size} × "
                        f"{ACCUM_STEPS} = {EFFECTIVE_BATCH}\n\n")
                f.write(f"## Lineage\n\n{experiment_config['lineage']}\n\n")
                f.write(f"## Re-run command\n\n```bash\n"
                        f"accelerate launch --num_processes {NUM_PROCESSES} "
                        f"train.py --rebuttal --model-size {args.model_size} "
                        f"--num-runs {args.num_runs}\n```\n")
            print(f"Rebuttal README written to: {readme_path}")

    # Initialize lists to store metrics and best metrics across runs
    all_runs_metrics = []
    all_best_metrics = {
        'r_squared': [],
        'rmse': [],
        'mae': [],
        'rpiq': []
    }
    min_distance_stats_all = []
    wandb_runs_info = []

    # **Add these variables to track the best model across all runs**
    best_overall_model_state = None
    best_overall_r_squared = -float('inf')
    best_overall_run_number = None
    best_overall_metrics = {}

    # Loop through the specified number of runs
    for run in range(args.num_runs):
        if accelerator.is_main_process:
            print(f"\nStarting Run {run + 1}/{args.num_runs}")

        # Initialize wandb for this run
        if accelerator.is_main_process:
            wandb_run = wandb.init(
                project="socmapping-SimpleSGT",
                name=f"{experiment_name}_run_{run+1}",
                config={
                    "experiment_name": experiment_name,
                    "experiment_dir": experiment_dir,
                    "run_number": run + 1,
                    "max_oc": MAX_OC,
                    "time_beginning": TIME_BEGINNING,
                    "time_end": TIME_END,
                    "window_size": window_size,
                    "time_before": time_before,
                    "bands": len(bands_list_order),
                    "epochs": num_epochs,
                    "batch_size": 256,
                    "learning_rate": args.lr,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout_rate": args.dropout_rate,
                    "loss_type": args.loss_type,
                    "target_transform": args.target_transform,
                    "use_validation": args.use_validation
                }
            )
            wandb_runs_info.append({
                'project': wandb_run.project,
                'name': wandb_run.name,
                'id': wandb_run.id
            })

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

        train_dataset_features_norm = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, df)
        train_df, _ = create_balanced_dataset(df, min_ratio=3/4,use_validation=False)
        train_dataset_std_means = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        train_dataset_std_means.set_feature_means(train_dataset_features_norm.get_feature_means())
        train_dataset_std_means.set_feature_stds(train_dataset_features_norm.get_feature_stds())
        feature_means = train_dataset_features_norm.get_feature_means()
        feature_stds = train_dataset_features_norm.get_feature_stds()
        target_mean, target_std = compute_training_statistics_oc()

        # Create train/validation split
        if args.use_validation:
            val_df, train_df, min_distance_stats = create_validation_train_sets(
                df=df,
                output_dir=experiment_dir,  # Use experiment directory
                target_val_ratio=args.target_val_ratio,
                use_gpu=args.use_gpu,
                distance_threshold=args.distance_threshold
            )
            min_distance_stats_all.append(min_distance_stats)
            # train_df, val_df = create_balanced_dataset(df, min_ratio=3/4,use_validation=True)

        if args.save_train_and_val:
            # Create data subfolder within experiment directory
            data_dir = os.path.join(experiment_dir, "data")
            if accelerator.is_main_process:
                os.makedirs(data_dir, exist_ok=True)

            # Create a descriptive filename based on run context
            parquet_filename = os.path.join(data_dir, f'train_val_data_run_{run+1}.parquet')

            # Combine train and val dataframes if validation is used
            if args.use_validation:
                # Add a column to identify train vs validation rows
                train_df['dataset_type'] = 'train'
                val_df['dataset_type'] = 'val'
                combined_df = pd.concat([train_df, val_df], ignore_index=True)
            else:
                # Just use train_df and mark all as train
                train_df['dataset_type'] = 'train'
                combined_df = train_df

            # Save to parquet file
            if accelerator.is_main_process:
                combined_df.to_parquet(parquet_filename)

            # Save normalization statistics to a separate file
            stats_filename = os.path.join(data_dir, f'normalization_stats_run_{run+1}.pkl')

            normalization_stats = {
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'target_mean': target_mean,
                'target_std': target_std,
                'experiment_config': experiment_config
            }

            if accelerator.is_main_process:
                import pickle
                with open(stats_filename, 'wb') as f:
                    pickle.dump(normalization_stats, f)

                print(f"Train and validation data saved to: {parquet_filename}")
                print(f"Normalization statistics saved to: {stats_filename}")

        # Create datasets
        if args.use_validation:
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            val_dataset.set_feature_means(train_dataset_features_norm.get_feature_means())
            val_dataset.set_feature_stds(train_dataset_features_norm.get_feature_stds())
            train_dataset.set_feature_means(train_dataset_features_norm.get_feature_means())
            train_dataset.set_feature_stds(train_dataset_features_norm.get_feature_stds())
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset)}")
                print(f"Run {run + 1} Length of val_dataset: {len(val_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=args.per_gpu_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.per_gpu_batch_size, shuffle=False)
        else:
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset_std_means)}")
            train_loader = DataLoader(train_dataset_std_means, batch_size=args.per_gpu_batch_size, shuffle=True)
            val_loader = None

        if accelerator.is_main_process:
            wandb_run.summary["train_size"] = len(train_df)
            wandb_run.summary["val_size"] = len(val_df) if args.use_validation else 0

        # Get batch size info
        for batch in train_loader:
            _, _, first_batch, _ = batch
            break
        first_batch_size = first_batch.shape
        if accelerator.is_main_process:
            print(f"Run {run + 1} Size of the first batch: {first_batch_size}")

        # Initialize model (variant selected by --model-size; defaults to 'big'
        # = EnhancedSGT, Model A architecture). build_sgt_model() respects
        # the --rebuttal preset overrides applied earlier.
        model = build_sgt_model(args)
        print(f"Model variant: {type(model).__name__} "
              f"(--model-size={args.model_size}); "
              f"parameters: {model.count_parameters():,}")

        if accelerator.is_main_process:
            wandb_run.summary["model_parameters"] = model.count_parameters()

        # Train model
        model, val_outputs, val_targets, best_model_state, best_r2, epoch_metrics = train_model(
            model,
            train_loader,
            val_loader,
            target_mean=target_mean, 
            target_std=target_std,
            num_epochs=num_epochs,
            accelerator=accelerator,
            lr=args.lr,
            loss_type=args.loss_type,
            target_transform=args.target_transform,
            min_r2=0.40,
            use_validation=args.use_validation,
            accum_steps=ACCUM_STEPS,
        )

        # Store metrics
        all_runs_metrics.append(epoch_metrics)

        # Find best metrics for this run
        best_metrics = {'r_squared': -float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'rpiq': -float('inf')}
        for epoch_metric in epoch_metrics:
            if not np.isnan(epoch_metric['r_squared']) and epoch_metric['r_squared'] > best_metrics['r_squared']:
                best_metrics['r_squared'] = epoch_metric['r_squared']
                best_metrics['rmse'] = epoch_metric['rmse']
                best_metrics['mae'] = epoch_metric['mae']
                best_metrics['rpiq'] = epoch_metric['rpiq']

        # Store best metrics
        for metric in all_best_metrics:
            if not np.isnan(best_metrics[metric]):
                all_best_metrics[metric].append(best_metrics[metric])

        # **Check if this run has the best overall performance**
        if not np.isnan(best_metrics['r_squared']) and best_metrics['r_squared'] > best_overall_r_squared:
            best_overall_r_squared = best_metrics['r_squared']
            best_overall_model_state = best_model_state
            best_overall_run_number = run + 1
            best_overall_metrics = best_metrics.copy()

        # Create models subfolder within experiment directory
        models_dir = os.path.join(experiment_dir, "models")
        if accelerator.is_main_process:
            os.makedirs(models_dir, exist_ok=True)

        # Save model for this run
        if accelerator.is_main_process and best_model_state is not None:
            # In rebuttal mode use a short, structured filename so all 4 runs
            # land cleanly side-by-side in audited_runs/<exp>/models/ .
            if args.rebuttal:
                pth_name = (f'SGT_{args.model_size}_audited_run{run+1}_'
                            f'R2_{best_r2:.4f}.pth')
            else:
                pth_name = f'SGT_model_run_{run+1}_R2_{best_r2:.4f}.pth'
            run_model_path = os.path.join(models_dir, pth_name)

            # Save model with metadata
            model_with_run_metadata = {
                'model_state_dict': best_model_state,
                'run_number': run + 1,
                'best_r2': best_r2,
                'best_metrics': best_metrics,
                'experiment_name': experiment_name,
                'rebuttal_audited': bool(args.rebuttal),
                'git_sha': GIT_SHA,
                'model_class': type(model).__name__,
                'model_size': args.model_size,
                'model_config': {
                    'input_channels': len(bands_list_order),
                    'height': window_size,
                    'width': window_size,
                    'time_steps': time_before,
                    'd_model': args.hidden_size,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout_rate,
                },
                'normalization_stats': {
                    'feature_means': feature_means,
                    'feature_stds': feature_stds,
                    'target_mean': target_mean,
                    'target_std': target_std
                },
                'training_args': vars(args)
            }

            accelerator.save(model_with_run_metadata, run_model_path)
            wandb_run.save(run_model_path)
            print(f"Run {run + 1} Model with best R² ({best_r2:.4f}) saved at: {run_model_path}")
            print(f"Run {run + 1} - Best metrics: R²: {best_metrics['r_squared']:.4f}, MAE: {best_metrics['mae']:.4f}, RMSE: {best_metrics['rmse']:.4f}, RPIQ: {best_metrics['rpiq']:.4f}")
        elif accelerator.is_main_process:
            print(f"Run {run + 1} No model saved - R² threshold not met")

        if accelerator.is_main_process:
            wandb_run.finish()

    # Compute and log average metrics
    if accelerator.is_main_process:
        avg_metrics = compute_average_metrics(all_runs_metrics)
        min_distance_stats = compute_min_distance_stats(min_distance_stats_all)

        print("\nAverage Metrics Across Runs:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nAverage Min Distance Statistics:")
        for stat, value in min_distance_stats.items():
            print(f"{stat}: {value:.4f}")

        print("\nBest Metrics Across Runs:")
        for metric, values in all_best_metrics.items():
            print(f"{metric} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

        # Create results subfolder
        results_dir = os.path.join(experiment_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save all metrics to file in results directory
        output_file = os.path.join(results_dir, 'training_metrics_summary.txt')
        save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename=output_file)
        print(f"\nMetrics saved to: {output_file}")

        # Save detailed metrics as JSON
        detailed_metrics_file = os.path.join(results_dir, 'detailed_metrics.json')
        detailed_metrics = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': timestamp,
                'total_runs': args.num_runs
            },
            'all_runs_metrics': all_runs_metrics,
            'all_best_metrics': all_best_metrics,
            'average_metrics': avg_metrics,
            'min_distance_stats': min_distance_stats,
            'wandb_runs_info': wandb_runs_info
        }
        with open(detailed_metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        print(f"Detailed metrics saved to: {detailed_metrics_file}")

        # Log average metrics to a new wandb run
        wandb_run = wandb.init(project="socmapping-SimpleSGT", name=f"{experiment_name}_summary")
        wandb_runs_info.append({
            'project': wandb_run.project,
            'name': wandb_run.name,
            'id': wandb_run.id
        })
        wandb_run.log({"average_metrics": avg_metrics, "min_distance_stats": min_distance_stats})
        for metric, values in all_best_metrics.items():
            wandb_run.summary[f"avg_{metric}"] = np.mean(values)
            wandb_run.summary[f"std_{metric}"] = np.std(values)

        # **Save the best model from all runs with complete metadata**
        if best_overall_model_state is not None:
            models_dir = os.path.join(experiment_dir, "models")
            final_model_path = os.path.join(models_dir, f'SGT_model_BEST_OVERALL_run_{best_overall_run_number}_R2_{best_overall_r_squared:.4f}.pth')

            # Save model state with comprehensive metadata
            model_with_metadata = {
                'model_state_dict': best_overall_model_state,
                'best_run_number': best_overall_run_number,
                'best_metrics': best_overall_metrics,
                'average_metrics': {
                    'avg_r_squared': np.mean(all_best_metrics['r_squared']) if all_best_metrics['r_squared'] else 0,
                    'avg_mae': np.mean(all_best_metrics['mae']) if all_best_metrics['mae'] else 0,
                    'avg_rmse': np.mean(all_best_metrics['rmse']) if all_best_metrics['rmse'] else 0,
                    'avg_rpiq': np.mean(all_best_metrics['rpiq']) if all_best_metrics['rpiq'] else 0
                },
                'total_runs': args.num_runs,
                'experiment_info': {
                    'name': experiment_name,
                    'timestamp': timestamp,
                    'directory': experiment_dir
                },
                'model_config': {
                    'input_channels': len(bands_list_order),
                    'height': window_size,
                    'width': window_size,
                    'time_steps': time_before,
                    'd_model': args.hidden_size
                },
                'normalization_stats': {
                    'feature_means': feature_means,
                    'feature_stds': feature_stds,
                    'target_mean': target_mean,
                    'target_std': target_std
                },
                'training_config': {
                    'MAX_OC': MAX_OC,
                    'TIME_BEGINNING': TIME_BEGINNING,
                    'TIME_END': TIME_END,
                    'target_transform': args.target_transform,
                    'loss_type': args.loss_type,
                    'learning_rate': args.lr,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'dropout_rate': args.dropout_rate
                }
            }

            accelerator.save(model_with_metadata, final_model_path)
            wandb_run.save(final_model_path)

            print(f"\n**Best overall model from run {best_overall_run_number} saved**")
            print(f"Model path: {final_model_path}")
            print(f"Best R²: {best_overall_r_squared:.4f}")
            print(f"Best MAE: {best_overall_metrics['mae']:.4f}")
            print(f"Best RMSE: {best_overall_metrics['rmse']:.4f}")
            print(f"Best RPIQ: {best_overall_metrics['rpiq']:.4f}")
            print(f"Average R² across all runs: {np.mean(all_best_metrics['r_squared']) if all_best_metrics['r_squared'] else 0:.4f}")
        else:
            print("No final model saved - R² threshold not met for any run")

        wandb_run.finish()

        # Create a summary file with experiment information
        summary_file = os.path.join(experiment_dir, "EXPERIMENT_SUMMARY.txt")
        with open(summary_file, 'w') as f:
            f.write(f"EXPERIMENT SUMMARY\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Experiment Name: {experiment_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Directory: {experiment_dir}\n\n")

            f.write(f"CONFIGURATION:\n")
            f.write(f"-" * 20 + "\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nDATA PARAMETERS:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"MAX_OC: {MAX_OC}\n")
            f.write(f"TIME_BEGINNING: {TIME_BEGINNING}\n")
            f.write(f"TIME_END: {TIME_END}\n")
            f.write(f"Window Size: {window_size}\n")
            f.write(f"Time Before: {time_before}\n")
            f.write(f"Number of Bands: {len(bands_list_order)}\n")

            if best_overall_model_state is not None:
                f.write(f"\nBEST MODEL:\n")
                f.write(f"-" * 20 + "\n")
                f.write(f"Run Number: {best_overall_run_number}\n")
                f.write(f"R²: {best_overall_r_squared:.4f}\n")
                f.write(f"MAE: {best_overall_metrics['mae']:.4f}\n")
                f.write(f"RMSE: {best_overall_metrics['rmse']:.4f}\n")
                f.write(f"RPIQ: {best_overall_metrics['rpiq']:.4f}\n")

            f.write(f"\nFILES GENERATED:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Models: {models_dir}/\n")
            f.write(f"Results: {results_dir}/\n")
            if args.save_train_and_val:
                f.write(f"Data: {data_dir}/\n")

        print(f"\nExperiment completed successfully!")
        print(f"All outputs saved in: {experiment_dir}")
        print(f"Summary available at: {summary_file}")

    accelerator.print("All runs completed, average metrics and min distance statistics computed and saved!")
