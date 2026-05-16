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
from SimpleTFT import SimpleTFT
from EnhancedTFT import EnhancedTFT
import argparse
from balancedDataset import (
    create_validation_train_sets,
    create_balanced_dataset,
    create_stratified_validation_train_sets,
    create_spatial_kfold_splits,
)
import uuid
import os

# Uncomment and use this composite loss function if desired
def composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L1 and scaled chi-squared loss"""
    errors = targets - outputs
    l1_loss = torch.mean(torch.abs(errors))

    squared_errors = errors ** 2
    chi2_unscaled = (1/4) * squared_errors * torch.exp(-squared_errors / (2 * sigma))
    chi2_unscaled_mean = torch.mean(chi2_unscaled)

    chi2_unscaled_mean = torch.clamp(chi2_unscaled_mean, min=1e-8)
    scale_factor = l1_loss / chi2_unscaled_mean
    chi2_scaled = scale_factor * chi2_unscaled_mean

    return alpha * l1_loss + (1 - alpha) * chi2_scaled


def composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=0.5):
    """Composite loss combining L2 and scaled chi-squared loss"""
    errors = targets - outputs

    # L2 loss: mean squared error
    l2_loss = torch.mean(errors ** 2)

    # Standard chi-squared loss: errors^2 / sigma^2
    chi2_loss = torch.mean((errors ** 2) / (sigma ** 2))

    # Ensure chi2_loss is not too small to avoid division issues
    chi2_loss = torch.clamp(chi2_loss, min=1e-8)

    # Scale chi2_loss to match the magnitude of l2_loss
    scale_factor = l2_loss / chi2_loss
    chi2_scaled = scale_factor * chi2_loss

    # Combine the losses with the weighting factor alpha
    return alpha * l2_loss + (1 - alpha) * chi2_scaled


def parse_args():
    parser = argparse.ArgumentParser(description='Train SimpleTFT model with customizable parameters')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['composite_l1', 'l1', 'mse','composite_l2'], help='Type of loss function')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for L1 loss in composite loss (if used)')
    parser.add_argument('--target_transform', type=str, default='normalize', choices=['none', 'log', 'normalize'], help='Transformation to apply to targets')
    parser.add_argument('--use_validation', action='store_true', default=False, help='Whether to use validation set')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--target-val-ratio', type=float, default=0.08, help='Target validation ratio')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--distance-threshold', type=float, default=1.2, help='Minimum distance threshold for validation points')
    parser.add_argument('--target-fraction', type=float, default=0.75, help='Fraction of max bin count for resampling')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of times to run the process')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size for the model')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model')
    # Pick which TFT variant to build:
    #   small → SimpleTFT  (d_model=128, single transformer layer)       → 360,593 params
    #   big   → EnhancedTFT (multi-scale CNN num_scales=3, num_heads=4,
    #                        transformer encoder 3 layers)               → 1,126,417 params
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'big'],
                        help='TFT variant: "small" = SimpleTFT (360k); '
                             '"big" = EnhancedTFT multi-scale (1.10M).')
    # --- Smooth-training bundle -----------------------------------------
    # All defaults preserve the original 8dce131 behaviour bit-identically:
    #   --lr-scheduler none → fixed LR, no schedule
    #   --grad-clip 0.0     → no gradient clipping
    # Enable any subset to reduce metric volatility.
    parser.add_argument('--lr-scheduler', type=str, default='none',
                        choices=['none', 'cosine', 'warmup_cosine'],
                        help='LR schedule. "warmup_cosine" = LinearLR warmup '
                             'then CosineAnnealingLR to --lr-min over the run.')
    parser.add_argument('--lr-min', type=float, default=1e-6,
                        help='Final LR for cosine schedules (eta_min).')
    parser.add_argument('--lr-warmup-epochs', type=int, default=20,
                        help='Warmup length in epochs for warmup_cosine. '
                             'Typical: 5-10%% of --num-epochs.')
    parser.add_argument('--lr-warmup-start-factor', type=float, default=0.01,
                        help='Initial LR fraction at epoch 0 for warmup_cosine '
                             '(LR ramps from --lr * this to --lr).')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='Max gradient norm for clip_grad_norm_; 0 disables. '
                             '1.0 is a sensible default for stability.')
    # --- Anti-overfit bundle --------------------------------------------
    # Defaults preserve original Adam-with-no-weight-decay, no-early-stop
    # behaviour. Enable when the val R² curve peaks early then drifts down.
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 weight decay for AdamW. 0 keeps plain Adam '
                             '(default). 1e-4–1e-2 is the usual range; '
                             'start at 1e-4 for soft regularisation.')
    parser.add_argument('--early-stop-patience', type=int, default=0,
                        help='Stop training if val R² (r_squared, proper) has '
                             'not improved for this many epochs. 0 disables '
                             '(default). 20–50 is typical for 200-epoch runs.')
    # --- Dataset bundle: stratified split + KS gate + spatial K-fold -------
    # Defaults preserve legacy behaviour bit-identically.
    parser.add_argument('--split-mode', type=str, default='legacy',
                        choices=['legacy', 'stratified'],
                        help='"legacy" = inverse-gamma sampled val set (original '
                             '8dce131 behaviour). "stratified" = OC-quantile-'
                             'stratified val sampling that matches train/val '
                             'OC distributions before applying the spatial buffer.')
    parser.add_argument('--split-n-bins', type=int, default=10,
                        help='Number of OC quantile bins for --split-mode '
                             'stratified.')
    parser.add_argument('--ks-pvalue-min', type=float, default=0.0,
                        help='Reject stratified splits whose ks_2samp(train_oc, '
                             'val_oc).pvalue < this. 0 disables the gate. '
                             '0.05 means "require train/val OC distributions to '
                             'be statistically indistinguishable".')
    parser.add_argument('--kfold', type=int, default=0,
                        help='Spatial K-fold CV. 0 (default) = single split, '
                             'repeated --num-runs times. >0 = build N latitude-'
                             'decile folds; the outer run loop iterates over '
                             'them, replacing --num-runs.')
    # --- Train-set rebalancing (per-OC-bin upsampling AFTER the split) -----
    # Independent dimension from --split-mode/--kfold. Can be enabled with
    # any split strategy. Default off preserves 8dce131 behaviour: train_df
    # returned by the split function is used as-is (raw spatial half).
    parser.add_argument('--rebalance-train', action='store_true', default=False,
                        help='After the train/val split, upsample the TRAIN set '
                             'so every OC quantile bin has at least '
                             '--rebalance-min-ratio × max_bin_count rows '
                             '(duplicated samples for under-represented bins). '
                             'Helps the model see high-OC tails more often. '
                             'Stacks with any --split-mode and with --kfold.')
    parser.add_argument('--rebalance-min-ratio', type=float, default=0.75,
                        help='Per-bin floor as fraction of densest bin for '
                             '--rebalance-train.')
    parser.add_argument('--rebalance-n-bins', type=int, default=128,
                        help='Number of OC quantile bins for --rebalance-train.')
    # --- Augmentation bundle (satellite-safe subset) -----------------------
    # Only physically-meaningful augmentations for radiometric satellite
    # covariates (Elevation, LAI, LST, MODIS_NPP, SoilEvaporation,
    # TotalEvapotranspiration). All bands are scalar (no direction), so
    # D4 spatial symmetries are pure-win augmentations. Temporal-dropout
    # (zero-fill) and mixup were removed because they create samples that
    # don't correspond to any real Bavaria location/timestep.
    parser.add_argument('--aug-spatial-flip', action='store_true', default=False,
                        help='Random hflip + vflip + 90°k rotation of the 5×5 '
                             'spatial patch on each training sample. Up to 8× '
                             'effective dataset size. Safe for all 6 scalar bands.')
    # --- Model / eval bundle: stochastic depth + test-time augmentation ---
    parser.add_argument('--layer-drop-prob', type=float, default=0.0,
                        help='LayerDrop / stochastic depth probability for the '
                             'transformer encoder. 0 disables (default). 0.1 is '
                             'a sensible value for a 3-layer encoder. Only '
                             'applies during training; eval skips no layers. '
                             'Currently --model-size big only (EnhancedTFT).')
    parser.add_argument('--tta', action='store_true', default=False,
                        help='Test-time augmentation: at eval, average the model '
                             'output over 4 rotations (0/90/180/270°) of the '
                             'input. Costs 4× eval forward passes; typically '
                             'adds ~0.005-0.02 to RPIQ for free.')
    return parser.parse_args()

def train_model(model, train_loader, val_loader,target_mean,target_std, num_epochs=num_epochs, accelerator=None, lr=0.001,
                loss_type='l1', loss_alpha=0.5, target_transform='none', min_r2=0.5, use_validation=True,
                lr_scheduler='none', lr_min=1e-6, lr_warmup_epochs=20, lr_warmup_start_factor=0.01,
                grad_clip=0.0, weight_decay=0.0, early_stop_patience=0,
                tta=False):
    # Define loss function based on loss_type
    if loss_type == 'composite_l1':
        criterion = lambda outputs, targets: composite_l1_chi2_loss(outputs, targets, sigma=3.0, alpha=loss_alpha)
    elif loss_type == 'composite_l2':
        criterion = lambda outputs, targets: composite_l2_chi2_loss(outputs, targets, sigma=3.0, alpha=loss_alpha)
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Use AdamW (decoupled weight decay) when weight_decay > 0;
    # plain Adam otherwise (matches original 8dce131 behaviour bit-identically
    # when --weight-decay is left at its default 0).
    if weight_decay > 0:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Build LR scheduler BEFORE accelerator.prepare so the optimizer's
    # param groups are the un-wrapped ones the scheduler will mutate.
    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr_min)
    elif lr_scheduler == 'warmup_cosine':
        # LinearLR warmup → CosineAnnealingLR chained via SequentialLR.
        # epoch [0, warmup):  lr ramps lr*start_factor → lr
        # epoch [warmup, end): lr cosine-descends lr → lr_min
        warmup_epochs = max(1, min(int(lr_warmup_epochs), num_epochs - 1))
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=lr_warmup_start_factor,
            end_factor=1.0, total_iters=warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=lr_min)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = None
    if accelerator is not None and accelerator.is_main_process:
        if lr_scheduler == 'warmup_cosine':
            warmup_epochs = max(1, min(int(lr_warmup_epochs), num_epochs - 1))
            print(f"LR schedule: warmup_cosine  "
                  f"(warmup {warmup_epochs} ep: {lr*lr_warmup_start_factor:.2e} → {lr:.2e}, "
                  f"cosine {num_epochs - warmup_epochs} ep: {lr:.2e} → {lr_min:.2e})", flush=True)
        else:
            print(f"LR schedule: {lr_scheduler}  (lr={lr}, lr_min={lr_min})", flush=True)
        if grad_clip > 0:
            print(f"Gradient clipping: max_norm={grad_clip}", flush=True)
        if weight_decay > 0:
            print(f"Optimizer: AdamW (weight_decay={weight_decay})", flush=True)
        if early_stop_patience > 0:
            print(f"Early stopping: patience={early_stop_patience} (monitoring r_squared)", flush=True)
        if tta:
            print(f"Test-time augmentation: 4-rotation average on val forward", flush=True)

    # Early-stopping bookkeeping (only used when early_stop_patience > 0)
    epochs_since_best = 0

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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (longitudes, latitudes, features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(accelerator.device)
            targets = targets.to(accelerator.device).float()

            # Apply target transformation
            if target_transform == 'log':
                targets = torch.log(targets + 1e-10)  # Add small constant to avoid log(0)
            elif target_transform == 'normalize':
                targets = (targets - target_mean) / (target_std + 1e-10)
            # 'none' requires no transformation

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            # Gradient clipping (disabled when grad_clip == 0).
            # accelerator.clip_grad_norm_ unscales mixed-precision grads first
            # and operates on the wrapped DDP model, so it's the correct call.
            if grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item()

            if accelerator.is_main_process:
                wandb.log({
                    'train_loss': loss.item(),
                    'batch': batch_idx + 1 + epoch * len(train_loader),
                    'epoch': epoch + 1
                })

        train_loss = running_loss / len(train_loader)

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

                    if tta:
                        # Test-time augmentation: average outputs over the 4
                        # 90°k rotations of the spatial dims (H, W). Input
                        # layout is (B, C, H, W, T) — rotate on (2, 3).
                        out_acc = None
                        for k in range(4):
                            feat_k = features if k == 0 else torch.rot90(features, k=k, dims=(2, 3))
                            o = model(feat_k)
                            out_acc = o if out_acc is None else (out_acc + o)
                        outputs = out_acc / 4.0
                    else:
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
                # which is what scikit-learn's r2_score reports and what reviewers mean
                # by R² for regression on held-out data. Squared Pearson correlation is
                # kept alongside as `pearson_r2` for continuity with v1 reporting; the
                # two agree only when predictions are unbiased and identically scaled.
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
                epochs_since_best = 0  # reset patience counter on improvement
            elif not use_validation and epoch == num_epochs - 1:
                best_r2 = 1.0
                best_model_state = model.state_dict()
                wandb.run.summary['best_r2'] = best_r2
            else:
                # Only count toward patience when we DID compute a real val R²
                # (use_validation=True and val pass actually ran).
                if use_validation and not np.isnan(r_squared):
                    epochs_since_best += 1

        accelerator.print(f'Epoch {epoch+1}:')
        accelerator.print(f'Training Loss: {train_loss:.4f}')
        if use_validation:
            accelerator.print(f'Validation Loss: {val_loss:.4f}')
            accelerator.print(f'R²: {r_squared:.4f}')
            accelerator.print(f'RMSE: {rmse:.4f}')
            accelerator.print(f'MAE: {mae:.4f}')
            accelerator.print(f'RPIQ: {rpiq:.4f}\n')
            # Print the results


        # Step the LR scheduler at epoch boundary (after eval, after logging).
        if scheduler is not None:
            scheduler.step()

        # Early stopping check (after the LR step so the schedule still
        # advances as recorded in epoch_metrics for any partial run).
        # CRITICAL: epochs_since_best is only updated on rank 0 (inside the
        # is_main_process block above), so this decision MUST be broadcast
        # to all ranks. Otherwise rank 0 breaks out of the epoch loop while
        # other ranks keep training — causing NCCL allgather timeouts at the
        # start of the next run's accelerator.prepare(). Use accelerator.gather
        # to disseminate rank 0's stop decision; non-main ranks contribute 0,
        # so .any() returns True iff rank 0 decided to stop.
        if early_stop_patience > 0 and use_validation:
            stop_t = torch.tensor([0], device=accelerator.device, dtype=torch.int)
            if accelerator.is_main_process and epochs_since_best >= early_stop_patience:
                stop_t[0] = 1
            stop_all = accelerator.gather(stop_t)
            if stop_all.any().item():
                if accelerator.is_main_process:
                    print(f"Early stopping at epoch {epoch+1}: "
                          f"r_squared has not improved for {epochs_since_best} epochs "
                          f"(patience={early_stop_patience}). Best r_squared = {best_r2:.4f}.",
                          flush=True)
                break

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

if __name__ == "__main__":
    args = parse_args()
    # Set num_runs to 1 if use_validation is False
    if not args.use_validation:
        args.num_runs = 1
        num_epochs = NUM_EPOCHS_RUN
    # LayerDrop / stochastic depth means each rank may skip different transformer
    # encoder layers per batch, so the skipped layers' params get no gradient on
    # one rank while contributing on another — DDP raises
    #   "Expected to have finished reduction in the prior iteration before
    #    starting a new one. … Parameter indices which did not receive grad…"
    # The PyTorch-documented fix is to pass find_unused_parameters=True to DDP.
    # We only enable it when LayerDrop is actually on (it adds ~5% DDP overhead
    # per backward).
    if args.layer_drop_prob > 0:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator()

    # Pre-compute spatial K-fold splits if requested. Each fold = one
    # (val_df, train_df, dist_stats, fold_meta) tuple; the outer run loop
    # iterates over them instead of running --num-runs independent splits.
    kfold_splits = None
    if args.use_validation and args.kfold > 0:
        if accelerator.is_main_process:
            print(f"Building {args.kfold} spatial K-fold splits (lat-decile blocks, "
                  f"distance_threshold={args.distance_threshold} km)…")
        df_all = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
        kfold_splits = create_spatial_kfold_splits(
            df=df_all,
            output_dir=args.output_dir,
            n_folds=args.kfold,
            use_gpu=args.use_gpu,
            distance_threshold=args.distance_threshold,
            ks_pvalue_min=args.ks_pvalue_min,
        )
        if not kfold_splits:
            raise RuntimeError("create_spatial_kfold_splits returned 0 folds; "
                               "lower --distance-threshold or increase --kfold "
                               "to leave more usable points per block.")
        args.num_runs = len(kfold_splits)
        if accelerator.is_main_process:
            print(f"K-fold mode: --num-runs overridden to {args.num_runs} "
                  f"(one run per fold).")

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

    # Loop through the specified number of runs
    for run in range(args.num_runs):
        if accelerator.is_main_process:
            print(f"\nStarting Run {run + 1}/{args.num_runs}")

        # Initialize wandb for this run
        if accelerator.is_main_process:
            wandb_run = wandb.init(
                project="socmapping-SimpleTFT",
                name=f"run_{run+1}",
                config={
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
                    "loss_alpha": args.loss_alpha,
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
        #train_dataset_std_means = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        train_dataset_std_means = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
        train_dataset_std_means.set_feature_means(train_dataset_features_norm.get_feature_means())
        train_dataset_std_means.set_feature_stds(train_dataset_features_norm.get_feature_stds())
        target_mean, target_std =  compute_training_statistics_oc()
        # Create train/validation split — pick strategy based on flags.
        if args.use_validation:
            if kfold_splits is not None:
                # Spatial K-fold: pull the precomputed split for this fold.
                val_df, train_df, min_distance_stats, fold_meta = kfold_splits[run]
                if accelerator.is_main_process:
                    print(f"Using spatial fold {fold_meta['fold_idx']+1}/{args.kfold}: "
                          f"lat [{fold_meta['lat_lo']:.4f}, {fold_meta['lat_hi']:.4f}]  "
                          f"val n={fold_meta['val_size']}  train n={fold_meta['train_size']}  "
                          f"KS p={fold_meta['ks_pvalue']:.4f}")
            elif args.split_mode == 'stratified':
                val_df, train_df, min_distance_stats = create_stratified_validation_train_sets(
                    df=df,
                    output_dir=args.output_dir,
                    target_val_ratio=args.target_val_ratio,
                    use_gpu=args.use_gpu,
                    distance_threshold=args.distance_threshold,
                    n_bins=args.split_n_bins,
                    ks_pvalue_min=args.ks_pvalue_min,
                    seed=42 + run,
                )
            else:
                val_df, train_df, min_distance_stats = create_validation_train_sets(
                    df=df,
                    output_dir=args.output_dir,
                    target_val_ratio=args.target_val_ratio,
                    use_gpu=args.use_gpu,
                    distance_threshold=args.distance_threshold
                )
            min_distance_stats_all.append(min_distance_stats)

            # Optional: rebalance the (spatial-split) training set by per-OC-bin
            # upsampling. Stacks with whichever split strategy produced
            # train_df above — legacy, stratified, or K-fold. Val is never
            # rebalanced (we evaluate on the natural distribution).
            if args.rebalance_train:
                train_df_pre_rebalance = train_df
                train_df, _ = create_balanced_dataset(
                    train_df,
                    n_bins=args.rebalance_n_bins,
                    min_ratio=args.rebalance_min_ratio,
                    use_validation=False,
                )
                if accelerator.is_main_process:
                    print(f"Rebalanced train: {len(train_df_pre_rebalance)} → {len(train_df)} rows "
                          f"(+{len(train_df) - len(train_df_pre_rebalance)} from per-bin oversample; "
                          f"n_bins={args.rebalance_n_bins}, min_ratio={args.rebalance_min_ratio})")


        # Create datasets
        if args.use_validation:
            train_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            val_dataset = NormalizedMultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            val_dataset.set_feature_means(train_dataset_features_norm.get_feature_means())
            val_dataset.set_feature_stds(train_dataset_features_norm.get_feature_stds())
            train_dataset.set_feature_means(train_dataset_features_norm.get_feature_means())
            train_dataset.set_feature_stds(train_dataset_features_norm.get_feature_stds())
            # Enable augmentation on TRAINING dataset only (val stays deterministic).
            if args.aug_spatial_flip:
                train_dataset.set_augmentation(spatial_flip=True)
                if accelerator.is_main_process:
                    print(f"Augmentation on train: spatial_flip=True (D4 — hflip + vflip + 90°k rotation)")
            #train_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, train_df)
            #val_dataset = MultiRasterDatasetMultiYears(samples_coordinates_array_path, data_array_path, val_df)
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset)}")
                print(f"Run {run + 1} Length of val_dataset: {len(val_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        else:
            if accelerator.is_main_process:
                print(f"Run {run + 1} Length of train_dataset: {len(train_dataset_std_means)}")
            train_loader = DataLoader(train_dataset_std_means, batch_size=256, shuffle=True)
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

        # Initialize model — pick SimpleTFT or EnhancedTFT via --model-size.
        if args.model_size == 'big':
            # EnhancedTFT (single-scale CNN + GRN blocks + 3-layer transformer
            # encoder + expansion_factor=4) → 1,120,546 trainable params. This
            # is the architecture of the historic Model A v2 saved checkpoint
            # at residualModels1mil_normalize_composite_l2_v2 (R²=0.6909):
            # load_state_dict on that .pth gives 0 missing / 0 unexpected keys.
            model = EnhancedTFT(
                input_channels=len(bands_list_order),
                height=window_size,
                width=window_size,
                time_steps=time_before,
                d_model=args.hidden_size,
                num_heads=4,
                dropout=args.dropout_rate,
                num_encoder_layers=3,
                expansion_factor=4,
            )
            if args.layer_drop_prob > 0:
                model.set_layer_drop_prob(args.layer_drop_prob)
                if accelerator.is_main_process:
                    print(f"LayerDrop: enabled (p={args.layer_drop_prob}) on EnhancedTFT")
        else:
            model = SimpleTFT(
                input_channels=len(bands_list_order),
                height=window_size,
                width=window_size,
                time_steps=time_before,
                d_model=args.hidden_size
            )
        print(f" Model variant: {type(model).__name__} (--model-size={args.model_size});",
              "parameters:", model.count_parameters())

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
            loss_alpha=args.loss_alpha,
            target_transform=args.target_transform,
            min_r2=0.5,
            use_validation=args.use_validation,
            lr_scheduler=args.lr_scheduler,
            lr_min=args.lr_min,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_warmup_start_factor=args.lr_warmup_start_factor,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            tta=args.tta,
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

        # Save model
        if accelerator.is_main_process and best_model_state is not None:
            final_model_path = (f'TFT_model_run_{run+1}_MAX_OC_{MAX_OC}_TIME_BEGINNING_{TIME_BEGINNING}_'
                               f'TIME_END_{TIME_END}_R2_{best_r2:.4f}_TRANSFORM_{args.target_transform}_'
                               f'LOSS_{args.loss_type}.pth')
            accelerator.save(best_model_state, final_model_path)
            wandb_run.save(final_model_path)
            print(f"Run {run + 1} Model with best R² ({best_r2:.4f}) saved at: {final_model_path}")
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

        # Save all metrics to file
        output_file = os.path.join(args.output_dir, f'training_metrics_tft_{uuid.uuid4().hex[:8]}.txt')
        save_metrics_to_file(args, wandb_runs_info, avg_metrics, min_distance_stats, all_best_metrics, filename=output_file)
        print(f"\nMetrics saved to: {output_file}")

        # Log average metrics to a new wandb run
        wandb_run = wandb.init(project="socmapping-SimpleTFT", name="average_metrics")
        wandb_runs_info.append({
            'project': wandb_run.project,
            'name': wandb_run.name,
            'id': wandb_run.id
        })
        wandb_run.log({"average_metrics": avg_metrics, "min_distance_stats": min_distance_stats})
        for metric, values in all_best_metrics.items():
            wandb_run.summary[f"avg_{metric}"] = np.mean(values)
            wandb_run.summary[f"std_{metric}"] = np.std(values)
        wandb_run.finish()

    accelerator.print("All runs completed, average metrics and min distance statistics computed and saved!")
