#!/usr/bin/env python3
"""
rebuttal/final_models/train_full.py — train ONE neural-network architecture
on the entire LUCAS/LfL/LfU dataset (no spatial holdout) to produce a
production-ready mapping model.

The spatial-CV sweep (rebuttal/gpu_experiments/spatial_kfold/) is for
*evaluating* generalization. For *producing maps*, we re-train each
winning architecture using the full data — same hyperparameters, same
augmentation, same target transform, but train_idx = all rows. A small
5% random holdout monitors convergence; the saved checkpoint corresponds
to the highest-monitoring-R² epoch (best-state save, post b5c1cac fix).

Outputs (under rebuttal/final_models/checkpoints/<run-name>/):
    final_model.pth        weights + minimal metadata
    stats.json             target_mean, target_std, feature_means/stds
    train_log.txt          per-epoch training log
    config.json            full args used

Run:
    python rebuttal/final_models/train_full.py \\
        --run-name sgt_d128_h4_L1 \\
        --model-family sgt --model-size small \\
        --hidden_size 128 --num_heads 4 --num_layers 1 \\
        --dropout_rate 0.5 --lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 \\
        --loss_type composite_l2 --loss-alpha 0.5 --chi2-weight 0.1 \\
        --target_transform log --max-oc 150 \\
        --augment-train --num-epochs 60 \\
        --per-gpu-batch-size 256 --effective-batch-size 256
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('WANDB_MODE', 'disabled')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]
sys.path.insert(0, str(SOC_ROOT))
from _paths import SOC_REBUTTAL_DIR  # noqa: E402

SGT_DIR = SOC_ROOT / 'SpatiotemporalGatedTransformer'
sys.path.insert(0, str(SGT_DIR))
sys.path.insert(0, str(SGT_DIR / 'dataloader'))

# Reuse k-fold infrastructure where it fits.
KFOLD_DIR = SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'
sys.path.insert(0, str(KFOLD_DIR))
from run_kfold import (  # noqa: E402
    MODEL_READY, _build_model_ready_dataset, make_dataset, _build_model,
    _AugmentingWrapper, compute_density_weights,
)
from band_subsets import get_band_indices, band_suffix  # noqa: E402
import wandb  # noqa: E402  (disabled mode)
from accelerate import Accelerator  # noqa: E402
from train import train_model, _resolve_accum_steps, compute_training_statistics_oc  # noqa: E402
from config import bands_list_order, hidden_size, time_before, window_size, NUM_HEADS, NUM_LAYERS  # noqa: E402

CHECKPOINTS_ROOT = HERE / 'checkpoints'


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', type=str, required=True,
                   help='Identifier for the checkpoint folder, e.g. sgt_d128_h4_L1.')
    # Mirror run_kfold's CLI for the shared flags ----------------------------
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_heads', type=int, default=NUM_HEADS)
    p.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    p.add_argument('--loss_type', type=str, default='l1',
                   choices=['l1', 'mse', 'chi2', 'composite_l1', 'composite_l2'])
    p.add_argument('--loss-alpha', type=float, default=1.0)
    p.add_argument('--chi2-weight', type=float, default=0.1)
    p.add_argument('--target_transform', type=str, default='log',
                   choices=['none', 'log', 'normalize'])
    p.add_argument('--hidden_size', type=int, default=hidden_size)
    p.add_argument('--dropout_rate', type=float, default=0.5)
    p.add_argument('--model-size', type=str, default='small',
                   choices=['small', 'big'])
    p.add_argument('--model-family', type=str, default='sgt',
                   choices=['sgt', '3dcnn', 'cnnlstm', 'simpletransformer',
                            'vanilla_transformer', 'lightweight_transformer'])
    # crispness controls for the sgt/small flagship (consumed by
    # run_kfold._build_model; default ON = crisp linear-skip head).
    p.add_argument('--spatial-pool', type=str, default='avg',
                   choices=['avg', 'max', 'avgmax'])
    p.add_argument('--linear-skip', dest='linear_skip', action='store_true',
                   default=True)
    p.add_argument('--no-linear-skip', dest='linear_skip', action='store_false')
    p.add_argument('--static-head', dest='static_head', action='store_true',
                   default=True)
    p.add_argument('--no-static-head', dest='static_head', action='store_false')
    p.add_argument('--head-hidden', type=int, default=64)
    p.add_argument('--film', dest='film', action='store_true', default=False)
    p.add_argument('--no-film', dest='film', action='store_false')
    p.add_argument('--per-gpu-batch-size', type=int, default=256)
    p.add_argument('--effective-batch-size', type=int, default=256)
    p.add_argument('--accum-steps', type=int, default=0)
    p.add_argument('--num-epochs', type=int, default=60)
    p.add_argument('--lr-scheduler', type=str, default='cosine',
                   choices=['none', 'cosine', 'cosine_warm_restarts', 'exponential'])
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--lr-gamma', type=float, default=0.99)
    p.add_argument('--lr-restart-T0', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-oc', type=float, default=120.0)
    p.add_argument('--augment-train', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--monitor-frac', type=float, default=0.05,
                   help='Fraction of full data held out at RANDOM for '
                        'best-epoch monitoring. NOT a spatial split — '
                        'this is monitoring only, the saved model is meant '
                        'for production mapping.')
    p.add_argument('--monitor-n', type=int, default=0,
                   help='Absolute number of rows to hold out at RANDOM for '
                        'convergence monitoring. When > 0 it OVERRIDES '
                        '--monitor-frac, so e.g. --monitor-n 300 fits on '
                        'all-but-300 samples. Monitoring only (not spatial); '
                        'the saved model is for production mapping.')
    p.add_argument('--bands-list', type=str, default='full_extended',
                   choices=['full_20', 'original_6', 'full_extended',
                            'full_extended_nosoil'],
                   help='Covariate subset (default full_extended, the canonical '
                        '43-band stack). "full_extended_nosoil" drops the 5 '
                        'co-measured soil properties (circularity ablation). '
                        'Run-name auto-appends a band suffix so variants do not '
                        'overwrite each other under checkpoints/<run-name>/.')
    p.add_argument('--band-arch', type=str, default='two_path',
                   choices=['none', 'two_path'],
                   help='Band-input wrapper forwarded to the shared _build_model. '
                        '"two_path" (DEFAULT) reduces the extended Tier 1/2/3 '
                        'bands so the canonical 43-band final model matches the '
                        'spatial-CV sweep recipe; "none" feeds all channels raw.')
    p.add_argument('--ext-reduced', type=int, default=8,
                   help='[--band-arch two_path] channels after extended-band '
                        'reduction (default 8).')
    p.add_argument('--sampler-mode', type=str, default='none',
                   choices=['none', 'kde'],
                   help='Training-time sampler. "none" (default): plain '
                        'shuffle, preserves the raw LUCAS SOC distribution '
                        '(heavily skewed toward low SOC). "kde": '
                        'WeightedRandomSampler with KDE-inverse-density '
                        'weights on log(SOC), oversampling rare-tail '
                        '(high-SOC) rows. Use kde for production maps so '
                        'Alpine peat / organic-rich regions are not '
                        'systematically under-predicted.')
    p.add_argument('--sampler-alpha', type=float, default=0.5,
                   help='[kde mode only] Exponent on KDE-density inversion. '
                        'alpha=0 → uniform (no rebalancing), alpha=1 → full '
                        'inverse-frequency. Default 0.5 = sqrt-inverse, the '
                        'Yang et al. ICML 2021 standard for imbalance '
                        'regression.')
    return p.parse_args()


def main():
    args = parse()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Auto-append the bands-list suffix unless the user already encoded it.
    # Use 'in' rather than endswith so a rebal-suffixed name (e.g.
    # sgt_d128_h4_L1_20band_rebal) is not double-suffixed.
    suf = band_suffix(args.bands_list)
    if '_20band' not in args.run_name and '_6band' not in args.run_name:
        args.run_name = args.run_name + suf
    # Auto-append _rebal when KDE sampling is on, so rebalanced and
    # non-rebalanced final models coexist under checkpoints/.
    if args.sampler_mode == 'kde' and not args.run_name.endswith('_rebal'):
        args.run_name = args.run_name + '_rebal'
    out_dir = CHECKPOINTS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[final] run_name = {args.run_name}  (bands_list={args.bands_list})',
          flush=True)
    print(f'[final] output dir = {out_dir}', flush=True)

    # ---- Data ----
    _build_model_ready_dataset()
    df = pd.read_parquet(MODEL_READY).reset_index(drop=True)
    if args.max_oc and args.max_oc > 0:
        n_before = len(df)
        df = df[df['OC'] <= args.max_oc].reset_index(drop=True)
        print(f'[final] max-oc {args.max_oc:.1f}: kept {len(df):,}/{n_before:,}',
              flush=True)

    # Random monitor holdout
    rng = np.random.default_rng(args.seed)
    n = len(df)
    perm = rng.permutation(n)
    # --monitor-n (absolute count) overrides --monitor-frac when set, so the
    # production fit can hold out exactly a few hundred rows and train on the
    # rest. Clamp to [1, n-1] so we never empty the train or monitor split.
    if getattr(args, 'monitor_n', 0) and args.monitor_n > 0:
        n_mon = int(np.clip(args.monitor_n, 1, n - 1))
    else:
        n_mon = int(np.clip(round(args.monitor_frac * n), 1, n - 1))
    train_idx = perm[n_mon:]
    mon_idx = perm[:n_mon]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    mon_df = df.iloc[mon_idx].reset_index(drop=True)
    print(f'[final] train n={len(train_df)}  monitor n={len(mon_df)} '
          f'(fraction {n_mon / n:.2%}, RANDOM — not spatial)', flush=True)

    # Feature statistics — computed once over the FULL df (matches the k-fold
    # convention and the paper's pipeline).
    from run_kfold import compute_full_feature_statistics
    feature_means, feature_stds = compute_full_feature_statistics()
    target_mean, target_std = compute_training_statistics_oc()
    print(f'[final] target_mean={target_mean:.4f}  target_std={target_std:.4f}',
          flush=True)

    _band_indices = get_band_indices(args.bands_list, list(bands_list_order))
    train_ds = make_dataset(train_df, feature_means, feature_stds,
                             band_indices=_band_indices)
    mon_ds = make_dataset(mon_df, feature_means, feature_stds,
                           band_indices=_band_indices)
    if args.augment_train:
        train_ds = _AugmentingWrapper(train_ds, seed=args.seed)

    num_workers = int(os.environ.get('SOC_KFOLD_NUM_WORKERS', 0))

    # Training sampler: shuffle by default; KDE-inverse-density weights when
    # --sampler-mode kde. The latter oversamples high-SOC tail rows so the
    # production map's predicted SOC range covers Bavaria's organic-rich
    # regions (Alpine peat, fen / bog soils) instead of regressing toward
    # the bulk mineral-soil mean.
    if args.sampler_mode == 'kde':
        weights = compute_density_weights(train_df['OC'].to_numpy(),
                                           alpha=args.sampler_alpha)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_df),
            replacement=True,
        )
        print(f'[final] sampler=kde  alpha={args.sampler_alpha}  '
              f'n_train={len(train_df)}  '
              f'w in [{weights.min():.3f}, {weights.max():.3f}]  '
              f'mean={weights.mean():.3f}', flush=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.per_gpu_batch_size,
            sampler=sampler,           # shuffle MUST be False when sampler is set
            num_workers=num_workers, pin_memory=True,
        )
    else:
        print(f'[final] sampler=none  (plain shuffle, raw LUCAS distribution)',
              flush=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.per_gpu_batch_size,
            shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
    mon_loader = DataLoader(mon_ds, batch_size=args.per_gpu_batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # ---- Accelerator & gradient accumulation ----
    accelerator = Accelerator()
    accum_steps = _resolve_accum_steps(args, accelerator.num_processes)
    effective = accelerator.num_processes * args.per_gpu_batch_size * accum_steps
    if accelerator.is_main_process:
        print(f'[final] num_gpus={accelerator.num_processes}  '
              f'per_gpu_batch={args.per_gpu_batch_size}  '
              f'accum_steps={accum_steps}  effective_batch={effective}',
              flush=True)

    # ---- Model ----
    model = _build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[final] Model: {type(model).__name__}  family={args.model_family}  '
          f'({n_params:,} trainable params)', flush=True)

    wandb_run = wandb.init(project='socmapping-final',
                            name=args.run_name,
                            config=vars(args), reinit=True)

    # ---- Train ----
    t0 = time.time()
    (model, _, _, best_state, best_r2, epoch_metrics
     ) = train_model(
        model, train_loader, mon_loader,
        target_mean=target_mean, target_std=target_std,
        num_epochs=args.num_epochs,
        accelerator=accelerator,
        lr=args.lr,
        loss_type=args.loss_type,
        target_transform=args.target_transform,
        min_r2=-float('inf'),
        use_test=True,
        accum_steps=accum_steps,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_gamma=args.lr_gamma,
        lr_restart_T0=args.lr_restart_T0,
        loss_alpha=args.loss_alpha,
        chi2_weight=args.chi2_weight,
    )
    elapsed = time.time() - t0
    wandb_run.finish()
    print(f'[final] training done in {elapsed/60:.1f} min  best_r2 (monitor) = {best_r2:.4f}',
          flush=True)

    # ---- Save ----
    pth = out_dir / 'final_model.pth'
    accelerator.save({
        'model_state_dict': best_state,
        'family': args.model_family,
        'model_size': args.model_size,
        'd_model': args.hidden_size,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'best_r2_monitor': float(best_r2),
        'n_train': int(len(train_df)),
        'n_monitor': int(len(mon_df)),
        'args': vars(args),
    }, pth)
    print(f'[final] saved {pth}', flush=True)

    stats = {
        'target_mean': float(target_mean),
        'target_std': float(target_std),
        'feature_means': feature_means.tolist() if hasattr(feature_means, 'tolist') else list(feature_means),
        'feature_stds':  feature_stds.tolist() if hasattr(feature_stds, 'tolist') else list(feature_stds),
        'bands_list_order': list(bands_list_order),
        'time_before': int(time_before),
        'window_size': int(window_size),
    }
    (out_dir / 'stats.json').write_text(json.dumps(stats, indent=2, default=str))
    (out_dir / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str))
    (out_dir / 'train_log.txt').write_text(
        '\n'.join(f'epoch {m.get("epoch", "?"):>3}  '
                  f'train_loss={m.get("train_loss_avg", float("nan")):.4f}  '
                  f'test_loss={m.get("test_loss", float("nan")):.4f}  '
                  f'r2={m.get("r_squared", float("nan")):+.4f}  '
                  f'rmse={m.get("rmse", float("nan")):.3f}  '
                  f'mae={m.get("mae", float("nan")):.3f}'
                  for m in epoch_metrics))
    print(f'[final] saved stats.json, config.json, train_log.txt to {out_dir}',
          flush=True)


if __name__ == '__main__':
    main()
