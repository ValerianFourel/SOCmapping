#!/usr/bin/env python3
"""
mc_dropout_uncertainty.py — Task T3.2 (R3.9 + R4.4).

Both R3 and R4 flag the absence of prediction-uncertainty estimates in
the SOC maps. Full Bayesian / ensemble UQ is a major retraining effort
we don't have budget for before the May 29 deadline. As a defensible
cheap-win we generate **Monte Carlo dropout** uncertainty estimates from
the already-trained Model A — Gal & Ghahramani 2016 — by running N
stochastic forward passes with dropout active at inference time, then
reporting the per-sample mean and standard deviation as a usable
prediction interval.

For each validation row we report:
    pred_mean      mean over N MC samples
    pred_std       standard deviation over N MC samples
    pred_p05/p95   5th/95th percentile = 90% credible interval
    PI_width       p95 - p05
    in_PI          1 if OC_actual lies in [p05, p95], else 0

Coverage (mean of in_PI) ≈ 0.90 indicates the MC dropout PI is
reasonably calibrated. If coverage is much below 0.90 the UQ is
underdispersed (common at low dropout rates) and we report that
honestly as a known calibration limitation.

Outputs (under rebuttal/):
    mc_dropout_predictions.parquet     full per-sample table
    mc_dropout_uncertainty.json        aggregate stats
    mc_dropout_uncertainty.md          markdown summary
    mc_dropout_map.png                 spatial uncertainty map

Run on the cluster from SOCmapping/ (needs CUDA + the Model A
checkpoint):
    python rebuttal/mc_dropout_uncertainty.py --n-mc 30

Compute cost: ~5-10 min for n_mc=30 on one GPU (single-batch inference
over ~1.4k validation points × 30 forward passes).
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parent
sys.path.insert(0, str(SOC_ROOT))
from _paths import SOC_REBUTTAL_DIR, SOC_WEIGHTS_DIR  # noqa: E402

SGT_DIR = SOC_ROOT / 'SpatiotemporalGatedTransformer'
sys.path.insert(0, str(SGT_DIR))
sys.path.insert(0, str(SGT_DIR / 'dataloader'))

from EnhancedSGT import EnhancedSGT                  # noqa: E402
from dataloaderMultiYears import MultiRasterDatasetMultiYears  # noqa: E402
from dataframe_loader import filter_dataframe, separate_and_add_data  # noqa: E402
from config import TIME_BEGINNING, TIME_END, MAX_OC, time_before      # noqa: E402

# --------------------------------------------------------------------------
# Model A checkpoint paths (Model A = EnhancedSGT(d=128, h=4, L=3) trained
# at MAX_OC=150 on the 1mil run-1 split, R²=0.6909). Resolved via
# SOC_WEIGHTS_DIR — set the env var if the weights live somewhere
# non-standard (e.g. on the cluster).
# --------------------------------------------------------------------------
DEFAULT_PTH = (
    SOC_WEIGHTS_DIR /
    'TemporalFusionTransformer' /
    'residualModels1mil_normalize_composite_l2_v2' /
    'TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_'
    'TRANSFORM_normalize_LOSS_composite_l2_R2_0.6909.pth'
)
DEFAULT_PKL = (
    SOC_WEIGHTS_DIR /
    'Archive' / 'residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer' /
    'analysis_results.pkl'
)


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-mc', type=int, default=30,
                   help='Number of MC dropout forward passes per sample.')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_PTH)
    p.add_argument('--analysis-pkl', type=Path, default=DEFAULT_PKL,
                   help='analysis_results.pkl with the saved val predictions '
                        '+ normalization stats.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Batch size for inference.')
    p.add_argument('--limit', type=int, default=0,
                   help='If > 0, only run on the first N validation samples '
                        '(for debugging).')
    p.add_argument('--device', type=str, default='cuda',
                   help='cuda or cpu.')
    return p.parse_args()


def enable_dropout(m: torch.nn.Module) -> int:
    """Switch every dropout layer in m back to training mode (so it
    samples). Returns the count of dropout layers found."""
    n = 0
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout1d,
                            torch.nn.Dropout2d, torch.nn.Dropout3d)):
            mod.train()
            n += 1
    return n


def main():
    args = parse()
    device = (torch.device(args.device)
              if (args.device == 'cpu' or torch.cuda.is_available())
              else torch.device('cpu'))
    print(f'[mc-dropout] device={device}  n_mc={args.n_mc}', flush=True)

    # --- Load checkpoint + saved val analysis ---
    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f'\n[ERROR] Model A checkpoint not found at:\n'
            f'    {args.checkpoint}\n'
            f'SOC_WEIGHTS_DIR currently resolves to:\n'
            f'    {SOC_WEIGHTS_DIR}\n'
            f'Either:\n'
            f'  (1) export SOC_WEIGHTS_DIR=/path/to/Weights-…/\n'
            f'  (2) scp the Model A checkpoint + analysis_results.pkl from the laptop\n'
            f'  (3) pass --checkpoint /path/to/model.pth --analysis-pkl /path/to/analysis_results.pkl\n'
        )
    if not Path(args.analysis_pkl).exists():
        raise SystemExit(
            f'\n[ERROR] analysis_results.pkl not found at:\n'
            f'    {args.analysis_pkl}\n'
            f'The pkl carries the target_mean / target_std / feature_means / '
            f'feature_stds used during Model A training. It must be transferred '
            f'along with the .pth.\n'
        )
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ckpt['model_state_dict'].items()}
    analysis = pickle.load(open(args.analysis_pkl, 'rb'))
    s = analysis['stats']
    target_mean = float(s['target_mean'])
    target_std = float(s['target_std'])
    feature_means = torch.as_tensor(s['feature_means'], dtype=torch.float32)
    feature_stds = torch.as_tensor(s['feature_stds'], dtype=torch.float32)

    val_lon_saved = np.asarray(analysis['val_results']['longitudes'], dtype=float)
    val_lat_saved = np.asarray(analysis['val_results']['latitudes'], dtype=float)
    val_preds_saved = np.asarray(analysis['val_results']['predictions'], dtype=float)
    val_targets_saved = np.asarray(analysis['val_results']['targets'], dtype=float)
    print(f'[mc-dropout] saved val set: {len(val_lon_saved):,} samples', flush=True)

    # --- Rebuild val_df aligned to saved-val coordinates ---
    df_full = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    df_full['_key'] = list(zip(np.round(df_full['GPS_LONG'], 6),
                                np.round(df_full['GPS_LAT'], 6)))
    saved_keys = list(zip(np.round(val_lon_saved, 6),
                          np.round(val_lat_saved, 6)))
    saved_key_to_pred = dict(zip(saved_keys, val_preds_saved))
    saved_key_to_target = dict(zip(saved_keys, val_targets_saved))

    val_df = (df_full[df_full['_key'].isin(set(saved_keys))]
              .drop(columns=['_key']).reset_index(drop=True))
    if args.limit > 0:
        val_df = val_df.head(args.limit).reset_index(drop=True)
    print(f'[mc-dropout] val_df reconstructed: {len(val_df)} rows', flush=True)

    sample_paths, data_paths = separate_and_add_data()

    def flatten(lst):
        out = []
        for x in lst:
            out += flatten(x) if isinstance(x, list) else [x]
        return out
    sample_paths = list(dict.fromkeys(flatten(sample_paths)))
    data_paths = list(dict.fromkeys(flatten(data_paths)))

    ds = MultiRasterDatasetMultiYears(sample_paths, data_paths,
                                       val_df, time_before=time_before)

    # --- Build model and load weights ---
    model = EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5,
                        d_model=128, num_heads=4, dropout=0.3,
                        num_encoder_layers=3, expansion_factor=4).to(device)
    model.load_state_dict(sd, strict=True)
    # Eval mode disables batchnorm running-stats updates; we only want dropout
    # active, so manually toggle dropout back to train mode after .eval().
    model.eval()
    n_dropout = enable_dropout(model)
    print(f'[mc-dropout] model loaded ({sum(p.numel() for p in model.parameters()):,} params), '
          f'{n_dropout} dropout layers active', flush=True)

    # --- Build all-features batch once (a few hundred MB at this scale) ---
    feats_all = []
    lons_all = []
    lats_all = []
    targets_orig = []
    saved_preds_orig = []
    for i in range(len(ds)):
        lon, lat, f, oc = ds[i]
        f_norm = (f - feature_means[:, None, None]) / feature_stds[:, None, None]
        feats_all.append(f_norm)
        lons_all.append(float(lon))
        lats_all.append(float(lat))
        targets_orig.append(float(oc))
        key = (round(lons_all[-1], 6), round(lats_all[-1], 6))
        saved_preds_orig.append(float(saved_key_to_pred.get(key, np.nan)))
    feats_all = torch.stack(feats_all).float()
    lons_all = np.array(lons_all)
    lats_all = np.array(lats_all)
    targets_orig = np.array(targets_orig)
    saved_preds_orig = np.array(saved_preds_orig)
    n_samples = feats_all.shape[0]
    print(f'[mc-dropout] features tensor: {tuple(feats_all.shape)}', flush=True)

    # --- MC dropout: n_mc forward passes, collect predictions ---
    all_mc_preds = np.zeros((args.n_mc, n_samples), dtype=np.float32)
    feats_all = feats_all.to(device)
    with torch.no_grad():
        for k in range(args.n_mc):
            # re-enable dropout each pass (defensive — model.eval() shouldn't
            # be called inside this loop, but make explicit)
            enable_dropout(model)
            batches = []
            for s_ in range(0, n_samples, args.batch_size):
                e_ = min(s_ + args.batch_size, n_samples)
                out = model(feats_all[s_:e_]).cpu().numpy()
                batches.append(out)
            preds_norm = np.concatenate(batches)
            # Inverse-transform: model output is normalized, so add target_mean × std back.
            # The trained model uses target_transform=normalize (per the analysis pkl
            # stats key), so saved preds in analysis_results.pkl are already in g/kg.
            # We mirror that: undo the normalize transform.
            preds_gkg = preds_norm * target_std + target_mean
            preds_gkg = np.clip(preds_gkg, 0.0, None)
            all_mc_preds[k] = preds_gkg
            print(f'  [pass {k+1:>3}/{args.n_mc}] mean={preds_gkg.mean():.2f}  '
                  f'std={preds_gkg.std():.2f}  range=[{preds_gkg.min():.2f}, {preds_gkg.max():.2f}]',
                  flush=True)

    # --- Aggregate per-sample MC statistics ---
    pred_mean = all_mc_preds.mean(axis=0)
    pred_std = all_mc_preds.std(axis=0, ddof=1)
    pred_p05 = np.percentile(all_mc_preds, 5, axis=0)
    pred_p95 = np.percentile(all_mc_preds, 95, axis=0)
    pi_width = pred_p95 - pred_p05
    in_pi = ((targets_orig >= pred_p05) & (targets_orig <= pred_p95)).astype(int)

    # Reliability metrics on the MC mean
    resid = pred_mean - targets_orig
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((targets_orig - targets_orig.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    coverage_90 = float(in_pi.mean())

    summary = {
        'n_samples': int(n_samples),
        'n_mc': int(args.n_mc),
        'r2_on_mc_mean': r2,
        'rmse_on_mc_mean': rmse,
        'mae_on_mc_mean': mae,
        'pi90_width_mean': float(pi_width.mean()),
        'pi90_width_median': float(np.median(pi_width)),
        'pi90_coverage_observed': coverage_90,
        'pi90_coverage_expected': 0.90,
        'mc_std_mean': float(pred_std.mean()),
        'mc_std_median': float(np.median(pred_std)),
    }
    print(f'\n[mc-dropout] Summary:')
    for k, v in summary.items():
        print(f'  {k:>26}: {v}', flush=True)

    out_df = pd.DataFrame({
        'GPS_LAT': lats_all, 'GPS_LONG': lons_all,
        'OC_actual': targets_orig,
        'pred_saved': saved_preds_orig,    # the paper's deterministic prediction
        'pred_mc_mean': pred_mean,
        'pred_mc_std': pred_std,
        'pred_mc_p05': pred_p05,
        'pred_mc_p95': pred_p95,
        'pi90_width': pi_width,
        'in_pi90': in_pi,
    })
    out_df.to_parquet(HERE / 'mc_dropout_predictions.parquet')
    print(f'Saved {HERE / "mc_dropout_predictions.parquet"}', flush=True)

    (HERE / 'mc_dropout_uncertainty.json').write_text(
        json.dumps(summary, indent=2, default=str))

    # --- Markdown summary ---
    md = ['# MC dropout uncertainty (Model A)', '']
    md.append(f'**Configuration:** n_mc = {args.n_mc} stochastic forward passes, '
              f'dropout p = 0.3 at all dropout layers ({n_dropout} layers), '
              f'evaluated on n = {n_samples} validation samples.')
    md.append('')
    md.append('## Aggregate metrics')
    md.append('')
    md.append(f'| Metric | Value |')
    md.append(f'|--------|-------|')
    md.append(f'| R² (on MC mean prediction) | {r2:.4f} |')
    md.append(f'| RMSE (on MC mean) | {rmse:.3f} g/kg |')
    md.append(f'| MAE (on MC mean) | {mae:.3f} g/kg |')
    md.append(f'| Mean MC stddev per sample | {summary["mc_std_mean"]:.3f} g/kg |')
    md.append(f'| Mean 90% PI width | {summary["pi90_width_mean"]:.3f} g/kg |')
    md.append(f'| Median 90% PI width | {summary["pi90_width_median"]:.3f} g/kg |')
    md.append(f'| Observed 90% PI coverage | **{coverage_90 * 100:.1f}%** '
              f'(target 90%) |')
    md.append('')
    md.append('## Interpretation')
    md.append('')
    if coverage_90 >= 0.85:
        md.append('Coverage of the 90% prediction interval is within 5 pp of the '
                  'nominal level — the MC dropout uncertainty is reasonably well '
                  'calibrated for this dataset.')
    else:
        md.append(f'Observed coverage ({coverage_90*100:.1f}%) is below the nominal '
                  '90%, indicating the MC dropout estimate is *under-dispersed*. '
                  'This is a known limitation of MC dropout at moderate dropout '
                  'rates (Gal & Ghahramani 2016). For operational uncertainty, '
                  'deep ensembles or conformal prediction would provide better '
                  'calibration; we report MC dropout here as a baseline UQ '
                  'estimate from the existing trained model without retraining.')
    md.append('')
    md.append('## Spatial uncertainty map')
    md.append('')
    md.append('![MC dropout uncertainty map](mc_dropout_map.png)')
    md.append('')
    (HERE / 'mc_dropout_uncertainty.md').write_text('\n'.join(md))
    print(f'Saved {HERE / "mc_dropout_uncertainty.md"}', flush=True)

    # --- Spatial map ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sc1 = axes[0].scatter(lons_all, lats_all, c=pred_mean, s=8,
                                cmap='viridis', alpha=0.85)
        axes[0].set_title(f'MC mean prediction (g/kg)\nR² = {r2:.3f}', fontsize=11)
        axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')
        axes[0].set_aspect('equal', adjustable='box')
        plt.colorbar(sc1, ax=axes[0], shrink=0.7)
        sc2 = axes[1].scatter(lons_all, lats_all, c=pred_std, s=8,
                                cmap='magma', alpha=0.85)
        axes[1].set_title(f'MC std (uncertainty, g/kg)\n'
                           f'mean = {pred_std.mean():.2f}, '
                           f'median = {np.median(pred_std):.2f}', fontsize=11)
        axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
        axes[1].set_aspect('equal', adjustable='box')
        plt.colorbar(sc2, ax=axes[1], shrink=0.7)
        fig.suptitle(f'MC dropout uncertainty (n_mc = {args.n_mc}, '
                      f'90% PI coverage = {coverage_90*100:.1f}%)',
                      fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(HERE / 'mc_dropout_map.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {HERE / "mc_dropout_map.png"}', flush=True)
    except Exception as e:
        print(f'[warn] map generation failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
