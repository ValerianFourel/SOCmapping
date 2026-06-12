#!/usr/bin/env python3
"""
rebuttal/final_models/infer_bavaria.py — generate a Bavaria-wide SOC
prediction map from one trained final model.

Auto-dispatches between two paths based on the checkpoint contents:
  * Neural network (.pth with EnhancedSGT / SimpleSGT / SimpleTransformerV2
    / Small3DCNN / RefittedCovLSTM weights) → loads via _build_model + the
    1mil-grid MultiRasterDataset1MilMultiYears feature extractor.
  * Tree ensemble (.joblib RandomForest / .json XGBRegressor) → loads via
    the corresponding library + the per-band-statistic feature extractor
    from run_baselines.py, on the same 1mil-grid mapping dataset.

For both, the prediction grid is the 1mil-point Bavaria reference grid;
each grid point's covariate window uses target_year = --year (default 2023)
with the natural 5-year history window {y-4, …, y}.

Outputs (under rebuttal/final_models/maps/<run-name>/):
    bavaria_<year>_predictions.parquet     lon/lat/predicted_soc per grid pt
    bavaria_<year>_map.png                 spatial map (300 dpi)
    bavaria_<year>_summary.json            per-year stats

Run:
    python rebuttal/final_models/infer_bavaria.py \\
        --run-name sgt_d128_h4_L1 --year 2023

    python rebuttal/final_models/infer_bavaria.py \\
        --run-name rf_default --year 2023
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

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]
sys.path.insert(0, str(SOC_ROOT))
from _paths import SOC_DATA_DIR, SOC_REBUTTAL_DIR  # noqa: E402

SGT_DIR = SOC_ROOT / 'SpatiotemporalGatedTransformer'
sys.path.insert(0, str(SGT_DIR))
sys.path.insert(0, str(SGT_DIR / 'dataloader'))

KFOLD_DIR = SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'
sys.path.insert(0, str(KFOLD_DIR))
from run_kfold import _build_model, make_dataset  # noqa: E402
from run_baselines import _aggregate_cube, inverse_y  # noqa: E402
from band_subsets import get_band_indices  # noqa: E402
# Production-mapping dataloader (built for the 1mil Bavaria grid via
# MatrixCoordinates_1mil_* paths), NOT the training dataloader. The
# training loader requires every (lon, lat) to be pre-indexed in the
# LUCAS coordinates.npy files; the mapping loader has its own
# 1mil-grid-specific coordinates.npy per band.
from dataloaderMapping import MultiRasterDataset1MilMultiYears  # noqa: E402
from dataframe_loader import separate_and_add_data_1mil_inference  # noqa: E402
from config import time_before, bands_list_order  # noqa: E402

CHECKPOINTS_ROOT = HERE / 'checkpoints'
MAPS_ROOT = HERE / 'maps'
DEFAULT_GRID = SOC_DATA_DIR / 'Coordinates1Mil' / 'coordinates_Bavaria_1mil.csv'


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', type=str, required=True)
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--grid-csv', type=Path, default=DEFAULT_GRID)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--limit', type=int, default=0,
                   help='Predict only the first N grid points (dry-run sanity).')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_grid(grid_csv: Path, limit: int) -> pd.DataFrame:
    """Load the 1mil Bavaria grid.

    The mapping dataloader reads lower-case 'longitude'/'latitude'. We
    also keep GPS_LONG/GPS_LAT aliases so downstream code (parquet
    writer, scatter plotting) doesn't need to know which loader was used.
    """
    if not grid_csv.exists():
        raise SystemExit(f'\n[ERROR] grid CSV not found: {grid_csv}\n'
                         f'        Pass --grid-csv /path/to/coordinates_Bavaria_1mil.csv\n')
    df = pd.read_csv(grid_csv)
    cols = {c.lower(): c for c in df.columns}
    lon_col = cols.get('gps_long') or cols.get('lon') or cols.get('longitude') or df.columns[0]
    lat_col = cols.get('gps_lat') or cols.get('lat') or cols.get('latitude') or df.columns[1]
    df = df.rename(columns={lon_col: 'longitude', lat_col: 'latitude'})
    df = df[['longitude', 'latitude']].copy()
    # Aliases for downstream consumers that expect GPS_LONG/GPS_LAT.
    df['GPS_LONG'] = df['longitude']
    df['GPS_LAT'] = df['latitude']
    if limit > 0:
        df = df.head(limit).reset_index(drop=True)
    return df


def _is_tree_run(run_dir: Path) -> bool:
    return ((run_dir / 'final_model.joblib').exists()
            or (run_dir / 'final_model.json').exists())


def predict_nn(args, run_dir: Path, grid_df: pd.DataFrame, device) -> np.ndarray:
    """Neural-network inference path."""
    pth = run_dir / 'final_model.pth'
    if not pth.exists():
        raise SystemExit(f'\n[ERROR] checkpoint missing: {pth}\n')
    stats_p = run_dir / 'stats.json'
    if not stats_p.exists():
        raise SystemExit(f'\n[ERROR] stats.json missing: {stats_p}\n')
    stats = json.loads(stats_p.read_text())
    ckpt = torch.load(pth, map_location='cpu', weights_only=False)

    feature_means = torch.as_tensor(stats['feature_means'], dtype=torch.float32)
    feature_stds = torch.as_tensor(stats['feature_stds'], dtype=torch.float32)
    # Mirror run_kfold._NormalizingWrapper: clamp stds and sanitize means
    # so a constant-valued covariate (std == 0) doesn't divide-by-zero into
    # ±inf and then NaN through every NN layer. Training already clamps to
    # 1e-8; inference must do the same or every prediction comes out NaN.
    feature_means = torch.nan_to_num(feature_means, nan=0.0, posinf=0.0, neginf=0.0)
    feature_stds = torch.clamp(
        torch.nan_to_num(feature_stds, nan=1.0, posinf=1.0, neginf=1.0),
        min=1e-8,
    )
    n_zero_std = int((feature_stds <= 1e-8).sum().item())
    if n_zero_std:
        print(f'[infer-nn] WARNING: {n_zero_std} band(s) had std≈0 in stats.json; '
              f'clamped to 1e-8 (matches training-time normalizer). '
              f'These channels contribute ~0 signal.', flush=True)
    target_mean = float(stats['target_mean'])
    target_std = float(stats['target_std'])

    # Reconstruct the model with the same args used at training time.
    fake = argparse.Namespace(**ckpt['args'])
    model = _build_model(fake).to(device)
    sd = {k.replace('module.', '', 1): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[infer-nn] loaded {type(model).__name__}  family={ckpt.get("family")}  '
          f'({n_params:,} params)', flush=True)

    # Build a dataloader-friendly DataFrame: every grid point gets
    # year=args.year (the mapping dataset uses (lon, lat, year, season)
    # to locate covariate tiles; no OC required, this is inference).
    grid = grid_df.copy()
    grid['year'] = args.year
    grid['season'] = f'{args.year}_summer'

    # 1mil-grid paths from config (MatrixCoordinates_1mil_*) — these point
    # to per-band coordinates.npy files that index the 1mil grid, not
    # the LUCAS training coords. Using separate_and_add_data() here would
    # silently fail every coordinate lookup.
    sample_paths, data_paths = separate_and_add_data_1mil_inference()
    def flatten(lst):
        out = []
        for x in lst:
            out += flatten(x) if isinstance(x, list) else [x]
        return out
    sample_paths = list(dict.fromkeys(flatten(sample_paths)))
    data_paths = list(dict.fromkeys(flatten(data_paths)))

    ds = MultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=sample_paths,
        data_array_subfolders=data_paths,
        dataframe=grid,
        time_before=time_before,
    )

    # Mirror the training-time bands-list subsetting for inference.
    band_indices_t = torch.as_tensor(
        get_band_indices(
            ckpt['args'].get('bands_list', 'full_20'),
            list(bands_list_order),
        ),
        dtype=torch.long,
    )
    do_slice = band_indices_t.numel() < len(bands_list_order)

    n = len(ds)
    preds = np.zeros(n, dtype=np.float32)
    apply_target_transform = ckpt['args'].get('target_transform', 'normalize')
    n_missing = 0
    with torch.no_grad():
        batch_feats, batch_idx = [], []
        t0 = time.time()
        first_errors: list[str] = []   # collect a few examples to surface
        for i in range(n):
            try:
                # MultiRasterDataset1MilMultiYears returns (lon, lat, features) —
                # 3 values, no OC. Same tensor shape as the training loader
                # (both end with permute(0, 2, 3, 1)).
                _, _, f = ds[i]
                f_norm = (f - feature_means[:, None, None]) / feature_stds[:, None, None]
                # Defensive: replace any leftover NaN/inf with 0 (matches the
                # behaviour of a well-normalized constant band). A single bad
                # pixel propagates through every conv/linear layer otherwise.
                if not torch.isfinite(f_norm).all():
                    f_norm = torch.nan_to_num(f_norm, nan=0.0, posinf=0.0, neginf=0.0)
                if do_slice:
                    f_norm = f_norm.index_select(0, band_indices_t)
                batch_feats.append(f_norm)
                batch_idx.append(i)
            except Exception as e:
                preds[i] = np.nan
                n_missing += 1
                if len(first_errors) < 3:
                    lon = float(grid_df.longitude.iloc[i]); lat = float(grid_df.latitude.iloc[i])
                    first_errors.append(
                        f'i={i}  lon={lon:.4f} lat={lat:.4f}  '
                        f'{type(e).__name__}: {e}')
            if len(batch_feats) >= args.batch_size or (i == n - 1 and batch_feats):
                x = torch.stack(batch_feats).float().to(device)
                out = model(x).cpu().numpy()
                # Inverse-transform back to g/kg
                if apply_target_transform == 'log':
                    y_gkg = np.exp(np.clip(out, -3.0, 6.0))
                elif apply_target_transform == 'normalize':
                    y_gkg = out * target_std + target_mean
                else:
                    y_gkg = out
                # Upper clip at 2× training cap so a degenerate model
                # (e.g. exp(log_pred) running away in early epochs) can't
                # produce predictions far outside the training-target range
                # that would dominate downstream visualization colour scales.
                _max_oc_train = float(ckpt['args'].get('max_oc', 150.0))
                y_gkg = np.clip(y_gkg, 0.0, 2.0 * _max_oc_train)
                for j, gi in enumerate(batch_idx):
                    preds[gi] = float(y_gkg[j])
                batch_feats.clear(); batch_idx.clear()
            if (i + 1) % 50_000 == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-6)
                print(f'  [{i+1:>7}/{n}] rate {rate:.0f} pts/s', flush=True)
    if n_missing:
        print(f'  [warn] {n_missing} grid points missing covariate tiles', flush=True)
        if first_errors:
            print(f'  [warn] first few exceptions raised by the dataset '
                  f'(use these to diagnose path / coordinate / year mismatch):',
                  flush=True)
            for e in first_errors:
                print(f'    {e}', flush=True)
        if n_missing == n:
            raise SystemExit(
                '\n[ERROR] EVERY grid point failed the dataset lookup '
                '— there is no actual prediction to save. Diagnose the '
                'dataset error above before re-running infer_bavaria.\n')
    mean_pred = float(np.nanmean(preds))
    max_oc_train = float(ckpt['args'].get('max_oc', 150.0))
    if mean_pred > max_oc_train * 1.2:
        print(f'[infer-nn] WARNING: mean prediction = {mean_pred:.1f} g/kg '
              f'exceeds max_oc_train × 1.2 = {max_oc_train * 1.2:.1f}. '
              f'Model likely overshooting in log-target space.', flush=True)
    return preds


def predict_tree(args, run_dir: Path, grid_df: pd.DataFrame) -> np.ndarray:
    """Tree-ensemble (RF / XGB) inference path."""
    config_p = run_dir / 'config.json'
    stats_p = run_dir / 'stats.json'
    if not config_p.exists() or not stats_p.exists():
        raise SystemExit(f'\n[ERROR] config.json or stats.json missing in {run_dir}\n')
    cfg = json.loads(config_p.read_text())
    stats = json.loads(stats_p.read_text())
    target_transform = cfg['target_transform']
    mu = float(stats.get('target_mean_for_normalize', 0.0))
    sd = float(stats.get('target_std_for_normalize', 1.0))

    joblib_p = run_dir / 'final_model.joblib'
    xgb_p = run_dir / 'final_model.json'
    if joblib_p.exists():
        import joblib
        model = joblib.load(joblib_p)
        family = 'rf'
    elif xgb_p.exists():
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(xgb_p))
        family = 'xgb'
    else:
        raise SystemExit(f'\n[ERROR] no tree model file in {run_dir}\n')
    print(f'[infer-tree] loaded {family} from {run_dir}', flush=True)

    # Build per-band features at each grid point — same 80-d recipe as
    # run_baselines.extract_features_for_df, but driven by the 1mil-grid
    # mapping dataset (which has the right coordinates.npy per band).
    grid = grid_df.copy()
    grid['year'] = args.year
    grid['season'] = f'{args.year}_summer'

    sample_paths, data_paths = separate_and_add_data_1mil_inference()
    def _flatten(lst):
        out = []
        for x in lst:
            out += _flatten(x) if isinstance(x, list) else [x]
        return out
    sample_paths = list(dict.fromkeys(_flatten(sample_paths)))
    data_paths = list(dict.fromkeys(_flatten(data_paths)))
    ds = MultiRasterDataset1MilMultiYears(
        samples_coordinates_array_subfolders=sample_paths,
        data_array_subfolders=data_paths,
        dataframe=grid,
        time_before=time_before,
    )
    n = len(ds)
    # Per-band {mean, std, min, max}. The 1mil mapping dataset serves the
    # full bands_list_order stack (43 bands in the extended setup), so
    # _aggregate_cube returns 4*len(bands_list_order) features; we slice down
    # to the tree's band subset below. Hardcoding 80 assumed a 20-band cube
    # and made every point fail the X[i] broadcast on the 43-band grid.
    n_feat = 4 * len(bands_list_order)
    X = np.empty((n, n_feat), dtype=np.float32)
    n_missing = 0
    first_errors: list[str] = []   # collect first few exceptions for diagnosis
    t0 = time.time()
    for i in range(n):
        try:
            _, _, f = ds[i]    # mapping loader returns (lon, lat, features) — 3-tuple
            X[i] = _aggregate_cube(f)
        except Exception as e:
            X[i] = 0.0
            n_missing += 1
            if len(first_errors) < 3:
                lon = float(grid_df.longitude.iloc[i]); lat = float(grid_df.latitude.iloc[i])
                first_errors.append(
                    f'i={i}  lon={lon:.4f} lat={lat:.4f}  '
                    f'{type(e).__name__}: {e}')
        if (i + 1) % 50_000 == 0:
            rate = (i + 1) / (time.time() - t0 + 1e-6)
            print(f'  [{i+1:>7}/{n}] rate {rate:.0f} pts/s', flush=True)
    if n_missing:
        print(f'  [warn] {n_missing} grid points missing covariate tiles', flush=True)
        if first_errors:
            print(f'  [warn] first few exceptions raised by the dataset:', flush=True)
            for e in first_errors:
                print(f'    {e}', flush=True)
        if n_missing == n:
            # Trees would silently "predict" the model's response on X=zeros,
            # which is a meaningless constant masquerading as a real map.
            # Refuse to write that.
            raise SystemExit(
                '\n[ERROR] EVERY grid point failed the dataset lookup — '
                'predictions on X=zeros would be a meaningless constant. '
                'Diagnose the dataset error above before re-running.\n')

    # If the trained tree was on a band subset, slice the full per-band
    # feature vector (4*len(bands_list_order)) down to its matching columns
    # before prediction — e.g. full_20 → first 80 cols, original_6 → 24 cols.
    bands_used = cfg.get('bands_list', 'full_20')
    band_indices = get_band_indices(bands_used, list(bands_list_order))
    if len(band_indices) < len(bands_list_order):
        col_indices = []
        for b in band_indices:
            col_indices.extend([b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3])
        X = X[:, col_indices]
        print(f'[infer-tree] sliced X to {X.shape} for bands_list={bands_used}',
              flush=True)

    pred_raw = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
    pred = inverse_y(pred_raw, target_transform, mean=mu, std=sd)
    # Upper clip at 2× training cap. Tree models trained on small
    # feature subsets (e.g. 6-band XGB shallow) can collapse to
    # predicting near the training max in log-target space, which after
    # exp() becomes a mean ≈ exp(log(max_oc)) — well above the
    # physically-meaningful range. Clip for visualization sanity; the
    # broken-model warning is surfaced separately by compare_maps.
    max_oc_train = float(cfg.get('max_oc', 150.0))
    pred = np.clip(pred, 0.0, 2.0 * max_oc_train)
    mean_pred = float(np.nanmean(pred))
    if mean_pred > max_oc_train * 1.2:
        print(f'[infer-tree] WARNING: mean prediction = {mean_pred:.1f} g/kg '
              f'exceeds max_oc_train × 1.2 = {max_oc_train * 1.2:.1f}. '
              f'Model likely collapsed in log-target space on this feature set.',
              flush=True)
    return pred.astype(np.float32)


def main():
    args = parse()
    run_dir = CHECKPOINTS_ROOT / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f'\n[ERROR] checkpoint directory not found: {run_dir}\n'
                         f'        Train first via:\n'
                         f'          python rebuttal/final_models/train_full.py --run-name {args.run_name} ...\n')

    grid_df = load_grid(args.grid_csv, args.limit)
    print(f'[infer] grid: {len(grid_df):,} points  '
          f'lon [{grid_df.GPS_LONG.min():.2f}, {grid_df.GPS_LONG.max():.2f}]  '
          f'lat [{grid_df.GPS_LAT.min():.2f}, {grid_df.GPS_LAT.max():.2f}]', flush=True)

    is_tree = _is_tree_run(run_dir)
    if is_tree:
        preds = predict_tree(args, run_dir, grid_df)
    else:
        device = (torch.device(args.device)
                  if (args.device == 'cpu' or torch.cuda.is_available())
                  else torch.device('cpu'))
        print(f'[infer] device={device}', flush=True)
        preds = predict_nn(args, run_dir, grid_df, device)

    # Save outputs
    out_dir = MAPS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f'bavaria_{args.year}_predictions.parquet'
    out_df = grid_df.assign(predicted_soc=preds, year=args.year)
    out_df.to_parquet(pq)
    print(f'[infer] saved {pq}', flush=True)

    v = preds[np.isfinite(preds)]
    summary = {
        'run_name': args.run_name,
        'family': 'tree' if is_tree else 'neural',
        'year': args.year,
        'n_total': int(preds.size),
        'n_valid': int(v.size),
        'mean': float(v.mean()) if v.size else float('nan'),
        'std': float(v.std()) if v.size else float('nan'),
        'p05': float(np.percentile(v, 5)) if v.size else float('nan'),
        'p50': float(np.percentile(v, 50)) if v.size else float('nan'),
        'p95': float(np.percentile(v, 95)) if v.size else float('nan'),
        'min': float(v.min()) if v.size else float('nan'),
        'max': float(v.max()) if v.size else float('nan'),
    }
    (out_dir / f'bavaria_{args.year}_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))
    print(f'[infer] mean={summary["mean"]:.2f}  std={summary["std"]:.2f}  '
          f'p05={summary["p05"]:.2f}  p95={summary["p95"]:.2f}', flush=True)

    # Map
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(grid_df.GPS_LONG, grid_df.GPS_LAT, c=preds, s=2,
                          cmap='YlOrBr', vmin=0, vmax=80, alpha=0.85)
        ax.set_title(f'{args.run_name}  —  predicted SOC, Bavaria {args.year}\n'
                      f'mean = {summary["mean"]:.2f} g/kg, '
                      f'p05–p95 = [{summary["p05"]:.1f}, {summary["p95"]:.1f}]',
                      fontsize=11)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(sc, ax=ax, label='SOC (g/kg)', shrink=0.7)
        fig.tight_layout()
        out_png = out_dir / f'bavaria_{args.year}_map.png'
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[infer] saved {out_png}', flush=True)
    except Exception as e:
        print(f'[warn] map generation failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
