#!/usr/bin/env python3
"""map_baseline_extband.py — 43-band RF / XGBoost Bavaria SOC production maps,
CONSISTENT with the spatial-CV trees.

Trains on ALL sample points using run_baselines' per-band {mean,std,min,max}
features (43-band 'full_extended' stack, OC<=150, log target) — the exact same
feature construction that produced the tree R² in the result tables — then runs
inference on the 1-mil/400k Bavaria grid via the SGT-dir mapping loader (the same
one the NN extband maps used), aggregating each grid window the same way.

Writes <out>/bavaria_2023_predictions.parquet with GPS_LONG/GPS_LAT/predicted_soc
so figmaps.py renders it in the unified (thin-border, no-title, terrain) style.

RUN ON JUPITER (needs the 43-band raster tensors + 1-mil grid + a GPU for XGB).
    python rebuttal/gpu_experiments/spatial_kfold/map_baseline_extband.py \
        --model xgb --max-oc 150 \
        --out rebuttal/.../final_models/maps/xgb_extband/

NOTE: the grid-loader call (separate_and_add_data_1mil_inference / dataset ctor)
must match how your NN map driver invoked it. If the signature differs, paste the
NN driver's grid-loading lines and I'll align this exactly. A fallback using the
BaselinesXGBoostAndRF pattern is in the comments below.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# run_baselines puts the SGT dir on sys.path and gives us the SAME feature path.
from run_baselines import extract_features_for_df, _aggregate_cube     # noqa: E402
from run_kfold import MODEL_READY, _build_model_ready_dataset           # noqa: E402
from config import bands_list_order                                    # noqa: E402  (SGT 43-band config)


def build_model(kind, a):
    if kind == 'xgb':
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=a.xgb_n_estimators, max_depth=a.xgb_max_depth,
            learning_rate=a.xgb_lr, subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', device=a.device, n_jobs=-1)
    try:
        from cuml.ensemble import RandomForestRegressor as RF
        print('[map] cuML RandomForest (GPU)', flush=True)
        md = 16 if a.rf_max_depth in (0, None) else a.rf_max_depth
        return RF(n_estimators=a.rf_n_estimators, max_depth=md)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor as RF
        print('[map] sklearn RandomForest (CPU)', flush=True)
        md = None if a.rf_max_depth in (0, None) else a.rf_max_depth
        return RF(n_estimators=a.rf_n_estimators, max_depth=md, n_jobs=-1)


def load_grid():
    """Return a torch DataLoader over the 43-band 1-mil grid yielding
    (lon, lat, features). Mirrors rebuttal/final_models/infer_bavaria.py exactly:
    MultiRasterDataset1MilMultiYears + the 2-tuple separate_and_add_data_1mil_
    inference() + the Coordinates1Mil CSV for the grid coordinates."""
    from dataloader.dataloaderMapping import MultiRasterDataset1MilMultiYears
    from dataloader.dataframe_loader import separate_and_add_data_1mil_inference
    from config import time_before
    from _paths import SOC_DATA_DIR
    grid_csv = (Path(ARGS.grid_csv) if ARGS.grid_csv else
                SOC_DATA_DIR / 'Coordinates1Mil' / 'coordinates_Bavaria_1mil.csv')
    df = pd.read_csv(grid_csv)
    cols = {c.lower(): c for c in df.columns}
    lon_col = cols.get('gps_long') or cols.get('lon') or cols.get('longitude') or df.columns[0]
    lat_col = cols.get('gps_lat') or cols.get('lat') or cols.get('latitude') or df.columns[1]
    df = df.rename(columns={lon_col: 'longitude', lat_col: 'latitude'})[['longitude', 'latitude']].copy()
    df['GPS_LONG'] = df['longitude']; df['GPS_LAT'] = df['latitude']
    print(f'[map] grid: {len(df):,} locations from {grid_csv}', flush=True)
    sample_paths, data_paths = separate_and_add_data_1mil_inference()
    ds = MultiRasterDataset1MilMultiYears(sample_paths, data_paths, df, time_before)
    return DataLoader(ds, batch_size=ARGS.batch_size, shuffle=False,
                      num_workers=ARGS.num_workers)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['rf', 'xgb'], required=True)
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--target-transform', default='log', choices=['log', 'none'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--num-workers', type=int, default=0,
                    help='0 = single process (avoids each worker copying the '
                         'raster-tile cache → host-RAM OOM on the 1.3M grid). '
                         'Bump only if the node has plenty of RAM.')
    ap.add_argument('--out', required=True, help='output run dir (parquet written inside)')
    ap.add_argument('--grid-csv', default=None,
                    help='1-mil grid coords CSV (default: SOC_DATA_DIR/Coordinates1Mil/coordinates_Bavaria_1mil.csv)')
    ap.add_argument('--xgb-n-estimators', type=int, default=2000)
    ap.add_argument('--xgb-max-depth', type=int, default=6)
    ap.add_argument('--xgb-lr', type=float, default=0.05)
    ap.add_argument('--rf-n-estimators', type=int, default=500)
    ap.add_argument('--rf-max-depth', type=int, default=0)
    ARGS = ap.parse_args()

    # ---- 1. train on ALL sample points (43-band per-band stats) ----
    _build_model_ready_dataset()
    df = pd.read_parquet(MODEL_READY).reset_index(drop=True)
    oc_col = 'OC' if 'OC' in df.columns else ('OC_actual' if 'OC_actual' in df.columns else None)
    if oc_col:
        df = df[df[oc_col] <= ARGS.max_oc].reset_index(drop=True)
    cache = HERE / 'sweep' / f'_mapfeat_train_oc{int(ARGS.max_oc)}_extband.npz'
    X, y, lon, lat = extract_features_for_df(df, cache_path=cache)
    yt = np.log1p(y) if ARGS.target_transform == 'log' else y
    print(f'[map] train: X={X.shape}  bands={len(bands_list_order)}  '
          f'y[{y.min():.1f},{y.max():.1f}]', flush=True)
    model = build_model(ARGS.model, ARGS)
    t0 = time.time(); model.fit(X, yt)
    print(f'[map] fit done in {time.time()-t0:.0f}s', flush=True)

    # ---- 2. grid inference (43-band 1-mil) ----
    dl = load_grid()
    plon, plat, pred = [], [], []
    t0 = time.time()
    for bi, batch in enumerate(dl):
        lons, lats, feats = batch
        f = feats.detach().cpu()
        Xg = np.stack([_aggregate_cube(f[j]) for j in range(f.shape[0])])
        pred.append(model.predict(Xg))
        plon.append(np.asarray(lons)); plat.append(np.asarray(lats))
        if bi % 100 == 0:
            print(f'  grid batch {bi}  n~{(bi+1)*ARGS.batch_size}  '
                  f'elapsed {time.time()-t0:.0f}s', flush=True)
    pred = np.concatenate(pred); plon = np.concatenate(plon); plat = np.concatenate(plat)
    if ARGS.target_transform == 'log':
        pred = np.expm1(pred)

    out_dir = Path(ARGS.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'bavaria_2023_predictions.parquet'
    pd.DataFrame({'GPS_LONG': plon, 'GPS_LAT': plat,
                  'predicted_soc': pred, 'year': 2023}).to_parquet(out)
    print(f'[map] wrote {out}  n={len(pred)}  '
          f'mean={np.nanmean(pred):.2f} sd={np.nanstd(pred):.2f} '
          f'min={np.nanmin(pred):.2f} max={np.nanmax(pred):.2f}', flush=True)


if __name__ == '__main__':
    main()
