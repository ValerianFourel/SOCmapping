"""train_resaware_windowed.py — spatial-CV training of ResolutionAwareNet on the
NATIVE-resolution windowed sentinel2 dataset (windows/lucas/), one variant per run.

Uses the SAME lon-decile spatial folds as the rest of the rebuttal
(`run_kfold.build_folds_spatial_deciles`, axis=lon, seed=42, buffer) so the
native-res resaware ablation (fine | fine_med | fine_med_coarse | all_flat, plus
--ablate-group) is comparable across variants on identical folds. Reads the 45-band
cube straight from the pre-extracted windows — no 250 m RasterTensorData.

Writes kfold_results_summary.json (across_folds r2_mean/std + per-fold) in the same
shape sweep_summarize / resaware_table read.

Example (cluster):
  python train_resaware_windowed.py --data-root $WINDOWS --branches fine_med_coarse \
      --max-oc 150 --num-folds 10 --window-size 11 --time-before 5 --epochs 100 \
      --out runs/resaware_native_fmc
"""
from __future__ import annotations
import argparse, json, os, sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'SpatiotemporalGatedTransformer'))
from WindowedSentinel2Dataset import WindowedSentinel2Dataset          # noqa: E402
from ResolutionAwareNet import ResolutionAwareNet                      # noqa: E402

# --- spatial folds: copied VERBATIM from run_kfold (lat/lon-decile path) so the
#     protocol is byte-identical without pulling run_kfold's heavy import chain.
EARTH_RADIUS_KM = 6371.0


def _haversine_km_matrix(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1)); lon1 = np.deg2rad(np.asarray(lon1))
    lat2 = np.deg2rad(np.asarray(lat2)); lon2 = np.deg2rad(np.asarray(lon2))
    dlat = lat2[np.newaxis, :] - lat1[:, np.newaxis]
    dlon = lon2[np.newaxis, :] - lon1[:, np.newaxis]
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1)[:, np.newaxis] * np.cos(lat2)[np.newaxis, :]
         * np.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _min_distance_to_set_km(t_lat, t_lon, r_lat, r_lon, chunk=2048):
    n = len(t_lat); out = np.full(n, np.inf)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = _haversine_km_matrix(t_lat[s:e], t_lon[s:e], r_lat, r_lon).min(axis=1)
    return out


def build_folds_spatial_deciles(df, n_folds=10, buffer_km=1.2, axis='lon', seed=42):
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError('build_folds requires df.index == RangeIndex')
    lat_all = df['GPS_LAT'].to_numpy(float); lon_all = df['GPS_LONG'].to_numpy(float)
    coord = lat_all if axis == 'lat' else lon_all
    edges = np.quantile(coord, np.linspace(0, 1, n_folds + 1))
    edges[-1] = coord.max() + 1e-9; edges[0] = coord.min() - 1e-9
    folds = []
    for i in range(n_folds):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_test = (coord >= lo) & (coord < hi)
        test_idx = df.index[in_test].to_numpy(); train_pool = df.index[~in_test].to_numpy()
        d = _min_distance_to_set_km(lat_all[~in_test], lon_all[~in_test],
                                    lat_all[in_test], lon_all[in_test])
        keep = d >= buffer_km
        folds.append({'fold_id': i, 'split_axis': axis, 'edge_lo': lo, 'edge_hi': hi,
                      'test_idx': test_idx, 'train_idx': train_pool[keep],
                      'buffer_idx': train_pool[~keep]})
    return folds


def r2_score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss_res = float(((y - p) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def set_determinism(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _fit_inv(y_tr, transform):
    """Return (fwd, inv) for the target transform, fit on the TRAIN OC values.
    fwd: OC->model space; inv: model space->SOC. R^2 is always computed in SOC."""
    if transform == 'log':
        return (lambda y: torch.log(y.clamp_min(1e-3)),
                lambda p: torch.exp(p.clamp(-3.0, 9.0)))
    if transform == 'normalize':
        m, s = float(y_tr.mean()), float(y_tr.std() + 1e-10)
        return (lambda y: (y - m) / s, lambda p: p * s + m)
    return (lambda y: y, lambda p: p)                       # 'none'


def train_eval_fold(X, y_oc, tr, te, args, dev, dims):
    """Train ResolutionAwareNet on the PRECOMPUTED cube tensor X (no per-epoch
    assembly). Target transformed per --target-transform (default log); R^2 kept
    in ORIGINAL SOC space at the BEST epoch (canonical recipe)."""
    fine_idx, med_idx, coarse_idx, C = dims
    net = ResolutionAwareNet(C, height=args.window_size, width=args.window_size,
                             time_steps=args.time_before, branches=args.branches,
                             ablate_group=args.ablate_group, fine_idx=fine_idx,
                             med_idx=med_idx, coarse_idx=coarse_idx).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = torch.nn.L1Loss() if args.loss == 'l1' else torch.nn.MSELoss()
    tr = torch.as_tensor(tr, dtype=torch.long); te = torch.as_tensor(te, dtype=torch.long)
    fwd, inv = _fit_inv(y_oc[tr], args.target_transform)
    g = y_oc[te].to(dev)                                    # SOC ground truth (test)
    best = {'r2': -float('inf'), 'rmse': float('nan'), 'best_epoch': -1, 'n_test': int(len(te))}
    bs = args.batch_size
    for ep in range(args.epochs):
        net.train(); perm = tr[torch.randperm(len(tr))]
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            xb = X[b].to(dev).float(); yb = fwd(y_oc[b].to(dev))
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        net.eval(); P = []
        with torch.no_grad():
            for s in range(0, len(te), bs):
                P.append(inv(net(X[te[s:s + bs]].to(dev).float())))
        p = torch.cat(P)
        r = float(1 - ((g - p) ** 2).sum() / (((g - g.mean()) ** 2).sum() + 1e-12))
        if r > best['r2']:
            best.update(r2=r, rmse=float(((g - p) ** 2).mean().sqrt()), best_epoch=ep)
    return best, net.count_parameters()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True, help='windows/lucas dir (local or HF-snapshot)')
    ap.add_argument('--branches', default='fine_med_coarse',
                    choices=['fine', 'fine_med', 'fine_med_coarse', 'all_flat'])
    ap.add_argument('--ablate-group', default=None, choices=['fine', 'medium', 'coarse'])
    ap.add_argument('--num-folds', type=int, default=10)
    ap.add_argument('--split-axis', default='lon')
    ap.add_argument('--buffer-km', type=float, default=1.2)
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--window-size', type=int, default=11)
    ap.add_argument('--time-before', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--loss', default='l1', choices=['l1', 'mse'])
    ap.add_argument('--target-transform', default='log', choices=['log', 'normalize', 'none'],
                    help='Target transform; R^2 always reported in original SOC space.')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='cap to first N points (sanity runs).')
    ap.add_argument('--out', required=True, help='output dir for kfold_results_summary.json')
    args = ap.parse_args()
    set_determinism(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    import time
    # preload=False: with the precompute pass we touch each band once via mmap
    # (OS page cache), so we never hold the 4 GB band cache + the 3.3 GB cube
    # simultaneously — peak RAM is just the cube.
    ds = WindowedSentinel2Dataset(args.data_root, time_before=args.time_before,
                                  window=args.window_size, preload=False)
    if args.limit:
        ds.labels = ds.labels.iloc[:args.limit].reset_index(drop=True); ds.N = len(ds.labels)
    dims = (ds.fine_idx, ds.med_idx, ds.coarse_idx, len(ds.bands))
    lab = ds.labels.copy()
    keep = (lab['oc'] > 0) & (lab['oc'] <= args.max_oc) & np.isfinite(lab['lat']) & np.isfinite(lab['lon'])
    cap = lab[keep].reset_index().rename(columns={'index': 'orig_idx', 'lat': 'GPS_LAT', 'lon': 'GPS_LONG'})
    cap.index = pd.RangeIndex(len(cap))
    print(f"dataset N={len(ds)}  after oc<= {args.max_oc}: {len(cap)}  bands={len(ds.bands)} "
          f"(fine={len(ds.fine_idx)} med={len(ds.med_idx)} coarse={len(ds.coarse_idx)})  dev={dev}", flush=True)

    # Precompute EVERY sample's cube ONCE (~3.3 GB) so the per-epoch Python
    # assembly disappears — training becomes pure GPU compute on a tiny net.
    t0 = time.time(); C, W, T = len(ds.bands), args.window_size, args.time_before
    X = torch.empty((len(ds), C, W, W, T), dtype=torch.float32)
    for i in range(len(ds)):
        X[i] = ds[i][0]
    y_oc = torch.tensor(ds.labels['oc'].to_numpy(float), dtype=torch.float32)
    ds._cache.clear()                                       # free the ~4 GB band arrays
    print(f"precomputed cubes X={tuple(X.shape)} ({X.element_size()*X.nelement()/1e9:.1f} GB) in {time.time()-t0:.0f}s", flush=True)

    folds = build_folds_spatial_deciles(cap, n_folds=args.num_folds, buffer_km=args.buffer_km,
                                        axis=args.split_axis, seed=args.seed)
    results = []; nparam = 0
    for f in folds:
        tr = cap.loc[f['train_idx'], 'orig_idx'].to_numpy()
        te = cap.loc[f['test_idx'], 'orig_idx'].to_numpy()
        res, nparam = train_eval_fold(X, y_oc, tr, te, args, dev, dims)
        res.update({'fold_id': f['fold_id'], 'lon_lo': f.get('edge_lo'), 'lon_hi': f.get('edge_hi')})
        results.append(res)
        print(f"  fold {f['fold_id']}: r2={res['r2']:+.4f} rmse={res['rmse']:.3f} n_test={res['n_test']}", flush=True)
    r2s = np.array([r['r2'] for r in results], float)
    summ = {'model_family': 'resaware', 'branches': args.branches, 'ablate_group': args.ablate_group,
            'split_axis': args.split_axis, 'n_folds': args.num_folds, 'fold_geometry': 'lon-deciles',
            'distance_threshold_km': args.buffer_km,
            'recipe': {'model_family': 'resaware', 'max_oc': args.max_oc, 'loss_type': args.loss,
                       'target_transform': args.target_transform, 'window_size': args.window_size,
                       'time_before': args.time_before, 'n_bands': len(ds.bands),
                       'n_params': nparam, 'native_windowed': True},
            'across_folds': {'r2_mean': float(r2s.mean()), 'r2_std': float(r2s.std(ddof=1)),
                             'rmse_mean': float(np.mean([r['rmse'] for r in results]))},
            'fold_results': results}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'kfold_results_summary.json').write_text(json.dumps(summ, indent=2))
    print(f"\nRESULT branches={args.branches} ablate={args.ablate_group} "
          f"R2={r2s.mean():+.4f} +/- {r2s.std(ddof=1):.4f}  (score={r2s.mean()-0.5*r2s.std(ddof=1):+.4f}) "
          f"-> {out}/kfold_results_summary.json", flush=True)


if __name__ == '__main__':
    main()
