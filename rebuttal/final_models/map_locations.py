#!/usr/bin/env python3
"""
rebuttal/final_models/map_locations.py — run one or more trained final
models over a (random) subset of Bavaria mapping locations and write the
predicted-SOC maps.

Why this exists alongside infer_bavaria.py
------------------------------------------
infer_bavaria.py maps ONE model over the WHOLE 1.3M-point reference grid,
in CSV order, and its --limit option takes the *head* of the grid (which is
spatially biased — the first N rows are one corner of Bavaria). For the
production comparison we want, instead:

  * a *random* sample of N locations (default 400,000) drawn once with a
    fixed seed, so every model is scored on the SAME representative spread
    of Bavaria rather than a corner; and
  * several top models mapped in ONE invocation, producing a combined
    wide parquet and a side-by-side comparison figure.

Feature extraction (the slow part) is the proven code in infer_bavaria:
we reuse its load_grid / _is_tree_run / predict_nn / predict_tree verbatim,
so NaN-handling, band-subset slicing, target inverse-transform and the
2x-max-oc visualization clip all behave identically to the single-model
path. The grid is the same coordinates_Bavaria_1mil.csv; --n-locations is
just a random row subsample of it.

Outputs (under rebuttal/final_models/maps/_locations_<tag>/):
    <run_name>_<year>_predictions.parquet   lon/lat/predicted_soc per model
    <run_name>_<year>_map.png               per-model scatter map (300 dpi)
    combined_<year>_predictions.parquet     lon/lat + one column per model
    combined_<year>_maps.png                side-by-side comparison panel
    combined_<year>_summary.json            per-model stats

Run (after the models are trained under checkpoints/<run_name>/):
    python rebuttal/final_models/map_locations.py \\
        --run-names sgt_d32_h2_L1_extband,vanilla_transformer_d64_h4_L1_extband,rf_deep_20band \\
        --year 2023 --n-locations 400000

    # map the full grid instead of a random subsample:
    python rebuttal/final_models/map_locations.py \\
        --run-names sgt_d32_h2_L1_extband --year 2023 --n-locations 0
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the battle-tested single-model inference internals. Importing the
# module also runs its sys.path setup for the SGT dirs / dataloaders.
import infer_bavaria as ib  # noqa: E402


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-names', type=str, required=True,
                   help='Comma-separated checkpoint run-names to map, e.g. '
                        '"sgt_d32_h2_L1_extband,rf_deep_20band". Each must '
                        'exist under rebuttal/final_models/checkpoints/. '
                        'NN and tree runs may be mixed; each is auto-detected.')
    p.add_argument('--year', type=int, default=2023,
                   help='Target year; each point uses the 5-year window '
                        '{year-4, ..., year} (default 2023).')
    p.add_argument('--n-locations', type=int, default=400_000,
                   help='Number of grid points to RANDOMLY sample from the '
                        'reference grid (default 400000). 0 = use every '
                        'point in the grid (full ~1.3M map).')
    p.add_argument('--sample-seed', type=int, default=42,
                   help='RNG seed for the random location subsample, so all '
                        'models share the SAME set of points and reruns are '
                        'reproducible (default 42).')
    p.add_argument('--grid-csv', type=Path, default=ib.DEFAULT_GRID,
                   help='Reference grid CSV (default: the 1mil Bavaria grid).')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out-tag', type=str, default=None,
                   help='Sub-directory tag under maps/_locations_<tag>/. '
                        'Default: "<N>rand_seed<seed>" or "full".')
    p.add_argument('--vmax', type=float, default=80.0,
                   help='Upper bound of the SOC colour scale in the maps '
                        '(g/kg, default 80 — matches the paper figures).')
    return p.parse_args()


def sample_grid(grid_csv: Path, n_locations: int, seed: int) -> pd.DataFrame:
    """Load the full reference grid, then draw a reproducible random subset.

    Reuses infer_bavaria.load_grid (limit=0 → whole grid) so the column
    normalization (longitude/latitude + GPS_LONG/GPS_LAT aliases) is
    identical to the single-model path.
    """
    grid = ib.load_grid(grid_csv, limit=0)
    n_total = len(grid)
    if n_locations and 0 < n_locations < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_locations, replace=False)
        idx.sort()  # keep grid order for cache-friendlier tile reads
        grid = grid.iloc[idx].reset_index(drop=True)
        print(f'[map] sampled {len(grid):,} / {n_total:,} grid points '
              f'(random, seed={seed})', flush=True)
    else:
        print(f'[map] using ALL {n_total:,} grid points', flush=True)
    return grid


def predict_one(args, run_name: str, grid_df: pd.DataFrame) -> np.ndarray:
    """Dispatch a single run-name to the NN or tree inference path."""
    run_dir = ib.CHECKPOINTS_ROOT / run_name
    if not run_dir.is_dir():
        raise SystemExit(
            f'\n[ERROR] checkpoint directory not found: {run_dir}\n'
            f'        Train it first (see fit_and_map_top_models.py / '
            f'train_full.py).\n')
    # infer_bavaria.predict_nn / predict_tree read only .year / .batch_size /
    # .device off the args object and take run_dir + grid_df explicitly, so a
    # small per-call namespace is all they need.
    call_args = argparse.Namespace(year=args.year,
                                   batch_size=args.batch_size,
                                   device=args.device)
    if ib._is_tree_run(run_dir):
        print(f'[map] {run_name}: tree path', flush=True)
        return ib.predict_tree(call_args, run_dir, grid_df)
    import torch
    device = (torch.device(args.device)
              if (args.device == 'cpu' or torch.cuda.is_available())
              else torch.device('cpu'))
    print(f'[map] {run_name}: neural path  device={device}', flush=True)
    return ib.predict_nn(call_args, run_dir, grid_df, device)


def _summary(run_name: str, preds: np.ndarray, year: int) -> dict:
    v = preds[np.isfinite(preds)]
    return {
        'run_name': run_name,
        'year': year,
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


def save_single_map(out_dir: Path, run_name: str, grid_df: pd.DataFrame,
                    preds: np.ndarray, year: int, vmax: float) -> dict:
    """Per-model parquet + scatter PNG (mirrors infer_bavaria's output)."""
    pq = out_dir / f'{run_name}_{year}_predictions.parquet'
    grid_df.assign(predicted_soc=preds, year=year).to_parquet(pq)
    summary = _summary(run_name, preds, year)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(grid_df.GPS_LONG, grid_df.GPS_LAT, c=preds, s=2,
                        cmap='YlOrBr', vmin=0, vmax=vmax, alpha=0.85)
        ax.set_title(f'{run_name}  —  predicted SOC, Bavaria {year}\n'
                     f'mean = {summary["mean"]:.2f} g/kg, '
                     f'p05–p95 = [{summary["p05"]:.1f}, {summary["p95"]:.1f}]  '
                     f'(n={summary["n_valid"]:,})', fontsize=11)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(sc, ax=ax, label='SOC (g/kg)', shrink=0.7)
        fig.tight_layout()
        fig.savefig(out_dir / f'{run_name}_{year}_map.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] per-model map failed for {run_name}: {e}',
              file=sys.stderr)
    return summary


def save_combined(out_dir: Path, grid_df: pd.DataFrame,
                  preds_by_run: dict, year: int, vmax: float) -> None:
    """Wide parquet (one prediction column per model) + a comparison panel."""
    combined = grid_df[['longitude', 'latitude']].copy()
    for run_name, preds in preds_by_run.items():
        combined[run_name] = preds
    combined.to_parquet(out_dir / f'combined_{year}_predictions.parquet')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        runs = list(preds_by_run)
        ncol = min(3, len(runs))
        nrow = int(np.ceil(len(runs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5.5 * nrow),
                                 squeeze=False)
        sc = None
        for k, run_name in enumerate(runs):
            ax = axes[k // ncol][k % ncol]
            preds = preds_by_run[run_name]
            v = preds[np.isfinite(preds)]
            sc = ax.scatter(grid_df.GPS_LONG, grid_df.GPS_LAT, c=preds, s=1,
                            cmap='YlOrBr', vmin=0, vmax=vmax, alpha=0.85)
            mean = float(v.mean()) if v.size else float('nan')
            ax.set_title(f'{run_name}\nmean {mean:.1f} g/kg', fontsize=10)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
        for k in range(len(runs), nrow * ncol):       # hide empty panels
            axes[k // ncol][k % ncol].axis('off')
        if sc is not None:
            fig.colorbar(sc, ax=axes, label='SOC (g/kg)', shrink=0.6,
                         location='right')
        fig.suptitle(f'Predicted SOC — Bavaria {year}  '
                     f'(n={len(grid_df):,} locations)', fontsize=13)
        fig.savefig(out_dir / f'combined_{year}_maps.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] combined map failed: {e}', file=sys.stderr)


def main():
    args = parse()
    run_names = [r.strip() for r in args.run_names.split(',') if r.strip()]
    if not run_names:
        raise SystemExit('[ERROR] --run-names is empty.')

    tag = args.out_tag or (
        f'{args.n_locations}rand_seed{args.sample_seed}'
        if args.n_locations else 'full')
    out_dir = ib.MAPS_ROOT / f'_locations_{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[map] output dir = {out_dir}', flush=True)

    grid_df = sample_grid(args.grid_csv, args.n_locations, args.sample_seed)
    print(f'[map] grid lon [{grid_df.GPS_LONG.min():.2f}, '
          f'{grid_df.GPS_LONG.max():.2f}]  '
          f'lat [{grid_df.GPS_LAT.min():.2f}, {grid_df.GPS_LAT.max():.2f}]',
          flush=True)

    preds_by_run, summaries = {}, []
    for run_name in run_names:
        t0 = time.time()
        preds = predict_one(args, run_name, grid_df)
        preds_by_run[run_name] = preds
        summ = save_single_map(out_dir, run_name, grid_df, preds,
                               args.year, args.vmax)
        summ['elapsed_s'] = round(time.time() - t0, 1)
        summaries.append(summ)
        print(f'[map] {run_name}: mean={summ["mean"]:.2f}  '
              f'p05={summ["p05"]:.2f}  p95={summ["p95"]:.2f}  '
              f'valid={summ["n_valid"]:,}/{summ["n_total"]:,}  '
              f'({summ["elapsed_s"]}s)', flush=True)

    if len(preds_by_run) > 1:
        save_combined(out_dir, grid_df, preds_by_run, args.year, args.vmax)

    (out_dir / f'combined_{args.year}_summary.json').write_text(
        json.dumps({'year': args.year,
                    'n_locations': int(len(grid_df)),
                    'sample_seed': args.sample_seed,
                    'models': summaries}, indent=2, default=str))
    print(f'[map] done — wrote {len(run_names)} model map(s) to {out_dir}',
          flush=True)


if __name__ == '__main__':
    main()
