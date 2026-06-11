#!/usr/bin/env python3
"""
rebuttal/final_models/combine_location_maps.py — rebuild the combined
comparison panel/parquet/summary for a maps/_locations_<tag>/ directory
from the per-model prediction parquets already on disk.

Why: map_locations.py writes the combined_<year>_* files at the END of its
run, so if two map jobs share the same output directory (same --n-locations
/ --sample-seed / --year) the last one to finish overwrites the combined
figure — leaving it with only that job's models. The per-model
<run>_<year>_predictions.parquet files are uniquely named and all survive,
so this script just re-reads them and regenerates the combined outputs over
EVERY model present, with no re-inference.

Run (after all map jobs for a tag have finished):
    python rebuttal/final_models/combine_location_maps.py \\
        --dir rebuttal/final_models/maps/_locations_400000rand_seed42 \\
        --year 2023
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', type=Path, required=True,
                   help='A maps/_locations_<tag>/ directory containing '
                        '<run>_<year>_predictions.parquet files.')
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--vmax', type=float, default=80.0,
                   help='Upper bound of the SOC colour scale (g/kg).')
    return p.parse_args()


def main():
    a = parse()
    d: Path = a.dir
    if not d.is_dir():
        raise SystemExit(f'[ERROR] not a directory: {d}')

    suffix = f'_{a.year}_predictions.parquet'
    parts = sorted(p for p in d.glob(f'*{suffix}')
                   if not p.name.startswith('combined_'))
    if not parts:
        raise SystemExit(f'[ERROR] no per-model parquets matching *{suffix} in {d}')

    merged = None
    summaries = []
    for pq in parts:
        run_name = pq.name[:-len(suffix)]
        df = pd.read_parquet(pq)
        if 'predicted_soc' not in df.columns:
            print(f'[skip] {pq.name}: no predicted_soc column')
            continue
        sub = df[['longitude', 'latitude', 'predicted_soc']].rename(
            columns={'predicted_soc': run_name})
        merged = sub if merged is None else merged.merge(
            sub, on=['longitude', 'latitude'], how='outer')
        v = df['predicted_soc'].to_numpy()
        v = v[np.isfinite(v)]
        summaries.append({
            'run_name': run_name,
            'year': a.year,
            'n_valid': int(v.size),
            'mean': float(v.mean()) if v.size else float('nan'),
            'std': float(v.std()) if v.size else float('nan'),
            'p05': float(np.percentile(v, 5)) if v.size else float('nan'),
            'p50': float(np.percentile(v, 50)) if v.size else float('nan'),
            'p95': float(np.percentile(v, 95)) if v.size else float('nan'),
            'min': float(v.min()) if v.size else float('nan'),
            'max': float(v.max()) if v.size else float('nan'),
        })
        print(f'[combine] {run_name:>40}: n={v.size:,}  '
              f'mean={summaries[-1]["mean"]:.2f}')

    runs = [s['run_name'] for s in summaries]
    merged.to_parquet(d / f'combined_{a.year}_predictions.parquet')

    # Multi-panel comparison figure (shared colour scale).
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ncol = min(3, len(runs))
        nrow = int(np.ceil(len(runs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5.5 * nrow),
                                 squeeze=False)
        sc = None
        for k, run_name in enumerate(runs):
            ax = axes[k // ncol][k % ncol]
            preds = merged[run_name].to_numpy()
            v = preds[np.isfinite(preds)]
            sc = ax.scatter(merged.longitude, merged.latitude, c=preds, s=1,
                            cmap='YlOrBr', vmin=0, vmax=a.vmax, alpha=0.85)
            mean = float(v.mean()) if v.size else float('nan')
            ax.set_title(f'{run_name}\nmean {mean:.1f} g/kg', fontsize=10)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
        for k in range(len(runs), nrow * ncol):
            axes[k // ncol][k % ncol].axis('off')
        if sc is not None:
            fig.colorbar(sc, ax=axes, label='SOC (g/kg)', shrink=0.6,
                         location='right')
        fig.suptitle(f'Predicted SOC — Bavaria {a.year}  '
                     f'({len(merged):,} locations, {len(runs)} models)',
                     fontsize=13)
        fig.savefig(d / f'combined_{a.year}_maps.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] combined figure failed: {e}')

    (d / f'combined_{a.year}_summary.json').write_text(
        json.dumps({'year': a.year, 'n_locations': int(len(merged)),
                    'models': summaries}, indent=2, default=str))
    print(f'[combine] wrote combined_{a.year}_(predictions.parquet|maps.png|'
          f'summary.json) over {len(runs)} models to {d}')


if __name__ == '__main__':
    main()
