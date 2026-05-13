#!/usr/bin/env python3
"""
plot_uncertainty.py — Three-panel publication figure for the MC dropout map.

Run after mc_dropout_inference.py. Reads the two GeoTIFFs (or, as fallback,
the .npy/parquet outputs if rasterio wasn't available) and emits:

    figure_uncertainty_3panel.png   (300 dpi, 18 × 7 in)
    figure_uncertainty_3panel.pdf   (vector, for journal submission)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal/gpu_experiments/uncertainty')
MEAN_TIF = OUT_DIR / 'SGT_1mil_2023_mean_mc30.tif'
STD_TIF = OUT_DIR / 'SGT_1mil_2023_std_mc30.tif'

# ASSUMPTION: the existing Figures 14/15 in the paper use the viridis
# colormap (confirmed: SOCmapping/SpatiotemporalGatedTransformer/mapping.py
# uses cmap='viridis'). Panel 1 mirrors that exactly. If the paper actually
# uses a different cmap (e.g. YlOrBr, custom SOC palette), swap CMAP_MEAN.
CMAP_MEAN = 'viridis'
CMAP_STD = 'OrRd'
CMAP_CV = 'RdYlGn_r'


def load_geotiff(path: Path):
    try:
        import rasterio
    except ImportError:
        print(f'rasterio missing; cannot read {path}.', file=sys.stderr)
        return None, None, None
    with rasterio.open(path) as src:
        arr = src.read(1, masked=False).astype(np.float32)
        extent = (src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top)
        crs = src.crs.to_string()
    arr[~np.isfinite(arr)] = np.nan
    return arr, extent, crs


def maybe_load_bavaria_outline():
    """Try to load Bavaria boundary for overlay; return None if unavailable."""
    try:
        import geopandas as gpd
        # ASSUMPTION: bavaria.geojson exists in the project dir
        for cand in (
            Path('/home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer/bavaria.geojson'),
            Path('/home/valerian/SGTPublication/SOCmapping/Maps/bavaria.geojson'),
            Path('/home/valerian/SGTPublication/SOCmapping/balancedDataset/bavaria.geojson'),
        ):
            if cand.exists():
                bav = gpd.read_file(cand)
                if 'EPSG:32632' not in str(bav.crs):
                    bav = bav.to_crs('EPSG:32632')
                return bav
    except Exception as e:
        print(f'Bavaria outline load failed: {e}', file=sys.stderr)
    return None


def add_decorations(ax, bavaria, extent):
    """Bavaria outline, scale bar, north arrow, nodata-as-white."""
    if bavaria is not None:
        bavaria.boundary.plot(ax=ax, color='black', linewidth=0.5)
    # ASSUMPTION: simple scale bar 50 km long-bar in lower-left.
    if extent is not None:
        x0 = extent[0] + 0.05 * (extent[1] - extent[0])
        y0 = extent[2] + 0.05 * (extent[3] - extent[2])
        bar_len = 50_000  # 50 km in metres (UTM)
        ax.plot([x0, x0 + bar_len], [y0, y0], color='black', linewidth=2)
        ax.text(x0 + bar_len / 2, y0 + (extent[3] - extent[2]) * 0.015,
                '50 km', ha='center', va='bottom', fontsize=8)
        # North arrow top-right
        nx = extent[0] + 0.94 * (extent[1] - extent[0])
        ny = extent[2] + 0.88 * (extent[3] - extent[2])
        dy = (extent[3] - extent[2]) * 0.05
        ax.annotate('N', xy=(nx, ny + dy), xytext=(nx, ny),
                    ha='center', fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5))


def main():
    mean_arr, extent, _ = load_geotiff(MEAN_TIF)
    std_arr, _, _ = load_geotiff(STD_TIF)

    if mean_arr is None or std_arr is None:
        # Fallback to the parquet (point scatter rendering instead of raster)
        # ASSUMPTION: at least one of GeoTIFF or parquet is present after
        # running mc_dropout_inference.py.
        import pandas as pd
        df = pd.read_parquet(OUT_DIR / 'mc_dropout_points.parquet')
        # Fall back to a scatter-render at low DPI; this branch only fires
        # if rasterio failed at write time, which would be a setup bug.
        print('GeoTIFF missing — falling back to scatter; result may be sparse.',
              file=sys.stderr)
        mean_arr = std_arr = None
        cv_arr = None
        # We'll just plot scatter points in original lon/lat space.
        scatter_mode = True
    else:
        scatter_mode = False
        with np.errstate(invalid='ignore', divide='ignore'):
            cv_arr = np.where(mean_arr > 1e-6,
                              np.clip(100 * std_arr / mean_arr, 0, 100),
                              np.nan)

    bavaria = maybe_load_bavaria_outline() if not scatter_mode else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=300)
    plt.subplots_adjust(wspace=0.15)

    if scatter_mode:
        # --- scatter fallback (rasterio missing) ---
        import pandas as pd
        df = pd.read_parquet(OUT_DIR / 'mc_dropout_points.parquet')
        for ax, key, cmap, label, title in [
            (axes[0], 'mean_pred_g_per_kg', CMAP_MEAN,
             'Mean predicted SOC (g/kg)',
             'SGT 1.1M — 2023 SOC prediction (MC mean, n=30)'),
            (axes[1], 'std_pred_g_per_kg', CMAP_STD,
             'Prediction uncertainty — std (g/kg)',
             'MC Dropout uncertainty (n=30 passes)'),
            (axes[2], 'cv_pct', CMAP_CV,
             'Coefficient of variation (%)',
             'Relative uncertainty (CV = std/mean × 100%)'),
        ]:
            sc = ax.scatter(df['longitude'], df['latitude'],
                            c=df[key], cmap=cmap, s=1, marker='s')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(label, fontsize=9)
            ax.set_aspect('equal', adjustable='box')
    else:
        panel_specs = [
            (axes[0], mean_arr, CMAP_MEAN,
             'Mean predicted SOC (g/kg)',
             'SGT 1.1M — 2023 SOC prediction (MC mean, n=30)',
             dict(vmin=0, vmax=float(np.nanpercentile(mean_arr, 99)))),
            (axes[1], std_arr, CMAP_STD,
             'Prediction uncertainty — std (g/kg)',
             'MC Dropout uncertainty (n=30 passes)',
             dict(vmin=0, vmax=float(np.nanpercentile(std_arr, 99)))),
            (axes[2], cv_arr, CMAP_CV,
             'Coefficient of variation (%)',
             'Relative uncertainty (CV = std/mean × 100%)',
             dict(vmin=0, vmax=100)),
        ]
        for ax, arr, cmap, label, title, kwargs in panel_specs:
            im = ax.imshow(arr, extent=extent, cmap=cmap, origin='upper',
                            interpolation='nearest', **kwargs)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('UTM Easting (m)')
            ax.set_ylabel('UTM Northing (m)')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(label, fontsize=9)
            ax.set_facecolor('white')
            add_decorations(ax, bavaria, extent)

    out_png = OUT_DIR / 'figure_uncertainty_3panel.png'
    out_pdf = OUT_DIR / 'figure_uncertainty_3panel.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Wrote {out_png}', flush=True)
    print(f'Wrote {out_pdf}', flush=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
