#!/usr/bin/env python3
"""
rebuttal/figures/figstyle.py — shared style for the Geoderma revision figure
set (GEODER-D-26-01032, 43-band / lon-blocked results).

Imported by every figure script. Provides:
  - setup()                : elsarticle-matching rcParams (serif, ~9-10pt, 300dpi)
  - FAMILIES / fam_style() : one colorblind-safe palette + marker per model family
  - FOLD_COLORS            : fixed palette for the 10 spatial folds
  - SOC_CMAP/VMIN/VMAX     : YlOrBr clipped 0-80 g/kg, shared across every map
  - bavaria_scatter()      : Bavaria map helper (UTM 32N if pyproj present, else
                             lon/lat), scale bar + N arrow, consistent SOC colour
  - save()                 : write <name>.pdf (vector) + <name>.png (300dpi),
                             optional provenance stamp

No hard dependency on cartopy/geopandas (not installed in this env). UTM
reprojection is best-effort via pyproj; falls back to lon/lat gracefully.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# --- Colorblind-safe per-family identity (Okabe-Ito based) ------------------
# (label, color, marker). SGT is the flagship — strongest/first color.
FAMILIES: dict[str, tuple[str, str, str]] = {
    'sgt':               ('SGT (gated CNN+GRN+Transformer)', '#D55E00', 'o'),
    'vanilla':           ('Vanilla (CNN+Transformer, no gate)', '#0072B2', 's'),
    'simpletransformer': ('Simple Transformer (no CNN)',        '#E69F00', '^'),
    'lightweight':       ('Lightweight Transformer (no CNN)',   '#CC79A7', 'v'),
    'cnnlstm':           ('CNN-LSTM',                            '#009E73', 'D'),
    '3dcnn':             ('3D-CNN',                              '#999999', 'X'),
    'rf':                ('Random Forest',                       '#56B4E9', 'P'),
    'xgb':               ('XGBoost',                             '#117733', '*'),
}
FLAGSHIP = 'sgt'


def fam_style(family: str) -> tuple[str, str, str]:
    """(label, color, marker) for a canonical family key; gray fallback."""
    return FAMILIES.get(family, (family, '#444444', '.'))


def fam_label(family: str) -> str:
    return fam_style(family)[0]


# Fixed palette for up to 12 spatial folds (viridis-sampled, stable order).
FOLD_COLORS = [cm.viridis(x) for x in np.linspace(0.05, 0.95, 12)]

# Shared SOC colour scale for ALL maps.
SOC_CMAP = 'YlOrBr'
SOC_VMIN = 0.0
SOC_VMAX = 80.0


def setup():
    """elsarticle-matching rcParams. Call once at the top of each figure."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Nimbus Roman'],
        'font.size': 9.5,
        'axes.titlesize': 10,
        'axes.labelsize': 9.5,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'legend.frameon': False,
    })


def _utm32n(lon, lat):
    """Reproject lon/lat -> EPSG:32632 (m) if pyproj is available, else None."""
    try:
        from pyproj import Transformer
        t = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        x, y = t.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y)
    except Exception:
        return None


def bavaria_scatter(ax, lon, lat, values, *, s=2, vmin=SOC_VMIN, vmax=SOC_VMAX,
                    cmap=SOC_CMAP, scalebar=True, north=True, utm=True):
    """Plot a Bavaria SOC map on `ax`. Returns the scatter handle.

    Uses UTM 32N (metres) when pyproj is available so distances/scale bars are
    true; otherwise plots lon/lat with an equal-aspect approximation. Same
    SOC colour scale everywhere by default.
    """
    lon = np.asarray(lon); lat = np.asarray(lat); values = np.asarray(values)
    xy = _utm32n(lon, lat) if utm else None
    if xy is not None:
        x, y = xy
        unit = 'm'
    else:
        x, y = lon, lat
        # ~aspect correction at Bavaria's mean latitude (~48.9N)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(48.9)))
        unit = 'deg'
    sc = ax.scatter(x, y, c=values, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=0.9, linewidths=0, rasterized=True)
    if unit == 'm':
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Easting (UTM 32N, km)'); ax.set_ylabel('Northing (km)')
        ax.xaxis.set_major_formatter(lambda v, _p: f'{v/1000:.0f}')
        ax.yaxis.set_major_formatter(lambda v, _p: f'{v/1000:.0f}')
        if scalebar:
            _scalebar_m(ax, x, y, 50_000, '50 km')
    else:
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    if north:
        ax.annotate('N', xy=(0.96, 0.96), xycoords='axes fraction',
                    ha='center', va='center', fontsize=10, fontweight='bold')
        ax.annotate('', xy=(0.96, 0.95), xytext=(0.96, 0.88),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.0))
    return sc


def _scalebar_m(ax, x, y, length_m, label):
    x0 = np.nanmin(x) + 0.06 * (np.nanmax(x) - np.nanmin(x))
    y0 = np.nanmin(y) + 0.06 * (np.nanmax(y) - np.nanmin(y))
    ax.plot([x0, x0 + length_m], [y0, y0], color='k', lw=2.5,
            solid_capstyle='butt')
    ax.text(x0 + length_m / 2, y0 + 0.012 * (np.nanmax(y) - np.nanmin(y)),
            label, ha='center', va='bottom', fontsize=8)


def provenance(fig, script: str, commit: str = '', note: str = ''):
    """Small bottom-left provenance stamp for your own tracking (toggle off
    for final submission by passing note=None)."""
    if note is None:
        return
    import datetime  # noqa: imported lazily; figures pass a fixed stamp if needed
    txt = f'{script}'
    if commit:
        txt += f' @ {commit}'
    if note:
        txt += f'  {note}'
    fig.text(0.005, 0.005, txt, fontsize=5, color='#999999', ha='left', va='bottom')


def save(fig, out_dir, name: str, *, script: str = '', commit: str = '',
         stamp: bool = False):
    """Write <name>.pdf and <name>.png (300 dpi) to out_dir. Returns paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if stamp:
        provenance(fig, script, commit, note='draft')
    pdf = out_dir / f'{name}.pdf'
    png = out_dir / f'{name}.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png
