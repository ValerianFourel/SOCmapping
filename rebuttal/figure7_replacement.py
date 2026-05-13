#!/usr/bin/env python3
"""
figure7_replacement.py — Task D (T1.16).

Produces a 2D replacement for Figure 7 of the paper using the 16,514-row
model-ready dataset (with altitude joined).

Outputs:
    rebuttal/figure7_replacement.png   — 2-panel figure, 300 dpi
    rebuttal/figure7_data.csv           — SOC, year, altitude, altitude_bin,
                                          year_bin per sample
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_READY = Path('/home/valerian/SGTPublication/rebuttal/model_ready_dataset.parquet')
OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal')

ALTITUDE_EDGES = [0, 300, 700, 1500, np.inf]
ALTITUDE_LABELS = ['<300 m', '300–700 m', '700–1500 m', '>1500 m']
YEAR_BIN_EDGES = [2006, 2011, 2016, 2023]      # right-inclusive: 2007–2011 etc.
YEAR_BIN_LABELS = ['2007–2011', '2012–2016', '2017–2023']

ALT_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
YEAR_COLORS = ['#7570b3', '#1b9e77', '#d95f02']


def fit_line(x, y):
    """Simple OLS slope + intercept; returns (m, b, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 2 or x.std() == 0:
        return float('nan'), float('nan'), int(x.size)
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b), int(x.size)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(MODEL_READY).copy()
    df['altitude_bin'] = pd.cut(df['elevation'], bins=ALTITUDE_EDGES,
                                labels=ALTITUDE_LABELS, right=False)
    df['year_bin'] = pd.cut(df['year'], bins=YEAR_BIN_EDGES,
                            labels=YEAR_BIN_LABELS, right=True,
                            include_lowest=True)

    # Save the CSV
    csv_cols = ['POINTID', 'GPS_LONG', 'GPS_LAT', 'year', 'OC',
                'elevation', 'altitude_bin', 'year_bin']
    csv_cols = [c for c in csv_cols if c in df.columns]
    df[csv_cols].to_csv(OUT_DIR / 'figure7_data.csv', index=False)
    print(f'Wrote {OUT_DIR / "figure7_data.csv"}  ({len(df):,} rows)')

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
    rng = np.random.default_rng(2026)

    # ----- Left panel: SOC vs year, coloured by altitude bin -----
    ax = axes[0]
    jitter = rng.uniform(-0.15, 0.15, size=len(df))
    for i, bin_label in enumerate(ALTITUDE_LABELS):
        sub = df[df['altitude_bin'] == bin_label]
        if sub.empty:
            continue
        ax.scatter(sub['year'] + jitter[sub.index] if False else sub['year']
                   + rng.uniform(-0.15, 0.15, size=len(sub)),
                   sub['OC'], s=6, alpha=0.20, color=ALT_COLORS[i],
                   linewidths=0, label=None)
    # Trend lines per altitude bin (drawn last so they sit on top)
    trend_summary_left = []
    for i, bin_label in enumerate(ALTITUDE_LABELS):
        sub = df[df['altitude_bin'] == bin_label]
        if len(sub) < 3:
            continue
        m, b, n = fit_line(sub['year'], sub['OC'])
        years = np.array([2007, 2023])
        ax.plot(years, m * years + b, color=ALT_COLORS[i], linewidth=2.6,
                label=f'{bin_label}  (n={n}, slope={m:+.2f} g/kg/yr)')
        trend_summary_left.append({'altitude_bin': bin_label, 'n': n,
                                   'slope_g_per_kg_per_yr': m, 'intercept': b})
    ax.set_xlabel('Survey year', fontsize=11)
    ax.set_ylabel('SOC (g/kg)', fontsize=11)
    ax.set_title('SOC vs year, stratified by altitude band',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(2006.5, 2023.5)
    ax.set_ylim(-2, 152)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ----- Right panel: SOC vs altitude, coloured by year bin -----
    ax = axes[1]
    for i, bin_label in enumerate(YEAR_BIN_LABELS):
        sub = df[df['year_bin'] == bin_label]
        if sub.empty:
            continue
        ax.scatter(sub['elevation'], sub['OC'], s=6, alpha=0.18,
                   color=YEAR_COLORS[i], linewidths=0)
    trend_summary_right = []
    for i, bin_label in enumerate(YEAR_BIN_LABELS):
        sub = df[df['year_bin'] == bin_label]
        if len(sub) < 3:
            continue
        m, b, n = fit_line(sub['elevation'], sub['OC'])
        elev = np.linspace(df['elevation'].min(), df['elevation'].max(), 200)
        ax.plot(elev, m * elev + b, color=YEAR_COLORS[i], linewidth=2.6,
                label=f'{bin_label}  (n={n}, slope={m * 100:+.2f} g/kg per 100 m)')
        trend_summary_right.append({'year_bin': bin_label, 'n': n,
                                    'slope_g_per_kg_per_100m': m * 100,
                                    'intercept': b})
    ax.set_xlabel('Altitude (m)', fontsize=11)
    ax.set_ylabel('SOC (g/kg)', fontsize=11)
    ax.set_title('SOC vs altitude, stratified by year band',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(df['elevation'].min() - 20, df['elevation'].max() + 20)
    ax.set_ylim(-2, 152)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'SOC concentration vs year and vs altitude  '
                 f'— n = {len(df):,} Bavaria samples, 2007–2023',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig_path = OUT_DIR / 'figure7_replacement.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_path}')

    # Persist trend summaries as JSON for the master numbers doc
    import json
    (OUT_DIR / 'figure7_trends.json').write_text(json.dumps({
        'left_panel_trends_by_altitude_bin': trend_summary_left,
        'right_panel_trends_by_year_bin': trend_summary_right,
        'altitude_bin_counts': df['altitude_bin'].value_counts().to_dict(),
        'year_bin_counts': df['year_bin'].value_counts().to_dict(),
    }, indent=2, default=str))


if __name__ == '__main__':
    main()
