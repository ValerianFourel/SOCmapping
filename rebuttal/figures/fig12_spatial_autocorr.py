#!/usr/bin/env python3
"""fig12_spatial_autocorr.py — spatial structure of the flagship's residuals.

From the flagship SGT's pooled hold-out fold predictions (canonical protocol:
split-axis lon, 43-band 'full_extended', oc150), residual = OC_predicted -
OC_actual. We then ask whether residuals carry spatial structure beyond the
spatial-CV buffer:

  * Empirical semivariogram of residuals vs separation distance (binned over
    a random subsample of <=N points; distances in UTM metres via pyproj if
    available, else great-circle km). Nugget (intercept) and sill (plateau)
    are estimated and annotated.
  * A global Moran's-I-style lag-1 statistic using a k-nearest-neighbour
    (row-standardised) weight matrix, with a permutation p-value, to give a
    single "are residuals spatially independent?" number.

The point: residuals are ~spatially independent (low Moran's I, nugget ~= sill)
beyond the spatial buffer, so the held-out R2 is not inflated by leakage.

CANONICAL data is JUPITER-only -> pass --sweep-dir; defaults select lon/43/oc150.
Smoke-test with the STALE local bundle:
  --sweep-dir /home/.../SOCrebuttal_HF/sweep --axis lat --bands 20 --max-oc 150
Every metric is read from disk via figdata; nothing hardcoded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/valerian/SGTPublication/SOCmapping/rebuttal/figures')
import figstyle as fs
import figdata as fd
fs.setup()

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / 'out'


def resolve_flagship(rows, *, axis, bands, max_oc):
    cand = fd.select(rows, axis=axis, bands=str(bands), max_oc=max_oc,
                     family=fs.FLAGSHIP)
    if not cand:
        return None
    return max(cand, key=lambda r: r['score'])


def to_metres(lon, lat):
    """(x,y) in metres via UTM 32N if pyproj present, else local equirect approx.

    Returns (x, y, unit_label, to_km_factor).
    """
    try:
        from pyproj import Transformer
        t = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        x, y = t.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y), 'UTM 32N', 1.0 / 1000.0
    except Exception:
        # equirectangular metres at mean latitude (great-circle approx)
        R = 6371000.0
        lat0 = np.deg2rad(np.mean(lat))
        x = np.deg2rad(np.asarray(lon)) * R * np.cos(lat0)
        y = np.deg2rad(np.asarray(lat)) * R
        return x, y, 'great-circle approx', 1.0 / 1000.0


def empirical_variogram(x, y, z, n_bins=18, max_frac=0.5, rng=None):
    """Binned semivariogram from pairwise distances.

    gamma(h) = 0.5 * mean[(z_i - z_j)^2] over pairs in distance bin h.
    Returns (centres_m, gamma, counts, hmax_m).
    """
    n = z.size
    # pairwise via broadcasting (n<=subsample so this is fine)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    dz2 = (z[:, None] - z[None, :]) ** 2
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    g = dz2[iu]
    hmax = max_frac * float(np.nanmax(d))
    sel = d <= hmax
    d, g = d[sel], g[sel]
    edges = np.linspace(0.0, hmax, n_bins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    centres, gamma, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c < 30:            # skip sparsely populated bins
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        gamma.append(0.5 * float(np.mean(g[m])))
        counts.append(c)
    return (np.array(centres), np.array(gamma), np.array(counts), hmax)


def nugget_sill(centres, gamma):
    """Crude nugget (first bin) and sill (overall residual variance proxy).

    Sill ~ mean gamma of the far half (plateau); nugget ~ first-bin gamma.
    """
    if centres.size == 0:
        return np.nan, np.nan
    nugget = float(gamma[0])
    far = gamma[centres >= np.median(centres)]
    sill = float(np.mean(far)) if far.size else float(np.mean(gamma))
    return nugget, sill


def morans_I_knn(x, y, z, k=8, n_perm=499, rng=None):
    """Global Moran's I with a row-standardised k-NN weight matrix.

    Returns (I, p_perm, k). Uses a vectorised pairwise distance (subsample-sized).
    Permutation p-value is two-sided-ish (fraction of permuted |I'| >= |I|).
    """
    rng = rng or np.random.default_rng(0)
    n = z.size
    if n <= k + 1:
        return np.nan, np.nan, k
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    # k nearest neighbours per row
    nn = np.argpartition(dist, kth=k, axis=1)[:, :k]
    zc = z - z.mean()
    denom = float(np.sum(zc * zc))
    if denom == 0:
        return np.nan, np.nan, k

    def stat(zc_):
        # row-standardised weights -> each row contributes mean of its neighbours
        neigh = zc_[nn]                       # (n, k)
        num = float(np.sum(zc_ * neigh.mean(axis=1)))
        # W (sum of weights) = n for row-standardised; n/W = 1
        return num / float(np.sum(zc_ * zc_))

    I = stat(zc)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = stat(rng.permutation(zc))
    p = (1.0 + np.sum(np.abs(perm) >= np.abs(I))) / (n_perm + 1.0)
    return float(I), float(p), k


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT))
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat', 'cluster'])
    ap.add_argument('--bands', default='43')
    ap.add_argument('--max-oc', type=float, default=150.0)
    ap.add_argument('--subsample', type=int, default=3000,
                    help='max points for pairwise distances (tractability)')
    ap.add_argument('--n-bins', type=int, default=18)
    ap.add_argument('--knn', type=int, default=8,
                    help="k for Moran's-I nearest-neighbour weights")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    ap.add_argument('--name', default='fig12_spatial_autocorr')
    a = ap.parse_args()

    man = fd.Manifest()
    sweep_dir = Path(a.sweep_dir)
    rng = np.random.default_rng(a.seed)

    rows = fd.load_ranking(sweep_dir)
    flag = resolve_flagship(rows, axis=a.axis, bands=a.bands, max_oc=a.max_oc)
    if flag is None:
        reason = (f'no SGT flagship row at axis={a.axis} bands={a.bands} '
                  f'max_oc={a.max_oc}')
        man.block('fig12_spatial_autocorr', reason, str(sweep_dir))
        man.write(Path(a.out_dir) / 'fig12_spatial_autocorr_manifest.md')
        print(f'BLOCKED {reason}')
        return 1

    cd = Path(flag['config_dir'])
    df = fd.load_fold_predictions(cd)
    if df is None:
        reason = 'load_fold_predictions returned None (no fold parquet)'
        man.block('fig12_spatial_autocorr', reason, str(cd))
        man.write(Path(a.out_dir) / 'fig12_spatial_autocorr_manifest.md')
        print(f'BLOCKED {reason}: {cd}')
        return 1

    resid = (df['OC_predicted'].to_numpy(float)
             - df['OC_actual'].to_numpy(float))
    lon = df['GPS_LONG'].to_numpy(float)
    lat = df['GPS_LAT'].to_numpy(float)
    ok = np.isfinite(resid) & np.isfinite(lon) & np.isfinite(lat)
    resid, lon, lat = resid[ok], lon[ok], lat[ok]
    n_all = resid.size

    # subsample for tractable O(n^2) pairwise work
    if n_all > a.subsample:
        sel = rng.choice(n_all, size=a.subsample, replace=False)
        resid_s, lon_s, lat_s = resid[sel], lon[sel], lat[sel]
    else:
        resid_s, lon_s, lat_s = resid, lon, lat
    n_use = resid_s.size

    x, y, proj_label, to_km = to_metres(lon_s, lat_s)

    centres_m, gamma, counts, hmax = empirical_variogram(
        x, y, resid_s, n_bins=a.n_bins, rng=rng)
    nugget, sill = nugget_sill(centres_m, gamma)
    var_resid = float(np.var(resid_s, ddof=1))

    I, p_perm, k = morans_I_knn(x, y, resid_s, k=a.knn, n_perm=499, rng=rng)

    # ---- plot: variogram (left) + Moran's-I panel (right) ----
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                  gridspec_kw={'width_ratios': [1.5, 1.0]})

    if centres_m.size:
        ck = centres_m * to_km
        ax.plot(ck, gamma, 'o-', color=fs.fam_style(fs.FLAGSHIP)[1],
                ms=5, lw=1.6, label='empirical $\\gamma(h)$')
        # sample variance reference (the variogram plateaus near it if random)
        ax.axhline(var_resid, color='k', ls=':', lw=1.2,
                   label=f'sample variance = {var_resid:.1f}')
        if np.isfinite(nugget):
            ax.axhline(nugget, color='#888888', ls='--', lw=1.0)
        if np.isfinite(sill):
            ax.axhline(sill, color='#cc5500', ls='--', lw=1.0)
        ax.annotate(
            f'nugget $\\approx$ {nugget:.1f}\nsill $\\approx$ {sill:.1f}\n'
            f'nugget/sill $\\approx$ {nugget/sill:.2f}'
            if np.isfinite(sill) and sill != 0 else
            f'nugget $\\approx$ {nugget:.1f}',
            xy=(0.97, 0.06), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=8.5,
            bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.9))
    ax.set_xlabel(f'Separation distance $h$ (km, {proj_label})')
    ax.set_ylabel('Semivariance $\\gamma(h)$ (g/kg)$^2$')
    ax.set_title('Residual semivariogram', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)

    # Moran's I bar with permutation null band
    ax2.axhline(0.0, color='k', lw=1.0)
    col = '#cc3311' if (np.isfinite(p_perm) and p_perm < 0.05) else '#4477aa'
    ax2.bar([0], [I if np.isfinite(I) else 0.0], width=0.5, color=col,
            edgecolor='k', lw=1.0)
    # null expectation ~ -1/(n-1)
    e_I = -1.0 / (n_use - 1)
    ax2.axhline(e_I, color='#888888', ls='--', lw=1.0,
                label=f'E[I] under null = {e_I:+.4f}')
    ax2.set_xticks([0])
    ax2.set_xticklabels([f'lag-1\nk={k} NN'])
    ax2.set_ylabel("Global Moran's I")
    ax2.set_ylim(min(-0.05, (I if np.isfinite(I) else 0) - 0.05),
                 max(0.05, (I if np.isfinite(I) else 0) + 0.05))
    ax2.set_title("Residual spatial autocorrelation", fontsize=10)
    ax2.annotate(
        f"I = {I:+.4f}\np(perm) = {p_perm:.3f}\n"
        + ("structured" if (np.isfinite(p_perm) and p_perm < 0.05)
           else "~independent"),
        xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.9))
    ax2.legend(loc='lower center', fontsize=7.5)

    fig.suptitle(
        f'{fs.fam_label(fs.FLAGSHIP)} — spatial structure of CV residuals  '
        f'(n_used={n_use} of {n_all})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    pdf, png = fs.save(fig, a.out_dir, a.name,
                       script='fig12_spatial_autocorr.py')

    man.done(
        'fig12_spatial_autocorr',
        files=[str(pdf), str(png)],
        source=str(cd),
        numbers=(f'morans_I={I:+.4f} p_perm={p_perm:.3f} k={k} '
                 f'nugget={nugget:.3f} sill={sill:.3f} var={var_resid:.3f} '
                 f'n_used={n_use}/{n_all} flagship={flag["tag"]}'))
    man.write(Path(a.out_dir) / 'fig12_spatial_autocorr_manifest.md')

    print(f'flagship={flag["tag"]} cd={cd}')
    print(f'morans_I={I:+.4f} p_perm={p_perm:.3f} nugget={nugget:.3f} '
          f'sill={sill:.3f} var={var_resid:.3f} n_used={n_use}/{n_all}')
    print(f'OK {pdf}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
