"""WindowedSentinel2Dataset — tensor-ready loader for the native-resolution
sentinel2 windowed dataset (ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel2,
`windows/lucas/`).

Instead of full RasterTensorData tiles, this reads the pre-extracted 11x11
windows per (band, year) — `<band>_<year>.npy` (SRC, yearly) and
`<band>_static.npy` (terrain + S2 SWIR) — each `float32 [N, 11, 11]`, row-aligned
to `_labels.parquet` (pointid, lat, lon, year, oc) and `_points.npy`.

It assembles the FINE-band cube `x [C_fine, H, W, T]` per LUCAS point that
ResolutionAwareNet(branches='fine') consumes directly:
  - SRC bands (yearly): the T years ending at the sample year.
  - terrain + S2 SWIR (static composites): repeated across T.
NaN (sparse bare-soil) -> 0 after per-band standardisation, matching -large.

The 23 coarse/medium bands are NOT here (reused from -large; the net reads only
their centre pixel) — add them via a separate centre-value source for the full
fine_med_coarse model. This loader covers the fine branch end-to-end.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Fine-band order (S2 SWIR ×2, Landsat SRC ×11, SRTM/terrain ×9) = 22.
SRC_BANDS = ['SRC_Blue', 'SRC_Green', 'SRC_Red', 'SRC_NIR', 'SRC_SWIR1', 'SRC_SWIR2',
             'SRC_RCC', 'SRC_BCC', 'SRC_NBR2', 'SRC_BSI', 'SRC_ExposureCount']
TERRAIN_BANDS = ['Elevation', 'Slope', 'Aspect', 'TWI', 'TPI_90', 'TPI_300',
                 'TPI_1000', 'TRI', 'Roughness']
S2_BANDS = ['S2SRC_SWIR1', 'S2SRC_SWIR2']
FINE_ORDER = S2_BANDS + SRC_BANDS + TERRAIN_BANDS          # 22
_YEARLY = set(SRC_BANDS)                                   # the rest are static
YEARS = list(range(2002, 2024))


class WindowedSentinel2Dataset(Dataset):
    def __init__(self, root, time_before=5, window=11, bands=None,
                 standardize=True, fill=0.0):
        """root = .../windows/lucas ; time_before = T years (incl. sample year)."""
        self.root = Path(root)
        self.T = int(time_before)
        self.W = int(window)
        self.bands = list(bands) if bands else [b for b in FINE_ORDER
                                                if self._exists(b, YEARS[0])]
        self.fill = float(fill)
        self.labels = pd.read_parquet(self.root / '_labels.parquet').reset_index(drop=True)
        self.N = len(self.labels)
        self._cache: dict[str, np.ndarray] = {}
        self._stats: dict[str, tuple[float, float]] = {}
        if standardize:
            self._fit_stats()

    # --- file helpers -------------------------------------------------------
    def _path(self, band, year):
        return self.root / (f"{band}_{year}.npy" if band in _YEARLY else f"{band}_static.npy")

    def _exists(self, band, year):
        return self._path(band, year).exists()

    def _arr(self, band, year):
        key = f"{band}_{year}" if band in _YEARLY else f"{band}_static"
        a = self._cache.get(key)
        if a is None:
            a = np.load(self._path(band, year), mmap_mode='r')
            self._cache[key] = a
        return a

    def _years_for(self, sample_year):
        y1 = int(min(max(sample_year, YEARS[0]), YEARS[-1]))
        ys = [max(YEARS[0], y1 - k) for k in range(self.T - 1, -1, -1)]
        return ys                                          # length T, oldest..newest

    # --- per-band standardisation over valid (finite) pixels ----------------
    def _fit_stats(self, sample=4000):
        idx = np.linspace(0, self.N - 1, min(sample, self.N)).astype(int)
        for b in self.bands:
            yrs = YEARS if b in _YEARLY else [YEARS[0]]
            vals = []
            for y in yrs[:: max(1, len(yrs) // 3)]:        # a few years is enough
                a = np.asarray(self._arr(b, y)[idx])
                vals.append(a[np.isfinite(a)])
            v = np.concatenate(vals) if vals else np.array([0.0])
            mu = float(np.nanmean(v)) if v.size else 0.0
            sd = float(np.nanstd(v)) or 1.0
            self._stats[b] = (mu, sd)

    def _norm(self, win, band):
        mu, sd = self._stats.get(band, (0.0, 1.0))
        out = (win.astype(np.float32) - mu) / sd
        out[~np.isfinite(out)] = self.fill
        return out

    # --- Dataset API --------------------------------------------------------
    def __len__(self):
        return self.N

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        ys = self._years_for(int(row['year']) if pd.notna(row['year']) else YEARS[-1])
        cube = np.empty((len(self.bands), self.W, self.W, self.T), dtype=np.float32)
        for bi, b in enumerate(self.bands):
            if b in _YEARLY:
                for ti, y in enumerate(ys):
                    cube[bi, :, :, ti] = self._norm(np.asarray(self._arr(b, y)[i]), b)
            else:                                          # static: repeat across T
                w = self._norm(np.asarray(self._arr(b, YEARS[0])[i]), b)
                cube[bi] = w[:, :, None].repeat(self.T, axis=2)
        return torch.from_numpy(cube), torch.tensor(float(row['oc']), dtype=torch.float32)


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SOC_DATA_DIR', '.') + '/windows/lucas'
    ds = WindowedSentinel2Dataset(root, time_before=5, window=11)
    print(f"dataset: N={len(ds)}  fine bands={len(ds.bands)} -> {ds.bands}")
    x, y = ds[0]
    print(f"sample cube {tuple(x.shape)} (C,H,W,T)  oc={y.item():.1f}  finite={torch.isfinite(x).float().mean():.2f}")
    # smoke-test the fine branch of ResolutionAwareNet
    from ResolutionAwareNet import ResolutionAwareNet
    C = len(ds.bands)
    net = ResolutionAwareNet(C, height=11, width=11, time_steps=5, branches='fine',
                             fine_idx=list(range(C)))
    xb = torch.stack([ds[k][0] for k in range(4)])
    out = net(xb)
    print(f"ResolutionAwareNet(branches=fine) forward: in {tuple(xb.shape)} -> out {tuple(out.shape)}  params={net.count_parameters():,}")
