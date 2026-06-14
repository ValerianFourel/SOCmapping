"""WindowedSentinel2Dataset — tensor-ready loader for the native-resolution
sentinel2 windowed dataset (ValerianFourel/sgt-bavaria-soc-2002-2023-large-sentinel2,
`windows/lucas/`).

Instead of full RasterTensorData tiles, this reads pre-extracted per-point data:
  - FINE bands (22, native res)  -> `<band>_<year>.npy` (SRC, yearly) or
    `<band>_static.npy` (terrain, S2), each `float32 [N, 11, 11]` (a real window).
  - MED/COARSE bands (23, 250 m) -> `<band>_<year>.npy` / `<band>_static.npy`,
    `float32 [N]` (the CENTRE value only — the net reads centre pixel for these).
All row-aligned to `_labels.parquet` (pointid, lat, lon, year, oc).

It assembles `x [C, H, W, T]` in `_bands.FULL_EXTENDED_S2_BANDS` order (45 bands):
fine bands give the real 11x11 window over the T years ending at the sample year
(static fine repeat across T); med/coarse bands broadcast their centre value
across HxW x T. Exposes `.fine_idx / .med_idx / .coarse_idx` (from
`_bands.resolution_groups`) to wire ResolutionAwareNet directly.

If the coarse `[N]` arrays are absent, pass `bands=_bands.SENTINEL_FINE_BANDS`
for the fine-only branch.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root for _bands
import _bands as B

FULL = B.FULL_EXTENDED_S2_BANDS                      # 45, canonical order
FINE = set(B.SENTINEL_FINE_BANDS)                    # 22
YEARS = list(range(2002, 2024))


class WindowedSentinel2Dataset(Dataset):
    def __init__(self, root, bands=None, time_before=5, window=11,
                 standardize=True, fill=0.0, preload=True):
        self.root = Path(root)
        self.T = int(time_before); self.W = int(window); self.fill = float(fill)
        self.preload = bool(preload)
        self.bands = list(bands) if bands else [b for b in FULL if self._has(b)]
        self.labels = pd.read_parquet(self.root / '_labels.parquet').reset_index(drop=True)
        self.N = len(self.labels)
        self._cache: dict[str, np.ndarray] = {}
        self._stats: dict[str, tuple[float, float]] = {}
        # resolution-group indices INTO self.bands (for ResolutionAwareNet)
        f, m, c = B.resolution_groups(self.bands)
        self.fine_idx, self.med_idx, self.coarse_idx = f, m, c
        if self.preload:                       # load every band-array into RAM once
            self._warm()                       # (~4 GB) — kills the per-sample np.load I/O
        if standardize:
            self._fit_stats()

    def _warm(self):
        for b in self.bands:
            for y in (YEARS if self._yearly(b) else [YEARS[0]]):
                self._arr(b, y)

    # --- file helpers -------------------------------------------------------
    def _yearly(self, band):
        return (self.root / f"{band}_{YEARS[0]}.npy").exists()

    def _has(self, band):
        return self._yearly(band) or (self.root / f"{band}_static.npy").exists()

    def _path(self, band, year):
        return self.root / (f"{band}_{year}.npy" if self._yearly(band) else f"{band}_static.npy")

    def _arr(self, band, year):
        key = f"{band}_{year}" if self._yearly(band) else f"{band}_static"
        a = self._cache.get(key)
        if a is None and key not in self._cache:
            p = self._path(band, year)
            # preload → full array in RAM (fast per-sample); else mmap (lazy)
            a = (np.load(p) if self.preload else np.load(p, mmap_mode='r')) if p.exists() else None
            self._cache[key] = a
        return a

    def _years_for(self, y):
        y1 = int(min(max(int(y), YEARS[0]), YEARS[-1]))
        return [max(YEARS[0], y1 - k) for k in range(self.T - 1, -1, -1)]   # oldest..newest

    def _fit_stats(self, sample=4000):
        idx = np.linspace(0, self.N - 1, min(sample, self.N)).astype(int)
        for b in self.bands:
            yrs = (YEARS if self._yearly(b) else [YEARS[0]])
            vals = []
            for y in yrs[:: max(1, len(yrs) // 3)]:
                a = self._arr(b, y)
                if a is None: continue
                v = np.asarray(a[idx]); vals.append(v[np.isfinite(v)])
            v = np.concatenate(vals) if vals else np.array([0.0], np.float32)
            self._stats[b] = (float(np.nanmean(v)) if v.size else 0.0,
                              float(np.nanstd(v)) or 1.0)

    def _norm(self, x, band):
        mu, sd = self._stats.get(band, (0.0, 1.0))
        out = (np.asarray(x, np.float32) - mu) / sd
        return np.where(np.isfinite(out), out, self.fill).astype(np.float32)   # 0-d safe

    # --- Dataset API --------------------------------------------------------
    def __len__(self):
        return self.N

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        ys = self._years_for(row['year'] if pd.notna(row['year']) else YEARS[-1])
        cube = np.empty((len(self.bands), self.W, self.W, self.T), dtype=np.float32)
        for bi, b in enumerate(self.bands):
            is_fine = b in FINE
            if self._yearly(b):
                for ti, y in enumerate(ys):
                    a = self._arr(b, y)
                    val = a[i] if a is not None else (np.full((self.W, self.W), self.fill)
                                                      if is_fine else self.fill)
                    cube[bi, :, :, ti] = self._cell(val, b, is_fine)
            else:
                a = self._arr(b, YEARS[0])
                val = a[i] if a is not None else (np.full((self.W, self.W), self.fill)
                                                  if is_fine else self.fill)
                cube[bi] = self._cell(val, b, is_fine)[:, :, None].repeat(self.T, axis=2)
        return torch.from_numpy(cube), torch.tensor(float(row['oc']), dtype=torch.float32)

    def _cell(self, val, band, is_fine):
        """Return an [W,W] patch: fine = the real window; coarse = centre broadcast."""
        if is_fine:
            return self._norm(val, band)                       # [W,W]
        return np.full((self.W, self.W), self._norm(np.asarray(val), band), np.float32)


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SOC_DATA_DIR', '.') + '/windows/lucas'
    ds = WindowedSentinel2Dataset(root, time_before=5, window=11)
    print(f"N={len(ds)}  bands={len(ds.bands)}  fine={len(ds.fine_idx)} med={len(ds.med_idx)} coarse={len(ds.coarse_idx)}")
    x, y = ds[0]
    print(f"cube {tuple(x.shape)} (C,H,W,T) oc={y.item():.1f} finite={torch.isfinite(x).float().mean():.2f}")
    from ResolutionAwareNet import ResolutionAwareNet
    C = len(ds.bands)
    for br in ['fine', 'fine_med_coarse', 'all_flat']:
        net = ResolutionAwareNet(C, height=11, width=11, time_steps=5, branches=br,
                                 fine_idx=ds.fine_idx, med_idx=ds.med_idx, coarse_idx=ds.coarse_idx)
        xb = torch.stack([ds[k][0] for k in range(4)])
        out = net(xb)
        print(f"  branches={br:16s} in {tuple(xb.shape)} -> out {tuple(out.shape)}  params={net.count_parameters():,}")
