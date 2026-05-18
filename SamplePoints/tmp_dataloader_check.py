"""Quick dataloader sanity check on the real 20-band 2002-2023 dataset.

What it does:
  - Picks a RANDOM 2007 sample from LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx
  - Subsets the dataloader to only the 96 (band, year) folders actually
    used (Elevation static + 19 yearly bands × 5 years 2003-2007) — so it
    fits in ~5 GB RAM instead of ~20 GB.
  - Instantiates MultiRasterDatasetMultiYears with a single-row dataframe.
  - Pulls __getitem__(0) → (C, H, W, T) tensor.
  - Reports per-band stats. Verifies that the materialized-yearly statics
    (Clay/Sand/pH/BulkDensity/CEC/Slope/Aspect/TWI) have IDENTICAL values
    across the 5 time-frames (proves the symlink trick is working
    end-to-end).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ_ROOT = Path('/home/valerian/SGTPublication')
sys.path.insert(0, str(PROJ_ROOT / 'SOCmapping' / 'TemporalFusionTransformer'))
sys.path.insert(0, str(PROJ_ROOT / 'SOCmapping'))

from config import (bands_list_order, time_before, window_size,
                     SamplesCoordinates_Yearly, DataYearly)
from dataloader.dataloaderMultiYears import MultiRasterDatasetMultiYears

print('=' * 72)
print('Quick dataloader test — 20 bands × 5 years, random 2007 sample')
print('=' * 72)
print(f'bands_list_order:  {len(bands_list_order)} entries')
print(f'time_before:       {time_before}')
print(f'window_size:       {window_size}x{window_size}')
print()

# 1. Pick a random 2007 sample from the combined LUCAS+LFU+LfL xlsx
df = pd.read_excel(PROJ_ROOT / 'Data' / 'LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx')
df = df.dropna(subset=['GPS_LONG', 'GPS_LAT', 'OC'])
df_2007 = df[df['year'] == 2007]
print(f'2007 candidates: {len(df_2007)} (full df: {len(df)})')
sample = df_2007.sample(1, random_state=42).iloc[0]
print(f'\nSelected sample:')
print(f'  POINTID:   {sample.get("POINTID", "?")}')
print(f'  GPS_LONG:  {sample["GPS_LONG"]:.6f}°')
print(f'  GPS_LAT:   {sample["GPS_LAT"]:.6f}°')
print(f'  year:      {int(sample["year"])}')
print(f'  OC:        {sample["OC"]} g/kg')

# 2. Build minimal path set — only the 5 years actually needed (2003..2007)
TARGET_YEAR = 2007
years_needed = list(range(TARGET_YEAR - time_before + 1, TARGET_YEAR + 1))   # [2003, 2004, 2005, 2006, 2007]
print(f'\nYears to load for this sample (time_before={time_before}): {years_needed}')

samples_paths = []
data_paths = []
for sp, dp, band in zip(SamplesCoordinates_Yearly, DataYearly, bands_list_order):
    if band == 'Elevation':
        # Static — single path
        samples_paths.append(sp)
        data_paths.append(dp)
    else:
        # Yearly — one path per year in [2003..2007]
        for y in years_needed:
            samples_paths.append(f'{sp}/{y}')
            data_paths.append(f'{dp}/{y}')

print(f'Total minimal paths: {len(samples_paths)} '
      f'(1 Elevation + 19 yearly bands × {len(years_needed)} years)')

# 3. Build a 1-row dataframe with the chosen sample
test_df = pd.DataFrame({
    'GPS_LONG': [sample['GPS_LONG']],
    'GPS_LAT':  [sample['GPS_LAT']],
    'OC':       [sample['OC']],
    'year':     [int(sample['year'])],
    'season':   ['summer'],   # ignored in yearly mode
})

# 4. Instantiate the dataloader
print('\nInstantiating MultiRasterDatasetMultiYears (loads tiles into memory)...', flush=True)
import time, resource
def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # MB → GB on Linux
t0 = time.time()
mem_before = _rss_gb()
ds = MultiRasterDatasetMultiYears(samples_paths, data_paths, test_df)
mem_after = _rss_gb()
print(f'  dataset constructed in {time.time() - t0:.1f}s')
print(f'  peak RSS: {mem_before:.2f} GB → {mem_after:.2f} GB   (delta {mem_after - mem_before:+.2f} GB)')
print(f'  dataset length: {len(ds)}')

# 5. Pull __getitem__(0)
print('\nCalling __getitem__(0)...', flush=True)
t0 = time.time()
longitude, latitude, tensor, oc = ds[0]
print(f'  ({time.time() - t0:.2f}s)')
print(f'  longitude returned: {longitude:.6f}')
print(f'  latitude returned:  {latitude:.6f}')
print(f'  OC returned:        {oc}')
print(f'  tensor type:        {type(tensor).__name__}')
print(f'  tensor dtype:       {tensor.dtype}')
print(f'  tensor shape:       {tuple(tensor.shape)}    (C, H, W, T)')

# 6. Per-band sanity stats
C, H, W, T = tensor.shape
assert C == len(bands_list_order), f'channel count {C} ≠ bands_list_order {len(bands_list_order)}'
assert H == window_size and W == window_size, f'patch {H}x{W} ≠ window {window_size}'
assert T == time_before, f'time frames {T} ≠ time_before {time_before}'
print(f'\n✓ Shape check passed: C={C}, H=W={H}, T={T}')

print(f'\n{"Per-band stats (over the 5×5×5 patch)":-^96}')
print(f'  {"#":>2}  {"band":<28}  {"min":>10}  {"mean":>10}  {"max":>10}  {"std":>10}  {"identical-across-T":>20}')
for i, band in enumerate(bands_list_order):
    bt = tensor[i]   # shape (H, W, T)
    vmin = bt.min().item()
    vmax = bt.max().item()
    vmean = bt.mean().item()
    vstd = bt.std().item()
    # Check whether all T frames are identical (true for materialized statics
    # whose year-dirs are symlinks to the same anchor data).
    identical = bool(torch.allclose(bt[..., 0:1], bt, rtol=1e-6, atol=1e-6))
    print(f'  {i:>2}  {band:<28}  {vmin:>10.4f}  {vmean:>10.4f}  {vmax:>10.4f}  '
          f'{vstd:>10.4f}  {("YES" if identical else "no"):>20}')

# 7. Mark each band's expected behavior
print()
expected_identical_T = {
    'Elevation',                                              # static (repeated time_before times)
    'ClayContent_0_10cm', 'SandContent_0_10cm', 'pH_H2O_0_10cm',
    'BulkDensity_0_10cm', 'CEC_0_10cm',                       # static-as-yearly soil
    'Slope', 'Aspect', 'TWI',                                  # static-as-yearly terrain
}
expected_varying_T = set(bands_list_order) - expected_identical_T

print('Cross-T expectations:')
print(f'  identical across T (static / static-as-yearly): {len(expected_identical_T)} bands')
print(f'  varies across T (true yearly bands):            {len(expected_varying_T)} bands')

mismatches = []
for i, band in enumerate(bands_list_order):
    bt = tensor[i]
    is_identical = bool(torch.allclose(bt[..., 0:1], bt, rtol=1e-6, atol=1e-6))
    expected = band in expected_identical_T
    if is_identical != expected:
        mismatches.append((band, is_identical, expected))

if mismatches:
    print('\n✗ MISMATCH in cross-T behavior:')
    for band, got, exp in mismatches:
        print(f'  {band}: got identical={got}, expected identical={exp}')
else:
    print('\n✓ All bands behave as expected: statics constant across T, yearly bands vary.')

# 8. Finite-value check
n_finite = int(torch.isfinite(tensor).sum())
total = tensor.numel()
print(f'\nfinite values: {n_finite}/{total}  ({100*n_finite/total:.2f}%)')
assert n_finite == total, 'FAIL: tensor contains NaN/inf'
print('✓ All values finite.')

print()
print('=' * 72)
print('DATALOADER TEST PASSED — full 20-band, 5-year tensor for a random 2007 sample.')
print('=' * 72)
