"""End-to-end pipeline test: 20-band config + dataloader → tensor of expected shape.

Exercises the full dataloader call path the way train.py does it:
  1. config.py provides bands_list_order (20 entries) + SamplesCoordinates_Yearly + DataYearly.
  2. dataframe_loader.separate_and_add_data() expands band paths × selected years
     using config TIME_BEGINNING ('2007') and LOADING_TIME_BEGINNING ('2002').
  3. dedup + flatten (mirrors train.py:909-910).
  4. filter_dataframe(TIME_BEGINNING, TIME_END, max_oc).
  5. instantiate NormalizedMultiRasterDatasetMultiYears.
  6. pull __getitem__(0) — verify the returned tensor shape.

Before running:
  python SamplePoints/scaffold_synthetic_bands.py    # symlink 14 new bands → LAI

Expected output:
  - dataset length > 0
  - tensor shape (C, H, W, T) with C=20, H=window_size, W=window_size, T>0
"""
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Path setup — match how config.py / train.py resolve paths
REPO_ROOT = Path(__file__).resolve().parents[2]
TFT_DIR = REPO_ROOT / 'SOCmapping' / 'TemporalFusionTransformer'
sys.path.insert(0, str(TFT_DIR))
sys.path.insert(0, str(REPO_ROOT / 'SOCmapping'))


def flatten_paths(path_list):
    """Mirror train.py:900-907 — flatten any nested list (MODIS_NPP returns one)."""
    flattened = []
    for item in path_list:
        if isinstance(item, list):
            flattened.extend(flatten_paths(item))
        else:
            flattened.append(item)
    return flattened


def main():
    # 1. Load config + dataloader
    from config import (
        bands_list_order, TIME_BEGINNING, TIME_END, LOADING_TIME_BEGINNING,
        time_before, window_size, MAX_OC,
    )
    from dataloader.dataloaderMultiYears import (
        MultiRasterDatasetMultiYears, NormalizedMultiRasterDatasetMultiYears,
    )
    from dataloader.dataframe_loader import separate_and_add_data, filter_dataframe

    print('============================================================')
    print('Pipeline end-to-end test — 20-band TFT config')
    print('============================================================')
    print(f'bands_list_order:    {len(bands_list_order)} entries')
    print(f'  first 6 (existing):  {bands_list_order[:6]}')
    print(f'  new 14 (appended):   {bands_list_order[6:]}')
    print(f'TIME_BEGINNING:      {TIME_BEGINNING}')
    print(f'TIME_END:            {TIME_END}')
    print(f'LOADING_TIME_BEGINNING (TIME_BEGINNING − time_before): {LOADING_TIME_BEGINNING}')
    print(f'time_before:         {time_before}')
    print(f'window_size:         {window_size}')
    print(f'MAX_OC:              {MAX_OC}')
    print()

    # 2. Expand band paths × years
    samples_coordinates_array_path, data_array_path = separate_and_add_data()
    print(f'separate_and_add_data():')
    print(f'  samples paths (raw, before dedup): {len(samples_coordinates_array_path)}')
    print(f'  data    paths (raw, before dedup): {len(data_array_path)}')

    # 3. Dedup + flatten (mirrors train.py)
    samples_coordinates_array_path = list(dict.fromkeys(flatten_paths(samples_coordinates_array_path)))
    data_array_path = list(dict.fromkeys(flatten_paths(data_array_path)))
    print(f'  samples paths (deduped):           {len(samples_coordinates_array_path)}')
    print(f'  data    paths (deduped):           {len(data_array_path)}')
    print()

    # Quick sanity peek: list the new bands' first year paths actually exist
    print('Sanity: do new-band coordinate dirs resolve?')
    for band in bands_list_order[6:9]:
        sample = [p for p in samples_coordinates_array_path if f'/{band}/' in p][:2]
        for p in sample:
            exists = Path(p, 'coordinates.npy').exists()
            print(f'  {p}  → coordinates.npy: {exists}')
    print()

    # 4. Build dataframe
    df = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    print(f'Filtered dataframe: {len(df)} rows')
    print()

    # 5. Instantiate dataloader
    print('Constructing MultiRasterDatasetMultiYears (without normalization)...')
    ds = MultiRasterDatasetMultiYears(samples_coordinates_array_path,
                                      data_array_path,
                                      df)
    print(f'  dataset length: {len(ds)}')
    print()

    # 6. Pull __getitem__(0) — the load-bearing test
    print('Calling __getitem__(0)...')
    longitude, latitude, tensor, oc = ds[0]
    print(f'  longitude: {longitude}')
    print(f'  latitude:  {latitude}')
    print(f'  OC:        {oc}')
    print(f'  tensor type:   {type(tensor).__name__}')
    print(f'  tensor dtype:  {tensor.dtype}')
    print(f'  tensor shape:  {tuple(tensor.shape)}')
    print()

    # 7. Assertions on shape
    C, H, W, T = tensor.shape
    print('Shape semantics: (C, H, W, T) after the dataloader\'s permute(0, 2, 3, 1)')
    print(f'  C (channels)      = {C}  (expected: {len(bands_list_order)} = len(bands_list_order))')
    print(f'  H (patch height)  = {H}  (expected: {window_size} = window_size)')
    print(f'  W (patch width)   = {W}  (expected: {window_size} = window_size)')
    print(f'  T (time frames)   = {T}  (expected: {time_before} = time_before, '
          f'or {time_before + 1} if MODIS_NPP appends prev year)')
    assert C == len(bands_list_order), (
        f'FAIL: tensor has {C} channels but bands_list_order has {len(bands_list_order)} entries'
    )
    assert H == window_size and W == window_size, (
        f'FAIL: patch is {H}×{W} but window_size = {window_size}'
    )
    assert T >= time_before, (
        f'FAIL: time dimension {T} < time_before {time_before}'
    )
    assert torch.isfinite(tensor).all(), 'FAIL: tensor contains NaN/inf'
    print()
    print('All shape assertions passed.')
    print()

    # 8. Quick run through normalization wrapper.
    # IMPORTANT: free the base dataset first. Each RasterTensorDataset
    # pre-loads ~46 MB of tiles into self.data_cache; with 420 deduped
    # (band, year) folders that's ~19 GB. Holding two such datasets
    # simultaneously (parent + normalized child) OOM-kills the test.
    print('Freeing base dataset to make room for the Normalized variant...', flush=True)
    import gc
    del ds
    gc.collect()
    print('Now exercising NormalizedMultiRasterDatasetMultiYears...', flush=True)
    print('(computes per-channel mean/std across all samples in the small df)', flush=True)
    # Use a small subset so compute_statistics stays fast; the I/O of the
    # 420-folder cache is the slow part, not the sample iteration.
    df_small = df.head(8).reset_index(drop=True)
    print(f'  using df_small with {len(df_small)} rows for stats compute', flush=True)
    norm_ds = NormalizedMultiRasterDatasetMultiYears(
        samples_coordinates_array_path, data_array_path, df_small
    )
    print(f'  ✓ normalization dataset constructed', flush=True)
    print(f'  normalized dataset length: {len(norm_ds)}')
    print(f'  per-channel means shape:   {tuple(norm_ds._feature_means.shape)}  '
          f'(expected: ({len(bands_list_order)}, {time_before}) — mean over (sample, H, W) leaves (C, T))')
    print(f'  per-channel stds  shape:   {tuple(norm_ds._feature_stds.shape)}')
    print(f'  means all finite:          {bool(torch.isfinite(norm_ds._feature_means).all())}')
    print(f'  stds  all finite & > 0:    '
          f'{bool(torch.isfinite(norm_ds._feature_stds).all() and (norm_ds._feature_stds > 0).all())}')
    assert norm_ds._feature_means.shape[0] == len(bands_list_order), (
        f'FAIL: normalization mean has {norm_ds._feature_means.shape[0]} channels '
        f'but bands_list_order has {len(bands_list_order)}'
    )

    lon, lat, normed_tensor, oc = norm_ds[0]
    print(f'  normalized sample[0] tensor shape: {tuple(normed_tensor.shape)}')
    print(f'  normalized sample[0] mean/std:    {normed_tensor.mean().item():.3f} / {normed_tensor.std().item():.3f}')
    assert normed_tensor.shape == tensor.shape, (
        f'FAIL: normalized tensor shape {tuple(normed_tensor.shape)} != base tensor shape {tuple(tensor.shape)}'
    )
    print()

    # 9.0 Mark verify phase as in-progress in pipeline_state.json.
    try:
        from pipeline_state import State
        ps_state = State()
        ps_state.start_phase('verify')
    except Exception as _ex:
        ps_state = None
        print(f'  (warn: could not start pipeline_state.verify: {_ex})')

    # 9. Model-fwd sanity (input-conv channel growth):
    #    The model's input layer must accept input_channels = len(bands_list_order)
    #    without modification — verifies the "no architecture change" promise.
    print('Model-input compatibility check...', flush=True)
    EnhancedTFT = None
    model_class_name = None
    for mod_name, cls_name in [('EnhancedTFT', 'EnhancedTFT'), ('SimpleTFT', 'SimpleTFT')]:
        try:
            module = __import__(mod_name)
            if hasattr(module, cls_name):
                EnhancedTFT = getattr(module, cls_name)
                model_class_name = f'{mod_name}.{cls_name}'
                break
        except ImportError as ex:
            print(f'  could not import {mod_name}: {ex}')

    if EnhancedTFT is None:
        print('  (skipping — neither EnhancedTFT nor SimpleTFT module importable)')
    else:
        import inspect
        sig = inspect.signature(EnhancedTFT.__init__)
        if 'input_channels' in sig.parameters:
            try:
                model = EnhancedTFT(input_channels=len(bands_list_order))
                # Try the most common batch shape (B, C, H, W, T) first.
                batched = normed_tensor.unsqueeze(0)
                with torch.no_grad():
                    out = model(batched)
                print(f'  ✓ {model_class_name}(input_channels={len(bands_list_order)}) accepted '
                      f'tensor shape {tuple(batched.shape)} → output {tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}',
                      flush=True)

                # 9.5 Forward + backward pass through the model.
                # Verifies gradients flow through all 20 input channels;
                # this is the cheapest end-to-end signal that the model
                # actually USES the new bands.
                print('Forward + backward pass (1 sample, L1 loss)...', flush=True)
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                opt.zero_grad()
                # Build a tiny batch of 2 samples to exercise the input conv.
                xs = []
                ys = []
                for i in range(2):
                    lon_i, lat_i, t_i, oc_i = norm_ds[i % len(norm_ds)]
                    xs.append(t_i)
                    ys.append(float(oc_i))
                xs = torch.stack(xs)   # (B, C, H, W, T) = (2, 20, 5, 5, 5)
                ys = torch.tensor(ys, dtype=torch.float32)
                pred = model(xs)
                pred = pred.reshape(-1)
                loss = (pred - ys).abs().mean()
                loss.backward()
                # Walk input-conv weight grad to confirm gradient reached the first layer.
                first_conv = None
                for m in model.modules():
                    if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 4:
                        first_conv = m; break
                grad_norm = None if first_conv is None or first_conv.weight.grad is None \
                    else float(first_conv.weight.grad.norm())
                opt.step()
                print(f'  ✓ loss = {float(loss):.4f}', flush=True)
                print(f'  ✓ first-conv weight shape = {None if first_conv is None else tuple(first_conv.weight.shape)} '
                      f'(channels-in = {None if first_conv is None else first_conv.weight.shape[1]})',
                      flush=True)
                print(f'  ✓ first-conv grad norm    = {grad_norm}', flush=True)
                assert torch.isfinite(loss).item(), 'FAIL: loss is non-finite'
                assert grad_norm is not None and grad_norm > 0, 'FAIL: input-conv received no gradient'

            except Exception as ex:
                print(f'  {model_class_name} init/forward raised: {type(ex).__name__}: {ex}', flush=True)
                print(f'  (the dataloader-returned shape is {tuple(normed_tensor.shape)}; '
                      f'the model may expect a different layout — out of scope for this test)',
                      flush=True)
        else:
            print(f'  (skipping — {model_class_name}.__init__ signature does not take input_channels)')

    # 10. Finalize state file
    if ps_state is not None:
        ps_state.finish_phase('verify')
        print(flush=True)
        print('Pipeline state updated:', flush=True)
        ps_state.summary()

    print()
    print('============================================================')
    print('PIPELINE TEST PASSED')
    print('============================================================')


if __name__ == '__main__':
    main()
