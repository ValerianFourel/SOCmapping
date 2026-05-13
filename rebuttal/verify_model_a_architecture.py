#!/usr/bin/env python3
"""
verify_model_a_architecture.py — definitive test of which EnhancedSGT
configuration produced the Model A checkpoint (`…_R2_0.6909.pth`).

The state_dict alone is ambiguous on `num_heads`: nn.MultiheadAttention
has the same `in_proj_weight` shape (3*d_model, d_model) regardless of
how many heads d_model is split into. So we can't read `num_heads` from
the saved weights directly.

The definitive test: re-run inference on a subset of the saved val rows
through each candidate `num_heads`, compare against the predictions
recorded in `analysis_results.pkl`. Only the correct `num_heads` value
will reproduce the saved predictions (the model was trained with that
head dimension; any other splitting produces a different forward pass
on the same weights).

Result (run on 2026-05-13):
    num_heads=4 → max abs diff 2e-6 → ✅ MATCH
    num_heads=8 → max abs diff 0.67 g/kg → ❌

Combined with the unambiguous state_dict facts:
    transformer_encoder has layers [0, 1, 2]  → num_encoder_layers = 3
    grn_blocks has indices [0, 1, 2]          → 3 GRN blocks (EnhancedSGT)
    spatial_encoder.0/1/3/4/6/7              → 3 (Conv2d, BatchNorm2d) pairs

→ Model A is EnhancedSGT(d_model=128, num_heads=4, num_encoder_layers=3,
                         dropout=0.3, expansion_factor=4)
  = EnhancedSGT CLASS DEFAULTS at d_model=128.

Wandb logged `num_heads=8, num_layers=2` from the CLI args at training
time, but the train.py code path at that commit called
`EnhancedSGT(d_model=hidden_size)` without forwarding the args — so the
class defaults are what the model actually used. The wandb config field
captures intent (CLI), not realisation (model).
"""
from __future__ import annotations
import pickle
import sys
import numpy as np
import torch
from pathlib import Path

SGT_DIR = Path('/home/valerian/SGTPublication/SOCmapping/SpatiotemporalGatedTransformer')
sys.path.insert(0, str(SGT_DIR))
sys.path.insert(0, str(SGT_DIR / 'dataloader'))

from EnhancedSGT import EnhancedSGT
from dataloaderMultiYears import MultiRasterDatasetMultiYears
from dataframe_loader import filter_dataframe, separate_and_add_data
from config import TIME_BEGINNING, TIME_END, MAX_OC, time_before

PTH = (
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/'
    'TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/'
    'TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_'
    'TRANSFORM_normalize_LOSS_composite_l2_R2_0.6909.pth'
)
PKL = (
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/'
    'Archive/residual_analysis1mil_normalize_composite_l2_v2_TemporalFusionTransformer/'
    'analysis_results.pkl'
)

N_TEST = 32
CANDIDATES = (4, 8)        # num_heads to test
N_LAYERS = 3               # forced by state_dict


def main():
    # Load checkpoint and strip DDP "module." prefix
    ckpt = torch.load(PTH, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ckpt['model_state_dict'].items()}

    # Normalization stats live in the analysis_results.pkl, not the .pth
    analysis = pickle.load(open(PKL, 'rb'))
    s = analysis['stats']
    target_mean, target_std = s['target_mean'], s['target_std']
    feature_means, feature_stds = s['feature_means'], s['feature_stds']

    val_preds_saved = np.asarray(analysis['val_results']['predictions'], dtype=float)
    val_lon = np.asarray(analysis['val_results']['longitudes'], dtype=float)
    val_lat = np.asarray(analysis['val_results']['latitudes'], dtype=float)

    # Rebuild the val dataframe by joining (lon, lat) against the master filter
    df_full = filter_dataframe(TIME_BEGINNING, TIME_END, MAX_OC)
    df_full['_key'] = list(zip(np.round(df_full['GPS_LONG'], 6),
                               np.round(df_full['GPS_LAT'], 6)))
    val_keys = set(zip(np.round(val_lon, 6), np.round(val_lat, 6)))
    val_df = (df_full[df_full['_key'].isin(val_keys)]
              .drop(columns=['_key']).reset_index(drop=True))
    print(f'val df reconstructed: {len(val_df)} rows')

    sample_paths, data_paths = separate_and_add_data()

    def flatten(lst):
        out = []
        for x in lst:
            if isinstance(x, list):
                out += flatten(x)
            else:
                out.append(x)
        return out

    sample_paths = list(dict.fromkeys(flatten(sample_paths)))
    data_paths = list(dict.fromkeys(flatten(data_paths)))

    ds = MultiRasterDatasetMultiYears(
        sample_paths, data_paths,
        val_df.head(N_TEST).reset_index(drop=True),
        time_before=time_before,
    )

    # Map first N_TEST test points back to saved-val indices
    test_lon = val_df.head(N_TEST)['GPS_LONG'].to_numpy()
    test_lat = val_df.head(N_TEST)['GPS_LAT'].to_numpy()
    idx_in_saved = np.array([
        (np.where((np.round(val_lon, 6) == round(lo, 6))
                  & (np.round(val_lat, 6) == round(la, 6)))[0][:1].tolist() or [-1])[0]
        for lo, la in zip(test_lon, test_lat)
    ])
    matched = idx_in_saved >= 0
    ref_preds = val_preds_saved[idx_in_saved[matched]]

    print(f'\n--- Disambiguating num_heads (num_encoder_layers fixed at {N_LAYERS}) ---')
    for nh in CANDIDATES:
        m = EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5,
                        d_model=128, num_heads=nh, dropout=0.3,
                        num_encoder_layers=N_LAYERS, expansion_factor=4)
        m.load_state_dict(sd, strict=True)
        m.eval()

        feats = []
        for i in range(N_TEST):
            _, _, f, _ = ds[i]
            f_norm = (f - feature_means[:, None, None]) / feature_stds[:, None, None]
            feats.append(f_norm)
        x = torch.stack(feats).float()
        with torch.no_grad():
            out = m(x).cpu().numpy()
        preds = out * target_std + target_mean
        preds_matched = preds[matched]
        diff = np.abs(preds_matched - ref_preds)
        verdict = '✅ MATCH' if diff.max() < 1e-3 else '❌ MISMATCH'
        print(f'  num_heads={nh}: max abs diff = {diff.max():.6f}  '
              f'mean = {diff.mean():.6f}  → {verdict}')


if __name__ == '__main__':
    main()
