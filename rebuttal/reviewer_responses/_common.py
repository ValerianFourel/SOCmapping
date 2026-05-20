"""
_common.py — shared defaults and helpers for the reviewer-response scripts.

Every script in this folder targets the same flagship + comparison set,
defined here as a single source of truth. Edit FLAGSHIP / COMPARISONS to
re-target the entire batch.
"""
from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]
REBUTTAL = SOC_ROOT / 'rebuttal'
SWEEP_DIR = REBUTTAL / 'gpu_experiments' / 'spatial_kfold' / 'sweep'
RESULTS_DIR = HERE / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The model-ready dataset (16,514 LUCAS+LfL+LfU samples with altitude).
# Built by rebuttal/temporal_regression_corrected.py and reused by every
# spatial-CV run via run_kfold._build_model_ready_dataset.
MODEL_READY = REBUTTAL / 'model_ready_dataset.parquet'


# ---------------------------------------------------------------------------
# Architecture catalog. Each entry IS a row in the comparison tables.
# `nice` is the publication-ready label; `desc` is the one-line description
# for the response letter; `params` is the trained parameter count (loaded
# from a checkpoint if available, else None — populated lazily).
# ---------------------------------------------------------------------------
@dataclass
class ArchEntry:
    tag: str
    group: str
    family: str
    nice: str
    desc: str
    is_flagship: bool = False


# The flagship is FIRST in every table.
FLAGSHIP = ArchEntry(
    tag='vanilla_transformer_d128_h4_L1',
    group='oc150_vanilla',
    family='vanilla_transformer',
    nice='Vanilla (CNN + Transformer, 215k params)',
    desc='New recommended architecture. SimpleSGT minus the GRN gate. '
         'CNN spatial encoder + 1-layer Transformer encoder + MLP head.',
    is_flagship=True,
)

# Comparison set, ordered for the table. Each line is "what the row tells
# us". 3DCNN is included as the failed family (negative R²) for
# completeness — drop with --no-broken if you want a clean table.
COMPARISONS = [
    ArchEntry(tag='small_d128_h4_L1', group='oc150', family='sgt-small',
              nice='SGT (CNN + GRN + Transformer, 363k params)',
              desc='Original architecture. Gating ablation: SGT ≈ Vanilla, '
                   'so the +148k params for the GRN buy nothing.'),
    ArchEntry(tag='lightweight_transformer_d128_h4_L1', group='oc150',
              family='lightweight_transformer',
              nice='Lightweight @ 30 epochs (Transformer only, 240k params)',
              desc='Vanilla minus the CNN frontend, undertrained at 30 epochs. '
                   'R² mean = -0.064. Compare to the long-train row below.'),
    ArchEntry(tag='simpletransformer_d64_h4_L1', group='oc150',
              family='simpletransformer',
              nice='SimpleTransformer (Transformer only, 11.2M params)',
              desc='Reference transformer-alone at large scale. Achieves '
                   'Vanilla-equivalent R² at 50× the parameters.'),
    ArchEntry(tag='cnnlstm_d64_h4_L1', group='oc150', family='cnnlstm',
              nice='CNNLSTM (CNN + LSTM, 93k params)',
              desc='Recurrent counterpart to Vanilla. Slightly worse mean R², '
                   'fails on Alpine fold.'),
    ArchEntry(tag='3dcnn_d64_h4_L1', group='oc150', family='3dcnn',
              nice='3DCNN (CNN only)',
              desc='Failed family: R² ≈ -0.76 across all fold/band combinations.'),
    ArchEntry(tag='baseline_rf_default', group='oc150', family='baseline_rf',
              nice='Random Forest (80-d per-band stats)',
              desc='Classical baseline. Mid-table R², fails on Alpine fold.'),
    ArchEntry(tag='baseline_xgb_shallow', group='oc150', family='baseline_xgb',
              nice='XGBoost (80-d per-band stats)',
              desc='Classical baseline. R² ≈ RF; same Alpine-fold failure.'),
]


def all_entries(include_broken: bool = True,
                 include_longtrain: bool = False) -> list[ArchEntry]:
    """Flagship first, comparisons after. Optionally exclude 3DCNN (broken)
    and/or include the 200-epoch defensive long-train Lightweight result
    as an additional comparison row."""
    out = [FLAGSHIP] + COMPARISONS
    if include_longtrain:
        # Insert right after the 30-epoch Lightweight row so the comparison
        # is immediately visible in the table.
        long_entry = ArchEntry(
            tag='lightweight_transformer_d128_h4_L1',
            group='oc150_longtrain',
            family='lightweight_transformer',
            nice='Lightweight @ 200 epochs (Transformer only, 240k params)',
            desc='Long-training defensive run. Converged at median best-epoch '
                 '52/200, with overfitting after. R² mean = +0.149, σ = 0.128 '
                 '— the architectural ceiling. Still trails Vanilla by +0.021 R² '
                 'and -0.169 R² on the Alpine fold even at 6.6× more training.',
        )
        # Insert just after the 30-epoch Lightweight entry
        for i, e in enumerate(out):
            if (e.family == 'lightweight_transformer'
                    and e.group == 'oc150'
                    and e.tag == 'lightweight_transformer_d128_h4_L1'):
                out = out[:i + 1] + [long_entry] + out[i + 1:]
                break
        else:
            out.append(long_entry)
    if not include_broken:
        out = [e for e in out if e.family != '3dcnn']
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def cfg_dir(entry: ArchEntry) -> Path:
    return SWEEP_DIR / entry.group / entry.tag


def load_summary(entry: ArchEntry) -> dict | None:
    p = cfg_dir(entry) / 'kfold_results_summary.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_fold_predictions(entry: ArchEntry) -> 'pandas.DataFrame | None':
    """Concatenate per-fold prediction parquets if they exist.

    Returns a DataFrame with columns at minimum: longitude/latitude/predicted/actual/fold.
    Returns None if no per-fold parquet exists for this config.
    """
    import pandas as pd
    parts = []
    d = cfg_dir(entry)
    for pq in sorted(d.glob('fold_*_predictions.parquet')):
        try:
            df = pd.read_parquet(pq)
            # Extract fold id from filename: fold_<i>_predictions.parquet
            m = re.search(r'fold_(\d+)_predictions', pq.name)
            if m:
                df['fold'] = int(m.group(1))
            parts.append(df)
        except Exception as e:
            print(f'[warn] could not read {pq}: {e}', file=sys.stderr)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def load_count_params(entry: ArchEntry) -> int | None:
    """Sum state_dict tensor elements from any one fold's best .pth."""
    d = cfg_dir(entry)
    for pth in sorted(d.glob('fold_*_best.pth')):
        try:
            import torch
            ck = torch.load(str(pth), map_location='cpu', weights_only=False)
            sd = ck.get('model_state_dict', ck.get('state_dict', ck))
            if not isinstance(sd, dict):
                continue
            return sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
        except Exception:
            continue
    return None


def write_pair(name: str, data_json: dict, md: str) -> None:
    """Write both .json and .md to results/ under the same stem."""
    (RESULTS_DIR / f'{name}.json').write_text(
        json.dumps(data_json, indent=2, default=str))
    (RESULTS_DIR / f'{name}.md').write_text(md)
    print(f'[write] {RESULTS_DIR / f"{name}.json"}')
    print(f'[write] {RESULTS_DIR / f"{name}.md"}')


def fmt_pm(mean: float | None, ci_lo: float | None, ci_hi: float | None,
           prec: int = 3) -> str:
    """Format mean [lo, hi] as 'M.MMM [L.LLL, H.HHH]'."""
    if mean is None:
        return '—'
    s = f'{mean:+.{prec}f}'
    if ci_lo is not None and ci_hi is not None:
        s += f' [{ci_lo:+.{prec}f}, {ci_hi:+.{prec}f}]'
    return s


def banner(title: str, width: int = 80) -> str:
    return '=' * width + f'\n  {title}\n' + '=' * width
