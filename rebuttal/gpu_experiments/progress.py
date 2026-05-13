#!/usr/bin/env python3
"""
progress.py — live progress monitor for in-flight rebuttal GPU runs.

Reads the per-worker log files written by mc_dropout_inference.py and
run_kfold.py, parses the most recent progress line per worker, and
prints per-shard / per-fold status plus an aggregate %.

Usage (from any terminal — does not need to share state with the run):

    # snapshot (one-shot)
    python rebuttal/gpu_experiments/progress.py

    # auto-refresh every 5 s
    python rebuttal/gpu_experiments/progress.py --watch
    python rebuttal/gpu_experiments/progress.py --watch --interval 10

    # only one experiment
    python rebuttal/gpu_experiments/progress.py --mc
    python rebuttal/gpu_experiments/progress.py --kfold

No third-party dependencies — pure stdlib so it works without the venv.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

# Resolve worker-log dirs relative to this file's location
_THIS = Path(__file__).resolve()
_BASE = _THIS.parent
MC_LOG_GLOB = str(_BASE / 'uncertainty' / 'worker_logs' / 'shard_*.log')
KFOLD_LOG_GLOB = str(_BASE / 'spatial_kfold' / 'worker_logs' / 'fold_*.log')

# Regexes for the progress signals each script emits
RE_MC_BATCH = re.compile(r'batch (\d+)/(\d+)\s+point (\d+)/(\d+)\s+elapsed ([\d.]+)s\s+ETA (\d+)s')
RE_KFOLD_EPOCH = re.compile(r'Fold (\d+) \| Epoch (\d+)/(\d+) \| train_loss=([\d.]+) \| val_R²=([-\d.]+) \| val_RMSE=([\d.]+)')


def _tail(path: str, max_bytes: int = 16384) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            return f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def _detect_phase(txt: str) -> str:
    if 'wrote _shard_' in txt or 'wrote fold_' in txt or ('wrote ' in txt and 'results.pkl' in txt):
        return 'done'
    if 'Streaming' in txt or 'Total batches' in txt:
        # Streaming started but no batch_N/M line yet → first batch hasn't
        # completed. Use a distinct phase so the print code doesn't try to
        # read batches_done / batches_total fields that aren't populated.
        return 'mc_starting'
    if 'points in shard' in txt:
        return 'loading_data'
    if 'recomputing' in txt or 'survey_date' in txt:
        return 'norm_recompute'
    if txt.strip():
        return 'starting'
    return 'launching'


def status_mc(globpat: str = MC_LOG_GLOB) -> list[dict]:
    logs = sorted(glob.glob(globpat))
    rows = []
    for f in logs:
        txt = _tail(f)
        short = os.path.basename(f)
        matches = list(RE_MC_BATCH.finditer(txt))
        if matches:
            m = matches[-1]
            rows.append({
                'file': short, 'kind': 'shard',
                'batches_done': int(m.group(1)),
                'batches_total': int(m.group(2)),
                'points_done': int(m.group(3)),
                'points_total': int(m.group(4)),
                'elapsed_s': float(m.group(5)),
                'eta_s': int(m.group(6)),
                'phase': 'mc_sampling',
            })
        else:
            rows.append({'file': short, 'kind': 'shard',
                         'phase': _detect_phase(txt)})
    return rows


def status_kfold(globpat: str = KFOLD_LOG_GLOB) -> list[dict]:
    logs = sorted(glob.glob(globpat))
    rows = []
    for f in logs:
        txt = _tail(f)
        short = os.path.basename(f)
        matches = list(RE_KFOLD_EPOCH.finditer(txt))
        if matches:
            m = matches[-1]
            rows.append({
                'file': short, 'kind': 'fold',
                'fold_id': int(m.group(1)),
                'epoch': int(m.group(2)),
                'epoch_total': int(m.group(3)),
                'train_loss': float(m.group(4)),
                'val_r2': float(m.group(5)),
                'val_rmse': float(m.group(6)),
                'phase': 'training',
            })
        else:
            rows.append({'file': short, 'kind': 'fold',
                         'phase': _detect_phase(txt)})
    return rows


def _print_mc(rows: list[dict]) -> None:
    print('=' * 70)
    print(f'MC dropout inference  ({len(rows)} shards)')
    print('-' * 70)
    if not rows:
        print('  (no worker logs found — has the script started?)')
        return
    total_d = total_t = 0
    for r in rows:
        # Only treat as mc_sampling if we actually parsed a batch_N/M line.
        if r['phase'] == 'mc_sampling' and 'batches_done' in r:
            d, t = r['batches_done'], r['batches_total']
            total_d += d; total_t += t
            print(f'  {r["file"]:<22}  batch {d:>4}/{t:<4}  '
                  f'({100*d/t:5.1f}%)  ETA {r["eta_s"]}s')
        else:
            tag = {
                'norm_recompute': 'recomputing normalisation stats',
                'loading_data':   'building hashmap / loading dataset',
                'mc_starting':    'streaming started, first batch in flight',
                'launching':      'launching / importing',
                'starting':       'starting up',
                'done':           'DONE ✓',
            }.get(r['phase'], r['phase'])
            print(f'  {r["file"]:<22}  [{tag}]')
    if total_t:
        print('-' * 70)
        print(f'  AGGREGATE          {total_d:>5}/{total_t:<5}  '
              f'({100*total_d/total_t:5.1f}%)')


def _print_kfold(rows: list[dict]) -> None:
    print('=' * 70)
    print(f'Spatial k-fold CV  ({len(rows)} fold workers)')
    print('-' * 70)
    if not rows:
        print('  (no worker logs found)')
        return
    total_d = total_t = 0
    for r in rows:
        if r['phase'] == 'training' and 'epoch' in r:
            e, et = r['epoch'], r['epoch_total']
            total_d += e; total_t += et
            print(f'  {r["file"]:<22}  fold {r["fold_id"]}  '
                  f'epoch {e:>3}/{et:<3} ({100*e/et:5.1f}%)  '
                  f'val R²={r["val_r2"]:.4f}  RMSE={r["val_rmse"]:.3f}')
        else:
            tag = {
                'norm_recompute': 'normalisation stats',
                'loading_data':   'building hashmap / loading dataset',
                'mc_starting':    'first batch in flight',
                'launching':      'launching / importing',
                'starting':       'starting up',
                'done':           'DONE ✓',
            }.get(r['phase'], r['phase'])
            print(f'  {r["file"]:<22}  [{tag}]')
    if total_t:
        print('-' * 70)
        print(f'  AGGREGATE          {total_d:>4}/{total_t:<4} epochs  '
              f'({100*total_d/total_t:5.1f}%)')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--watch', action='store_true',
                   help='Re-print every --interval seconds until Ctrl+C')
    p.add_argument('--interval', type=int, default=5,
                   help='Refresh interval for --watch (seconds, default 5)')
    p.add_argument('--mc', action='store_true', help='Only show MC dropout')
    p.add_argument('--kfold', action='store_true', help='Only show k-fold')
    args = p.parse_args()

    def render():
        if args.watch:
            # ANSI clear screen
            sys.stdout.write('\x1b[2J\x1b[H')
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'[progress.py  {ts}]')
        if args.kfold and not args.mc:
            _print_kfold(status_kfold())
        elif args.mc and not args.kfold:
            _print_mc(status_mc())
        else:
            _print_mc(status_mc())
            print()
            _print_kfold(status_kfold())
        sys.stdout.flush()

    if args.watch:
        try:
            while True:
                render()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('\n(stopped)')
    else:
        render()


if __name__ == '__main__':
    main()
