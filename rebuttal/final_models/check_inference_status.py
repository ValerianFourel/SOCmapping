#!/usr/bin/env python3
"""
check_inference_status.py — status checker for the Bavaria inference sbatch.

Reports:
  1. Slurm job state (via sacct, if available)
  2. Whether the log file reached the "[infer-all] all done" marker
  3. Per-run map status: ok / broken_nan / no_output / no_ckpt
  4. Tail of the latest log

Modes:
  --once         single snapshot then exit (default)
  --watch        loop every 30 s until completion marker fires (Ctrl-C to stop)
  --quiet        exit 0 if all expected runs are 'ok', else exit 1.
                 useful in shell pipelines.

Examples:
  python rebuttal/final_models/check_inference_status.py
  python rebuttal/final_models/check_inference_status.py --watch
  python rebuttal/final_models/check_inference_status.py --quiet && \
      python rebuttal/final_models/compare_maps.py
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE / 'maps'
CHECKPOINTS_ROOT = HERE / 'checkpoints'

# Must mirror NN_RUNS + TREE_RUNS in run_all_inferences.sbatch.
EXPECTED_RUNS: list[tuple[str, str]] = [
    ('sgt_d128_h4_L1_20band',                    'NN'),
    ('sgt_d128_h4_L1_6band',                     'NN'),
    ('vanilla_transformer_d128_h4_L1_20band',    'NN'),
    ('vanilla_transformer_d128_h4_L1_6band',     'NN'),
    ('lightweight_transformer_d128_h4_L1_20band', 'NN'),
    ('lightweight_transformer_d128_h4_L1_6band',  'NN'),
    ('simpletransformer_d64_h4_L1_20band',       'NN'),
    ('simpletransformer_d64_h4_L1_6band',        'NN'),
    ('cnnlstm_d64_h4_L1_20band',                 'NN'),
    ('cnnlstm_d64_h4_L1_6band',                  'NN'),
    ('3dcnn_d64_h4_L1_20band',                   'NN'),
    ('3dcnn_d64_h4_L1_6band',                    'NN'),
    ('xgb_shallow_20band',                       'tree'),
    ('xgb_shallow_6band',                        'tree'),
    ('rf_default_20band',                        'tree'),
    ('rf_default_6band',                         'tree'),
]


def find_latest_log() -> Path | None:
    """The newest infer_all_*.out file in MAPS_ROOT, if any."""
    logs = sorted(MAPS_ROOT.glob('infer_all_*.out'),
                   key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors='replace')
    m = re.search(r'job=(\d+)', text)
    jobid = m.group(1) if m else None
    done = '[infer-all] all done' in text
    # Count how many "=== <run> ===" markers fired vs how many [skip]/[warn]
    started = re.findall(r'^=== (\S+) ===', text, re.MULTILINE)
    skipped = re.findall(r'^\[skip\] (\S+)', text, re.MULTILINE)
    warned = re.findall(r'^\[warn\] (\S+) failed', text, re.MULTILINE)
    return {
        'log': log_path,
        'jobid': jobid,
        'done': done,
        'started': started,
        'skipped': skipped,
        'warned': warned,
        'tail': text.splitlines()[-30:],
    }


def slurm_state(jobid: str) -> str | None:
    """sacct lookup for the given Slurm job ID; returns one-line state or None."""
    if not jobid:
        return None
    try:
        out = subprocess.run(
            ['sacct', '-j', jobid, '-X', '--noheader',
             '--format=State,ExitCode,Elapsed,End'],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return ' '.join(out.stdout.split())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def check_run(run_name: str, year: int) -> tuple[str, dict | None]:
    """Per-run status. One of: ok, broken_nan, no_output, no_ckpt, parse_error."""
    ckpt_dir = CHECKPOINTS_ROOT / run_name
    map_dir = MAPS_ROOT / run_name
    summary_p = map_dir / f'bavaria_{year}_summary.json'

    if not ckpt_dir.exists():
        return 'no_ckpt', None
    if not summary_p.exists():
        return 'no_output', None
    try:
        s = json.loads(summary_p.read_text())
    except Exception as e:
        return f'parse_error({type(e).__name__})', None
    mean = s.get('mean')
    if mean is None or (isinstance(mean, float) and mean != mean):
        return 'broken_nan', s
    return 'ok', s


def snapshot(args) -> dict:
    """Single-pass status snapshot. Returns a dict with counts for --quiet."""
    log_path = find_latest_log()
    log_info = parse_log(log_path) if log_path else None

    print()
    print('=' * 76)
    print('  BAVARIA INFERENCE STATUS')
    print('=' * 76)

    if log_info is None:
        print('  No log files in', MAPS_ROOT)
        print('  Sbatch not yet started, or output goes elsewhere.')
    else:
        sl = slurm_state(log_info['jobid']) if log_info['jobid'] else None
        print(f"  Latest log : {log_info['log'].name}")
        print(f"  Job ID     : {log_info['jobid'] or '(none parsed)'}")
        print(f"  Slurm state: {sl or '(sacct unavailable or no job)'}")
        marker = ('COMPLETE — reached "[infer-all] all done"'
                  if log_info['done']
                  else 'still running OR crashed before final marker')
        print(f"  Log marker : {marker}")
        if log_info['started']:
            print(f"  Runs started in this batch: {len(log_info['started'])}")
        if log_info['skipped']:
            print(f"  Skipped (no checkpoint dir): {log_info['skipped']}")
        if log_info['warned']:
            print(f"  Reported failures: {log_info['warned']}")

    # Per-run map outputs
    print()
    print('  Per-run map output (status / mean SOC g/kg):')
    print(f'  {"Run":<48}  {"Kind":<5}  {"Status":<16}  {"Mean":>8}')
    print('  ' + '-' * 82)
    counts = {'ok': 0, 'broken_nan': 0, 'no_output': 0, 'no_ckpt': 0, 'other': 0}
    for run, kind in EXPECTED_RUNS:
        status, s = check_run(run, args.year)
        if status in counts:
            counts[status] += 1
        else:
            counts['other'] += 1
        mean_str = '—'
        if s and isinstance(s.get('mean'), (int, float)) and s['mean'] == s['mean']:
            mean_str = f"{s['mean']:.2f}"
        elif status == 'broken_nan':
            mean_str = 'nan'
        print(f'  {run:<48}  {kind:<5}  {status:<16}  {mean_str:>8}')

    print('  ' + '-' * 82)
    total = sum(counts.values())
    print(f"  Summary: ok={counts['ok']}/{total}  broken_nan={counts['broken_nan']}  "
          f"no_output={counts['no_output']}  no_ckpt={counts['no_ckpt']}"
          + (f"  other={counts['other']}" if counts['other'] else ''))
    print('=' * 76)

    if args.tail and log_info and log_info['tail']:
        print(f"\n  Last {args.tail} lines of log:")
        for line in log_info['tail'][-args.tail:]:
            print('    ', line)

    return {'log_info': log_info, 'counts': counts, 'total': total}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--tail', type=int, default=20,
                   help='Lines to print from the latest log (0 to disable).')
    p.add_argument('--watch', action='store_true',
                   help='Refresh every --interval seconds until the log marker fires.')
    p.add_argument('--interval', type=int, default=30,
                   help='Seconds between refreshes in --watch mode (default 30).')
    p.add_argument('--quiet', action='store_true',
                   help='No output; exit 0 if all expected runs are ok, else 1.')
    a = p.parse_args()

    if a.quiet:
        # Silent mode for shell-pipeline use
        log_path = find_latest_log()
        all_ok = all(check_run(run, a.year)[0] == 'ok' for run, _ in EXPECTED_RUNS)
        log_done = bool(log_path and '[infer-all] all done'
                         in log_path.read_text(errors='replace'))
        sys.exit(0 if (all_ok and log_done) else 1)

    if not a.watch:
        snapshot(a)
        return

    # Watch mode
    while True:
        snap = snapshot(a)
        info = snap['log_info']
        if info and info['done']:
            print(f'\n  → Completion marker found. Stopping watch.')
            return
        print(f"\n  ... sleeping {a.interval}s (Ctrl-C to stop)")
        try:
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print('\n  → Interrupted.')
            return


if __name__ == '__main__':
    main()
