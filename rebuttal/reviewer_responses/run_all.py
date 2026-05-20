#!/usr/bin/env python3
"""
run_all.py — orchestrator for the reviewer-response analyses.

Runs the six focused scripts in sequence, sharing defaults:

  1. flagship_summary.py     — executive table (Vanilla as flagship)
  2. bootstrap_ci.py         — T2.1: 95 % CIs on R²/RMSE/MAE
  3. validation_set_stats.py — T2.4: explain the R²/RMSE paradox
  4. temporal_sensitivity.py — T2.2/T2.3/T2.9: β_year sensitivities
  5. nn_distances.py         — T2.6: nearest-neighbour distance distribution
  6. residual_audit.py       — T2.7: stratified residual SD by SOC bin

Each script writes its own .json + .md under
rebuttal/reviewer_responses/results/. They are independent — failures
in one don't block the others. Total wall ~3-5 min depending on disk
and on whether per-fold prediction parquets are present locally.

Re-run safely: each script overwrites its OWN outputs only.

Usage:
    python rebuttal/reviewer_responses/run_all.py            # everything, defaults
    python rebuttal/reviewer_responses/run_all.py --skip nn  # everything except nn_distances
    python rebuttal/reviewer_responses/run_all.py --only bootstrap,temporal
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    ('flagship',     'flagship_summary.py',     []),
    ('bootstrap',    'bootstrap_ci.py',         ['--n-boot', '10000']),
    ('validation',   'validation_set_stats.py', []),
    ('temporal',     'temporal_sensitivity.py', []),
    ('nn',           'nn_distances.py',         []),
    ('residual',     'residual_audit.py',       []),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--skip', type=str, default=None,
                   help='Comma-separated short names to skip (e.g. nn,residual).')
    p.add_argument('--only', type=str, default=None,
                   help='Comma-separated short names to run (exclusive with --skip).')
    p.add_argument('--continue-on-error', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Continue after a script failure (default ON).')
    a = p.parse_args()

    if a.only:
        wanted = set(s.strip() for s in a.only.split(','))
        scripts = [s for s in SCRIPTS if s[0] in wanted]
    else:
        skip = set(s.strip() for s in a.skip.split(',')) if a.skip else set()
        scripts = [s for s in SCRIPTS if s[0] not in skip]

    print('=' * 72)
    print(f'  RUN ALL — {len(scripts)} scripts to run')
    print('=' * 72)
    results = []
    t_total = time.time()
    for name, fname, args in scripts:
        path = HERE / fname
        print(f'\n[run_all] === {name} ({fname}) ===')
        t0 = time.time()
        rc = subprocess.run([sys.executable, str(path)] + args).returncode
        dt = time.time() - t0
        ok = (rc == 0)
        results.append((name, fname, ok, dt))
        print(f'[run_all] {name}: {"OK" if ok else f"FAILED (rc={rc})"}  ({dt:.1f}s)')
        if not ok and not a.continue_on_error:
            print(f'[run_all] aborting (--no-continue-on-error)', file=sys.stderr)
            sys.exit(rc)

    print('\n' + '=' * 72)
    print(f'  SUMMARY ({time.time() - t_total:.1f}s total)')
    print('=' * 72)
    for name, fname, ok, dt in results:
        marker = '✓' if ok else '✗'
        print(f'  {marker}  {name:<12} {fname:<28} {dt:.1f}s')
    failures = [r for r in results if not r[2]]
    if failures:
        print(f'\n{len(failures)} script(s) failed. See above for details.')
        sys.exit(1)
    print(f'\nAll outputs under: {HERE / "results"}')


if __name__ == '__main__':
    main()
