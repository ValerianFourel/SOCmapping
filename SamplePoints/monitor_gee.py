"""Status summary for the Bavaria 2002-2023 GEE export batch.

Lists all tasks owned by the authenticated GEE account, filters to the
curated batch's description patterns (modis_/chirps_/era5_/soil_/topo_),
and prints a per-state and per-category summary.

Use:
    python SamplePoints/monitor_gee.py                  # summary
    python SamplePoints/monitor_gee.py --failed         # show failed only
    python SamplePoints/monitor_gee.py --running        # show running only
    python SamplePoints/monitor_gee.py --watch          # live refresh every 30s
"""
import argparse
import re
import sys
import time
from collections import Counter, defaultdict

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed. pip install earthengine-api")
    sys.exit(1)


CATEGORIES = ('modis', 'chirps', 'era5', 'soil', 'topo')
_DESC_RE = re.compile(rf'^({"|".join(CATEGORIES)})_(.+?)(?:_(static|\d{{4}}))?$')

STATE_ORDER = ['COMPLETED', 'RUNNING', 'READY', 'CANCEL_REQUESTED',
               'CANCELLED', 'FAILED']


# State priority — when multiple tasks share a description (e.g. a retry
# after a previous FAILED submit), we report the BEST outcome.
_STATE_PRIORITY = {
    'COMPLETED':         100,
    'RUNNING':            80,
    'READY':              60,
    'CANCEL_REQUESTED':   40,
    'CANCELLED':          20,
    'FAILED':              0,
    'UNSUBMITTED':       -10,
}


def list_relevant_tasks(dedupe: bool = True):
    """Return list of (state, desc, task) for our pipeline-style descriptions.

    By default, deduplicates by `desc`: when a description has multiple
    server-side records (a retry after a previous failure), the entry with
    the highest state priority wins. This prevents stale FAILED records
    from inflating the "FAILED" count after a successful resubmission.
    """
    ee.Initialize()
    tasks = ee.batch.Task.list()
    raw = []
    for t in tasks:
        desc = t.config.get('description', '')
        if not _DESC_RE.match(desc):
            continue
        raw.append((t.state, desc, t))
    if not dedupe:
        return raw
    best: dict[str, tuple] = {}
    for state, desc, t in raw:
        prev = best.get(desc)
        if prev is None or _STATE_PRIORITY.get(state, -99) > _STATE_PRIORITY.get(prev[0], -99):
            best[desc] = (state, desc, t)
    return list(best.values())


def summarize(tasks):
    by_state = Counter(state for state, _, _ in tasks)
    by_cat = defaultdict(Counter)
    for state, desc, _ in tasks:
        m = _DESC_RE.match(desc)
        if m:
            by_cat[m.group(1)][state] += 1

    print(f'\nBavaria 2002-2023 GEE batch — {len(tasks)} tasks found\n')
    print(f'  {"State":<22} count')
    print(f'  {"-" * 22} -----')
    for state in STATE_ORDER:
        if by_state.get(state):
            print(f'  {state:<22} {by_state[state]:>5}')

    print(f'\nPer category:')
    print(f'  {"category":<10} {"  ".join(s[:5] for s in STATE_ORDER)}   total')
    print(f'  {"-" * 10} {"  ".join("-----" for _ in STATE_ORDER)}   -----')
    for cat in CATEGORIES:
        if cat not in by_cat: continue
        row = '  '.join(f'{by_cat[cat][s]:>5}' for s in STATE_ORDER)
        total = sum(by_cat[cat].values())
        print(f'  {cat:<10} {row}   {total:>5}')

    done = by_state.get('COMPLETED', 0)
    pct = 100.0 * done / len(tasks) if tasks else 0
    print(f'\n  Progress: {done}/{len(tasks)} ({pct:.1f}%) COMPLETED')

    if by_state.get('FAILED'):
        print(f'  ⚠ {by_state["FAILED"]} task(s) FAILED — re-run with --failed to see which.')


def show_failed(tasks):
    failed = [(d, t) for s, d, t in tasks if s == 'FAILED']
    if not failed:
        print('No failed tasks.')
        return
    print(f'\n{len(failed)} failed tasks:\n')
    for desc, t in sorted(failed):
        # Real error message lives in t.status()['error_message'] — one API call per task.
        try:
            err = t.status().get('error_message') or '(GEE returned no error_message)'
        except Exception as ex:
            err = f'(status() failed: {type(ex).__name__}: {ex})'
        print(f'  {desc}')
        print(f'      {err[:250]}')


def show_running(tasks):
    running = [(d, t) for s, d, t in tasks if s == 'RUNNING']
    if not running:
        print('No running tasks.')
        return
    print(f'\n{len(running)} running tasks:')
    for desc, _ in sorted(running)[:50]:
        print(f'  {desc}')
    if len(running) > 50:
        print(f'  … and {len(running) - 50} more')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--failed', action='store_true', help='list failed tasks with their error messages')
    ap.add_argument('--running', action='store_true', help='list currently RUNNING tasks')
    ap.add_argument('--watch', action='store_true', help='refresh every 30s until all COMPLETED or only FAILED remain')
    ap.add_argument('--no-dedupe', action='store_true',
                    help='Show every server-side task record, including stale history. '
                         'Default deduplicates by description (best state wins).')
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                tasks = list_relevant_tasks(dedupe=not args.no_dedupe)
                print('\033[2J\033[H', end='')  # clear screen
                summarize(tasks)
                states = {s for s, _, _ in tasks}
                if states <= {'COMPLETED', 'FAILED', 'CANCELLED'}:
                    print('\n✓ No more pending work — all tasks have terminated.')
                    break
                print('\n(refreshing every 30s — Ctrl+C to exit)')
                time.sleep(30)
        except KeyboardInterrupt:
            print('\nstopped.')
        return

    tasks = list_relevant_tasks(dedupe=not args.no_dedupe)
    if args.failed:
        show_failed(tasks)
    elif args.running:
        show_running(tasks)
    else:
        summarize(tasks)


if __name__ == '__main__':
    main()
