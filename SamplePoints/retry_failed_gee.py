"""Remove server-side FAILED GEE tasks from pipeline_state so a re-run resubmits them.

When `gee_download_all_bands.py` submits a task and `task.start()` returns OK,
the state tracker marks it `done`. If GEE *later* fails the task server-side
(memory, asset gap, etc.), the state still says "done" — so a vanilla re-run
would skip it.

This helper queries the live GEE task list, identifies FAILED tasks whose
descriptions match our curated batch pattern, and unsets their entry in
`pipeline_state.json` under the `gee` phase's `done` set. After running it,
re-invoke `bash run_full_pipeline.sh gee` (or the underlying script directly)
and the failed tasks will be resubmitted.

Usage:
    # Dry-run — show what would be removed:
    python SamplePoints/retry_failed_gee.py --dry-run

    # Actually remove (default after confirmation):
    python SamplePoints/retry_failed_gee.py

    # Skip the prompt (CI / scripted use):
    python SamplePoints/retry_failed_gee.py --yes
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_state import State

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed. pip install earthengine-api")
    sys.exit(1)


CATEGORIES = ('modis', 'chirps', 'era5', 'soil', 'topo')
_DESC_RE = re.compile(rf'^({"|".join(CATEGORIES)})_(.+?)(?:_(static|\d{{4}}))?$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='List failed tasks but do not modify pipeline_state.json')
    ap.add_argument('--yes', action='store_true',
                    help='Skip the y/n confirmation')
    args = ap.parse_args()

    print('Fetching GEE task list…')
    ee.Initialize()
    tasks = ee.batch.Task.list()
    failed = [t for t in tasks
              if t.state == 'FAILED'
              and _DESC_RE.match(t.config.get('description', ''))]

    if not failed:
        print('No FAILED tasks matching the curated batch pattern. Nothing to do.')
        return

    print(f'\n{len(failed)} FAILED tasks found:')
    for t in sorted(failed, key=lambda x: x.config['description']):
        desc = t.config['description']
        try:
            err = t.status().get('error_message') or '(no error_message)'
        except Exception:
            err = '(could not fetch error_message)'
        print(f'  - {desc}')
        print(f'      {err[:200]}')

    state = State()
    in_state = [t.config['description'] for t in failed
                if state.is_done('gee', t.config['description'])]
    print(f'\n{len(in_state)}/{len(failed)} are currently recorded as done in pipeline_state.json.')
    if not in_state:
        print('Nothing to remove. (Re-running gee will not skip these; they may already be retry-ready.)')
        return

    if args.dry_run:
        print('\n[dry-run] would remove the following descs from pipeline_state.gee.done:')
        for d in in_state:
            print(f'  - {d}')
        return

    if not args.yes:
        ans = input(f'\nRemove these {len(in_state)} entries from pipeline_state? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    # Mutate state: read the underlying list and drop matching entries.
    # We bypass mark_done (which only adds) by directly editing the done list.
    done_list = state._state['phases']['gee'].get('done', [])
    before = len(done_list)
    state._state['phases']['gee']['done'] = [d for d in done_list if d not in set(in_state)]
    state._save()
    after = len(state._state['phases']['gee']['done'])
    print(f'\nRemoved {before - after} entries from pipeline_state.gee.done.')
    print(f'\nNow re-submit the failed tasks (state-tracker will pick them up):')
    print(f'  bash SOCmapping/SamplePoints/run_full_pipeline.sh gee')


if __name__ == '__main__':
    main()
