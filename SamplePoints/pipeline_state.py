"""Pipeline state tracker — single source of truth for resumable runs.

Stores progress in `Data/pipeline_state.json` as a flat JSON blob so any
phase script can read what's been done and skip duplicate work. Each
unit of work (a TIFF, a (band, year), a GEE task) becomes a string key
in a per-phase set. Writes are atomic (tempfile + rename) so power-loss
mid-update can't leave a half-written state.

Use:
    from pipeline_state import State
    state = State()                                  # loads or initializes
    if not state.is_done('cut', 'modis_NDVI_2002.tif'):
        # do the work
        state.mark_done('cut', 'modis_NDVI_2002.tif')

    state.start_phase('verify')
    # ... do work ...
    state.finish_phase('verify')

    state.summary()                                  # print progress
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import SOC_DATA_DIR_STR

_DEFAULT_PATH = Path(SOC_DATA_DIR_STR) / 'pipeline_state.json'
_SCHEMA = 1

PHASES = ('plan', 'gee', 'pull', 'cut', 'project', 'verify', 'derive')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _empty_state() -> dict:
    return {
        'schema_version': _SCHEMA,
        'started_at': _now(),
        'last_updated': _now(),
        'phases': {p: {'status': 'pending', 'done': []} for p in PHASES},
    }


class State:
    """Resumable JSON-backed state.

    Two units of state per phase:
      - `status`: 'pending' | 'in_progress' | 'done'
      - `done`:   list of string IDs of completed work units (TIFF names,
                  band/year combos, GEE task descs, etc.)

    A phase script that processes many items should call
    `is_done(phase, item)` before each, and `mark_done(phase, item)`
    after. Use `start_phase` / `finish_phase` for phase-level bookkeeping.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        self._state: dict = self._load()

    # ── persistence ────────────────────────────────────────────────
    def _load(self) -> dict:
        if not self.path.exists():
            return _empty_state()
        try:
            with self.path.open() as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError) as ex:
            print(f'[pipeline_state] WARN: {self.path} unreadable ({ex}); reinitializing', file=sys.stderr)
            return _empty_state()
        # Tolerate older states / new phases — fill in missing keys.
        s.setdefault('schema_version', _SCHEMA)
        s.setdefault('started_at', _now())
        s.setdefault('phases', {})
        for p in PHASES:
            s['phases'].setdefault(p, {'status': 'pending', 'done': []})
            s['phases'][p].setdefault('status', 'pending')
            s['phases'][p].setdefault('done', [])
        return s

    def _save(self) -> None:
        self._state['last_updated'] = _now()
        # Atomic write: tempfile in the same directory, then rename.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w', dir=str(self.path.parent), prefix='.pipeline_state.', suffix='.tmp',
            delete=False,
        ) as f:
            json.dump(self._state, f, indent=2, sort_keys=True)
            tmpname = f.name
        os.replace(tmpname, str(self.path))

    # ── phase-level ────────────────────────────────────────────────
    def phase_status(self, phase: str) -> str:
        return self._state['phases'].get(phase, {}).get('status', 'pending')

    def start_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f'unknown phase {phase!r}; valid: {PHASES}')
        self._state['phases'][phase]['status'] = 'in_progress'
        self._state['phases'][phase].setdefault('started_at', _now())
        self._save()

    def finish_phase(self, phase: str) -> None:
        self._state['phases'][phase]['status'] = 'done'
        self._state['phases'][phase]['finished_at'] = _now()
        self._save()

    def reset_phase(self, phase: str) -> None:
        """Clear all progress for a phase. Use with care."""
        self._state['phases'][phase] = {'status': 'pending', 'done': []}
        self._save()

    # ── item-level (within a phase) ────────────────────────────────
    def is_done(self, phase: str, item: str) -> bool:
        return item in self._state['phases'].get(phase, {}).get('done', [])

    def mark_done(self, phase: str, item: str) -> None:
        done = self._state['phases'][phase].setdefault('done', [])
        if item not in done:
            done.append(item)
            self._save()

    def mark_many_done(self, phase: str, items: Iterable[str]) -> None:
        done = self._state['phases'][phase].setdefault('done', [])
        added = False
        for it in items:
            if it not in done:
                done.append(it)
                added = True
        if added:
            self._save()

    def done_items(self, phase: str) -> list[str]:
        return list(self._state['phases'].get(phase, {}).get('done', []))

    # ── reporting ──────────────────────────────────────────────────
    def summary(self, fh=sys.stdout) -> None:
        fh.write(f'\nPipeline state — {self.path}\n')
        fh.write(f'  started_at:   {self._state.get("started_at")}\n')
        fh.write(f'  last_updated: {self._state.get("last_updated")}\n')
        for p in PHASES:
            ph = self._state['phases'].get(p, {})
            status = ph.get('status', 'pending')
            n = len(ph.get('done', []))
            note = f'  ({n} items done)' if n else ''
            fh.write(f'  [{status:>11}] {p:<8}{note}\n')
        fh.write('\n')


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description='Inspect / manipulate pipeline state.')
    ap.add_argument('cmd', choices=['summary', 'show', 'reset-phase', 'init'])
    ap.add_argument('--phase', help='phase name (for reset-phase)')
    ap.add_argument('--path', type=Path, default=None, help='override state file path')
    args = ap.parse_args()

    s = State(args.path)
    if args.cmd == 'summary':
        s.summary()
    elif args.cmd == 'show':
        json.dump(s._state, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write('\n')
    elif args.cmd == 'reset-phase':
        if not args.phase:
            ap.error('--phase required')
        s.reset_phase(args.phase)
        print(f'reset phase {args.phase!r}')
    elif args.cmd == 'init':
        # Force a save to materialize the file even if all-empty.
        s._save()
        print(f'state file initialized at {s.path}')


if __name__ == '__main__':
    _cli()
