"""
_paths.py — single source of truth for the four abstract roots used
across the SOCmapping codebase:

    SOC_CODE_DIR     → the SOCmapping/ package (this file's parent)
    SOC_DATA_DIR     → the Data/ directory   (Bavaria rasters, LUCAS xlsx)
    SOC_WEIGHTS_DIR  → the Weights-ResidualsModels-MappingInference-SOCmapping/
                       directory (Model A, Model B, residual checkpoints)
    SOC_REBUTTAL_DIR → the rebuttal/ directory (this session's analyses
                       and GPU experiments — usually SOC_CODE_DIR/rebuttal)

Resolution order for each root, first to last:

    1. The environment variable named above
    2. A walk-up search from this file looking for a sibling directory
       matching the "marker" (e.g. for SOC_DATA_DIR: any parent whose
       sibling "Data" exists)
    3. The historical absolute default the codebase was written with

This file has no third-party dependencies and never touches the disk
beyond `Path.is_dir()` probes, so it is safe to import from
balancedDataset/config.py, SpatiotemporalGatedTransformer/config.py,
the rebuttal scripts, and any future code.

Override examples:

    # All paths under /workspace/SOC (Runpod default after the symlinks)
    export SOC_PROJECT_ROOT=/workspace/SOC

    # Or per-component:
    export SOC_DATA_DIR=/mnt/external/SOC_data
    export SOC_WEIGHTS_DIR=/mnt/external/SOC_weights
"""
from __future__ import annotations

import os
from pathlib import Path

# Historical hardcoded defaults — used only if no env var matches AND
# no walk-up candidate exists.  Kept so the existing absolute paths
# still work for the project author's laptop without env vars set.
_LEGACY_PROJECT_ROOT = Path('/home/valerian/SGTPublication')
_LEGACY_DATA_DIR     = _LEGACY_PROJECT_ROOT / 'Data'
_LEGACY_WEIGHTS_DIR  = _LEGACY_PROJECT_ROOT / 'Weights-ResidualsModels-MappingInference-SOCmapping'
_LEGACY_CODE_DIR     = _LEGACY_PROJECT_ROOT / 'SOCmapping'
_LEGACY_REBUTTAL_DIR = _LEGACY_CODE_DIR / 'rebuttal'

# Marker dir to look for at each level when walking up
_DATA_MARKER     = 'Data'
_WEIGHTS_MARKER  = 'Weights-ResidualsModels-MappingInference-SOCmapping'
_CODE_MARKER     = 'SOCmapping'


def _walk_up_find(start: Path, marker: str) -> Path | None:
    """Climb from `start` looking for a sibling directory named `marker`.
    Returns the *matching directory itself* (not its parent)."""
    here = start.resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / marker
        if candidate.is_dir():
            return candidate.resolve()
        # Also try one level up (sibling of parent)
        if parent.parent != parent:
            candidate = parent.parent / marker
            if candidate.is_dir():
                return candidate.resolve()
    return None


def _resolve(env_name: str, walk_marker: str, project_root: Path | None,
             legacy: Path) -> Path:
    """Resolution priority:
       1. The specific env var (SOC_DATA_DIR / SOC_WEIGHTS_DIR / SOC_CODE_DIR)
       2. SOC_PROJECT_ROOT-derived child if set AND that child exists
       3. Walk-up search from this file looking for `walk_marker` as a sibling
       4. Historical hardcoded `legacy` default
    """
    # 1. Component-specific env var wins
    env = os.environ.get(env_name)
    if env:
        return Path(env).expanduser().resolve()

    # 2. SOC_PROJECT_ROOT (only when explicitly set by user)
    if project_root is not None:
        candidate = project_root / walk_marker
        if candidate.is_dir():
            return candidate.resolve()

    # 3. Walk-up
    found = _walk_up_find(Path(__file__), walk_marker)
    if found:
        return found

    # 4. Legacy
    return legacy


_PROJECT_ROOT_ENV = os.environ.get('SOC_PROJECT_ROOT')
if _PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(_PROJECT_ROOT_ENV).expanduser().resolve()
    _PROJECT_HINT: Path | None = PROJECT_ROOT
else:
    _PROJECT_HINT = None
    _maybe_code = _walk_up_find(Path(__file__), _CODE_MARKER)
    PROJECT_ROOT = _maybe_code.parent if _maybe_code else _LEGACY_PROJECT_ROOT


# Resolve the four roots
SOC_CODE_DIR    = _resolve('SOC_CODE_DIR',    _CODE_MARKER,    _PROJECT_HINT, _LEGACY_CODE_DIR)
SOC_DATA_DIR    = _resolve('SOC_DATA_DIR',    _DATA_MARKER,    _PROJECT_HINT, _LEGACY_DATA_DIR)
SOC_WEIGHTS_DIR = _resolve('SOC_WEIGHTS_DIR', _WEIGHTS_MARKER, _PROJECT_HINT, _LEGACY_WEIGHTS_DIR)

# REBUTTAL_DIR is special — it's nested inside SOC_CODE_DIR
_REBUTTAL_ENV = os.environ.get('SOC_REBUTTAL_DIR')
if _REBUTTAL_ENV:
    SOC_REBUTTAL_DIR = Path(_REBUTTAL_ENV).expanduser().resolve()
else:
    candidate = SOC_CODE_DIR / 'rebuttal'
    SOC_REBUTTAL_DIR = candidate.resolve() if candidate.is_dir() else _LEGACY_REBUTTAL_DIR


# Public string aliases for legacy code that wants the strings, not Paths
SOC_CODE_DIR_STR     = str(SOC_CODE_DIR)
SOC_DATA_DIR_STR     = str(SOC_DATA_DIR)
SOC_WEIGHTS_DIR_STR  = str(SOC_WEIGHTS_DIR)
SOC_REBUTTAL_DIR_STR = str(SOC_REBUTTAL_DIR)


def describe() -> str:
    """Diagnostic dump of all four resolved roots and where each came from."""
    lines = []
    lines.append('SOCmapping path resolution:')
    for env_name, value, legacy in (
        ('SOC_PROJECT_ROOT', PROJECT_ROOT, _LEGACY_PROJECT_ROOT),
        ('SOC_CODE_DIR',     SOC_CODE_DIR, _LEGACY_CODE_DIR),
        ('SOC_DATA_DIR',     SOC_DATA_DIR, _LEGACY_DATA_DIR),
        ('SOC_WEIGHTS_DIR',  SOC_WEIGHTS_DIR, _LEGACY_WEIGHTS_DIR),
        ('SOC_REBUTTAL_DIR', SOC_REBUTTAL_DIR, _LEGACY_REBUTTAL_DIR),
    ):
        env = os.environ.get(env_name)
        if env:
            origin = f'env={env}'
        elif value == legacy:
            origin = 'legacy default'
        else:
            origin = 'walk-up search'
        flag = '✓' if value.exists() else '✗ MISSING'
        lines.append(f'  {env_name:<18} = {value}  [{origin}]  {flag}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe())
