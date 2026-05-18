"""Pull the GEE-exported GeoTIFFs from Google Drive to a local folder.

Default backend: **rclone** (recommended; resumable + real OAuth + no public-link rate limit).
Fallback backend: **gdown** (works only on PUBLIC folders; hits "Cannot retrieve the
public link" errors on private folders).

Both backends are idempotent — re-running picks up where it stopped.

ONE-TIME RCLONE SETUP (do this once per machine):

    # Install
    sudo apt install rclone           # Debian/Ubuntu
    # OR: curl -fsSL https://rclone.org/install.sh | sudo bash
    # OR: brew install rclone         (macOS)

    rclone config
        n   (new remote)
        name> gdrive                  ← MUST be this exact name (or pass --remote)
        Storage> drive                ← type the word, or the number for "Google Drive"
        client_id>                    ← leave blank, press Enter
        client_secret>                ← leave blank
        scope> 1                      ← Full access (read+write)
        service_account_file>         ← blank
        Edit advanced config? n
        Use auto config? y            ← opens a browser to authenticate
        Configure as Shared Drive? n
        y                             ← save
        q                             ← quit

Examples:

    # Pull, default (rclone), from folder by ID:
    python SamplePoints/pull_from_drive.py --folder-id 12X21qgpt1tqmDr-nfNyZe-7m7DBW2wlP

    # Specify a custom rclone remote name:
    python SamplePoints/pull_from_drive.py --method rclone --remote myremote \\
        --folder-name bavaria_bands_2002_2023

    # Fall back to gdown (only for public folders):
    python SamplePoints/pull_from_drive.py --method gdown --folder-id <ID>

    # Dry run:
    python SamplePoints/pull_from_drive.py --dry-run --folder-id <ID>
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── Backend probes ────────────────────────────────────────────────────────

def _have_rclone() -> bool:
    return shutil.which('rclone') is not None


def _have_gdown() -> bool:
    if shutil.which('gdown') is not None:
        return True
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def _rclone_remote_configured(remote: str) -> bool:
    try:
        r = subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True)
        return f'{remote}:' in (r.stdout or '')
    except Exception:
        return False


def _rclone_setup_hint(remote: str) -> None:
    print(f"""ERROR: rclone is installed but the remote '{remote}' is not configured.

Run this once to set it up:

    rclone config
      n                              # new remote
      name> {remote}
      Storage> drive                  # Google Drive
      client_id>                      # blank
      client_secret>                  # blank
      scope> 1                        # full access
      service_account_file>           # blank
      Edit advanced config? n
      Use auto config? y              # opens a browser to authenticate
      Configure as Shared Drive? n
      y                               # save
      q

Then re-run this script.
""")


# ── Backend: rclone ────────────────────────────────────────────────────────

def pull_rclone(folder_id: str | None, folder_name: str | None,
                remote: str, out_dir: Path, dry_run: bool) -> int:
    """Use `rclone copy` with proper OAuth. Skips already-downloaded files."""
    if not _have_rclone():
        print("ERROR: rclone not installed.")
        print("       Install with: sudo apt install rclone   (or brew install rclone)")
        sys.exit(1)
    if not _rclone_remote_configured(remote):
        _rclone_setup_hint(remote)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # rclone refers to Drive entities by name path (gdrive:folder/sub) OR
    # by folder ID with --drive-root-folder-id. ID-based is robust because
    # it doesn't depend on the folder's location in the Drive hierarchy.
    if folder_id:
        cmd = [
            'rclone', 'copy',
            f'{remote}:',
            str(out_dir),
            '--drive-root-folder-id', folder_id,
            '--progress',
            '--transfers', '4',
            '--checkers', '8',
            '--drive-acknowledge-abuse',          # tolerate the "abuse" warning for COG TIFFs
            '--ignore-existing',                   # already-downloaded files are skipped
        ]
    elif folder_name:
        cmd = [
            'rclone', 'copy',
            f'{remote}:{folder_name}',
            str(out_dir),
            '--progress',
            '--transfers', '4',
            '--checkers', '8',
            '--drive-acknowledge-abuse',
            '--ignore-existing',
        ]
    else:
        sys.exit('ERROR: rclone backend requires --folder-id or --folder-name')

    if dry_run:
        print(f'[dry-run] would run: {" ".join(cmd)}')
        return 0

    print(f'Running: {" ".join(cmd)}\n')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'\nERROR: rclone exited with code {r.returncode}.')
        sys.exit(r.returncode)
    n = sum(1 for _ in out_dir.rglob('*.tif'))
    return n


# ── Backend: gdown ─────────────────────────────────────────────────────────

def pull_gdown(folder_id: str, out_dir: Path, dry_run: bool) -> int:
    """gdown's --folder mode. Works only on PUBLIC folders. Not resumable."""
    if not _have_gdown():
        print("ERROR: gdown not installed. Run: pip install gdown")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.google.com/drive/folders/{folder_id}'
    cmd = ['gdown', '--folder', url, '--output', str(out_dir)]
    if dry_run:
        print(f'[dry-run] would run: {" ".join(cmd)}')
        return 0
    print(f'Running: {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'\nERROR: gdown exited with code {r.returncode}.')
        print('Common fix: rclone instead (set --method rclone).')
        sys.exit(r.returncode)
    n = sum(1 for _ in out_dir.rglob('*.tif'))
    return n


# ── State recording ───────────────────────────────────────────────────────

def record_state(out_dir: Path) -> int:
    """Record each local TIFF under pipeline_state.pull.done. Mark the phase
    `done` ONLY if every desc in pipeline_state.gee.done has a matching local
    TIFF — otherwise keep status as `in_progress` so a future run knows to
    resume rather than skip.
    """
    try:
        from pipeline_state import State
        state = State()
        state.start_phase('pull')
        tiffs = list(out_dir.rglob('*.tif'))
        for f in tiffs:
            state.mark_done('pull', f.name)

        # Compare to what GEE was supposed to deliver. Each gee.done entry
        # is a description like 'modis_NDVI_2002' → expected file 'modis_NDVI_2002.tif'.
        expected = {f'{d}.tif' for d in state.done_items('gee')}
        present = {f.name for f in tiffs}
        missing = expected - present
        n_missing = len(missing)

        if expected and n_missing == 0:
            state.finish_phase('pull')
            print(f'  state: pull complete — {len(tiffs)} TIFFs match all {len(expected)} gee.done entries')
        else:
            # Don't finish — the orchestrator's resume check will treat 'in_progress'
            # as still-pending and re-enter on next run.
            if expected:
                print(f'  state: pull INCOMPLETE — {n_missing} TIFFs missing '
                      f'(expected {len(expected)}, have {len(tiffs)})')
                if n_missing <= 25:
                    for m in sorted(missing):
                        print(f'    missing: {m}')
                else:
                    print(f'    (first 25 missing of {n_missing}):')
                    for m in sorted(missing)[:25]:
                        print(f'    missing: {m}')
                    print(f'    … and {n_missing - 25} more')
            else:
                # No gee.done entries — happens if state was reset; we have no expectations.
                # Mark done by default (count check is impossible).
                state.finish_phase('pull')
                print(f'  state: pull complete — {len(tiffs)} TIFFs recorded '
                      f'(no gee.done to compare against)')
        return len(tiffs)
    except Exception as ex:
        print(f'  (warn: could not update pipeline_state: {ex})')
        return -1


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['rclone', 'gdown', 'auto'], default='auto',
                    help='Backend (default auto: rclone if available + configured, else gdown).')
    ap.add_argument('--remote', default='gdrive',
                    help='rclone remote name (default "gdrive"; only used with --method rclone).')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--folder-id', type=str,
                   help='Drive folder ID (the bit after /folders/ in the URL)')
    g.add_argument('--folder-url', type=str,
                   help='Full Drive folder URL')
    g.add_argument('--folder-name', type=str,
                   help='Folder name in your Drive root (rclone only)')
    ap.add_argument('--out', type=Path,
                    default=Path.home() / 'bavaria_tiffs',
                    help='Local destination directory (default ~/bavaria_tiffs)')
    ap.add_argument('--dry-run', action='store_true', help='Print the command without downloading')
    args = ap.parse_args()

    # Normalize folder URL → ID
    folder_id = args.folder_id
    if args.folder_url:
        marker = '/folders/'
        if marker not in args.folder_url:
            sys.exit(f"ERROR: --folder-url must contain '{marker}'")
        folder_id = args.folder_url.split(marker, 1)[1].split('?', 1)[0].split('/', 1)[0]

    # Pick a backend
    method = args.method
    if method == 'auto':
        if _have_rclone() and _rclone_remote_configured(args.remote):
            method = 'rclone'
            print(f'[auto] using rclone with remote "{args.remote}"')
        elif _have_gdown():
            method = 'gdown'
            print('[auto] rclone not configured — falling back to gdown')
            print('       (consider setting up rclone for private folders; see this script\'s docstring)')
        else:
            sys.exit("ERROR: neither rclone nor gdown is available.\n"
                     "       Install one: sudo apt install rclone   OR   pip install gdown")

    if method == 'rclone':
        n = pull_rclone(folder_id, args.folder_name, args.remote, args.out, args.dry_run)
    elif method == 'gdown':
        if args.folder_name:
            print('NOTE: gdown does not support --folder-name; you need the folder ID.')
            sys.exit(2)
        n = pull_gdown(folder_id, args.out, args.dry_run)
    else:
        sys.exit(f'unknown --method {method}')

    if args.dry_run:
        return

    print(f'\n✓ {n} TIFFs present at {args.out}')
    n_recorded = record_state(args.out)
    if n_recorded >= 0:
        print(f'  state: {n_recorded} TIFFs recorded under pipeline_state.pull.done')
    print(f'\nNext: bash SOCmapping/SamplePoints/run_full_pipeline.sh cut')


if __name__ == '__main__':
    main()
