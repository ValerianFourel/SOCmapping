#!/usr/bin/env python3
"""
rebuttal/figures/run_all_figures.py — driver for the Geoderma revision
figure set.

Discovers every figXX_*.py / figFx_*.py in this directory, runs each as a
subprocess with the SHARED --sweep-dir / --maps-dir / --out-dir (and the
axis/bands/max-oc protocol flags), scrapes each one's final
'OK <pdf>' / 'BLOCKED <reason>' line, and writes a consolidated
FIGURE_MANIFEST.md to <out-dir> with a DISCREPANCIES section pre-seeded
with the 4 known on-disk-vs-manuscript mismatches.

Reality check: the CANONICAL numbers (43-band full_extended, oc150, lon,
10-fold) live ONLY on JUPITER — pass --sweep-dir there. The local bundle
(/home/valerian/SGTPublication/SOCrebuttal_HF/sweep) is the STALE
lat/20-band experiment, schema-identical only; use it for smoke testing.

Smoke test:
  python rebuttal/figures/run_all_figures.py \
      --sweep-dir /home/valerian/SGTPublication/SOCrebuttal_HF/sweep \
      --axis lat --bands 20
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figdata as fd  # noqa: E402  (Manifest writer)

OUT_DEFAULT = HERE / 'out'

# Pre-seeded known mismatches between what's ON DISK (canonical revision
# results) and what the SUBMITTED manuscript / older draft reported. The
# driver always writes these so reviewers see them next to every figure.
# (what, on_disk, manuscript, dependent_fig)
KNOWN_DISCREPANCIES = [
    ('Flagship experiment / split / model / R2',
     '43-band full_extended, lon-blocked CV, SGT ~84k params, R2 ~0.34-0.39',
     '6-band, lat-blocked CV, Vanilla ~215k params, R2 ~0.20',
     'all R2 / ranking figures'),
    ('Gate ablation direction (SGT gate vs no-gate Vanilla)',
     'gated SGT now ABOVE ungated Vanilla on lon-blocked 43-band',
     'reported as gate hurting / reversal (no-gate Vanilla on top)',
     'family-comparison figure'),
    ('Parameter counts (flagship + baseline drift)',
     'SGT ~84k; baselines re-counted to ~115k / ~317k',
     'SGT ~2.4M->11M; Vanilla baseline ~215k',
     'params-vs-R2 figure'),
    ('Covariate stack width',
     '43 bands (full_extended)',
     '6 bands',
     'covariate / data figures'),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Run all revision figure scripts.')
    ap.add_argument('--sweep-dir', default=str(fd.SWEEP_DIR_DEFAULT),
                    help='sweep results dir (JUPITER for canonical; local bundle is stale)')
    ap.add_argument('--maps-dir', default=str(fd.MAPS_DIR_DEFAULT),
                    help='Bavaria prediction maps dir (JUPITER-only)')
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT),
                    help='where figures + FIGURE_MANIFEST.md are written')
    ap.add_argument('--axis', default='lon', choices=['lon', 'lat'])
    ap.add_argument('--bands', default='43')
    ap.add_argument('--max-oc', default='150')
    ap.add_argument('--only', nargs='+', default=None,
                    help='run only these script basenames (e.g. fig01_ranking.py)')
    ap.add_argument('--timeout', type=int, default=600,
                    help='per-figure subprocess timeout (s)')
    ap.add_argument('--python', default=sys.executable)
    return ap.parse_args(argv)


def discover(only):
    scripts = sorted(p for p in HERE.glob('fig*_*.py')
                     if re.match(r'fig(\d+|F\w+)_', p.name))
    # keep only fig\d\d_*.py and figFx_*.py; drop figstyle/figdata (no underscore-digit)
    scripts = [p for p in scripts if p.name not in ('figstyle.py', 'figdata.py')]
    if only:
        keep = set(only)
        scripts = [p for p in scripts if p.name in keep or p.stem in keep]
    return scripts


# Accept flags only if a script's --help advertises them, so a figure that
# doesn't take, say, --maps-dir isn't handed an unknown arg.
def _supported_flags(py, script):
    try:
        h = subprocess.run([py, str(script), '--help'],
                           capture_output=True, text=True, timeout=60)
        txt = (h.stdout or '') + (h.stderr or '')
    except Exception:
        return set()
    flags = set(re.findall(r'(--[a-zA-Z][a-zA-Z0-9-]*)', txt))
    return flags


def build_argv(py, script, args):
    supported = _supported_flags(py, script)
    candidate = {
        '--sweep-dir': args.sweep_dir,
        '--maps-dir': args.maps_dir,
        '--out-dir': args.out_dir,
        '--axis': args.axis,
        '--bands': args.bands,
        '--max-oc': args.max_oc,
    }
    cmd = [py, str(script)]
    for flag, val in candidate.items():
        if flag in supported:
            cmd += [flag, str(val)]
    return cmd


_RESULT_RE = re.compile(r'^(OK|BLOCKED)\b(.*)$')


def scrape_result(stdout: str, stderr: str):
    """Return (status, detail) from the LAST OK/BLOCKED line across stdout+stderr."""
    status, detail = None, ''
    for line in (stdout + '\n' + stderr).splitlines():
        m = _RESULT_RE.match(line.strip())
        if m:
            status, detail = m.group(1), m.group(2).strip()
    return status, detail


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts = discover(args.only)

    man = fd.Manifest()
    for d in KNOWN_DISCREPANCIES:
        man.discrepancy(*d)

    if not scripts:
        print('[run_all] no fig*_*.py scripts found yet in '
              f'{HERE} — writing manifest with discrepancies only',
              file=sys.stderr)

    n_ok = n_blocked = n_err = 0
    for script in scripts:
        cmd = build_argv(args.python, script, args)
        print(f'[run_all] $ {" ".join(cmd)}')
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired:
            man.block(script.name, f'timeout after {args.timeout}s', str(script))
            n_err += 1
            continue
        except Exception as e:  # noqa: BLE001
            man.block(script.name, f'failed to launch: {e}', str(script))
            n_err += 1
            continue

        status, detail = scrape_result(proc.stdout, proc.stderr)
        if status == 'OK':
            files = detail or '(pdf+png)'
            man.done(script.name, files,
                     f'sweep={args.sweep_dir} axis={args.axis} bands={args.bands}',
                     'see script')
            n_ok += 1
            print(f'[run_all]   OK {script.name}: {detail}')
        elif status == 'BLOCKED':
            man.block(script.name, detail or 'blocked (see log)', str(script))
            n_blocked += 1
        else:
            # No OK/BLOCKED emitted -> treat nonzero exit as error.
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            reason = ('exit %d; ' % proc.returncode) + ' | '.join(tail)
            man.block(script.name, reason[:300], str(script))
            n_err += 1
            print(f'[run_all]   no OK/BLOCKED line from {script.name} '
                  f'(exit {proc.returncode})', file=sys.stderr)

    manifest_path = out_dir / 'FIGURE_MANIFEST.md'
    man.write(manifest_path)
    print(f'[run_all] {len(scripts)} scripts: {n_ok} OK, '
          f'{n_blocked} BLOCKED, {n_err} ERROR')

    if scripts and n_ok == 0:
        print(f'BLOCKED no figure produced output; manifest {manifest_path}')
        return 1
    print(f'OK {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
