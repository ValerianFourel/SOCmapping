#!/usr/bin/env python3
"""
rebuttal/final_models/fit_and_map_top_models.py — fit the spatial-CV top
models on the FULL labelled dataset (holding out only a few hundred rows to
watch convergence), then map every fitted model over a random ~400k-location
sample of Bavaria in a single dependent job.

This is the production counterpart to the spatial-CV sweep: the sweep
*ranks* architectures under leave-one-block-out CV; here we re-fit the
winners on all the data and turn them into maps.

Pipeline per run (submitted on the JUPITER login node):
    NN model :  4-GPU `accelerate launch train_full.py --monitor-n <K> ...`
    tree     :  1-GPU `train_full_baselines.py ...`  (fits on all data)
Then ONE 1-GPU job runs map_locations.py over the random location sample
for ALL fitted models at once (it waits on every training job), producing
the per-model maps plus the side-by-side comparison panel.

Holdout: NN training keeps a random --monitor-n holdout (default 300) purely
to save the best-converged epoch; it does NOT bound performance (the
spatial-CV sweep already did that). Trees fit on all rows (no epoch
monitoring concept).

Band asymmetry: the NN feature pipeline supports the 43-band extended stack
(full_extended), so the neural top models map at 43 bands. The tree
feature extractor (train_full_baselines.py / infer_bavaria.predict_tree) is
hardwired to the 80-dim 20-band summary, so RF/XGB map at 20 bands. Each
config below carries its own bands_list accordingly.

Run on the login node:
    python rebuttal/final_models/fit_and_map_top_models.py            # submit
    python rebuttal/final_models/fit_and_map_top_models.py --dry-run  # preview
    python rebuttal/final_models/fit_and_map_top_models.py --only sgt_d32_h2_L1
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOC_ROOT = HERE.parents[1]
SBATCH_DIR = HERE / 'sbatch'
LOG_DIR = HERE / 'slurm_logs'

sys.path.insert(0, str(SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'))
from band_subsets import band_suffix  # noqa: E402


# ---------------------------------------------------------------------------
# TOP_MODELS — the per-family spatial-CV winners (sweep_summarize ranking).
#   run_name : checkpoint stem (train_full auto-appends the band suffix, e.g.
#              _extband / _20band, so the trained dir and the map run-name
#              match without us hardcoding it).
#   kind     : 'nn' (train_full.py, 4 GPU) or 'tree' (train_full_baselines.py)
#   bands    : --bands-list passed to the trainer.
#   cmd      : trainer flags EXCEPT --run-name / --bands-list / --monitor-n
#              (those are added by the sbatch builders).
# ---------------------------------------------------------------------------
TOP_MODELS = [
    # SGT — gated CNN+Transformer, overall spatial-CV winner (R2=0.387,
    # 43-band, longitude-blocked). Compact d=32, h=2, L=1, plain L1.
    {
        'run_name': 'sgt_d32_h2_L1',
        'kind': 'nn',
        'bands': 'full_extended',
        'cmd': ('--model-family sgt --model-size small '
                '--hidden_size 32 --num_heads 2 --num_layers 1 '
                '--dropout_rate 0.5 '
                '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
                '--loss_type l1 --target_transform log --max-oc 150 '
                '--per-gpu-batch-size 256 --effective-batch-size 256 '
                '--num-epochs 100 --seed 42 --augment-train'),
    },
    # CNN+Transformer without the gate (the ablation companion; 2nd overall).
    {
        'run_name': 'vanilla_transformer_d64_h4_L1',
        'kind': 'nn',
        'bands': 'full_extended',
        'cmd': ('--model-family vanilla_transformer --model-size small '
                '--hidden_size 64 --num_heads 4 --num_layers 1 '
                '--dropout_rate 0.5 '
                '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
                '--loss_type l1 --target_transform log --max-oc 150 '
                '--per-gpu-batch-size 256 --effective-batch-size 256 '
                '--num-epochs 100 --seed 42 --augment-train'),
    },
    # Transformer-only (no CNN) — the pure-attention reference.
    {
        'run_name': 'simpletransformer_d64_h4_L1',
        'kind': 'nn',
        'bands': 'full_extended',
        'cmd': ('--model-family simpletransformer --model-size small '
                '--hidden_size 64 --num_heads 4 --num_layers 1 '
                '--dropout_rate 0.5 '
                '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
                '--loss_type l1 --target_transform log --max-oc 150 '
                '--per-gpu-batch-size 256 --effective-batch-size 256 '
                '--num-epochs 100 --seed 42 --augment-train'),
    },
    # Random Forest (deep) — best tree baseline; 20-band (tree extractor cap).
    {
        'run_name': 'rf_deep',
        'kind': 'tree',
        'bands': 'full_20',
        'cmd': ('--model rf --rf-n-estimators 1000 --rf-max-depth 0 '
                '--max-oc 150 --target-transform log'),
    },
    # XGBoost (deep) — best XGB baseline; 20-band.
    {
        'run_name': 'xgb_deep',
        'kind': 'tree',
        'bands': 'full_20',
        'cmd': ('--model xgb --xgb-n-estimators 1000 --xgb-max-depth 8 '
                '--xgb-lr 0.05 --max-oc 150 --target-transform log'),
    },
]

DEFAULTS = {
    'partition': 'booster',
    'account': 'scifi',
    'time_train': '02:00:00',
    'time_baseline': '02:30:00',
    'time_map': '03:00:00',
    'venv_activate': str(SOC_ROOT.parent / 'venv' / 'bin' / 'activate'),
    'year': 2023,
    'monitor_n': 300,
    'n_locations': 400_000,
    'sample_seed': 42,
}


def mapped_run_name(cfg: dict) -> str:
    """The checkpoint dir / map run-name after train_full's auto band suffix."""
    return cfg['run_name'] + band_suffix(cfg['bands'])


def _preamble(opts: dict, job_name: str, log: Path, gpus: int,
              ntasks: int, time: str) -> str:
    venv = (f'source {shlex.quote(str(opts["venv_activate"]))}'
            if opts['venv_activate'] else 'true')
    return f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={opts["partition"]}
#SBATCH --account={opts["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --output={log}
#SBATCH --error={log}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv}
'''


def build_nn_train(cfg: dict, opts: dict) -> str:
    rn = mapped_run_name(cfg)
    log = LOG_DIR / f'fit_{rn}_train_%j.out'
    return _preamble(opts, f'fit-train-{rn}', log, gpus=4, ntasks=4,
                     time=opts['time_train']) + f'''
echo "[fit-train] run_name={rn}  bands={cfg["bands"]}  monitor_n={opts["monitor_n"]}  node=$(hostname)  job=$SLURM_JOB_ID"

accelerate launch --num_processes 4 \\
    rebuttal/final_models/train_full.py \\
    --run-name {cfg["run_name"]} \\
    --bands-list {cfg["bands"]} \\
    --monitor-n {opts["monitor_n"]} \\
    {cfg["cmd"]}
'''


def build_tree_train(cfg: dict, opts: dict) -> str:
    rn = mapped_run_name(cfg)
    log = LOG_DIR / f'fit_{rn}_train_%j.out'
    return _preamble(opts, f'fit-train-{rn}', log, gpus=1, ntasks=1,
                     time=opts['time_baseline']) + f'''
echo "[fit-train-tree] run_name={rn}  bands={cfg["bands"]}  node=$(hostname)  job=$SLURM_JOB_ID"

PYTHONUNBUFFERED=1 \\
python rebuttal/final_models/train_full_baselines.py \\
    --run-name {cfg["run_name"]} \\
    --bands-list {cfg["bands"]} \\
    {cfg["cmd"]}
'''


def build_map(run_names: list[str], opts: dict) -> str:
    log = LOG_DIR / f'fit_map_{opts["n_locations"]}rand_%j.out'
    rn_csv = ','.join(run_names)
    return _preamble(opts, 'fit-map-top', log, gpus=1, ntasks=1,
                     time=opts['time_map']) + f'''
echo "[fit-map] run_names={rn_csv}  n_locations={opts["n_locations"]}  year={opts["year"]}  node=$(hostname)  job=$SLURM_JOB_ID"

PYTHONUNBUFFERED=1 \\
python rebuttal/final_models/map_locations.py \\
    --run-names {rn_csv} \\
    --year {opts["year"]} \\
    --n-locations {opts["n_locations"]} \\
    --sample-seed {opts["sample_seed"]}
'''


def submit(script_path: Path, depends_on: list[str] | None = None) -> str | None:
    cmd = ['sbatch']
    if depends_on:
        cmd += ['--dependency=afterok:' + ':'.join(depends_on)]
    cmd += [str(script_path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f'[submit] FAILED for {script_path.name}: {out.stderr.strip()}',
              file=sys.stderr)
        return None
    return out.stdout.strip().split()[-1]


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Write the sbatch scripts but do not submit them.')
    p.add_argument('--only', type=str, default=None,
                   help='Comma-separated run_names (the stem, e.g. '
                        '"sgt_d32_h2_L1") to include (default: all).')
    p.add_argument('--partition', default=DEFAULTS['partition'])
    p.add_argument('--account', default=DEFAULTS['account'])
    p.add_argument('--venv-activate', default=DEFAULTS['venv_activate'])
    p.add_argument('--year', type=int, default=DEFAULTS['year'])
    p.add_argument('--monitor-n', type=int, default=DEFAULTS['monitor_n'],
                   help='Random convergence holdout for NN training '
                        '(default 300). Trees ignore it.')
    p.add_argument('--n-locations', type=int, default=DEFAULTS['n_locations'],
                   help='Random map-location sample size (default 400000; '
                        '0 = full ~1.3M grid).')
    p.add_argument('--sample-seed', type=int, default=DEFAULTS['sample_seed'])
    p.add_argument('--time-train', default=DEFAULTS['time_train'])
    p.add_argument('--time-baseline', default=DEFAULTS['time_baseline'])
    p.add_argument('--time-map', default=DEFAULTS['time_map'])
    return p.parse_args()


def main():
    a = parse()
    opts = {
        'partition': a.partition, 'account': a.account,
        'venv_activate': a.venv_activate, 'year': a.year,
        'monitor_n': a.monitor_n, 'n_locations': a.n_locations,
        'sample_seed': a.sample_seed, 'time_train': a.time_train,
        'time_baseline': a.time_baseline, 'time_map': a.time_map,
    }
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    only = set(a.only.split(',')) if a.only else None
    selected = [c for c in TOP_MODELS
                if not only or c['run_name'] in only
                or mapped_run_name(c) in only]
    if not selected:
        raise SystemExit(f'[ERROR] --only {a.only} matched no TOP_MODELS.')

    print(f'[fit] partition={a.partition} account={a.account} '
          f'monitor_n={a.monitor_n} n_locations={a.n_locations} '
          f'year={a.year}', flush=True)
    print(f'[fit] models: {[mapped_run_name(c) for c in selected]}', flush=True)

    train_jids: list[str] = []
    map_run_names: list[str] = []
    for cfg in selected:
        rn = mapped_run_name(cfg)
        map_run_names.append(rn)
        builder = build_nn_train if cfg['kind'] == 'nn' else build_tree_train
        sb = SBATCH_DIR / f'fit_{rn}_train.sbatch'
        sb.write_text(builder(cfg, opts)); sb.chmod(0o755)
        if a.dry_run:
            print(f'[dry-run] train sbatch → {sb}')
            continue
        jid = submit(sb)
        if jid:
            train_jids.append(jid)
            print(f'[submit] train {rn:>40}: job={jid}')

    # One mapping job over the random location sample, for ALL fitted models,
    # gated on every training job finishing successfully.
    map_sb = SBATCH_DIR / f'fit_map_{a.n_locations}rand.sbatch'
    map_sb.write_text(build_map(map_run_names, opts)); map_sb.chmod(0o755)
    if a.dry_run:
        print(f'[dry-run] map sbatch → {map_sb}  (run_names={map_run_names})')
        print(f'\n[dry-run] scripts in {SBATCH_DIR}/. Re-run without --dry-run.')
        return

    map_jid = submit(map_sb, depends_on=train_jids or None)
    if map_jid:
        dep = f' (waits on {len(train_jids)} train jobs)' if train_jids else ''
        print(f'[submit] map {a.n_locations} locations: job={map_jid}{dep}')
    print('\n[fit] queued. Watch with:  squeue -u $USER')
    print(f'[fit] maps will land under rebuttal/final_models/maps/'
          f'_locations_{a.n_locations}rand_seed{a.sample_seed}/')


if __name__ == '__main__':
    main()
