#!/usr/bin/env python3
"""
rebuttal/final_models/submit_finals.py — submit the full set of final-model
training+inference jobs for the Geoderma rebuttal pivot.

Each entry in FINAL_GRID is one production-mapping model. For neural-net
families, we submit one 4-GPU training job followed by a dependent
1-GPU inference job. For tree baselines (RF/XGB) we bundle train+infer
into one 1-GPU job.

Output structure (under rebuttal/final_models/):
    checkpoints/<run_name>/    trained weights + stats + config + log
    maps/<run_name>/           bavaria_<year>_predictions.parquet + .png + .json

Run on the login node:
    python rebuttal/final_models/submit_finals.py
    # or dry-run:
    python rebuttal/final_models/submit_finals.py --dry-run
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

# Import band_subsets via the spatial_kfold module path
sys.path.insert(0, str(SOC_ROOT / 'rebuttal' / 'gpu_experiments' / 'spatial_kfold'))
from band_subsets import band_suffix  # noqa: E402


# ---------------------------------------------------------------------------
# FINAL_GRID — one production mapping model per family. Hyperparameters
# are the spatial-CV winners from the sweep_summarize ranking. All trained
# on the FULL 16k dataset (only a 5% RANDOM monitor holdout for best-state
# tracking; not used to bound performance).
# ---------------------------------------------------------------------------
NN_CONFIGS = [
    # SGT (winner): SimpleSGT d=128, h=4, L=1 with composite_l2 at max-oc 150
    {
        'run_name': 'sgt_d128_h4_L1',
        'cmd': (
            '--model-family sgt --model-size small '
            '--hidden_size 128 --num_heads 4 --num_layers 1 '
            '--dropout_rate 0.5 '
            '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
            '--loss_type composite_l2 --loss-alpha 0.5 --chi2-weight 0.1 '
            '--target_transform log --max-oc 150 '
            '--per-gpu-batch-size 256 --effective-batch-size 256 '
            '--num-epochs 60 --seed 42 --augment-train'
        ),
    },
    # Parameter-matched-by-hyperparameter ablation: SimpleSGT minus the
    # Gated Residual Network. Same d_model=128, h=4, L=1 as the SGT winner;
    # vanilla is 215k params vs SGT's 363k, so it's *smaller*. If R²
    # ties SGT, the gating doesn't earn its 148k overhead. If SGT beats
    # this baseline, the gating's value is empirically demonstrated.
    {
        'run_name': 'vanilla_transformer_d128_h4_L1',
        'cmd': (
            '--model-family vanilla_transformer --model-size small '
            '--hidden_size 128 --num_heads 4 --num_layers 1 '
            '--dropout_rate 0.5 '
            '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
            '--loss_type composite_l2 --loss-alpha 0.5 --chi2-weight 0.1 '
            '--target_transform log --max-oc 150 '
            '--per-gpu-batch-size 256 --effective-batch-size 256 '
            '--num-epochs 60 --seed 42 --augment-train'
        ),
    },
    # SimpleTransformerV2 (11.2M params; comparison transformer at max-oc 150,
    # plain L1 because composite_l2 destabilizes it per the spatial-CV sweep)
    {
        'run_name': 'simpletransformer_d64_h4_L1',
        'cmd': (
            '--model-family simpletransformer --model-size small '
            '--hidden_size 64 --num_heads 4 --num_layers 1 '
            '--dropout_rate 0.5 '
            '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
            '--loss_type l1 --target_transform log --max-oc 150 '
            '--per-gpu-batch-size 256 --effective-batch-size 256 '
            '--num-epochs 60 --seed 42 --augment-train'
        ),
    },
    # CNNLSTM (93k params; competitive baseline)
    {
        'run_name': 'cnnlstm_d64_h4_L1',
        'cmd': (
            '--model-family cnnlstm --model-size small '
            '--hidden_size 64 --num_heads 4 --num_layers 1 '
            '--dropout_rate 0.5 '
            '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
            '--loss_type l1 --target_transform log --max-oc 150 '
            '--per-gpu-batch-size 256 --effective-batch-size 256 '
            '--num-epochs 60 --seed 42 --augment-train'
        ),
    },
    # LightweightTransformer — true transformer-only baseline at matched
    # parameter scale (~240k at 20-band, ~194k at 6-band). The CNN-frontend
    # ablation companion to vanilla_transformer. Same recipe as vanilla
    # (composite_l2) for clean A/B in the maps figure.
    {
        'run_name': 'lightweight_transformer_d128_h4_L1',
        'cmd': (
            '--model-family lightweight_transformer --model-size small '
            '--hidden_size 128 --num_heads 4 --num_layers 1 '
            '--dropout_rate 0.5 '
            '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
            '--loss_type composite_l2 --loss-alpha 0.5 --chi2-weight 0.1 '
            '--target_transform log --max-oc 150 '
            '--per-gpu-batch-size 256 --effective-batch-size 256 '
            '--num-epochs 60 --seed 42 --augment-train'
        ),
    },
    # 3DCNN (failed family — included for completeness in the comparison
    # figure; spatial-CV showed R²=-0.76 so the map will be poor but the
    # negative result IS the result).
    {
        'run_name': '3dcnn_d64_h4_L1',
        'cmd': (
            '--model-family 3dcnn --model-size small '
            '--hidden_size 64 --num_heads 4 --num_layers 1 '
            '--dropout_rate 0.5 '
            '--lr 1e-4 --lr-scheduler cosine --lr-min 1e-6 '
            '--loss_type l1 --target_transform log --max-oc 150 '
            '--per-gpu-batch-size 256 --effective-batch-size 256 '
            '--num-epochs 60 --seed 42 --augment-train'
        ),
    },
]

TREE_CONFIGS = [
    {'run_name': 'rf_default',
     'cmd': ('--model rf --rf-n-estimators 500 --rf-max-depth 0 '
             '--max-oc 150 --target-transform log')},
    {'run_name': 'xgb_shallow',
     'cmd': ('--model xgb --xgb-n-estimators 2000 --xgb-max-depth 4 --xgb-lr 0.05 '
             '--max-oc 150 --target-transform log')},
]

DEFAULTS = {
    'partition': 'booster',
    'account': 'scifi',
    'time_train': '01:30:00',
    'time_infer': '02:00:00',
    'time_baseline': '02:30:00',
    'venv_activate': str(SOC_ROOT.parent / 'venv' / 'bin' / 'activate'),
    'year': 2023,
}


def _suffix(opts: dict) -> str:
    """Tag appended to run_name when --rebalance is on so rebalanced and
    non-rebalanced final models can coexist under checkpoints/maps."""
    return '_rebal' if opts.get('rebalance') else ''


def _sampler_args(opts: dict) -> str:
    if not opts.get('rebalance'):
        return ''
    return f' --sampler-mode kde --sampler-alpha {opts["sampler_alpha"]}'


def build_nn_train_sbatch(cfg: dict, opts: dict, bands_list: str) -> str:
    run_name = cfg["run_name"] + band_suffix(bands_list) + _suffix(opts)
    log = LOG_DIR / f'final_{run_name}_train_%j.out'
    venv = (f'source {shlex.quote(str(opts["venv_activate"]))}'
            if opts['venv_activate'] else 'true')
    return f'''#!/bin/bash
#SBATCH --job-name=final-train-{run_name}
#SBATCH --partition={opts["partition"]}
#SBATCH --account={opts["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time={opts["time_train"]}
#SBATCH --output={log}
#SBATCH --error={log}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv}

echo "[final-train] run_name={run_name}  bands_list={bands_list}  node=$(hostname)  job=$SLURM_JOB_ID"

WANDB_MODE=disabled PYTHONUNBUFFERED=1 \\
accelerate launch --num_processes 4 \\
    rebuttal/final_models/train_full.py \\
    --run-name {cfg["run_name"]} \\
    --bands-list {bands_list} \\
    {cfg["cmd"]}{_sampler_args(opts)}
'''


def build_nn_infer_sbatch(cfg: dict, opts: dict, bands_list: str) -> str:
    run_name = cfg["run_name"] + band_suffix(bands_list) + _suffix(opts)
    log = LOG_DIR / f'final_{run_name}_infer_%j.out'
    venv = (f'source {shlex.quote(str(opts["venv_activate"]))}'
            if opts['venv_activate'] else 'true')
    return f'''#!/bin/bash
#SBATCH --job-name=final-infer-{run_name}
#SBATCH --partition={opts["partition"]}
#SBATCH --account={opts["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time={opts["time_infer"]}
#SBATCH --output={log}
#SBATCH --error={log}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv}

echo "[final-infer] run_name={run_name}  year={opts["year"]}  node=$(hostname)  job=$SLURM_JOB_ID"

WANDB_MODE=disabled PYTHONUNBUFFERED=1 \\
python rebuttal/final_models/infer_bavaria.py \\
    --run-name {run_name} \\
    --year {opts["year"]}
'''


def build_tree_combined_sbatch(cfg: dict, opts: dict, bands_list: str) -> str:
    # Same _suffix + _sampler_args plumbing as the NN path — trees now
    # support sample_weight in train_full_baselines.py (passes through to
    # sklearn/XGB .fit) so the rebalanced production maps cover trees too.
    run_name = cfg["run_name"] + band_suffix(bands_list) + _suffix(opts)
    log = LOG_DIR / f'final_{run_name}_%j.out'
    venv = (f'source {shlex.quote(str(opts["venv_activate"]))}'
            if opts['venv_activate'] else 'true')
    return f'''#!/bin/bash
#SBATCH --job-name=final-{run_name}
#SBATCH --partition={opts["partition"]}
#SBATCH --account={opts["account"]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time={opts["time_baseline"]}
#SBATCH --output={log}
#SBATCH --error={log}

set -euo pipefail
cd {shlex.quote(str(SOC_ROOT))}
{venv}

echo "[final-tree] run_name={run_name}  bands_list={bands_list}  node=$(hostname)  job=$SLURM_JOB_ID"

PYTHONUNBUFFERED=1 \\
python rebuttal/final_models/train_full_baselines.py \\
    --run-name {cfg["run_name"]} \\
    --bands-list {bands_list} \\
    {cfg["cmd"]}{_sampler_args(opts)}

PYTHONUNBUFFERED=1 \\
python rebuttal/final_models/infer_bavaria.py \\
    --run-name {run_name} \\
    --year {opts["year"]}
'''


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--partition', default=DEFAULTS['partition'])
    p.add_argument('--account', default=DEFAULTS['account'])
    p.add_argument('--venv-activate', default=DEFAULTS['venv_activate'])
    p.add_argument('--year', type=int, default=DEFAULTS['year'])
    p.add_argument('--time-train', default=DEFAULTS['time_train'])
    p.add_argument('--time-infer', default=DEFAULTS['time_infer'])
    p.add_argument('--time-baseline', default=DEFAULTS['time_baseline'])
    p.add_argument('--only', type=str, default=None,
                   help='Comma-separated run_names to submit (default: all).')
    p.add_argument('--bands-lists', type=str, default='full_20,original_6',
                   help='Comma-separated covariate subsets to train on. '
                        'Default: "full_20,original_6" generates BOTH variants '
                        'of every entry in NN_CONFIGS / TREE_CONFIGS, with '
                        'run_name auto-suffixed with _20band / _6band. '
                        'Pass just "full_20" or "original_6" to run a single '
                        'variant.')
    p.add_argument('--rebalance', action='store_true',
                   help='Train ALL configs (NN + trees) with --sampler-mode '
                        'kde (KDE-inverse-density on log(SOC), oversampling '
                        'high-SOC rare-tail rows). NN models use a '
                        'WeightedRandomSampler; tree baselines (RF, XGB) '
                        'pass the same weights via sample_weight to their '
                        'native .fit(). Run_name gets a "_rebal" suffix so '
                        'rebalanced and non-rebalanced outputs coexist. Use '
                        'this for the production comparison figure: the '
                        "resulting maps cover Bavaria's organic-rich regions "
                        'instead of regressing toward the bulk mineral-soil '
                        'mean.')
    p.add_argument('--sampler-alpha', type=float, default=0.5,
                   help='[--rebalance only] KDE inversion exponent. '
                        'Default 0.5 = sqrt-inverse density (Yang et al. '
                        'ICML 2021). Higher → more aggressive upweighting.')
    return p.parse_args()


def submit(script_path: Path, depends_on: str | None = None) -> str | None:
    cmd = ['sbatch']
    if depends_on:
        cmd += ['--dependency=afterok:' + depends_on]
    cmd += [str(script_path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f'[submit] FAILED for {script_path.name}: {out.stderr.strip()}',
              file=sys.stderr)
        return None
    return out.stdout.strip().split()[-1]


def main():
    a = parse()
    opts = {
        'partition': a.partition,
        'account': a.account,
        'venv_activate': a.venv_activate,
        'year': a.year,
        'time_train': a.time_train,
        'time_infer': a.time_infer,
        'time_baseline': a.time_baseline,
        'rebalance': a.rebalance,
        'sampler_alpha': a.sampler_alpha,
    }
    if a.rebalance:
        print(f'[submit] --rebalance ON  →  NN training uses '
              f'WeightedRandomSampler(KDE, alpha={a.sampler_alpha}); '
              f'trees use sample_weight (KDE, alpha={a.sampler_alpha}) '
              f'passed to .fit(). All run_names auto-suffixed with '
              f'"_rebal" so outputs do not collide with non-rebalanced runs.')
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    only = set(a.only.split(',')) if a.only else None
    bands_lists = [b.strip() for b in a.bands_lists.split(',') if b.strip()]
    print(f'[submit] band variants to train: {bands_lists}')
    submitted = []

    # NN configs: train then infer with dependency — done for each bands-list
    for bands_list in bands_lists:
        suf = band_suffix(bands_list)
        for cfg in NN_CONFIGS:
            full_run_name = cfg['run_name'] + suf
            if only and not (cfg['run_name'] in only or full_run_name in only):
                continue
            train_sbatch = SBATCH_DIR / f'{full_run_name}_train.sbatch'
            infer_sbatch = SBATCH_DIR / f'{full_run_name}_infer.sbatch'
            train_sbatch.write_text(build_nn_train_sbatch(cfg, opts, bands_list))
            infer_sbatch.write_text(build_nn_infer_sbatch(cfg, opts, bands_list))
            train_sbatch.chmod(0o755); infer_sbatch.chmod(0o755)

            if a.dry_run:
                print(f'[dry-run] would submit (train→infer): '
                      f'{train_sbatch.name}, {infer_sbatch.name}')
                continue

            train_jid = submit(train_sbatch)
            if not train_jid:
                continue
            infer_jid = submit(infer_sbatch, depends_on=train_jid)
            submitted.append((full_run_name, train_jid, infer_jid))
            print(f'[submit] {full_run_name:>35}: train job={train_jid}  '
                  f'infer job={infer_jid} (waits on {train_jid})')

    # Tree configs: combined train+infer — also done for each bands-list
    for bands_list in bands_lists:
        suf = band_suffix(bands_list)
        for cfg in TREE_CONFIGS:
            full_run_name = cfg['run_name'] + suf
            if only and not (cfg['run_name'] in only or full_run_name in only):
                continue
            sbatch = SBATCH_DIR / f'{full_run_name}.sbatch'
            sbatch.write_text(build_tree_combined_sbatch(cfg, opts, bands_list))
            sbatch.chmod(0o755)
            if a.dry_run:
                print(f'[dry-run] would submit: {sbatch.name}')
                continue
            jid = submit(sbatch)
            if jid:
                submitted.append((full_run_name, jid, None))
                print(f'[submit] {full_run_name:>35}: job={jid} (train+infer)')

    if a.dry_run:
        print(f'\n[dry-run] scripts in {SBATCH_DIR}/. Re-run without --dry-run.')
        return
    if submitted:
        print(f'\n[submit] {len(submitted)} pipelines queued. Watch with:')
        print(f'  squeue -u $USER')
        print(f'[submit] When all infer jobs finish, generate the comparison figure with:')
        print(f'  python rebuttal/final_models/compare_maps.py')


if __name__ == '__main__':
    main()
