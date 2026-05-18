#!/usr/bin/env python3
"""
run_folds_parallel.py — parallel orchestrator for run_kfold.py.

Each fold of run_kfold.py is small enough to fit comfortably on one GH200
(model ~1M params + tiny batches). Running them sequentially wastes 3 of
the 4 GPUs. This orchestrator launches `--num-parallel` fold subprocesses
concurrently, each pinned to one GPU via CUDA_VISIBLE_DEVICES, and dispatches
the remaining folds as GPUs free up.

Usage (drop-in replacement for run_kfold.py):

    python run_folds_parallel.py \
        --num-folds 10 \
        --num-parallel 4 \
        -- \
        --model-size big --num_heads 4 --num_layers 3 \
        --hidden_size 128 --dropout_rate 0.3 \
        --lr 2e-4 --lr-scheduler cosine --lr-min 1e-6 \
        --loss_type l1 --target_transform log \
        --per-gpu-batch-size 512 --effective-batch-size 512 \
        --num-epochs 300 --seed-base 42

Everything AFTER the `--` is forwarded verbatim to each run_kfold.py call.
The `--fold N`, `--num-folds`, and CUDA_VISIBLE_DEVICES are injected per
subprocess.

Logs per fold are written to:
    {output-dir}/fold_{i}_console.log

The orchestrator prints a one-line status update for each fold start and
finish so you can follow progress in real time.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_KFOLD = HERE / "run_kfold.py"


def parse():
    p = argparse.ArgumentParser(
        description="Parallel orchestrator for spatial k-fold CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--num-folds', type=int, default=10,
                   help="Total number of folds (default 10).")
    p.add_argument('--num-parallel', type=int, default=4,
                   help="How many folds to run concurrently. Defaults to "
                        "min(num_folds, num_gpus_visible).")
    p.add_argument('--output-dir', type=str,
                   default=str(HERE),
                   help="Directory for per-fold log files (default: this dir).")
    p.add_argument('--num-gpus', type=int, default=None,
                   help="Override detected GPU count. Default: torch.cuda.device_count().")
    return p.parse_known_args()


def detect_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def main():
    args, passthrough = parse()
    # Strip leading '--' from passthrough if present
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    num_gpus = args.num_gpus or detect_gpus()
    if num_gpus == 0:
        print("ERROR: no CUDA GPUs detected.", file=sys.stderr)
        sys.exit(2)

    n_parallel = min(args.num_parallel, num_gpus, args.num_folds)
    print(f"[orchestrator] {args.num_folds} folds, "
          f"{num_gpus} GPUs detected, "
          f"running {n_parallel} in parallel.")
    print(f"[orchestrator] passthrough args: {' '.join(passthrough)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the fold queue
    pending = list(range(args.num_folds))
    running: dict[int, subprocess.Popen] = {}   # gpu_id -> Popen
    fold_for_gpu: dict[int, int] = {}            # gpu_id -> fold_idx
    started_at: dict[int, float] = {}            # fold_idx -> ts

    def launch(fold_idx: int, gpu_id: int) -> subprocess.Popen:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Ensure each fold's wandb subdir doesn't collide
        env.setdefault('WANDB_MODE', 'disabled')
        env.setdefault('PYTHONUNBUFFERED', '1')
        cmd = [sys.executable, str(RUN_KFOLD),
               '--fold', str(fold_idx),
               '--num-folds', str(args.num_folds)] + passthrough
        log_path = out_dir / f'fold_{fold_idx}_console.log'
        log_fh = open(log_path, 'w', buffering=1)
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            cwd=str(HERE.parents[2]),   # SOCmapping root
        )
        proc._log_fh = log_fh
        proc._log_path = log_path
        started_at[fold_idx] = time.time()
        print(f"[orchestrator] launched fold {fold_idx} on GPU {gpu_id} "
              f"(PID {proc.pid}, log: {log_path.name})")
        return proc

    # Initial fill
    for gpu_id in range(n_parallel):
        if not pending:
            break
        fold = pending.pop(0)
        running[gpu_id] = launch(fold, gpu_id)
        fold_for_gpu[gpu_id] = fold

    # Poll loop
    rc_by_fold: dict[int, int] = {}
    while running:
        time.sleep(2)
        finished = []
        for gpu_id, proc in running.items():
            rc = proc.poll()
            if rc is not None:
                fold = fold_for_gpu[gpu_id]
                dt = time.time() - started_at[fold]
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"[orchestrator] fold {fold} on GPU {gpu_id} "
                      f"{status}  ({dt/60:.1f} min)")
                proc._log_fh.close()
                rc_by_fold[fold] = rc
                finished.append(gpu_id)
        for gpu_id in finished:
            running.pop(gpu_id)
            fold_for_gpu.pop(gpu_id)
            if pending:
                next_fold = pending.pop(0)
                running[gpu_id] = launch(next_fold, gpu_id)
                fold_for_gpu[gpu_id] = next_fold

    # Summary
    n_ok = sum(1 for rc in rc_by_fold.values() if rc == 0)
    n_fail = len(rc_by_fold) - n_ok
    print()
    print(f"[orchestrator] DONE — {n_ok}/{len(rc_by_fold)} folds succeeded, "
          f"{n_fail} failed.")
    if n_fail:
        failed = sorted(f for f, rc in rc_by_fold.items() if rc != 0)
        print(f"[orchestrator] failed folds: {failed}")
        print(f"[orchestrator] inspect logs: {out_dir}/fold_<i>_console.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
