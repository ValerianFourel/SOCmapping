#!/usr/bin/env python3
"""
run_folds_parallel.py — parallel orchestrator for run_kfold.py.

Each fold of run_kfold.py is small enough to fit comfortably on one GH200
(model ~1.2M params + tiny batches, < 5 GB per fold of 96 GB HBM). With
--folds-per-gpu N, the orchestrator packs N fold subprocesses on each GPU
so all 10 folds can run simultaneously on 4 GH200s. Each subprocess is
pinned to one GPU via CUDA_VISIBLE_DEVICES; CUDA time-slices among the
processes sharing a device.

Usage — 4 folds at a time, one per GPU (legacy default):

    python run_folds_parallel.py \
        --num-folds 10 --num-parallel 4 \
        -- <run_kfold.py args>

Usage — all 10 folds at once on 4 GPUs (3 folds per GPU):

    python run_folds_parallel.py \
        --num-folds 10 --num-parallel 10 --folds-per-gpu 3 \
        -- \
        --model-size big --num_heads 4 --num_layers 3 \
        --hidden_size 128 --dropout_rate 0.3 \
        --lr 2e-4 --lr-scheduler cosine --lr-min 1e-6 \
        --loss_type l1 --target_transform log \
        --per-gpu-batch-size 256 --effective-batch-size 256 \
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
                        "min(num_folds, num_gpus_visible * folds_per_gpu).")
    p.add_argument('--folds-per-gpu', type=int, default=1,
                   help="How many fold processes to pack on each GPU. Default 1. "
                        "Set to 3 to run all 10 folds across 4 GPUs simultaneously "
                        "(slot_idx %% num_gpus picks the GPU). GH200s have ~96 GB "
                        "and each fold uses well under 5 GB, so 3 is safe.")
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

    total_slots = num_gpus * args.folds_per_gpu
    n_parallel = min(args.num_parallel, total_slots, args.num_folds)
    slot_to_gpu = {slot: slot % num_gpus for slot in range(total_slots)}
    print(f"[orchestrator] {args.num_folds} folds, "
          f"{num_gpus} GPUs, {args.folds_per_gpu} folds/GPU "
          f"({total_slots} slots), running {n_parallel} in parallel.")
    print(f"[orchestrator] passthrough args: {' '.join(passthrough)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the fold queue
    pending = list(range(args.num_folds))
    running: dict[int, subprocess.Popen] = {}    # slot_idx -> Popen
    fold_for_slot: dict[int, int] = {}            # slot_idx -> fold_idx
    started_at: dict[int, float] = {}             # fold_idx -> ts

    def launch(fold_idx: int, slot_idx: int) -> subprocess.Popen:
        gpu_id = slot_to_gpu[slot_idx]
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
        print(f"[orchestrator] launched fold {fold_idx} on slot {slot_idx} "
              f"(GPU {gpu_id}, PID {proc.pid}, log: {log_path.name})")
        return proc

    # Initial fill
    for slot_idx in range(n_parallel):
        if not pending:
            break
        fold = pending.pop(0)
        running[slot_idx] = launch(fold, slot_idx)
        fold_for_slot[slot_idx] = fold

    # Poll loop
    rc_by_fold: dict[int, int] = {}
    while running:
        time.sleep(2)
        finished = []
        for slot_idx, proc in running.items():
            rc = proc.poll()
            if rc is not None:
                fold = fold_for_slot[slot_idx]
                gpu_id = slot_to_gpu[slot_idx]
                dt = time.time() - started_at[fold]
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"[orchestrator] fold {fold} on slot {slot_idx} "
                      f"(GPU {gpu_id}) {status}  ({dt/60:.1f} min)")
                proc._log_fh.close()
                rc_by_fold[fold] = rc
                finished.append(slot_idx)
        for slot_idx in finished:
            running.pop(slot_idx)
            fold_for_slot.pop(slot_idx)
            if pending:
                next_fold = pending.pop(0)
                running[slot_idx] = launch(next_fold, slot_idx)
                fold_for_slot[slot_idx] = next_fold

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

    # Aggregate: read all per-fold predictions and write the cross-fold tables.
    # Passthrough is reused so the recipe metadata (max_oc, sampler_mode, etc.)
    # ends up in kfold_results.md / summary.json.
    print(f"[orchestrator] aggregating cross-fold results …")
    agg_cmd = [sys.executable, str(RUN_KFOLD),
               '--aggregate-only',
               '--num-folds', str(args.num_folds)] + passthrough
    agg_env = os.environ.copy()
    agg_env.setdefault('WANDB_MODE', 'disabled')
    rc = subprocess.call(agg_cmd, env=agg_env, cwd=str(HERE.parents[2]))
    if rc != 0:
        print(f"[orchestrator] aggregate-only step failed (rc={rc}). "
              f"Per-fold parquets are still on disk; rerun manually with "
              f"`python {RUN_KFOLD.name} --aggregate-only`.")
        sys.exit(rc)
    print(f"[orchestrator] kfold_results.md + summary.json written to "
          f"{out_dir}")


if __name__ == "__main__":
    main()
