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
                        "20-band runs are fine with 3; 43-band runs OOM at 3 "
                        "(simultaneous .to(device) spike), use 2.")
    p.add_argument('--launch-stagger', type=float, default=15.0,
                   help="Seconds to sleep between successive fold launches in "
                        "the initial fill phase. Stops the simultaneous "
                        "model.to(device) / dataloader-init spike from racing "
                        "CUDA OOM. Default 15s — enough for one process's "
                        "CUDA context + dataset init to settle before the "
                        "next lands on the same GPU. Set 0 to disable.")
    p.add_argument('--max-fold-retries', type=int, default=2,
                   help="If a fold exits non-zero (CUDA OOM, SIGKILL by the "
                        "OS / Slurm cgroup, etc.), requeue it to the END of "
                        "the pending list and try again. This is meant for "
                        "transient launch-time failures — the retry runs in "
                        "the quieter post-initial-fill phase when other folds "
                        "are mid-training, so GPU/host memory pressure is "
                        "lower. Default 2 (i.e. 1 original attempt + 2 "
                        "retries = 3 total). Set 0 to disable retries.")
    p.add_argument('--output-dir', type=str,
                   default=str(HERE),
                   help="Directory for per-fold log files (default: this dir).")
    p.add_argument('--num-gpus', type=int, default=None,
                   help="Override detected GPU count. Default: torch.cuda.device_count().")
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                   help="Skip folds whose fold_<i>_predictions.parquet already "
                        "exists in --output-dir and run only the missing ones. "
                        "Lets a timed-out / OOM-killed job be re-run or "
                        "resubmitted until every fold is present, without "
                        "redoing finished folds. Default on; --no-resume forces "
                        "a full re-run from scratch.")
    return p.parse_known_args()


def detect_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def fold_done(out_dir: Path, fold: int) -> bool:
    """A fold counts as complete iff its per-fold predictions parquet exists
    and is non-empty — that's exactly the artifact the --aggregate-only step
    consumes, so this is the same notion of 'done' as the final tables use."""
    p = out_dir / f'fold_{fold}_predictions.parquet'
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


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

    # Build the fold queue. With --resume (default) drop folds whose
    # predictions parquet is already on disk, so a re-run / resubmit only
    # fills the gaps a previous timed-out or OOM-killed job left behind.
    all_folds = list(range(args.num_folds))
    if args.resume:
        already = [f for f in all_folds if fold_done(out_dir, f)]
        pending = [f for f in all_folds if f not in already]
        if already:
            print(f"[orchestrator] resume: {len(already)}/{args.num_folds} folds "
                  f"already complete {already} — skipping. "
                  f"{len(pending)} to run: {pending}")
        if not pending:
            print("[orchestrator] all folds already present — nothing to train, "
                  "going straight to aggregation.")
    else:
        pending = list(all_folds)
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

    # Initial fill — stagger launches so concurrent fold processes don't
    # race CUDA memory allocation at startup (model.to(device), dataset
    # tensor staging). 43-band runs are particularly sensitive.
    for slot_idx in range(n_parallel):
        if not pending:
            break
        fold = pending.pop(0)
        running[slot_idx] = launch(fold, slot_idx)
        fold_for_slot[slot_idx] = fold
        if args.launch_stagger > 0 and slot_idx + 1 < n_parallel and pending:
            time.sleep(args.launch_stagger)

    # Poll loop. We track per-fold attempt counts so transient launch-time
    # failures (CUDA OOM races, host-OOM SIGKILLs, flaky GPUs) get requeued
    # to the END of the pending list and retried in the quieter post-initial-
    # fill phase. rc_by_fold holds the LATEST attempt's rc per fold.
    rc_by_fold: dict[int, int] = {}
    attempts: dict[int, int] = {f: 1 for f in fold_for_slot.values()}
    # ^ only folds actually launched in the initial fill are counted; folds
    #   still pending (or skipped by --resume) get attempt set when launched.
    requeued: list[tuple[int, int]] = []  # (fold, attempt) — book-keeping only.
    while running:
        time.sleep(2)
        finished = []
        for slot_idx, proc in running.items():
            rc = proc.poll()
            if rc is not None:
                fold = fold_for_slot[slot_idx]
                gpu_id = slot_to_gpu[slot_idx]
                dt = time.time() - started_at[fold]
                attempt = attempts.get(fold, 1)
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                attempt_tag = f" attempt={attempt}" if attempt > 1 else ""
                print(f"[orchestrator] fold {fold}{attempt_tag} on slot "
                      f"{slot_idx} (GPU {gpu_id}) {status}  ({dt/60:.1f} min)")
                proc._log_fh.close()
                # Requeue if failure and retries left. The retry runs at the
                # end of the queue → after the initial fill has trained for
                # a while, so the GPU + host memory pressure has eased.
                if (rc != 0 and attempt <= args.max_fold_retries):
                    pending.append(fold)
                    attempts[fold] = attempt + 1
                    requeued.append((fold, attempt))
                    print(f"[orchestrator]   ↳ requeued fold {fold} for "
                          f"attempt {attempt + 1}/{args.max_fold_retries + 1}")
                    # Preserve the previous attempt's console log so we don't
                    # overwrite it when launch() reopens fold_{i}_console.log.
                    old_log = out_dir / f'fold_{fold}_console.log'
                    if old_log.exists():
                        old_log.rename(out_dir / f'fold_{fold}_attempt{attempt}.log')
                else:
                    rc_by_fold[fold] = rc
                finished.append(slot_idx)
        for slot_idx in finished:
            running.pop(slot_idx)
            fold_for_slot.pop(slot_idx)
            if pending:
                next_fold = pending.pop(0)
                # Initial-fill folds were counted at attempts={f:1} above;
                # any fold first launched HERE (none in current sweep config —
                # initial-fill covers all 10 — but safe for n_parallel < n_folds)
                # also gets attempt=1.
                attempts.setdefault(next_fold, 1)
                running[slot_idx] = launch(next_fold, slot_idx)
                fold_for_slot[slot_idx] = next_fold

    # Summary — report completeness from what is actually on disk now
    # (folds resumed from a prior run + folds trained this run), not just this
    # invocation's successes, so resume / resubmit converges transparently.
    n_ran_ok = sum(1 for rc in rc_by_fold.values() if rc == 0)
    n_ran_fail = len(rc_by_fold) - n_ran_ok
    if requeued:
        print(f"[orchestrator] retried folds: "
              f"{sorted({f for f, _ in requeued})} ({len(requeued)} retries total)")
    present = [f for f in range(args.num_folds) if fold_done(out_dir, f)]
    missing = [f for f in range(args.num_folds) if f not in present]
    print()
    print(f"[orchestrator] this run: {n_ran_ok} ok, {n_ran_fail} failed.")
    print(f"[orchestrator] folds present on disk: {len(present)}/{args.num_folds} "
          f"{present}")
    if missing:
        print(f"[orchestrator] STILL MISSING: {missing} — re-run / resubmit this "
              f"job to fill them (resume skips the {len(present)} already done).")
        print(f"[orchestrator] inspect: {out_dir}/fold_<i>_console.log "
              f"(earlier tries: fold_<i>_attempt<N>.log)")

    # Best-effort: aggregate whatever folds are present. Losing the whole
    # config because one fold got SIGKILL'd is too punishing — a 9/10 cross-fold
    # R² is still publishable. Only abort if nothing survived at all.
    if not present:
        print("[orchestrator] no folds present — skipping aggregation.")
        sys.exit(1)

    # Aggregate: read all per-fold predictions and write the cross-fold tables.
    # Passthrough is reused so the recipe metadata (max_oc, sampler_mode, etc.)
    # ends up in kfold_results.md / summary.json.
    print(f"[orchestrator] aggregating cross-fold results "
          f"({len(present)}/{args.num_folds} folds present) …")
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
    print(f"[orchestrator] kfold_results.md + summary.json written to {out_dir}")
    # Distinct non-zero exit (3) when folds remain, so a resubmit wrapper can
    # tell "aggregated but still incomplete" from "fully done" (0).
    sys.exit(0 if not missing else 3)


if __name__ == "__main__":
    main()
