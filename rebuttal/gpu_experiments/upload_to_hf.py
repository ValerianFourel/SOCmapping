#!/usr/bin/env python3
"""
upload_to_hf.py — push the GPU-experiment outputs to a single
public HF dataset repo:  https://huggingface.co/datasets/ValerianFourel/SOCrebuttal

Run this AFTER both experiments have finished, from inside the venv:

    source /workspace/SOC/SOCmapping/rebuttal/gpu_experiments/.venv/bin/activate
    export HF_TOKEN="hf_xxx_paste_your_write_token_here"
    python rebuttal/gpu_experiments/upload_to_hf.py

The token only needs `write` scope on this one repo
(https://huggingface.co/settings/tokens → "New token" → write).
You can also `hf auth login` once instead of exporting HF_TOKEN.

What gets uploaded:
  - rebuttal/gpu_experiments/spatial_kfold/{kfold_results.md,
        kfold_results_summary.json, kfold_predictions_all_folds.parquet,
        figure_kfold.png, fold_*_best.pth, fold_*_metrics.json}
  - rebuttal/gpu_experiments/uncertainty/{SGT_1mil_2023_mean_mc30.tif,
        SGT_1mil_2023_std_mc30.tif, figure_uncertainty_3panel.{png,pdf},
        mc_dropout_points.parquet, mc_dropout_metadata.json,
        mc_dropout_{mean,std,coords}.npy, validation_check.txt}
  - An auto-generated README.md with provenance

Anything that isn't on disk yet is silently skipped — so you can run
this after Experiment 2 alone (you'll just get a partial upload) or
after both.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ID = "ValerianFourel/SOCrebuttal"
REPO_TYPE = "dataset"
ROOT = Path(__file__).resolve().parent       # rebuttal/gpu_experiments/

# Fixed singletons we always expect (or skip silently).
EXPECTED_FIXED_PATHS = [
    # Experiment 1 — spatial k-fold CV (k can be 5, 10, or anything else)
    "spatial_kfold/kfold_results.md",
    "spatial_kfold/kfold_results_summary.json",
    "spatial_kfold/kfold_predictions_all_folds.parquet",
    "spatial_kfold/figure_kfold.png",

    # Experiment 2 — MC dropout uncertainty
    "uncertainty/SGT_1mil_2023_mean_mc30.tif",
    "uncertainty/SGT_1mil_2023_std_mc30.tif",
    "uncertainty/figure_uncertainty_3panel.png",
    "uncertainty/figure_uncertainty_3panel.pdf",
    "uncertainty/mc_dropout_points.parquet",
    "uncertainty/mc_dropout_metadata.json",
    "uncertainty/mc_dropout_mean.npy",
    "uncertainty/mc_dropout_std.npy",
    "uncertainty/mc_dropout_coords.npy",
    "uncertainty/validation_check.txt",
]

# Per-fold checkpoints + metrics — globbed at runtime so k=5/k=10/k=N all
# upload without code edits.
def _discover_fold_files() -> list[str]:
    kfold_dir = ROOT / "spatial_kfold"
    out = []
    if kfold_dir.exists():
        for p in sorted(kfold_dir.glob("fold_*_best.pth")):
            out.append(str(p.relative_to(ROOT)))
        for p in sorted(kfold_dir.glob("fold_*_metrics.json")):
            out.append(str(p.relative_to(ROOT)))
    return out


def _expected_paths() -> list[str]:
    """Combine the fixed list with discovered per-fold artefacts. Called
    each time we need the list, so glob is fresh on every invocation."""
    return EXPECTED_FIXED_PATHS + _discover_fold_files()


# Back-compat alias for the rest of the script
EXPECTED_PATHS = _expected_paths()

# Generated each run; pinned at the root of the HF repo
README_PATH = ROOT / "_SOCrebuttal_README.md"


# --------------------------------------------------------------------------
def find_existing() -> list[Path]:
    """Subset of EXPECTED_PATHS that actually exist on disk right now.
    Re-globs per-fold files each call so this works across k=5/k=10/k=N."""
    return [ROOT / p for p in _expected_paths() if (ROOT / p).exists()]


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} GB"


def build_readme(existing_files: list[Path]) -> str:
    """Generate the dataset README.md with provenance + file table."""
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    spatial = [p for p in existing_files if p.parent.name == "spatial_kfold"]
    uncertainty = [p for p in existing_files if p.parent.name == "uncertainty"]

    lines = []
    lines.append("---")
    lines.append("license: cc-by-4.0")
    lines.append("tags:")
    lines.append("  - soil-organic-carbon")
    lines.append("  - digital-soil-mapping")
    lines.append("  - bavaria")
    lines.append("  - geoderma-rebuttal")
    lines.append("  - cross-validation")
    lines.append("  - mc-dropout")
    lines.append("---")
    lines.append("")
    lines.append("# SOCrebuttal — Geoderma revision outputs")
    lines.append("")
    lines.append(f"_Last upload: {now}_")
    lines.append("")
    lines.append("Outputs from the two GPU-dependent reviewer-response experiments "
                 "for the Geoderma revision **GEODER-D-26-01032** of "
                 "*\"Machine Learning-Based Spatial Modeling of Soil Organic Carbon "
                 "in Bavaria\"*. CPU-side analyses live in the SOCmapping GitHub "
                 "repo at "
                 "[`rebuttal/`](https://github.com/ValerianFourel/SOCmapping/tree/main/rebuttal); "
                 "this HF dataset is only the GPU experiments' artefacts.")
    lines.append("")
    lines.append("Companion repos:")
    lines.append("- Code: <https://github.com/ValerianFourel/SOCmapping>")
    lines.append("- Sample data + rasters (~17 GB): "
                 "<https://huggingface.co/datasets/ValerianFourel/SOCmappingRastersAndSoilSamples>")
    lines.append("- Model weights: "
                 "<https://huggingface.co/ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping>")
    lines.append("")

    lines.append("## Experiment 1 — spatial 5-fold CV")
    lines.append("")
    lines.append("Answers reviewer concerns **R1.3, R3.6, R3.8** (single split is "
                 "weak; no confidence intervals on Table 2; no proper test set).")
    lines.append("")
    lines.append("Latitude-strip folds across Bavaria with a 1.2 km minimum-distance "
                 "buffer between train and test. EnhancedSGT (1.12 M params) trained "
                 "from scratch on each fold (270 epochs, Adam lr=2e-4, L1 on "
                 "log(OC + 1e-10), per-fold feature normalisation).")
    lines.append("")
    if spatial:
        lines.append("| file | size |")
        lines.append("|------|------|")
        for p in sorted(spatial):
            lines.append(f"| `{p.relative_to(ROOT)}` | {fmt_size(file_size(p))} |")
    else:
        lines.append("_(Experiment 1 outputs not present at upload time.)_")
    lines.append("")
    lines.append("- `kfold_results.md` is the **table to paste into the manuscript** "
                 "(per-fold + mean ± SD + 95% CI + original single-split row).")
    lines.append("- `figure_kfold.png` is the **figure for the response letter** "
                 "(2-panel: Bavaria fold map + per-fold metric comparison).")
    lines.append("- `kfold_predictions_all_folds.parquet` has per-sample "
                 "predictions (lon/lat/OC_actual/OC_predicted/fold_id/year/altitude) "
                 "for downstream analysis.")
    lines.append("- `fold_{i}_best.pth` are the 5 trained-from-scratch checkpoints "
                 "(can be re-loaded with EnhancedSGT for further inference).")
    lines.append("")

    lines.append("## Experiment 2 — MC Dropout uncertainty map")
    lines.append("")
    lines.append("Answers reviewer concerns **R3.9, R4.4** (no uncertainty "
                 "quantification; uncertainty maps expected in digital soil mapping).")
    lines.append("")
    lines.append("30 stochastic forward passes through the deployment model "
                 "(`finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1`) with "
                 "selective dropout activation (BatchNorm kept in eval mode). "
                 "Welford online accumulator computes per-pixel mean and SD in "
                 "original SOC units. The mean prediction matches the paper's "
                 "Figures 14/15.")
    lines.append("")
    if uncertainty:
        lines.append("| file | size |")
        lines.append("|------|------|")
        for p in sorted(uncertainty):
            lines.append(f"| `{p.relative_to(ROOT)}` | {fmt_size(file_size(p))} |")
    else:
        lines.append("_(Experiment 2 outputs not present at upload time.)_")
    lines.append("")
    lines.append("- `SGT_1mil_2023_mean_mc30.tif` — MC mean prediction, "
                 "EPSG:32632, 250 m, float32, LZW-compressed.")
    lines.append("- `SGT_1mil_2023_std_mc30.tif` — per-pixel prediction SD (g/kg), "
                 "same grid as the mean.")
    lines.append("- `figure_uncertainty_3panel.{png,pdf}` — **figure for the "
                 "response letter** (3 panels: mean, std, CV).")
    lines.append("- `mc_dropout_points.parquet` — per-grid-point predictions "
                 "(longitude, latitude, mean, std, cv_pct) for downstream use.")
    lines.append("- `mc_dropout_metadata.json` — model path, N_PASSES, target "
                 "transform, grid CRS / resolution / extent, validation Pearson r "
                 "vs the single-pass map.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("Both experiments are deterministic given the same fold seeds "
                 "(`SEED_BASE = 42` → 42..46 across folds) and the same MC sampling "
                 "RNG (the GPU's default dropout RNG, identical given the same "
                 "torch and CUDA versions). Re-running on the same hardware with "
                 "torch 2.0.1 / 2.2.x produces the same numbers; cross-hardware or "
                 "cross-torch-version may differ slightly at numerical-noise level.")
    lines.append("")
    lines.append("Launch from a fresh Runpod box following "
                 "[`RUNPOD_SETUP.md`](https://github.com/ValerianFourel/SOCmapping/blob/main/rebuttal/gpu_experiments/RUNPOD_SETUP.md).")
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("- Code commit (at upload time): see `_provenance.json` for the "
                 "exact SHA.")
    lines.append("- Model A: "
                 "`TFT_model_BEST_OVERALL_from_run_1_..._normalize_LOSS_composite_l2_R2_0.6909.pth` "
                 "(eval, 91/9 spatial split — used as the Table 2 baseline row "
                 "in `kfold_results.md`).")
    lines.append("- Model B: "
                 "`TFT_model_BEST_OVERALL_from_run_1_..._log_LOSS_l1_R2_1.0000.pth` "
                 "(full-data mapping model — input to Experiment 2).")
    lines.append("")
    return "\n".join(lines)


def build_provenance() -> dict:
    """Capture git SHA + uname for the upload's _provenance.json."""
    import subprocess
    sha = None
    branch = None
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True).strip()
    except Exception:
        pass
    return {
        "uploaded_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_sha": sha,
        "git_branch": branch,
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "uploaded_files": sorted(str(p.relative_to(ROOT)) for p in find_existing()),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
    }


# --------------------------------------------------------------------------
def main():
    token = os.environ.get("HF_TOKEN")
    # HfApi() will also fall back to `hf auth login`'s cached token if no env var
    api = HfApi(token=token)

    existing = find_existing()
    expected_now = _expected_paths()
    if not existing:
        print(f"ERROR: none of the expected output files exist under {ROOT}.")
        print("Run the experiments first, then re-run this upload script.")
        print("Expected:")
        for p in expected_now:
            print(f"  {p}  (missing)")
        sys.exit(1)

    missing = [p for p in expected_now if not (ROOT / p).exists()]
    print(f"Found {len(existing)} / {len(expected_now)} expected files.")
    if missing:
        print("\nMissing (will be skipped):")
        for p in missing:
            print(f"  - {p}")
    print()

    total = sum(file_size(p) for p in existing)
    print(f"Total to upload: {fmt_size(total)}")
    print()

    # 1. Generate the dataset README + provenance JSON
    README_PATH.write_text(build_readme(existing))
    print(f"Wrote {README_PATH}")
    prov_path = ROOT / "_provenance.json"
    prov_path.write_text(json.dumps(build_provenance(), indent=2))
    print(f"Wrote {prov_path}")

    # 2. Ensure repo exists (idempotent)
    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True,
                private=False, token=token)
    print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")

    # 3. Upload — pass two patterns through allow_patterns. We rename the
    #    local _SOCrebuttal_README.md to README.md inside the repo by uploading
    #    the file individually below; meanwhile upload_folder() takes the
    #    whitelisted experiment outputs and the _provenance.json.
    allow_patterns = expected_now + ["_provenance.json"]

    print("\n→ uploading experiment artefacts (this may take a few minutes)…")
    upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(ROOT),
        allow_patterns=allow_patterns,
        commit_message=f"Sync GPU-experiment outputs ({len(existing)} files, "
                       f"{fmt_size(total)})",
        token=token,
    )

    print("\n→ uploading README.md…")
    api.upload_file(
        path_or_fileobj=str(README_PATH),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update README with current upload manifest",
        token=token,
    )

    print(f"\n✅ Done.  https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
