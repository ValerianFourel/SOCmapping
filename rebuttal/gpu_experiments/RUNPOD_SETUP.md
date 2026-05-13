# Runpod step-by-step — code + data + weights for the rebuttal experiments

End-to-end recipe to go from a fresh Runpod pod to running
`spatial_kfold/run_kfold.py` and `uncertainty/mc_dropout_inference.py`.

Both Hugging Face repos used here are **public** — no token, no gating.

| What | Where | Size |
|---|---|---|
| Code | github.com/ValerianFourel/SOCmapping | < 200 MB |
| Data | huggingface.co/datasets/ValerianFourel/SOCmappingRastersAndSoilSamples | **17 GB** (single zip) |
| Model weights | huggingface.co/ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping | ~250 MB (full); ~10 MB if you only need Model A + Model B |

---

## 0. SSH in

```bash
ssh mnp2foff20udyf-64411310@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The rest of this guide is run **inside that SSH session**, in
`/workspace` (Runpod's default persistent volume). Adjust paths if your
pod uses a different mount.

```bash
cd /workspace
mkdir -p SOC && cd SOC
```

---

## 1. System prerequisites

Most Runpod base images already have these. The `apt update / install`
lines are only needed if the binary check just before each one fails.

```bash
# Probe what's already there
python3 --version    # need 3.10+; 3.10 or 3.11 ideal
git --version
which nvidia-smi && nvidia-smi | head -5

# If anything is missing:
apt update && apt install -y git git-lfs curl unzip build-essential python3.10-venv
git lfs install
```

---

## 2. Clone the code (GitHub)

```bash
cd /workspace/SOC
git clone https://github.com/ValerianFourel/SOCmapping.git
cd SOCmapping
git log --oneline -3       # confirm the most recent commit is the rebuttal one
```

Expected most-recent commits (from this session):

```
2339252 Add Runpod venv setup for rebuttal GPU experiments
3482ca1 Add rebuttal/ analyses for Geoderma revision (GEODER-D-26-01032)
5783e47 Refactor: archive legacy code; remove dead composite-loss functions
```

---

## 3. Install the Hugging Face CLI (one-line)

We use it to download both the dataset and the model weights. No auth
needed (both repos are public), but `hf auth login` works too if
you want faster CDN access via your account.

```bash
pip install --upgrade "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1   # 5–10× faster downloads (parallel chunks)
```

> **Note on the binary name.** Newer `huggingface_hub` releases ship the
> CLI as `hf` and print a deprecation warning for `huggingface-cli`.
> Use `hf` (not `huggingface-cli`) below. The flag `--local-dir-use-symlinks`
> was removed in the rewrite — the new default is to download actual files,
> which is what you want anyway.

---

## 4. Download the data (~17 GB)

The dataset is a single `SOCmappingData.zip` that unpacks to the
directory structure the codebase expects.

```bash
mkdir -p /workspace/SOC/Data && cd /workspace/SOC/Data

# Download the zip (uses HF_HUB_ENABLE_HF_TRANSFER for parallel chunks)
hf download ValerianFourel/SOCmappingRastersAndSoilSamples \
    SOCmappingData.zip \
    --repo-type dataset \
    --local-dir .

# Unzip (≈ 17 GB → ≈ 25 GB on disk)
unzip -q SOCmappingData.zip
rm SOCmappingData.zip    # free 17 GB once you've confirmed it unzipped cleanly

# After unzip you should have:
ls -lh /workspace/SOC/Data
# Expected top-level entries:
#   LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx
#   OC_LUCAS_LFU_LfL_Coordinates_v2/
#   RasterTensorData/
#   Coordinates1Mil/
#   Preprocessing/
```

---

## 5. Download the model weights (~10 MB if selective)

You need **two** checkpoints + the matching normalization stats for the
rebuttal:

- **Model A** (Table 2 baseline / evaluation): `residualModels1mil_normalize_composite_l2_v2/`
- **Model B** (operational mapping; used by Experiment 2): `finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/`

```bash
mkdir -p /workspace/SOC/Weights && cd /workspace/SOC/Weights

# Selective download — Model A directory (everything under it, ~5 MB)
hf download \
    ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping \
    --include "TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/*" \
    --local-dir .

# Selective download — Model B (single .pth file, ~4.3 MB)
hf download \
    ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping \
    "TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_log_LOSS_l1_R2_1.0000.pth" \
    --local-dir .

# Verify the two files landed
find . -name "*.pth" -size +1M
# expected:
#   ./TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/TFT_model_BEST_OVERALL_from_run_1_...R2_0.6909.pth
#   ./TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/TFT_model_BEST_OVERALL_from_run_1_...LOSS_l1_R2_1.0000.pth
```

If you want **all** model weights for completeness (other architectures,
residual runs, ~250 MB):

```bash
hf download \
    ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping \
    --local-dir .
```

---

## 6. Point the codebase at the data + weights paths

The `config.py` files hard-code `/home/valerian/SGTPublication/Data/`
and the scripts in `rebuttal/` reference
`/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/`.
Easiest fix: **symlink** instead of editing source. One pair of `ln -s`
keeps every script working unmodified.

```bash
sudo mkdir -p /home/valerian/SGTPublication
sudo chown -R "$(whoami)" /home/valerian/SGTPublication
ln -s /workspace/SOC/Data /home/valerian/SGTPublication/Data
ln -s /workspace/SOC/Weights /home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping
ln -s /workspace/SOC/SOCmapping /home/valerian/SGTPublication/SOCmapping
ln -s /workspace/SOC/SOCmapping/rebuttal /home/valerian/SGTPublication/rebuttal
```

After this:

```bash
ls -la /home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx
ls -la /home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/
```
Both must resolve (symlinks shown with `->`).

> **If you don't want symlinks:** open
> `SOCmapping/SpatiotemporalGatedTransformer/config.py` and change
> `base_path_data = '/home/valerian/SGTPublication/Data'` to
> `base_path_data = '/workspace/SOC/Data'`. Then also edit the four
> `MODEL_B_PATH` / `MODEL_READY` / `PRED_PATH` etc. constants at the top
> of each rebuttal script.

---

## 7. Create the Python venv (one command)

```bash
cd /workspace/SOC/SOCmapping/rebuttal/gpu_experiments
bash setup_venv.sh
```

The script auto-detects CUDA via `nvidia-smi`, picks a matching PyTorch
wheel (cu117 / cu118 / cu121 / cpu), installs everything in
`requirements.txt`, and runs a smoke test (EnhancedSGT instantiation +
forward pass on `(2, 6, 5, 5, 5)`).

Override CUDA detection if needed:

```bash
CUDA_TAG=cu118 bash setup_venv.sh
```

---

## 8. Sanity-check Model B loads

```bash
source /workspace/SOC/SOCmapping/rebuttal/gpu_experiments/.venv/bin/activate
python - <<'PY'
import sys, torch
sys.path.insert(0, '/workspace/SOC/SOCmapping/SpatiotemporalGatedTransformer')
from EnhancedSGT import EnhancedSGT
ck = torch.load(
    '/workspace/SOC/Weights/TemporalFusionTransformer/'
    'finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/'
    'TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_'
    'TIME_END_2023_TRANSFORM_log_LOSS_l1_R2_1.0000.pth',
    map_location='cuda' if torch.cuda.is_available() else 'cpu',
    weights_only=False)
sd = {k.replace('module.', ''): v for k, v in ck['model_state_dict'].items()}
m = EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5, d_model=128,
                num_heads=4, dropout=0.3, num_encoder_layers=3, expansion_factor=4)
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f'params={sum(p.numel() for p in m.parameters() if p.requires_grad):,} '
      f'missing={len(missing)} unexpected={len(unexpected)}')
if torch.cuda.is_available(): m = m.cuda()
x = torch.randn(4, 6, 5, 5, 5, device=next(m.parameters()).device)
y = m(x)
print('forward OK', y.shape, 'device', y.device)
PY
```

Expected: `params=1,120,546  missing=0  unexpected=0` and a `(4,)`
output. Anything else → check that the symlinks resolved and that the
selective HF download captured the right files.

---

## 9. Launch the experiments

```bash
# Single-GPU order: Experiment 2 first (≈ 3 h), then Experiment 1 (≈ 15 h)
cd /workspace/SOC/SOCmapping
source rebuttal/gpu_experiments/.venv/bin/activate

# Experiment 2 — MC dropout uncertainty map (R3.9, R4.4)
python rebuttal/gpu_experiments/uncertainty/mc_dropout_inference.py
python rebuttal/gpu_experiments/uncertainty/plot_uncertainty.py

# Experiment 1 — spatial 5-fold CV (R1.3, R3.6, R3.8)
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py
```

Outputs land alongside each script:

- `spatial_kfold/kfold_results.md` + `figure_kfold.png` + `kfold_predictions_all_folds.parquet`
- `uncertainty/SGT_1mil_2023_mean_mc30.tif` + `_std_mc30.tif` + `figure_uncertainty_3panel.{png,pdf}`

---

## 10. Pull the outputs back to your laptop

```bash
# From your laptop terminal (NOT the SSH session)
rsync -avzP \
  -e "ssh -i ~/.ssh/id_ed25519" \
  "mnp2foff20udyf-64411310@ssh.runpod.io:/workspace/SOC/SOCmapping/rebuttal/gpu_experiments/" \
  ~/SGTPublication/SOCmapping/rebuttal/gpu_experiments/
```

The `--exclude=.venv` flag is optional but recommended (the venv on
Runpod won't match your local CUDA).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi: command not found` | Pod has no GPU — pick a GPU pod template (RTX 3090 / 4090 / A100 etc.) |
| `hf` download stalls | `unset HF_HUB_ENABLE_HF_TRANSFER` and retry; falls back to plain requests |
| `huggingface-cli: command is deprecated` | The CLI was renamed to `hf` in `huggingface_hub` ≥ 0.27. Replace `huggingface-cli` with `hf` and drop `--local-dir-use-symlinks` (removed). |
| `Coordinates ({lon},{lat}) not found in Elevation` at training time | Data didn't unzip into the right structure — re-verify `/workspace/SOC/Data/OC_LUCAS_LFU_LfL_Coordinates_v2/StaticValue/Elevation/coordinates.npy` exists |
| `torch.cuda.OutOfMemoryError` in MC dropout | drop `BATCH_SIZE` in `mc_dropout_inference.py` from 256 → 128 |
| Push back to GitHub from Runpod fails | Same as locally — set up a PAT or SSH key on the pod before `git push` |
| `setup_venv.sh` smoke test prints `WARN: SGT source tree not at /home/...` | Symlinks (step 6) weren't created — make them or edit hard-coded paths |
