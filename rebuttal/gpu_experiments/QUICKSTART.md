# SOCmapping rebuttal — Runpod quickstart (copy-paste)

Single-page, command-only. For background / troubleshooting see
`RUNPOD_SETUP.md` (longer doc) and `checkpoint_note.md`.

**Pod requirement:** ≥ 60 GB `/workspace` volume, any NVIDIA GPU with
≥ 12 GB VRAM (RTX 3090 / 4090 / A100 / L40S all work).

---

## 1 — System prep

```bash
apt update && apt install -y \
    git git-lfs curl unzip build-essential \
    python3.10-venv python3-pip
git lfs install
nvidia-smi | head -5
```

> `python3.10-venv` is mandatory — Ubuntu's stock `python3.10` doesn't
> include `venv` and `bash setup_venv.sh` will fail with
> `No module named ensurepip` if it's missing.

## 2 — Clone code

```bash
mkdir -p /workspace/SOC && cd /workspace/SOC
git clone https://github.com/ValerianFourel/SOCmapping.git
cd SOCmapping && git log --oneline -5
```

## 3 — Install HF CLI

```bash
pip install --upgrade "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc
```

## 4 — Download dataset (≈ 17 GB zip → ≈ 25 GB unzipped)

```bash
mkdir -p /workspace/SOC/Data && cd /workspace/SOC/Data
hf download ValerianFourel/SOCmappingRastersAndSoilSamples \
    SOCmappingData.zip --repo-type dataset --local-dir .
df -h /workspace                              # confirm ≥ 30 GB free
unzip -q SOCmappingData.zip
rm SOCmappingData.zip
ls /workspace/SOC/Data/Data                   # the zip nests under Data/
```

## 5 — Download model weights (≈ 10 MB)

```bash
mkdir -p /workspace/SOC/Weights && cd /workspace/SOC/Weights

# Model A (eval / Table 2 baseline)
hf download ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping \
    --include "TemporalFusionTransformer/residualModels1mil_normalize_composite_l2_v2/*" \
    --local-dir .

# Model B (operational mapping — input to Experiment 2)
hf download ValerianFourel/Weights-ResidualsModels-MappingInference-SOCmapping \
    "TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023_TRANSFORM_log_LOSS_l1_R2_1.0000.pth" \
    --local-dir .

find . -name "*.pth" -size +1M                # expect 2 .pth files
```

## 6 — Symlinks (so hard-coded `/home/valerian/...` paths resolve)

```bash
mkdir -p /home/valerian/SGTPublication
ln -s /workspace/SOC/Data/Data /home/valerian/SGTPublication/Data
ln -s /workspace/SOC/Weights /home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping
ln -s /workspace/SOC/SOCmapping /home/valerian/SGTPublication/SOCmapping
ln -s /workspace/SOC/SOCmapping/rebuttal /home/valerian/SGTPublication/rebuttal

# Verify (no broken symlinks)
ls /home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx
ls /home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/
```

## 7 — Build venv (auto-detects CUDA, ≈ 3 GB)

```bash
cd /workspace/SOC/SOCmapping/rebuttal/gpu_experiments
bash setup_venv.sh
```

Expected last line: `forward pass OK, output shape: (2,) device=cuda:0`.

## 8 — Sanity-load Model B

```bash
source /workspace/SOC/SOCmapping/rebuttal/gpu_experiments/.venv/bin/activate
python - <<'PY'
import sys, torch
sys.path.insert(0, '/workspace/SOC/SOCmapping/SpatiotemporalGatedTransformer')
from EnhancedSGT import EnhancedSGT
ck = torch.load(
    '/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/'
    'TemporalFusionTransformer/finalResults2023_1milVersion_TRANSFORM_log_LOSS_l1/'
    'TFT_model_BEST_OVERALL_from_run_1_MAX_OC_150_TIME_BEGINNING_2007_'
    'TIME_END_2023_TRANSFORM_log_LOSS_l1_R2_1.0000.pth',
    map_location='cuda', weights_only=False)
sd = {k.replace('module.', ''): v for k, v in ck['model_state_dict'].items()}
m = EnhancedSGT(input_channels=6, height=5, width=5, time_steps=5, d_model=128,
                num_heads=4, dropout=0.3, num_encoder_layers=3,
                expansion_factor=4).cuda()
miss, extra = m.load_state_dict(sd, strict=False)
print(f'params={sum(p.numel() for p in m.parameters() if p.requires_grad):,}  '
      f'missing={len(miss)}  unexpected={len(extra)}')
print('forward OK', m(torch.randn(4, 6, 5, 5, 5, device="cuda")).shape)
PY
```

Expected: `params=1,120,546  missing=0  unexpected=0  forward OK torch.Size([4])`.

## 9 — Launch experiments

```bash
source /workspace/SOC/SOCmapping/rebuttal/gpu_experiments/.venv/bin/activate
cd /workspace/SOC/SOCmapping

# Experiment 2 — MC dropout uncertainty map (~3 GPU-hours)
python rebuttal/gpu_experiments/uncertainty/mc_dropout_inference.py
python rebuttal/gpu_experiments/uncertainty/plot_uncertainty.py

# Experiment 1 — spatial 5-fold CV (~15 GPU-hours)
python rebuttal/gpu_experiments/spatial_kfold/run_kfold.py
```

## 10 — Push outputs to the SOCrebuttal HF dataset

```bash
export HF_TOKEN="hf_xxx_paste_a_write_token_from_huggingface.co/settings/tokens"
python /workspace/SOC/SOCmapping/rebuttal/gpu_experiments/upload_to_hf.py
# → https://huggingface.co/datasets/ValerianFourel/SOCrebuttal
```

## 11 — (Optional) rsync outputs to your laptop

```bash
# From your LAPTOP, not the SSH session:
rsync -avzP --exclude=.venv \
  -e "ssh -i ~/.ssh/id_ed25519" \
  "<USER>@ssh.runpod.io:/workspace/SOC/SOCmapping/rebuttal/gpu_experiments/" \
  ~/SGTPublication/SOCmapping/rebuttal/gpu_experiments/
```

---

## If something breaks

| symptom | fix |
|---|---|
| `bash: unzip: command not found` | `apt install -y unzip` |
| `No module named ensurepip` from `setup_venv.sh` | `apt install -y python3.10-venv` |
| `nvidia-smi: command not found` | pod has no GPU — switch template |
| `hf` not found | `pip install --upgrade "huggingface_hub[cli,hf_transfer]"` |
| HF download stalls | `unset HF_HUB_ENABLE_HF_TRANSFER` and retry |
| `Coordinates (lon,lat) not found in Elevation` mid-training | symlink in step 6 is wrong — check it resolves to the inner `Data/Data/` not the outer `Data/` |
| `torch.cuda.OutOfMemoryError` in MC dropout | drop `BATCH_SIZE` in `mc_dropout_inference.py` from 256 → 128 |
| `huggingface-cli: command is deprecated` | use `hf` (CLI was renamed in `huggingface_hub` ≥ 0.27) |
| Disk full during unzip | `df -h /workspace`, resize pod volume to ≥ 60 GB, restart, re-run step 4 |
