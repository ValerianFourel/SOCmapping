# Next steps — what's left after the May 2026 session

All required steps are complete. Outstanding items are optional follow-ups.

For the full chronological log of what was done, see
[`SESSION_LOG_2026-05-17.md`](./SESSION_LOG_2026-05-17.md).

---

## ✅ Pipeline complete

| Phase | Status |
|---|---|
| plan | ✅ done |
| gee | ✅ done — 251 GEE tasks SUCCEEDED |
| pull | ✅ done — 251 TIFFs in `~/bavaria_tiffs/` |
| cut | ✅ done — 3 012 canonical npy tiles in `Data/RasterTensorData/` |
| project | ✅ done — `samples` (30 227) + `1mil` (1 300 000) coords for all 20 bands |
| verify | ✅ done — 250 OK / 1 minor WARN / 0 FAIL audit; dataloader test passed |
| **HF push** | ✅ **done — 9 366 files at `ValerianFourel/sgt-bavaria-soc-2002-2023`** |

---

## Optional follow-ups

### 1. Retrain the model with 20 channels

```bash
cd SOCmapping/TemporalFusionTransformer
python train.py --use_validation --model-size big --num-runs 1 --num_epochs 200
```

- `bands_list_order` is already 20 in every model's `config.py`
- `input_channels = len(bands_list_order)` auto-applies in `train.py`
- Pretrained 6-channel checkpoints will need `load_state_dict(strict=False)` + reinit of the first conv layer

### 2. Make the HF dataset public

If intended for publication, flip from private to public via either:
- HF web UI: settings → "Make public"
- CLI: `python3 -c "from huggingface_hub import HfApi; HfApi().update_repo_visibility('ValerianFourel/sgt-bavaria-soc-2002-2023', private=False, repo_type='dataset')"`

### 3. Slim the HF repo (optional, saves ~5 GB)

The `RasterTensorData/SeasonalValue/` subtree (4 608 files, ~5 GB) was uploaded because it exists in `Data/`. It's the legacy seasonal-aggregated tiles for LAI/LST/SoilEvap/TotalEvap from an older pipeline; it does NOT include the 14 new bands so it's only useful for seasonal-mode training on the 5 original bands.

To delete from HF if not needed:

```python
from huggingface_hub import HfApi
HfApi().delete_folder(
    'RasterTensorData/SeasonalValue',
    repo_id='ValerianFourel/sgt-bavaria-soc-2002-2023',
    repo_type='dataset',
)
```

### 4. Fix dataloader RAM usage (single-line edit)

`RasterTensorDataset.__init__` in
`SOCmapping/TemporalFusionTransformer/dataloader/dataloaderMultiYears.py`
eagerly pre-loads all 12 tiles per (band, year) into `self.data_cache`.
For 20 × 22 = 440 datasets that's ~20 GB — OOMs on 16 GB laptops.

To lazy-load:

```python
# In RasterTensorDataset.__init__, replace:
self.data_cache = {}
for id_num, filepath in self.id_to_file.items():
    self.data_cache[id_num] = np.load(filepath)
# with:
self.data_cache = {}   # populated on first access in get_tensor_by_location
```

The `get_tensor_by_location` method already handles the empty-cache path:
```python
if id_num in self.data_cache:
    data = self.data_cache[id_num]
else:
    data = np.load(self.id_to_file[id_num])   # ← this path
```

(Optionally cache `data` on first read for faster subsequent samples.)

### 5. Drive cleanup

Orphan CHIRPS TIFFs and aborted GEE task outputs are in Drive Trash.
Empty the trash via Drive web UI if you want to reclaim quota.

---

## Consumer reference — pulling the dataset

```bash
pip install huggingface_hub
huggingface-cli login    # required (repo is private)

python3 SOCmapping/SamplePoints/pull_from_hf.py \
    --repo-id ValerianFourel/sgt-bavaria-soc-2002-2023 \
    --out ./Data_HF_pulled

export SOC_DATA_DIR=$(realpath ./Data_HF_pulled)
# The SOCmapping pipeline picks this up via _paths.py — no other config needed.
```

---

## What's untouched (kept as-is by design)

- **Model code** — no architecture changes. Only `bands_list_order` + per-band paths in `config.py`.
- **Dataloader** (`dataloaderMultiYears.py`) — not modified. The static-as-yearly symlink trick avoided the only point where modification would have been needed.
- **Training loop** (`train.py`) — not modified.
- **Existing checkpoints** (`_hf_full_weights_repo/`) — not modified. Compatible with 20-channel input via `strict=False` + first-conv reinit.
- **Original `Data/`** — preserved. New tiles added, 6 pre-existing bands recut onto the canonical grid, but `RasterBandsData/` (legacy v1) untouched.
