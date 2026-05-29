#!/usr/bin/env bash
# Bavaria 2002-2023 data prep — end-to-end driver.
#
# Default CATEGORY=extended produces the 43-band on-disk layout (curated 20 +
# Tier 1/2/3 revision covariates). The on-disk folder layout is identical to
# before — one folder per band — so the data prep needs no dataloader change,
# but USING the new bands in a model requires adding them to each model's
# config.py band lists (see SOCmapping/_bands.py + bands_list_order).
#
#   Data/RasterTensorData/StaticValue/Elevation/ID*.npy            (existing)
#   Data/RasterTensorData/YearlyValue/<band>/<year>/ID*.npy        (yearly bands incl. SRC + Tier-3)
#   Data/OC_LUCAS_LFU_LfL_Coordinates_v2/...                       (per-band coordinates.npy)
#
# Each phase is gated by environment variables so you can re-enter
# the pipeline midway. Defaults assume an interactive run.
#
# Phases:
#   1. plan      — dry-run summary of the GEE batch (always runs first)
#   2. gee       — submit GEE export tasks (server-side; ~1-12h to complete).
#                  Includes Slope/Aspect/TWI as server-side derivations from
#                  SRTM + MERIT Hydro — no local DEM compute needed.
#   3. pull      — download exported TIFFs from Google Drive (gdown)
#   4. cut       — slice TIFFs into 12-tile npy layout
#   5. project   — project LUCAS GPS onto each (band, year) tile grid
#   6. verify    — sanity-test tile counts + dataloader instantiation
#   7. hf        — build Data_HF mirror + push to HuggingFace
#                  (opt-in: only runs when HF_REPO_ID is set)
#
# Skip any phase via env var, e.g. SKIP_GEE=1.
# Or run a single phase: bash run_full_pipeline.sh gee
#
# Legacy `derive` phase (local Slope/Aspect/TWI from existing Elevation tiles)
# is kept as a callable single-phase fallback for users who can't reach GEE:
#   bash run_full_pipeline.sh derive

set -e
set -u

# ── Configuration ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"
DRIVE_FOLDER="${DRIVE_FOLDER:-bavaria_bands_2002_2023}"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-}"   # required for the pull phase
TIFF_LOCAL_DIR="${TIFF_LOCAL_DIR:-${HOME}/bavaria_tiffs}"
YEAR_START="${YEAR_START:-2002}"
YEAR_END="${YEAR_END:-2023}"
# GEE band set to pull. 'extended' = curated 20 + Tier 1/2/3 revision
# covariates (Landsat SRC, multi-scale terrain, climate/phenology derivations).
# Override with CATEGORY=curated to reproduce the original 20-band pull.
CATEGORY="${CATEGORY:-extended}"
# HuggingFace publish (hf phase). Opt-in: the hf phase is skipped unless
# HF_REPO_ID is set (e.g. HF_REPO_ID=user/sgt-bavaria-soc). HF_CREATE_REPO=1
# creates the repo on first push; HF_PRIVATE=1 makes it private.
HF_REPO_ID="${HF_REPO_ID:-}"

# Bands to materialize-as-yearly (statics that need symlinks across years):
# SoilGrids soil properties, the SRTM terrain derivatives (Slope/Aspect/TWI,
# server-side from SRTM + MERIT Hydro), and the Tier-2 multi-scale terrain.
STATIC_AS_YEARLY=(
    ClayContent_0_10cm SandContent_0_10cm pH_H2O_0_10cm
    BulkDensity_0_10cm CEC_0_10cm
    Slope Aspect TWI
    TPI_90 TPI_300 TPI_1000 TRI Roughness
)
# Pure-yearly bands (one TIFF per year, regular cut): the original MODIS/ERA5
# family, plus Tier-1 Landsat SRC (11) and Tier-3 derivations (7).
YEARLY_BANDS=(
    LAI LST MODIS_NPP SoilEvaporation TotalEvapotranspiration
    NDVI EVI Precipitation AirTemperature SoilMoisture_layer1 SnowDepth
    SRC_Blue SRC_Green SRC_Red SRC_NIR SRC_SWIR1 SRC_SWIR2
    SRC_RCC SRC_BCC SRC_NBR2 SRC_BSI SRC_ExposureCount
    ClimaticWaterBalance SoilTemperature_layer1 FrostDays GrowingDegreeDays
    NDVI_Amplitude NDVI_Integral EVI_Amplitude
)
# Tier-1 SRC bands are sparse (bare soil exposed only on arable land): their
# NoData must be filled with 0 at cut time, NOT the tile mean (which would
# fabricate soil reflectance over grassland/forest). SRC_ExposureCount carries
# per-pixel validity (0 = no bare-soil observation that year).
FILL_ZERO_BANDS=(
    SRC_Blue SRC_Green SRC_Red SRC_NIR SRC_SWIR1 SRC_SWIR2
    SRC_RCC SRC_BCC SRC_NBR2 SRC_BSI SRC_ExposureCount
)
# Legacy local-derivation fallback (kept for the optional `derive` phase
# when GEE access is unavailable).
DERIVED_BANDS=(Slope Aspect TWI)

# ── Helpers ────────────────────────────────────────────────────────
log()  { printf '\n\033[1;34m[%s]\033[0m %s\n' "$1" "$2"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m⚠\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m✗\033[0m %s\n' "$1" >&2; }

# Returns the JSON status of a phase: 'done' | 'in_progress' | 'pending' | 'missing'.
phase_state() {
    python "${SCRIPT_DIR}/pipeline_state.py" show 2> /dev/null \
        | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('phases',{}).get(sys.argv[1],{}).get('status','missing'))" "$1" 2> /dev/null \
        || echo missing
}

run_phase() {
    local phase="$1"
    [ -n "${ONLY_PHASE:-}" ] && [ "$phase" != "${ONLY_PHASE}" ] && return 0
    local skip_var="SKIP_$(echo "$phase" | tr '[:lower:]' '[:upper:]')"
    [ -n "${!skip_var:-}" ] && { warn "skipping phase '$phase' (\$$skip_var set)"; return 0; }

    # Resume behavior: skip phases marked 'done' in pipeline_state.json
    # unless --force is passed (FORCE=1).
    if [ -z "${FORCE:-}" ]; then
        local st
        st="$(phase_state "$phase")"
        if [ "$st" = "done" ]; then
            ok "phase '$phase' already done — skipping (use FORCE=1 to re-run)"
            return 0
        fi
    fi
    "phase_${phase}"
}

# ── Phases ─────────────────────────────────────────────────────────

phase_plan() {
    log plan "GEE export dry-run for --category ${CATEGORY}, years ${YEAR_START}-${YEAR_END}"
    python "${SCRIPT_DIR}/gee_download_all_bands.py" \
        --category "${CATEGORY}" --years "${YEAR_START}" "${YEAR_END}" --dry-run
    python "${SCRIPT_DIR}/pipeline_state.py" init > /dev/null
    python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from pipeline_state import State
s = State(); s.start_phase('plan'); s.finish_phase('plan')
"
    ok "Plan printed."
}

phase_derive() {
    log derive "Computing Slope/Aspect/TWI from existing Elevation tiles"
    python "${SCRIPT_DIR}/compute_dem_derivatives.py"
    log derive "Projecting LUCAS onto derived bands"
    python "${SCRIPT_DIR}/project_lucas_coords.py" \
        --tier YearlyValue --bands "${DERIVED_BANDS[@]}" \
        --materialize-yearly "${DERIVED_BANDS[@]}" \
        --anchor-year "${YEAR_START}" --mat-years "${YEAR_START}" "${YEAR_END}"
    ok "Slope/Aspect/TWI ready under YearlyValue/."
}

phase_gee() {
    log gee "Submitting --category ${CATEGORY} GEE export batch for ${YEAR_START}-${YEAR_END} (extended ≈ 652 tasks: incl. 11×22 Landsat SRC; curated ≈ 251)"
    if ! python -c "import ee" 2> /dev/null; then
        warn "earthengine-api not installed — skipping GEE submission."
        warn "To enable: pip install earthengine-api && earthengine authenticate"
        warn "This is OK if you're testing the local phases against the synthetic-data scaffold."
        return 0
    fi
    warn "Tip: run 'gee_download_all_bands.py --category ${CATEGORY} --validate-assets' first to catch any bad asset/band names before this batch."
    python "${SCRIPT_DIR}/gee_download_all_bands.py" \
        --category "${CATEGORY}" --years "${YEAR_START}" "${YEAR_END}" \
        --drive-folder "${DRIVE_FOLDER}"
    ok "Submitted. Monitor at https://code.earthengine.google.com/tasks"
    warn "Wait for ALL tasks to finish before running the 'pull' phase."
    warn "Check with: earthengine task list | head"
}

phase_pull() {
    log pull "Downloading exported TIFFs from Drive to ${TIFF_LOCAL_DIR}"
    if [ -z "${DRIVE_FOLDER_ID}" ]; then
        warn "DRIVE_FOLDER_ID not set — skipping pull phase."
        warn "Find it in your browser URL: drive.google.com/drive/folders/<ID>"
        warn "Re-run with: DRIVE_FOLDER_ID=<id> bash $0 pull"
        return 0
    fi
    python "${SCRIPT_DIR}/pull_from_drive.py" \
        --folder-id "${DRIVE_FOLDER_ID}" --out "${TIFF_LOCAL_DIR}"
    ok "Downloaded to ${TIFF_LOCAL_DIR}"
}

phase_cut() {
    log cut "Cutting TIFFs into 12-tile npy layout"
    if [ ! -d "${TIFF_LOCAL_DIR}" ] || [ -z "$(ls -A "${TIFF_LOCAL_DIR}" 2>/dev/null)" ]; then
        warn "No TIFFs at ${TIFF_LOCAL_DIR} — skipping cut phase."
        warn "Run the gee + pull phases (or stage TIFFs manually) before cut."
        return 0
    fi
    python "${SCRIPT_DIR}/tiff_to_tiles.py" \
        --tiff-dir "${TIFF_LOCAL_DIR}" \
        --materialize-yearly "${STATIC_AS_YEARLY[@]}" \
        --fill-zero "${FILL_ZERO_BANDS[@]}" \
        --anchor-year "${YEAR_START}" --mat-years "${YEAR_START}" "${YEAR_END}"
    ok "TIFF → npy tiles done."
}

phase_project() {
    log project "Projecting GPS points onto each (band, year) tile grid (samples + 1.3M grid)"
    DATA_DIR="${SOC_DATA_DIR:-${PROJECT_ROOT}/Data}"
    if [ ! -d "${DATA_DIR}/RasterTensorData/YearlyValue/NDVI" ]; then
        warn "No NDVI tiles at ${DATA_DIR}/RasterTensorData/YearlyValue/NDVI — skipping project phase."
        warn "Run the cut phase first (or use scaffold_synthetic_bands.py for test mode)."
        return 0
    fi

    # Project against BOTH ground-truth (LUCAS+LFU+LfL combined ~30k samples)
    # for training AND the 1.3M Bavaria grid for inference.
    for mode in samples 1mil; do
        log project "  --mode ${mode}"

        # Existing Elevation lands in StaticValue/ (NOT materialize-yearly).
        # It's a separate projection call because the tier differs.
        if [ -d "${DATA_DIR}/RasterTensorData/StaticValue/Elevation" ]; then
            python "${SCRIPT_DIR}/project_lucas_coords.py" --mode "${mode}" \
                --tier StaticValue --bands Elevation
        fi

        # Pure yearly bands: one coordinates.npy anchor → symlinked across 22 years.
        python "${SCRIPT_DIR}/project_lucas_coords.py" --mode "${mode}" \
            --tier YearlyValue --bands "${YEARLY_BANDS[@]}" \
            --symlink-across-years

        # Materialized statics: project once at anchor year, symlink the rest.
        python "${SCRIPT_DIR}/project_lucas_coords.py" --mode "${mode}" \
            --tier YearlyValue --bands "${STATIC_AS_YEARLY[@]}" \
            --materialize-yearly "${STATIC_AS_YEARLY[@]}" \
            --anchor-year "${YEAR_START}" --mat-years "${YEAR_START}" "${YEAR_END}"
    done

    ok "All coordinates.npy files (LUCAS + 1mil) in place."
}

phase_verify() {
    log verify "Running sanity tests"
    DATA_DIR="${SOC_DATA_DIR:-${PROJECT_ROOT}/Data}"
    python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from pipeline_state import State
State().start_phase('verify')
"
    local total_bands=0 ok_bands=0
    for b in "${YEARLY_BANDS[@]}" "${STATIC_AS_YEARLY[@]}"; do
        total_bands=$((total_bands + 1))
        local missing_years=0
        for y in "${YEAR_START}" 2010 "${YEAR_END}"; do
            n=$(ls "${DATA_DIR}/RasterTensorData/YearlyValue/${b}/${y}/"ID*.npy 2> /dev/null | wc -l || echo 0)
            if [ "$n" -ne 12 ]; then
                warn "${b}/${y}: ${n} tiles (expect 12)"
                missing_years=$((missing_years + 1))
            fi
        done
        [ "$missing_years" -eq 0 ] && { ok "${b}: 12 tiles × 3 sampled years ✓"; ok_bands=$((ok_bands + 1)); }
    done
    echo
    echo "Band sanity: ${ok_bands}/${total_bands} pass tile-count check."

    log verify "Dataloader smoke test (TFT config)"
    python <<PY
import sys
sys.path.insert(0, '${PROJECT_ROOT}/SOCmapping/TemporalFusionTransformer')
from config import bands_list_order, SamplesCoordinates_Yearly, DataYearly
assert len(bands_list_order) == len(SamplesCoordinates_Yearly) == len(DataYearly), \\
    f'length mismatch: {len(bands_list_order)} vs {len(SamplesCoordinates_Yearly)} vs {len(DataYearly)}'
print(f'✓ bands_list_order has {len(bands_list_order)} entries, all lists aligned')
print('  bands:', bands_list_order)
PY
    log verify "End-to-end pipeline test (instantiates dataloader, pulls __getitem__(0))"
    python "${SCRIPT_DIR}/test_pipeline.py"

    python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from pipeline_state import State
State().finish_phase('verify')
"
    ok "All sanity tests passed."
}

phase_hf() {
    log hf "Build Data_HF mirror + push to HuggingFace"
    # Opt-in publish: gated on HF_REPO_ID so a normal pipeline run never
    # pushes to a shared dataset by accident.
    if [ -z "${HF_REPO_ID}" ]; then
        warn "HF_REPO_ID not set — skipping hf phase (publish is opt-in)."
        warn "Re-run with: HF_REPO_ID=<username>/<repo> bash $0 hf"
        warn "  HF_CREATE_REPO=1 on first push to create the repo; HF_PRIVATE=1 for private."
        return 0
    fi
    if ! python -c "import huggingface_hub" 2> /dev/null; then
        warn "huggingface_hub not installed — skipping hf phase."
        warn "To enable: pip install huggingface_hub && huggingface-cli login"
        return 0
    fi
    DATA_DIR="${SOC_DATA_DIR:-${PROJECT_ROOT}/Data}"
    if [ ! -d "${DATA_DIR}/RasterTensorData/YearlyValue" ]; then
        warn "No ${DATA_DIR}/RasterTensorData/YearlyValue — run cut/project first; nothing to publish."
        return 0
    fi

    log hf "  prepare_hf_export.py (rebuild Data_HF/ mirror + manifest.json/README)"
    python "${SCRIPT_DIR}/prepare_hf_export.py"

    local create_flag="" private_flag=""
    [ -n "${HF_CREATE_REPO:-}" ] && create_flag="--create-repo"
    [ -n "${HF_PRIVATE:-}" ] && private_flag="--private"

    log hf "  push_to_hf.py --repo-id ${HF_REPO_ID} ${create_flag} ${private_flag}"
    python "${SCRIPT_DIR}/push_to_hf.py" --repo-id "${HF_REPO_ID}" ${create_flag} ${private_flag}
    ok "Pushed to https://huggingface.co/datasets/${HF_REPO_ID}"
}

# ── Driver ─────────────────────────────────────────────────────────
# `derive` is no longer in the default sequence — Slope/Aspect/TWI come
# from GEE now. It remains callable as a single-phase fallback.
PHASES=(plan gee pull cut project verify hf)

if [ "$#" -gt 0 ]; then
    ONLY_PHASE="$1"
    case "${ONLY_PHASE}" in
        plan|derive|gee|pull|cut|project|verify|hf) ;;
        *) err "Unknown phase '${ONLY_PHASE}'. Valid: ${PHASES[*]}"; exit 2 ;;
    esac
    run_phase "${ONLY_PHASE}"
    exit 0
fi

echo "============================================================"
echo "Bavaria 2002-2023 data prep — full pipeline"
echo "  Drive folder:  ${DRIVE_FOLDER}"
echo "  Local TIFFs:   ${TIFF_LOCAL_DIR}"
echo "  Years:         ${YEAR_START}-${YEAR_END}"
echo "  GEE category:  ${CATEGORY}"
echo "  Yearly bands:      ${YEARLY_BANDS[*]}"
echo "  Statics-as-yearly: ${STATIC_AS_YEARLY[*]}"
echo "    (Slope/Aspect/TWI come from GEE — no local DEM compute)"
echo "  HF publish:    ${HF_REPO_ID:-<disabled; set HF_REPO_ID to enable hf phase>}"
echo "============================================================"

# Resume info — show what's already done so the user knows what'll run.
log resume "Pipeline state (resume info)"
python "${SCRIPT_DIR}/pipeline_state.py" summary || warn "no state file yet"

for p in "${PHASES[@]}"; do
    run_phase "$p"
done

log final "Final pipeline state"
python "${SCRIPT_DIR}/pipeline_state.py" summary
echo
ok "Pipeline complete."
