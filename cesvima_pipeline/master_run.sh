#!/bin/bash
# ==============================================================================
#      MASTER ORCHESTRATOR FOR CESVIMA HIGH-DIM FINSTOD PIPELINE
# ==============================================================================
# Usage:
#   1. Run ALL phases end-to-end (Canary + All Systems + Synthesis):
#      ./master_run.sh all
#      sbatch master_pipeline.slurm
#
#   2. Run selected phases (e.g. Phase 0, Phase 1, Phase 5):
#      ./master_run.sh --phases 0,1,5
#      sbatch --export=ALL,PHASES="0,1,5" master_pipeline.slurm
#
#   3. Run a single specific configuration:
#      ./master_run.sh configs/prod_lorenz96_m6.yaml
#      sbatch --export=ALL,CFG=configs/prod_lorenz96_m6.yaml master_pipeline.slurm
#
#   4. Run a tau series (new tags <prod>_tauNN; never overwrites the template):
#      ./master_run.sh configs/series_abc_fput_m8_tau.yaml
#      sbatch --export=ALL,PHASES="11" master_pipeline.slurm
#
#   5. FPUT modal S-bar aux on existing FinSTOD runs (not part of `all`):
#      ./master_run.sh configs/aux_fput_sbar.yaml
#      sbatch --export=ALL,PHASES="12" master_pipeline.slurm
#
#   6. Interpretation campaign (Bickley/Duffing/FPUT identity tests; new tags):
#      sbatch --export=ALL,PHASES="13" master_pipeline.slurm
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 1. SETUP ENVIRONMENT & LOGGING ---
mkdir -p logs

module --force purge > /dev/null 2>&1 || true
module load apps/2021 > /dev/null 2>&1 || true
module load Python/3.10.8-GCCcore-12.2.0 > /dev/null 2>&1 || true

if [[ -f "../generic_env/bin/activate" ]]; then
    source ../generic_env/bin/activate
elif [[ -f "generic_env/bin/activate" ]]; then
    source generic_env/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Detect Python executable
if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" >/dev/null 2>&1; then
    PY_BIN="python3"
elif command -v python >/dev/null 2>&1 && python -c "import numpy" >/dev/null 2>&1; then
    PY_BIN="python"
else
    PY_BIN="python3"
fi

export PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Integer-only counters so the watch line cannot inject a newline into python -c.
count_result_npzs() {
    local dir="$1"
    $PY_BIN -c "import glob,sys; print(len(glob.glob(sys.argv[1])))" "${dir}/result_*.npz"
}

count_squeue() {
    local jid="$1"
    squeue -h -j "$jid" 2>/dev/null | $PY_BIN -c "import sys; print(sum(1 for line in sys.stdin if line.strip()))"
}

array_watch_line() {
    $PY_BIN -c "from stod_nd.progress import format_array_watch; print(format_array_watch($1, $2, $3, $4))"
}

watch_slurm_array() {
    local JOB_ID="$1"
    local TAG="$2"
    local RESULT_DIR="$3"
    local NUM_SHARDS="$4"
    local SLEEP_S="${5:-12}"
    local T0 NOW DONE RUNNING
    mkdir -p "$RESULT_DIR"
    T0=$(date +%s)
    echo "    Tracking $NUM_SHARDS shards in $RESULT_DIR"
    while true; do
        RUNNING=$(count_squeue "$JOB_ID")
        RUNNING=${RUNNING:-0}
        DONE=$(count_result_npzs "$RESULT_DIR")
        DONE=${DONE:-0}
        NOW=$(date +%s)
        ELAPSED=$((NOW - T0))
        echo "    [$(date +%H:%M:%S)] $TAG  $(array_watch_line "$DONE" "$NUM_SHARDS" "$ELAPSED" "$RUNNING")"
        if [[ "$RUNNING" -eq 0 ]]; then
            break
        fi
        sleep "$SLEEP_S"
    done
    DONE=$(count_result_npzs "$RESULT_DIR")
    echo "    Array $JOB_ID finished: ${DONE:-0}/$NUM_SHARDS shard files on disk"
}

echo "======================================================================="
echo "   CESVIMA HIGH-DIMENSIONAL FINSTOD MASTER PIPELINE"
echo "   Host  : $(hostname)"
echo "   Start : $(date)"
echo "   Python: $($PY_BIN --version 2>&1) ($PY_BIN)"
echo "======================================================================="

# --- FUNCTION: Run a single configuration file end-to-end ---
run_single_config() {
    local CFG_FILE="$1"
    local RUN_TAG_OVERRIDE="${2:-}"
    local N_LEVELS_OVERRIDE="${3:-}"
    if [[ ! -f "$CFG_FILE" ]]; then
        echo "ERROR: Config file not found: $CFG_FILE" >&2
        return 1
    fi

    local GEN_ARGS=(--config "$CFG_FILE")
    if [[ -n "$RUN_TAG_OVERRIDE" ]]; then
        GEN_ARGS+=(--run_tag "$RUN_TAG_OVERRIDE")
    fi
    if [[ -n "$N_LEVELS_OVERRIDE" ]]; then
        GEN_ARGS+=(--n_levels "$N_LEVELS_OVERRIDE")
    fi

    local RUN_DIR
    RUN_DIR=$($PY_BIN -c "
from pipeline_core.generate_tasks import load_config, resolve_run_dir
cfg = load_config('$CFG_FILE')
tag = '${RUN_TAG_OVERRIDE}' or None
run_dir, _ = resolve_run_dir(cfg, run_tag=tag)
print(run_dir)
")

    if [[ -f "${RUN_DIR}/results_consolidated.npz" ]]; then
        echo "  SKIP (already complete, will not overwrite): ${RUN_DIR}"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " >>> EXECUTING CONFIG: $CFG_FILE"
    if [[ -n "$RUN_TAG_OVERRIDE" ]]; then
        echo "     run_tag=$RUN_TAG_OVERRIDE  n_levels=${N_LEVELS_OVERRIDE:-from yaml}"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. Generate task shards (skip if this tag already has meta from a crashed master)
    local META_FILE="${RUN_DIR}/meta.json"
    if [[ -f "$META_FILE" ]]; then
        echo "  [1/3] Resume: shards already present at $RUN_DIR"
    else
        echo "  [1/3] Sharding continuous grid domain into parallel tasks..."
        $PY_BIN -u pipeline_core/generate_tasks.py "${GEN_ARGS[@]}"
    fi
    if [[ ! -f "$META_FILE" ]]; then
        echo "ERROR: Could not locate generated meta.json at $META_FILE" >&2
        return 1
    fi

    local RUN_TAG
    RUN_TAG=$($PY_BIN -c "import json; print(json.load(open('$META_FILE')).get('run_tag', 'run_default'))")

    local NUM_SHARDS
    NUM_SHARDS=$($PY_BIN -c "import json; print(json.load(open('$META_FILE'))['num_shards'])")
    local MAX_TASK_ID=$((NUM_SHARDS - 1))
    local ARRAY_ID_FILE="${RUN_DIR}/slurm_array_id.txt"

    echo "  [2/3] Computing FinSTOD and FTLE for $NUM_SHARDS shards..."

    # 2. Worker Array Execution
    if command -v sbatch >/dev/null 2>&1; then
        local WORKER_JOB_ID=""
        if [[ -f "$ARRAY_ID_FILE" ]]; then
            WORKER_JOB_ID=$(tr -d '[:space:]' < "$ARRAY_ID_FILE")
            if [[ -z "$WORKER_JOB_ID" || "$(count_squeue "$WORKER_JOB_ID")" -eq 0 ]]; then
                WORKER_JOB_ID=""
            else
                echo "    Reattaching to running array $WORKER_JOB_ID"
            fi
        fi
        local DONE_NOW
        DONE_NOW=$(count_result_npzs "${RUN_DIR}/results")
        DONE_NOW=${DONE_NOW:-0}
        if [[ -z "$WORKER_JOB_ID" && "$DONE_NOW" -ge "$NUM_SHARDS" ]]; then
            echo "    All $NUM_SHARDS shard files present, skipping sbatch"
        elif [[ -z "$WORKER_JOB_ID" ]]; then
            echo "    Submitting Slurm array job (Task IDs: 0-$MAX_TASK_ID)..."
            WORKER_JOB_ID=$(sbatch --parsable \
                --array=0-${MAX_TASK_ID} \
                --export=ALL,RUN_DIR="${RUN_DIR}" \
                worker.slurm)
            echo "$WORKER_JOB_ID" > "$ARRAY_ID_FILE"
            echo "    Slurm Array Job ID: $WORKER_JOB_ID"
        fi
        if [[ -n "$WORKER_JOB_ID" ]]; then
            watch_slurm_array "$WORKER_JOB_ID" "$RUN_TAG" "${RUN_DIR}/results" "$NUM_SHARDS" 12
        fi
    else
        echo "    Running workers locally (sbatch not available)..."
        for ((task_id=0; task_id<NUM_SHARDS; task_id++)); do
            $PY_BIN -u pipeline_core/run_worker.py --run_dir "$RUN_DIR" --task_id "$task_id"
        done
    fi

    # 3. Stitch shards, calculate transforms, evaluate correlations & render figures
    echo "  [3/3] Aggregating results, computing transforms, generating publication figures..."
    $PY_BIN -u pipeline_core/aggregate_results.py --run_dir "$RUN_DIR"
    echo "  >>> Finished: $RUN_TAG"
}

run_tau_series() {
    local SERIES="$1"
    if [[ ! -f "$SERIES" ]]; then
        echo "ERROR: Series file not found: $SERIES" >&2
        return 1
    fi
    echo ""
    echo "======================================================================="
    echo " TAU SERIES: $SERIES"
    echo " New tags only (<prod>_tauNN). Existing productions are not opened."
    echo "======================================================================="
    local tmpl tag nlev tau
    while IFS=$'\t' read -r tmpl tag nlev tau; do
        echo ""
        echo " >>> series member  tau=$tau  L=$nlev  ->  $tag"
        run_single_config "$tmpl" "$tag" "$nlev"
    done < <($PY_BIN -c "
from pipeline_core.tau_series import iter_series_jobs
for j in iter_series_jobs('$SERIES'):
    print('\t'.join([j['template'], j['run_tag'], str(j['n_levels']), str(j['tau'])]))
")
}

run_aux_on_source() {
    local SOURCE_TAG="$1"
    local AUX_TAG="$2"
    local KIND="$3"
    local N_LEVELS="${4:-}"
    local SOURCE_DIR AUX_DIR
    SOURCE_DIR=$($PY_BIN -c "import os; print(os.path.abspath(os.path.join('$SCRIPT_DIR', '..', 'cesvima_output', '$SOURCE_TAG')))")
    AUX_DIR=$($PY_BIN -c "import os; print(os.path.abspath(os.path.join('$SCRIPT_DIR', '..', 'cesvima_output', '$AUX_TAG')))")
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " >>> AUX $KIND  source=$SOURCE_TAG  ->  $AUX_TAG"
    if [[ -n "$N_LEVELS" ]]; then
        echo "     n_levels override=$N_LEVELS"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ ! -f "$SOURCE_DIR/meta.json" ]]; then
        echo "ERROR: source run not found: $SOURCE_DIR/meta.json" >&2
        return 1
    fi
    if [[ -f "$AUX_DIR/results_consolidated.npz" ]]; then
        echo "  SKIP (aux already complete, will not overwrite): $AUX_DIR"
        return 0
    fi
    if [[ ! -d "$SOURCE_DIR/shards" ]]; then
        echo "ERROR: source shards missing: $SOURCE_DIR/shards" >&2
        return 1
    fi
    mkdir -p "$AUX_DIR/results"
    local NUM_SHARDS MAX_TASK_ID
    NUM_SHARDS=$($PY_BIN -c "import json; print(json.load(open('$SOURCE_DIR/meta.json'))['num_shards'])")
    MAX_TASK_ID=$((NUM_SHARDS - 1))
    local ARRAY_ID_FILE="${AUX_DIR}/slurm_array_id.txt"
    if command -v sbatch >/dev/null 2>&1; then
        local JOB_ID="" EXPORTS
        EXPORTS="ALL,SOURCE_DIR=${SOURCE_DIR},AUX_DIR=${AUX_DIR},AUX_KIND=${KIND}"
        if [[ -n "$N_LEVELS" ]]; then
            EXPORTS="${EXPORTS},AUX_N_LEVELS=${N_LEVELS}"
        fi
        if [[ -f "$ARRAY_ID_FILE" ]]; then
            JOB_ID=$(tr -d '[:space:]' < "$ARRAY_ID_FILE")
            if [[ -z "$JOB_ID" || "$(count_squeue "$JOB_ID")" -eq 0 ]]; then
                JOB_ID=""
            else
                echo "    Reattaching to running aux array $JOB_ID"
            fi
        fi
        local DONE_NOW
        DONE_NOW=$(count_result_npzs "$AUX_DIR/results")
        DONE_NOW=${DONE_NOW:-0}
        if [[ -z "$JOB_ID" && "$DONE_NOW" -ge "$NUM_SHARDS" ]]; then
            echo "    All $NUM_SHARDS aux shard files present, skipping sbatch"
        elif [[ -z "$JOB_ID" ]]; then
            JOB_ID=$(sbatch --parsable --array=0-${MAX_TASK_ID} \
                --export="$EXPORTS" \
                worker_aux.slurm)
            echo "$JOB_ID" > "$ARRAY_ID_FILE"
            echo "    Aux array job: $JOB_ID"
        fi
        if [[ -n "$JOB_ID" ]]; then
            watch_slurm_array "$JOB_ID" "$SOURCE_TAG" "$AUX_DIR/results" "$NUM_SHARDS" 15
        fi
    else
        local task_id
        for ((task_id=0; task_id<NUM_SHARDS; task_id++)); do
            $PY_BIN -u pipeline_core/run_aux_worker.py \
                --source_dir "$SOURCE_DIR" --aux_dir "$AUX_DIR" \
                --kind "$KIND" --task_id "$task_id" \
                ${N_LEVELS:+--n_levels "$N_LEVELS"}
        done
    fi
    $PY_BIN -u pipeline_core/run_aux_worker.py \
        --source_dir "$SOURCE_DIR" --aux_dir "$AUX_DIR" --kind "$KIND" --aggregate
}

run_fput_sbar_aux() {
    local CFG_FILE="$1"
    if [[ ! -f "$CFG_FILE" ]]; then
        echo "ERROR: Aux config not found: $CFG_FILE" >&2
        return 1
    fi
    echo ""
    echo "======================================================================="
    echo " FPUT S-BAR AUX: $CFG_FILE"
    echo " Re-integrates existing shards. FinSTOD directories are not opened."
    echo " Sequential 60-shard arrays (account cap 200)."
    echo " Wall time scales with L (tau=5 was ~2 min on 60 cores; tau=25 ~5x that)."
    echo "======================================================================="
    $PY_BIN -u pipeline_core/run_aux_worker.py --list_config "$CFG_FILE" | while IFS=$'\t' read -r src aux kind n_ov; do
        echo "    will do  $src  ->  $aux  ($kind)"
    done
    local src aux kind n_ov
    while IFS=$'\t' read -r src aux kind n_ov; do
        run_aux_on_source "$src" "$aux" "$kind" "$n_ov"
    done < <($PY_BIN -u pipeline_core/run_aux_worker.py --list_config "$CFG_FILE")
}

run_interp_aux() {
    local CFG_FILE="$1"
    if [[ ! -f "$CFG_FILE" ]]; then
        echo "ERROR: Aux config not found: $CFG_FILE" >&2
        return 1
    fi
    echo ""
    echo "======================================================================="
    echo " INTERP AUX: $CFG_FILE"
    echo " Re-integrates existing shards. FinSTOD directories are not opened."
    echo " Sequential 60-shard arrays (account cap 200)."
    echo "======================================================================="
    $PY_BIN -u pipeline_core/run_aux_worker.py --list_config "$CFG_FILE" | while IFS=$'\t' read -r src aux kind n_ov; do
        echo "    will do  $src  ->  $aux  ($kind)  n_levels=${n_ov:-from source}"
    done
    local src aux kind n_ov
    while IFS=$'\t' read -r src aux kind n_ov; do
        run_aux_on_source "$src" "$aux" "$kind" "$n_ov"
    done < <($PY_BIN -u pipeline_core/run_aux_worker.py --list_config "$CFG_FILE")
}

# --- 2. PARSE ARGUMENTS / PHASES ---
PHASE_MODE="all"
SINGLE_CFG=""

if [[ -n "${CFG:-}" ]]; then
    SINGLE_CFG="$CFG"
elif [[ $# -gt 0 ]]; then
    if [[ "$1" == "--phases" && $# -gt 1 ]]; then
        PHASE_MODE="$2"
    elif [[ "$1" == "all" || "$1" == "ALL" ]]; then
        PHASE_MODE="all"
    elif [[ -f "$1" ]]; then
        SINGLE_CFG="$1"
    else
        PHASE_MODE="$1"
    fi
elif [[ -n "${PHASES:-}" ]]; then
    PHASE_MODE="$PHASES"
fi

# Normalize and parse PHASE_MODE (supports commas, underscores, dashes/ranges like 3-7, colons, all)
PHASE_MODE=$($PY_BIN -c "
import sys, re
raw = '''$PHASE_MODE'''.strip()
if raw.lower() in ('all', ''):
    print('all')
    sys.exit(0)
tokens = re.split(r'[,:_+\s]+', raw)
phases = set()
for tok in tokens:
    if '-' in tok:
        parts = tok.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            for p in range(int(parts[0]), int(parts[1]) + 1):
                phases.add(str(p))
    elif tok.isdigit():
        phases.add(tok)
if not phases and raw.isdigit():
    for ch in raw:
        phases.add(ch)
print(','.join(sorted(phases, key=int)) if phases else 'all')
")

# Check if running a single configuration file directly
if [[ -n "$SINGLE_CFG" ]]; then
    KIND=$($PY_BIN -c "
from pipeline_core.generate_tasks import load_config
print(load_config('$SINGLE_CFG').get('kind') or '')
")
    if [[ "$KIND" == "tau_series" ]]; then
        run_tau_series "$SINGLE_CFG"
    elif [[ "$KIND" == "fput_sbar_aux" ]]; then
        run_fput_sbar_aux "$SINGLE_CFG"
    elif [[ "$KIND" == "interp_aux" ]]; then
        run_interp_aux "$SINGLE_CFG"
    else
        run_single_config "$SINGLE_CFG"
    fi
    echo ""
    echo "======================================================================="
    echo " Pipeline execution completed for $SINGLE_CFG: $(date)"
    echo "======================================================================="
    exit 0
fi

# --- 3. EXECUTE MULTI-PHASE PRODUCTION PIPELINE ---
echo "Configured Execution Phases: $PHASE_MODE"

should_run_phase() {
    local target="$1"
    if [[ "$PHASE_MODE" == "all" || "$PHASE_MODE" == "ALL" ]]; then
        return 0
    fi
    if [[ ",$PHASE_MODE," =~ ",$target," ]]; then
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------------------
# PHASE 0: Mathematical Canary Self-Test
# ------------------------------------------------------------------------------
if should_run_phase 0; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 0] MATHEMATICAL & LOGIC CANARY VERIFICATION"
    echo "======================================================================="
    $PY_BIN -u pipeline_core/sod_canary.py
    if [[ $? -ne 0 ]]; then
        echo "!!! CRITICAL: Canary tests failed! Aborting pipeline to save compute resources. !!!" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# PHASE 1: Lorenz-96 Dimension Suite (m=4, m=6, m=8)
# ------------------------------------------------------------------------------
if should_run_phase 1; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 1] LORENZ-96 HIGH-DIMENSIONAL SCALING SUITE (m=4, 6, 8)"
    echo "======================================================================="
    run_single_config "configs/prod_lorenz96_m4.yaml"
    run_single_config "configs/prod_lorenz96_m6.yaml"
    run_single_config "configs/prod_lorenz96_m8.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 2: 3-DoF Hénon-Heiles Energy Sweep (E=0.06, 0.10, 0.14, 0.16)
# ------------------------------------------------------------------------------
if should_run_phase 2; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 2] 3-DoF HÉNON-HEILES ENERGY SWEEP (m=6, E=0.06, 0.10, 0.14, 0.16)"
    echo "======================================================================="
    run_single_config "configs/prod_hh_energy_006.yaml"
    run_single_config "configs/prod_hh_energy_010.yaml"
    run_single_config "configs/prod_hh_energy_014.yaml"
    run_single_config "configs/prod_hh_energy_016.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 3: Coupled Duffing & Integrable Torus Flow (m=6)
# ------------------------------------------------------------------------------
if should_run_phase 3; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 3] COUPLED DUFFING & INTEGRABLE TORUS (m=6)"
    echo "======================================================================="
    run_single_config "configs/prod_duffing_m6.yaml"
    run_single_config "configs/prod_integrable_torus_m6.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 4: Canonical Benchmarks (Froeschle 4D/6D, ABC 3D Flow, CR3BP 6D)
# ------------------------------------------------------------------------------
if should_run_phase 4; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 4] CANONICAL BENCHMARKS (Froeschle, ABC Flow, CR3BP)"
    echo "======================================================================="
    run_single_config "configs/prod_froeschle_dof2_m4.yaml"
    run_single_config "configs/prod_froeschle_dof3_m6.yaml"
    run_single_config "configs/prod_abc_flow_m3.yaml"
    run_single_config "configs/prod_cr3bp_m6.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 5: 3D Volumetric Slices (ABC Flow, Henon-Heiles, Lorenz-96)
# ------------------------------------------------------------------------------
if should_run_phase 5; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 5] 3D VOLUMETRIC SUITE (ABC Flow, Henon-Heiles, Lorenz-96)"
    echo "======================================================================="
    run_single_config "configs/prod_abc_3d_volume.yaml"
    run_single_config "configs/prod_hh_m6_3d_volume.yaml"
    run_single_config "configs/prod_lorenz96_m6_3d_volume.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 6: Ultra-High-Dimensional Frontier (Lorenz-96 m=16, 32, FPUT, Kuramoto)
# ------------------------------------------------------------------------------
if should_run_phase 6; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 6] ULTRA-HIGH-DIMENSIONAL FRONTIER (m=16, 32, FPUT, Kuramoto)"
    echo "======================================================================="
    run_single_config "configs/prod_lorenz96_m16.yaml"
    run_single_config "configs/prod_lorenz96_m32.yaml"
    run_single_config "configs/prod_fput_m8.yaml"
    run_single_config "configs/prod_fput_m16.yaml"
    run_single_config "configs/prod_kuramoto_m8.yaml"
    run_single_config "configs/prod_kuramoto_m16.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 7: Cross-System Synthesis & Publication Paper Summary
# ------------------------------------------------------------------------------
if should_run_phase 7; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 7] CROSS-SYSTEM SYNTHESIS & PUBLICATION TABLES/FIGURES"
    echo "======================================================================="
    $PY_BIN -u pipeline_core/synthesize_all.py --output_dir "$(pwd)/../cesvima_output"
fi

# ------------------------------------------------------------------------------
# PHASE 8: Open-Problem Capability Tests (Shear Barriers, Noise, Calibrated Kuramoto)
# ------------------------------------------------------------------------------
if should_run_phase 8; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 8] OPEN-PROBLEM CAPABILITY BENCHMARK SUITE"
    echo "======================================================================="
    run_single_config "configs/test2_bickley_jet_shear.yaml"
    run_single_config "configs/test3_duffing_noise_001.yaml"
    run_single_config "configs/test3_duffing_noise_005.yaml"
    run_single_config "configs/prod_kuramoto_m8.yaml"
    run_single_config "configs/prod_kuramoto_m16.yaml"
    $PY_BIN -u pipeline_core/synthesize_tests.py
fi

# ------------------------------------------------------------------------------
# PHASE 9: Integrity reruns (new tags; does not overwrite Phase 2/8 output)
#   - Hénon-Heiles energy sweep with valid_cells active in slice mode
#   - Matched Duffing noise trio (σ = 0, 0.01, 0.05)
#   - Unperturbed Bickley jet at 1000×1000
# ------------------------------------------------------------------------------
if should_run_phase 9; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 9] INTEGRITY RERUNS AND PURE-SHEAR BICKLEY"
    echo "======================================================================="
    run_single_config "configs/prod_hh_energy_006_v2.yaml"
    run_single_config "configs/prod_hh_energy_010_v2.yaml"
    run_single_config "configs/prod_hh_energy_014_v2.yaml"
    run_single_config "configs/prod_hh_energy_016_v2.yaml"
    run_single_config "configs/test3_duffing_clean.yaml"
    run_single_config "configs/test3_duffing_noise_001_v2.yaml"
    run_single_config "configs/test3_duffing_noise_005_v2.yaml"
    run_single_config "configs/test2_bickley_pure_shear.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 10: Queue 1–10 at production quality (new tags only)
#   Laptop diagnostics were 40×40–125×125. These are 1000×1000, n_cells=1000,
#   60 shards, same L / dt as the rest of the CESVIMA suite.
#   Does not overwrite existing prod_* / test2_* / test3_* tags.
# ------------------------------------------------------------------------------
if [[ "$PHASE_MODE" != "all" && "$PHASE_MODE" != "ALL" ]] && should_run_phase 10; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 10] PRODUCTION-QUALITY QUEUE (K-sweep, L-envelope, graded d-hat,"
    echo "            type census, origin-shift, frequency plane, FPUT lifetime)"
    echo "======================================================================="

    # Post-process existing 1000×1000 production (no FinSTOD overwrite)
    run_aux_on_source "prod_froeschle_dof2_m4" "aux_froeschle_frequency_plane" "frequency_plane" \
        || echo "WARNING: frequency-plane aux failed; continuing Phase 10 FinSTOD jobs"
    run_aux_on_source "prod_fput_m8" "aux_fput_lifetime" "fput_lifetime" \
        || echo "WARNING: FPUT-lifetime aux failed; continuing Phase 10 FinSTOD jobs"

    # Froeschlé coupling sweep (K=0.30 already exists)
    run_single_config "configs/prod_froeschle_K005.yaml"
    run_single_config "configs/prod_froeschle_K010.yaml"
    run_single_config "configs/prod_froeschle_K020.yaml"
    run_single_config "configs/prod_froeschle_K050.yaml"
    run_single_config "configs/prod_froeschle_K100.yaml"

    # Froeschlé censoring envelope (L=2500 already exists)
    run_single_config "configs/prod_froeschle_L0250.yaml"
    run_single_config "configs/prod_froeschle_L0500.yaml"
    run_single_config "configs/prod_froeschle_L1000.yaml"
    run_single_config "configs/prod_froeschle_L2000.yaml"
    run_single_config "configs/prod_froeschle_L4000.yaml"
    run_single_config "configs/prod_froeschle_L8000.yaml"

    # Lorenz-96 type census vs L (L=2000 already exists)
    run_single_config "configs/prod_lorenz96_m4_L0250.yaml"
    run_single_config "configs/prod_lorenz96_m4_L0500.yaml"
    run_single_config "configs/prod_lorenz96_m4_L1000.yaml"
    run_single_config "configs/prod_lorenz96_m6_L0250.yaml"
    run_single_config "configs/prod_lorenz96_m6_L0500.yaml"
    run_single_config "configs/prod_lorenz96_m6_L1000.yaml"

    # Graded d-hat: pair_kstar persisted (identity + mixed torus)
    run_single_config "configs/prod_torus_id_d1_kstar.yaml"
    run_single_config "configs/prod_torus_id_d2_kstar.yaml"
    run_single_config "configs/prod_torus_id_d3_kstar.yaml"
    run_single_config "configs/prod_torus_id_d4_kstar.yaml"
    run_single_config "configs/prod_torus_id_d5_kstar.yaml"
    run_single_config "configs/prod_torus_mixed_d1_kstar.yaml"
    run_single_config "configs/prod_torus_mixed_d2_kstar.yaml"
    run_single_config "configs/prod_torus_mixed_d3_kstar.yaml"
    run_single_config "configs/prod_torus_mixed_d4_kstar.yaml"
    run_single_config "configs/prod_torus_mixed_d5_kstar.yaml"
    run_single_config "configs/prod_kuramoto_m8_kstar.yaml"

    # Bickley origin-shift (unperturbed 1000×1000 is Phase 9; these are +1 cell)
    run_single_config "configs/test2_bickley_jet_shear_shift1.yaml"
    run_single_config "configs/test2_bickley_pure_shear_shift1.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 11: Tau series (new tags only; not part of `all`)
#   ABC m=3 and FPUT m=8 at tau=5,10,15,20. tau=25 already exists.
# ------------------------------------------------------------------------------
if [[ "$PHASE_MODE" != "all" && "$PHASE_MODE" != "ALL" ]] && should_run_phase 11; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 11] TAU SERIES (ABC m=3, FPUT m=8; tau=5,10,15,20; FTLE on)"
    echo "======================================================================="
    run_tau_series "configs/series_abc_fput_m8_tau.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 12: FPUT modal S-bar aux (not part of `all`)
#   Full 1000×1000 mean_S + tau_eq on existing FPUT m=8 FinSTOD runs.
# ------------------------------------------------------------------------------
if [[ "$PHASE_MODE" != "all" && "$PHASE_MODE" != "ALL" ]] && should_run_phase 12; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 12] FPUT MODAL S-BAR AUX (tau series + production tau=25)"
    echo "======================================================================="
    run_fput_sbar_aux "configs/aux_fput_sbar.yaml"
fi

# ------------------------------------------------------------------------------
# PHASE 13: Interpretation campaign (not part of `all`)
#   New FinSTOD tags (never overwrite production) + LD/FLI/FTRN aux + basins.
# ------------------------------------------------------------------------------
if [[ "$PHASE_MODE" != "all" && "$PHASE_MODE" != "ALL" ]] && should_run_phase 13; then
    echo ""
    echo "======================================================================="
    echo " [PHASE 13] INTERPRETATION CAMPAIGN (filaments, Bickley, Duffing)"
    echo "======================================================================="

    # Origin-shift Bickley (Phase 10 tags; skip if already complete)
    run_single_config "configs/test2_bickley_jet_shear_shift1.yaml"
    run_single_config "configs/test2_bickley_pure_shear_shift1.yaml"

    # New FinSTOD: published Duffing plane, coupled (x1,v1), tau envelope, clip, longer shear
    run_single_config "configs/interp_duffing_d1.yaml"
    run_single_config "configs/interp_duffing_m6_xv.yaml"
    run_single_config "configs/interp_duffing_m6_tau10.yaml"
    run_single_config "configs/interp_duffing_m6_tau20.yaml"
    run_single_config "configs/interp_duffing_m6_tau40.yaml"
    run_single_config "configs/interp_duffing_m6_clip.yaml"
    run_single_config "configs/interp_bickley_pure_long.yaml"

    # LD / FLI / FTRN / time-aware FTLE on existing + new tags
    run_interp_aux "configs/aux_interp.yaml"
    run_interp_aux "configs/aux_duffing_basins.yaml"
fi

echo ""
echo "======================================================================="
echo " ALL REQUESTED PIPELINE PHASES COMPLETED SUCCESSFULLY!"
echo " End Time: $(date)"
echo " Results Directory: $(pwd)/../cesvima_output"
echo "======================================================================="
