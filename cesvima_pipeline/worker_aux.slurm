#!/bin/bash
#SBATCH --job-name=finstod_aux
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/aux_%A_%a.out
#SBATCH --error=logs/aux_%A_%a.err

set -euo pipefail

echo "======================================================================="
echo " CESVIMA FinSTOD aux worker: ${SLURM_ARRAY_TASK_ID:-0}  kind=${AUX_KIND:-}"
echo " Start : $(date)"
echo "======================================================================="

module --force purge > /dev/null 2>&1 || true
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

if [[ -f "../generic_env/bin/activate" ]]; then
    source ../generic_env/bin/activate
elif [[ -f "generic_env/bin/activate" ]]; then
    source generic_env/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/..:${SLURM_SUBMIT_DIR:-$PWD}:${PYTHONPATH:-}"

if [[ -z "${SOURCE_DIR:-}" || -z "${AUX_DIR:-}" || -z "${AUX_KIND:-}" ]]; then
    echo "ERROR: SOURCE_DIR, AUX_DIR, and AUX_KIND must be set." >&2
    exit 1
fi

python3 -u pipeline_core/run_aux_worker.py \
    --source_dir "$SOURCE_DIR" \
    --aux_dir "$AUX_DIR" \
    --kind "$AUX_KIND" \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    ${AUX_N_LEVELS:+--n_levels "$AUX_N_LEVELS"}

echo " Aux task completed: $(date)"
