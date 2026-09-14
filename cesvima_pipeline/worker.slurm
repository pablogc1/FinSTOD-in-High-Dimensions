#!/bin/bash
#SBATCH --job-name=finstod_nd_worker
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/worker_%A_%a.out
#SBATCH --error=logs/worker_%A_%a.err

set -euo pipefail

echo "======================================================================="
echo " CESVIMA FinSTOD High-Dim Worker Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo " Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-0}"
echo " Host  : $(hostname)"
echo " Start : $(date)"
echo "======================================================================="

# Load CESVIMA environment modules
module --force purge > /dev/null 2>&1 || true
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# Activate venv if present
if [[ -f "../generic_env/bin/activate" ]]; then
    source ../generic_env/bin/activate
elif [[ -f "generic_env/bin/activate" ]]; then
    source generic_env/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Ensure project root is in PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/..:${SLURM_SUBMIT_DIR:-$PWD}:${PYTHONPATH:-}"

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR environment variable must be set." >&2
    exit 1
fi

python3 -u pipeline_core/run_worker.py \
    --run_dir "$RUN_DIR" \
    --task_id "${SLURM_ARRAY_TASK_ID}"

echo "======================================================================="
echo " Worker task completed: $(date)"
echo "======================================================================="
