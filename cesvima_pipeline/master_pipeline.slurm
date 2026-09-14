#!/bin/bash
#SBATCH --job-name=finstod_master
#SBATCH --partition=standard
#SBATCH --time=160:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err

# ==============================================================================
# Usage on CESVIMA Magerit:
#   1. Run ALL phases automatically (Canary -> All Systems -> Synthesis):
#      sbatch master_pipeline.slurm
#
#   2. Run selected phases (e.g. Phase 0, 1, 5):
#      sbatch --export=ALL,PHASES="0,1,5" master_pipeline.slurm
#
#   3. Run a single specific configuration (e.g. pilot canary check):
#      sbatch --export=ALL,CFG=configs/pilot_test_lorenz96_m4.yaml master_pipeline.slurm
#
#   4. Tau series (ABC + FPUT m=8 at tau=5,10,15,20; new tags, no overwrite):
#      sbatch --export=ALL,PHASES="11" master_pipeline.slurm
#      sbatch --export=ALL,CFG=configs/series_abc_fput_m8_tau.yaml master_pipeline.slurm
#
#   5. FPUT modal S-bar aux (existing FinSTOD shards; not part of `all`):
#      sbatch --export=ALL,PHASES="12" master_pipeline.slurm
#      sbatch --export=ALL,CFG=configs/aux_fput_sbar.yaml master_pipeline.slurm
#
#   6. Interpretation campaign (not part of `all`; new tags only):
#      sbatch --export=ALL,PHASES="13" master_pipeline.slurm
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

chmod +x master_run.sh worker.slurm worker_aux.slurm 2>/dev/null || true

# Pass environment variable or argument to master_run.sh
if [[ -n "${CFG:-}" ]]; then
    ./master_run.sh "$CFG"
elif [[ $# -gt 0 ]]; then
    ./master_run.sh "$@"
elif [[ -n "${PHASES:-}" ]]; then
    ./master_run.sh --phases "$PHASES"
else
    ./master_run.sh all
fi
