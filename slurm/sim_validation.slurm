#!/bin/bash
#SBATCH --job-name=scfm-sim-val
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sim_validation_%j.out
#SBATCH --error=logs/sim_validation_%j.err

# Synthetic validation on real difficulty landscape + Fig 3 κ-vs-r sweep.
# CPU-only; no GPU.
#
#   sbatch slurm/sim_validation.slurm
#   SKIP_SWEEP=1 sbatch slurm/sim_validation.slurm   # Fig 2 + survival only

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs sim/results/validation

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONPATH=.

SWEEP_ARGS=()
if [[ "${SKIP_SWEEP:-0}" == "1" ]]; then
  SWEEP_ARGS+=(--skip-sweep)
fi

echo "[sim-validation] start $(date -Is)"
echo "[sim-validation] repo=$REPO_ROOT skip_sweep=${SKIP_SWEEP:-0}"

python sim/run_validation.py "${SWEEP_ARGS[@]}" 2>&1 | tee "sim/results/validation/run_${SLURM_JOB_ID}.log"

echo "[sim-validation] done $(date -Is)"
echo "[sim-validation] outputs -> sim/results/validation/"
