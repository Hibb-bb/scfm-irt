#!/bin/bash
#SBATCH --job-name=state-assets
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/state_assets_%j.out
#SBATCH --error=logs/state_assets_%j.err

# Download SE-600M (~12 GB) from Hugging Face. Run once before extraction jobs.
#
#   sbatch slurm/download_state_assets.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/state_workdir/SE-600M

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

python scripts/download_state_assets.py --out cache/state_workdir/SE-600M
echo "[state-assets] done"
