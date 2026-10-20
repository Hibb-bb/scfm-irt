#!/bin/bash
#SBATCH --job-name=scfm-irt-analyze
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err

# Light-side IRT fit after embeddings are cached. No GPU.
# Use --dependency=afterok:<extract_job_ids> when chaining after extraction.
#
#   CONFIG=configs/smoke_dev.yaml sbatch slurm/run_pipeline.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs runs

CONFIG="${CONFIG:-configs/smoke_dev.yaml}"

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[analyze] config=$CONFIG"
python scripts/run_pipeline.py --config "$CONFIG"
echo "[analyze] done"
