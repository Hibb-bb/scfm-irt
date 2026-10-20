#!/bin/bash
#SBATCH --job-name=scfm-lung-v3-10m
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lung_atlas_v3_partial_%j.out
#SBATCH --error=logs/lung_atlas_v3_partial_%j.err

# 10-model IRT ranking (cached embeddings). Requires transcriptformer-lung_atlas cache.
#
#   sbatch slurm/lung_atlas_v3_partial.slurm
#   sbatch --dependency=afterok:<transcriptformer_job_id> slurm/lung_atlas_v3_partial.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/run_pipeline.py --config configs/ranking_lung_atlas_v3_partial.yaml

echo "[lung-v3-10m] done -> runs/ranking_lung_atlas_v3_partial/report.md"
