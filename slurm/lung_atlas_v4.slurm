#!/bin/bash
#SBATCH --job-name=scfm-lung-v4
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lung_atlas_v4_%j.out
#SBATCH --error=logs/lung_atlas_v4_%j.err

# Full lung_atlas IRT ranking (all cached embedders + state).
# Chain after state extraction:
#
#   EXTRACT=$(sbatch --parsable --export="ALL,EMBEDDER=state,DATASET=lung_atlas,SPLIT=cross_donor" \
#     --time=12:00:00 --mem=128G slurm/extract_new_model.slurm)
#   sbatch --dependency=afterok:$EXTRACT slurm/lung_atlas_v4.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/run_pipeline.py --config configs/ranking_lung_atlas_v4.yaml

echo "[lung-v4] done -> runs/ranking_lung_atlas_v4/report.md"
