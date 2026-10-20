#!/bin/bash
#SBATCH --job-name=scfm-lung-v2-6m
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/lung_atlas_v2_pipeline_%j.out
#SBATCH --error=logs/lung_atlas_v2_pipeline_%j.err

# 6-model IRT ranking (cached embeddings). Then entropy-difficulty analysis.
#
#   sbatch slurm/lung_atlas_v2_pipeline.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/run_pipeline.py --config configs/ranking_lung_atlas_v2.yaml

python scripts/lung_entropy_difficulty_eval.py \
  --run runs/ranking_lung_atlas_v2 \
  --replicates 40 \
  --k 30

echo "[lung-v2-6m] done -> runs/ranking_lung_atlas_v2/report.md"
