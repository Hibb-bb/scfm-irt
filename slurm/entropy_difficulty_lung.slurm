#!/bin/bash
#SBATCH --job-name=scfm-entropy-lung
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/entropy_difficulty_lung_%j.out
#SBATCH --error=logs/entropy_difficulty_lung_%j.err

# Label-entropy difficulty eval on lung_atlas IRT run (CPU-heavy kNN; no GPU needed).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/lung_entropy_difficulty_eval.py \
  --run runs/ranking_lung_atlas_v1 \
  --replicates 40 \
  --k 30

echo "[entropy-lung] done -> runs/ranking_lung_atlas_v1/entropy_difficulty_eval/report.md"
