#!/bin/bash
#SBATCH --job-name=immune-v5-rank
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/immune_human_v5_%j.out
#SBATCH --error=logs/immune_human_v5_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs runs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/run_pipeline.py --config configs/ranking_immune_human_v5.yaml
echo "[immune-v5] done -> runs/ranking_immune_human_v5/report.md"
