#!/bin/bash
#SBATCH --job-name=scfm-pbmc-v3
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=logs/pbmc3k_v3_%j.out
#SBATCH --error=logs/pbmc3k_v3_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs runs

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/run_pipeline.py --config configs/ranking_pbmc3k_v3.yaml
echo "[pbmc-v3] done -> runs/ranking_pbmc3k_v3/report.md"
