#!/bin/bash
#SBATCH --job-name=scimilarity-assets
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/scimilarity_assets_%j.out
#SBATCH --error=logs/scimilarity_assets_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/scimilarity_workdir

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/download_scimilarity_assets.py --out cache/scimilarity_workdir
echo "[scimilarity-assets] done"
