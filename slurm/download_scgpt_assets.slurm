#!/bin/bash
#SBATCH --job-name=scgpt-assets
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scgpt_assets_%j.out
#SBATCH --error=logs/scgpt_assets_%j.err

# Download scGPT whole-human checkpoint via gdown (~200MB–1GB).
#   sbatch slurm/download_scgpt_assets.slurm

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/scgpt_workdir

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q gdown

python scripts/download_scgpt_assets.py --out cache/scgpt_workdir/scGPT_human
ls -lh cache/scgpt_workdir/scGPT_human/
echo "[scgpt-assets] done"
