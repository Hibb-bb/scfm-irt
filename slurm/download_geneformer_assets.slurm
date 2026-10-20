#!/bin/bash
#SBATCH --job-name=gf-assets
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/geneformer_assets_%j.out
#SBATCH --error=logs/geneformer_assets_%j.err

# Download Geneformer HF package + V1-10M weights (stdlib urllib — no pip at runtime).
#   sbatch slurm/download_geneformer_assets.slurm

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/geneformer_workdir

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/download_geneformer_assets.py --out cache/geneformer_workdir
ls -lh cache/geneformer_workdir/Geneformer/Geneformer-V1-10M/ | head -20
echo "[geneformer-assets] done"
