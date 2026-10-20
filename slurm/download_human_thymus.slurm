#!/bin/bash
#SBATCH --job-name=thymus-dl
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_human_thymus_%j.out
#SBATCH --error=logs/download_human_thymus_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs data/raw

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/download_data.py --dataset human_thymus --force
python -c "from scfm_irt.data.describe import describe_by_name; print(describe_by_name('human_thymus'))"
echo "[thymus-dl] done -> data/raw/human_thymus.h5ad"
