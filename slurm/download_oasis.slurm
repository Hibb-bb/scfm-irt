#!/bin/bash
#SBATCH --job-name=oasis-dl
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_oasis_%j.out
#SBATCH --error=logs/download_oasis_%j.err

# Normalize staged OASIS h5ad -> data/raw/oasis.h5ad (50k subsample).
# Requires data/staging/oasis*.h5ad or OASIS_H5AD_PATH (NBDC hum0197 / JGA controlled access).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs data/raw data/staging

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/download_data.py --dataset oasis --force
python -c "
from scfm_irt.data.describe import describe_by_name
info = describe_by_name('oasis')
print('oasis:', info)
"
echo "[oasis-dl] done -> data/raw/oasis.h5ad"
