#!/bin/bash
#SBATCH --job-name=uce-assets
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/uce_assets_%j.out
#SBATCH --error=logs/uce_assets_%j.err

# Download ~9GB of UCE Figshare assets with size checks (CPU only).
# Chain the GPU job with: sbatch --dependency=afterok:$JOBID slurm/uce_pbmc3k.slurm

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/uce_workdir

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

UCE_DIR="$REPO_ROOT/cache/uce_workdir/UCE"
if [[ ! -f "$UCE_DIR/eval_single_anndata.py" ]]; then
  module load git 2>/dev/null || true
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/snap-stanford/UCE.git "$UCE_DIR"
  else
    TMP_TGZ="$REPO_ROOT/cache/uce_workdir/uce_main.tar.gz"
    python - <<'PY'
import urllib.request, tarfile, shutil
from pathlib import Path
root = Path("cache/uce_workdir")
tg = root / "uce_main.tar.gz"
url = "https://github.com/snap-stanford/UCE/archive/refs/heads/main.tar.gz"
urllib.request.urlretrieve(url, tg)
with tarfile.open(tg) as t:
    t.extractall(root)
shutil.rmtree(root / "UCE", ignore_errors=True)
(root / "UCE-main").rename(root / "UCE")
print("cloned via tarball")
PY
  fi
fi

python scripts/download_uce_assets.py --uce-dir "$UCE_DIR"
echo "[uce-assets] done"
