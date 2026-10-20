#!/bin/bash
#SBATCH --job-name=scfm-lung-v2
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/lung_atlas_v2_%j.out
#SBATCH --error=logs/lung_atlas_v2_%j.err

# Extract dev_random + CellPLM on lung_atlas, then run 7-model IRT ranking (LOO calibration).
#
#   sbatch slurm/lung_atlas_v2.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs cache/embeddings cache/cellplm_workdir

echo "[lung-v2] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- CellPLM checkpoint (stdlib download; cache ~2.3 GB zip once) ----------
python scripts/download_cellplm_assets.py --out "$REPO_ROOT/cache/cellplm_workdir/ckpt"

# ---------- dev_random (CPU, analysis env) — skip if already cached ----------
if compgen -G "cache/embeddings/dev_random/lung_atlas/*.npz" > /dev/null; then
  echo "[lung-v2] dev_random cache present; skipping extraction"
else
  python scripts/extract_embeddings.py \
    --embedder dev_random \
    --dataset lung_atlas \
    --split cross_donor \
    --seed 0 \
    --device cpu \
    --force
fi

# ---------- CellPLM ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
CELLPLM_VENV="${CELLPLM_VENV:-$REPO_ROOT/.venv-cellplm}"
if [[ ! -x "$CELLPLM_VENV/bin/python" ]]; then
  /software/python/3.10.1/bin/python -m venv "$CELLPLM_VENV"
fi
# shellcheck disable=SC1091
source "$CELLPLM_VENV/bin/activate"
pip_retry install -U pip
install_torch_stack
pip_install_each scanpy anndata tqdm pyyaml hdf5plugin
pip_retry install cellplm || pip_retry install cellplm --no-deps
pip_retry install -e "$REPO_ROOT[sc]"
export SCFM_CELLPLM_WORKDIR="$REPO_ROOT/cache/cellplm_workdir"
python scripts/extract_embeddings.py \
  --embedder cellplm \
  --dataset lung_atlas \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

# ---------- IRT ranking (reuse v1 caches for other embedders) ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/run_pipeline.py --config configs/ranking_lung_atlas_v2.yaml

echo "[lung-v2] done -> runs/ranking_lung_atlas_v2/report.md"
