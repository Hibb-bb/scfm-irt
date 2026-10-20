#!/bin/bash
#SBATCH --job-name=scfm-lung-resume
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/atlas_lung_resume_%j.out
#SBATCH --error=logs/atlas_lung_resume_%j.err

# Resume lung_atlas after download + dev_pca: fix gene symbols, extract 4 FMs, IRT.
#
#   sbatch slurm/atlas_lung_resume.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings data/raw cache/scimilarity_workdir

echo "[lung-resume] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

test -f data/raw/lung_atlas.h5ad || {
  echo "[error] missing data/raw/lung_atlas.h5ad"
  exit 1
}

export SCFM_SCIMILARITY_WORKDIR="$REPO_ROOT/cache/scimilarity_workdir"
test -d "$SCFM_SCIMILARITY_WORKDIR/model_v1.1" || {
  echo "[error] SCimilarity model missing"
  exit 1
}

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
# Gene symbols are harmonized at load time (census numeric var index -> feature_name).

EXTRACT=(--dataset lung_atlas --split cross_donor --seed 0 --device cuda --force)

# ---------- UCE ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-uce/bin/activate"
python scripts/extract_embeddings.py --embedder uce "${EXTRACT[@]}"

# ---------- scGPT ----------
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" scgpt
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-scgpt/bin/activate"
python scripts/extract_embeddings.py --embedder scgpt "${EXTRACT[@]}"

# ---------- Geneformer ----------
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" geneformer
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-geneformer/bin/activate"
python scripts/extract_embeddings.py --embedder geneformer "${EXTRACT[@]}"

# ---------- SCimilarity ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
SCIM_VENV="${SCIM_VENV:-$REPO_ROOT/.venv-scimilarity}"
# shellcheck disable=SC1091
source "$SCIM_VENV/bin/activate"
python scripts/extract_embeddings.py --embedder scimilarity "${EXTRACT[@]}"

# ---------- IRT ranking ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

if [[ ! -d cache/embeddings/dev_pca/lung_atlas ]]; then
  python scripts/extract_embeddings.py \
    --embedder dev_pca \
    --dataset lung_atlas \
    --split cross_donor \
    --seed 0 \
    --device cpu \
    --force
fi

python scripts/run_pipeline.py --config configs/ranking_lung_atlas_v1.yaml

for model in dev_pca uce scgpt geneformer scimilarity; do
  python scripts/single_model_report.py --run runs/ranking_lung_atlas_v1 --model "$model"
done

echo "[lung-resume] done -> runs/ranking_lung_atlas_v1/report.md"
