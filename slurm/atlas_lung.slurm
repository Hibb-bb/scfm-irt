#!/bin/bash
#SBATCH --job-name=scfm-atlas-lung
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/atlas_lung_%j.out
#SBATCH --error=logs/atlas_lung_%j.err

# End-to-end atlas run: lung_atlas (50k HLCA subsample) + 5 embedders + IRT.
#
#   sbatch slurm/atlas_lung.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings data/raw cache/scimilarity_workdir

echo "[atlas-lung] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

export SCFM_SCIMILARITY_WORKDIR="$REPO_ROOT/cache/scimilarity_workdir"
test -d "$SCFM_SCIMILARITY_WORKDIR/model_v1.1" || {
  echo "[error] SCimilarity model missing; run slurm/download_scimilarity_assets.slurm first"
  exit 1
}

# ---------- download + dev baseline ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import cellxgene_census" 2>/dev/null || python scripts/install_census_deps.py
python scripts/download_data.py --dataset lung_atlas

python scripts/extract_embeddings.py \
  --embedder dev_pca \
  --dataset lung_atlas \
  --split cross_donor \
  --seed 0 \
  --device cpu \
  --force

# ---------- UCE ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-uce/bin/activate"
python scripts/extract_embeddings.py \
  --embedder uce \
  --dataset lung_atlas \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

# ---------- scGPT ----------
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" scgpt
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-scgpt/bin/activate"
python scripts/extract_embeddings.py \
  --embedder scgpt \
  --dataset lung_atlas \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

# ---------- Geneformer ----------
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" geneformer
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-geneformer/bin/activate"
python scripts/extract_embeddings.py \
  --embedder geneformer \
  --dataset lung_atlas \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

# ---------- SCimilarity ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
SCIM_VENV="${SCIM_VENV:-$REPO_ROOT/.venv-scimilarity}"
if [[ ! -x "$SCIM_VENV/bin/python" ]]; then
  /software/python/3.10.1/bin/python -m venv "$SCIM_VENV"
fi
# shellcheck disable=SC1091
source "$SCIM_VENV/bin/activate"
pip_retry install -U pip
pip_retry install scimilarity scanpy anndata tqdm pyyaml
pip_retry install -e "$REPO_ROOT[sc]"
python scripts/extract_embeddings.py \
  --embedder scimilarity \
  --dataset lung_atlas \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

# ---------- IRT ranking ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/run_pipeline.py --config configs/ranking_lung_atlas_v1.yaml

for model in dev_pca uce scgpt geneformer scimilarity; do
  python scripts/single_model_report.py --run runs/ranking_lung_atlas_v1 --model "$model"
done

echo "[atlas-lung] done -> runs/ranking_lung_atlas_v1/report.md"
