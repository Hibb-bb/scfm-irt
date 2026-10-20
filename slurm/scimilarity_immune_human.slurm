#!/bin/bash
#SBATCH --job-name=scfm-scim-immune
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/scimilarity_immune_human_%j.out
#SBATCH --error=logs/scimilarity_immune_human_%j.err

# SCimilarity on immune_human + supplementary 5-model ranking.
# Requires prior 4-model immune_human caches from atlas_immune_human.slurm.
#
#   sbatch slurm/scimilarity_immune_human.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings cache/scimilarity_workdir data/raw

echo "[scim-immune] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

export SCFM_SCIMILARITY_WORKDIR="$REPO_ROOT/cache/scimilarity_workdir"
test -d "$SCFM_SCIMILARITY_WORKDIR/model_v1.1" || {
  echo "[error] SCimilarity model missing; run download_scimilarity_assets.slurm first"
  exit 1
}

for model in dev_pca uce scgpt geneformer; do
  if [[ ! -d "cache/embeddings/$model/immune_human" ]]; then
    echo "[error] missing cache for $model on immune_human — run atlas_immune_human.slurm first"
    exit 1
  fi
done

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
  --dataset immune_human \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

python scripts/run_pipeline.py --config configs/ranking_immune_human_scimilarity_v1.yaml
python scripts/single_model_report.py --run runs/ranking_immune_human_scimilarity_v1 --model scimilarity

echo "[scim-immune] done -> runs/ranking_immune_human_scimilarity_v1/report.md"
