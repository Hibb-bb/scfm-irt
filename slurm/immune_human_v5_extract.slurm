#!/bin/bash
#SBATCH --job-name=immune-v5-extract
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/immune_human_v5_extract_%j.out
#SBATCH --error=logs/immune_human_v5_extract_%j.err

# Extract missing v5 embedder caches on immune_human (cross_donor split).
# Skips models whose cache already exists. State is handled separately (state-immune job).

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs cache/embeddings data/raw

DATASET=immune_human
SPLIT=cross_donor
SEED=0
EXTRACT=(--dataset "$DATASET" --split "$SPLIT" --seed "$SEED" --device cuda --force)

echo "[immune-v5-extract] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/download_data.py --dataset "$DATASET"

if ! compgen -G "cache/embeddings/dev_pca/$DATASET/*.npz" > /dev/null; then
  python scripts/extract_embeddings.py --embedder dev_pca --device cpu "${EXTRACT[@]}"
fi
if ! compgen -G "cache/embeddings/dev_random/$DATASET/*.npz" > /dev/null; then
  python scripts/extract_embeddings.py --embedder dev_random --device cpu "${EXTRACT[@]}"
fi
deactivate

# Legacy FMs (skip if cached)
if ! compgen -G "cache/embeddings/uce/$DATASET/*.npz" > /dev/null; then
  module load python/3.10.1 2>/dev/null || true
  export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-uce/bin/activate"
  python scripts/extract_embeddings.py --embedder uce "${EXTRACT[@]}"
  deactivate
fi

if ! compgen -G "cache/embeddings/scgpt/$DATASET/*.npz" > /dev/null; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-scgpt/bin/activate"
  python scripts/extract_embeddings.py --embedder scgpt "${EXTRACT[@]}"
  deactivate
fi

if ! compgen -G "cache/embeddings/geneformer/$DATASET/*.npz" > /dev/null; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-geneformer/bin/activate"
  python scripts/extract_embeddings.py --embedder geneformer "${EXTRACT[@]}"
  deactivate
fi

# immune_human is a 2k-HVG atlas; SCimilarity requires >=5000 overlapping genes (see scimilarity.py).
echo "[skip] scimilarity/$DATASET — incompatible with 2k-HVG immune_human"

NEW_MODELS=(
  geneformer_v2 cellplm transcriptformer scprint scprint2 langcell
  geneformer_v2_316m scprint_small_v1 scprint_medium_v1 scprint_large_v1
)

for model in "${NEW_MODELS[@]}"; do
  if compgen -G "cache/embeddings/$model/$DATASET/*.npz" > /dev/null; then
    echo "[skip] $model/$DATASET already cached"
    continue
  fi
  echo "[extract] $model on $DATASET"
  EMBEDDER="$model" DATASET="$DATASET" SPLIT="$SPLIT" SEED="$SEED" \
    bash "$REPO_ROOT/slurm/extract_new_model.slurm" \
    || echo "[warn] $model/$DATASET extraction failed — continuing"
done

echo "[immune-v5-extract] done"
