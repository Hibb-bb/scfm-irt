#!/bin/bash
#SBATCH --job-name=scfm-irt-extract
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_%x_%j.out
#SBATCH --error=logs/extract_%x_%j.err

# Template for a per-model embedding extraction job (Phase 5+).
# Submit one job per (embedder, dataset), each in that model's isolated env:
#
#   EMBEDDER=uce DATASET=zheng68k CONDA_ENV=scfm-irt-uce \
#     sbatch -J extract-uce-zheng slurm/extract_embeddings.slurm
#
# The analysis pipeline then consumes cache/embeddings/ and never imports the model.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/embeddings

EMBEDDER="${EMBEDDER:?set EMBEDDER, e.g. uce}"
DATASET="${DATASET:?set DATASET, e.g. zheng68k}"
SPLIT="${SPLIT:-cross_donor}"
SEED="${SEED:-0}"
CACHE_DIR="${CACHE_DIR:-cache/embeddings}"
CONDA_ENV="${CONDA_ENV:-}"

echo "[extract] embedder=$EMBEDDER dataset=$DATASET split=$SPLIT seed=$SEED"

if [[ -n "$CONDA_ENV" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No CONDA_ENV or .venv — activate the embedder env before sbatch, or set CONDA_ENV=" >&2
  exit 1
fi

python scripts/extract_embeddings.py \
  --embedder "$EMBEDDER" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --seed "$SEED" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --force

echo "[extract] done for $EMBEDDER / $DATASET"
