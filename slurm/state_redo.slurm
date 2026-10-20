#!/bin/bash
#SBATCH --job-name=state-redo
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/state_redo_%x_%j.out
#SBATCH --error=logs/state_redo_%x_%j.err

# Re-extract State embeddings with normalize+log1p preprocessing.
# Usage:
#   sbatch --export=ALL,DATASET=pbmc3k,SPLIT=stratified --job-name=state-pbmc3k slurm/state_redo.slurm
#   sbatch --export=ALL,DATASET=lung_atlas,SPLIT=cross_donor --job-name=state-lung slurm/state_redo.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache/embeddings cache/state_workdir data/raw

DATASET="${DATASET:?set DATASET=pbmc3k or lung_atlas}"
SPLIT="${SPLIT:-cross_donor}"

export EMBEDDER=state
export DATASET
export SPLIT
export SEED=0

echo "[state-redo] dataset=$DATASET split=$SPLIT (normalize+log1p preprocessing)"
bash slurm/extract_new_model.slurm

echo "[state-redo] done -> cache/embeddings/state/$DATASET"
