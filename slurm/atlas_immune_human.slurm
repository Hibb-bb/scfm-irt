#!/bin/bash
#SBATCH --job-name=scfm-atlas-immune
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/atlas_immune_human_%j.out
#SBATCH --error=logs/atlas_immune_human_%j.err

# End-to-end atlas run: immune_human + 4 embedders + IRT (cross-donor split).
#
#   sbatch slurm/atlas_immune_human.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings data/raw

echo "[atlas-immune] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# ---------- download + dev baseline ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
python scripts/download_data.py --dataset immune_human

python scripts/extract_embeddings.py \
  --embedder dev_pca \
  --dataset immune_human \
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
  --dataset immune_human \
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
  --dataset immune_human \
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
  --dataset immune_human \
  --split cross_donor \
  --seed 0 \
  --device cuda \
  --force

# ---------- IRT ranking ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/run_pipeline.py --config configs/ranking_immune_human_v1.yaml

for model in dev_pca uce scgpt geneformer; do
  python scripts/single_model_report.py --run runs/ranking_immune_human_v1 --model "$model"
done

echo "[atlas-immune] done -> runs/ranking_immune_human_v1/report.md"
