#!/bin/bash
#SBATCH --job-name=scfm-irt-smoke
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# End-to-end smoke path on the synthetic toy dataset + DevEmbedder.
# No GPU required. Run from the repo root:
#
#   mkdir -p logs
#   sbatch slurm/smoke_dev.slurm
#
# Or interactively after activating the venv:
#   bash slurm/smoke_dev.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings data/raw

module load python/3.12.10 2>/dev/null || true

if [[ ! -d .venv ]]; then
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install -e ".[sc,dev]"
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[smoke] host=$(hostname)  pwd=$PWD  python=$(which python)"
python -c "import scfm_irt, anndata, sklearn; print('scfm_irt', scfm_irt.__version__)"

python scripts/make_smoke_data.py

# Extract embeddings for each DevEmbedder (analysis env is fine for these).
for EMB in dev_pca dev_random; do
  python scripts/extract_embeddings.py \
    --embedder "$EMB" \
    --dataset smoke_toy \
    --split stratified \
    --seed 0 \
    --cache-dir cache/embeddings \
    --device cpu \
    --force
done

python scripts/run_pipeline.py --config configs/smoke_dev.yaml

echo "[smoke] done. See runs/smoke_dev/report.md"
