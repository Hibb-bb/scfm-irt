#!/bin/bash
#SBATCH --job-name=immune-scim-tf
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/immune_scimilarity_transcriptformer_%j.out
#SBATCH --error=logs/immune_scimilarity_transcriptformer_%j.err

# SCimilarity + TranscriptFormer on immune_human (full transcriptome), then append
# responses to ranking_immune_human_v5 and refresh the 3-dataset pooled ranking.
#
#   sbatch slurm/immune_scimilarity_transcriptformer.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings cache/scimilarity_workdir cache/tf_workdir data/raw

FULL_H5AD="$REPO_ROOT/data/raw/immune_human_full.h5ad"
EXTRACT_COMMON=(
  --dataset immune_human
  --adata-path "$FULL_H5AD"
  --split cross_donor
  --seed 0
  --device cuda
  --force
)

echo "[immune-scim-tf] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# ---------- download HVG + full transcriptome ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/download_data.py --dataset immune_human
python scripts/download_data.py --dataset immune_human_full

python - <<'PY'
import scanpy as sc
from pathlib import Path

repo = Path(".")
hvg = sc.read_h5ad(repo / "data/raw/immune_human.h5ad", backed="r")
full = sc.read_h5ad(repo / "data/raw/immune_human_full.h5ad", backed="r")
if hvg.n_obs != full.n_obs:
    raise SystemExit(f"cell count mismatch: hvg={hvg.n_obs} full={full.n_obs}")
if hvg.n_vars >= 5000:
    raise SystemExit(f"immune_human looks full-transcriptome ({hvg.n_vars} genes), expected HVG")
if full.n_vars < 5000:
    raise SystemExit(f"immune_human_full too few genes: {full.n_vars}")
# Spot-check obs alignment (same scIB object, different gene subsets).
for i in (0, 1000, hvg.n_obs - 1):
    if str(hvg.obs_names[i]) != str(full.obs_names[i]):
        raise SystemExit(f"obs_names diverge at index {i}")
print(f"[ok] immune alignment: {hvg.n_obs} cells, hvg={hvg.n_vars} genes, full={full.n_vars} genes")
PY
deactivate

# ---------- SCimilarity ----------
export SCFM_SCIMILARITY_WORKDIR="$REPO_ROOT/cache/scimilarity_workdir"
test -d "$SCFM_SCIMILARITY_WORKDIR/model_v1.1" || {
  echo "[error] SCimilarity model missing; run download_scimilarity_assets.slurm first"
  exit 1
}

module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"

SCIM_VENV="${SCIM_VENV:-$REPO_ROOT/.venv-scimilarity}"
# shellcheck disable=SC1091
source "$SCIM_VENV/bin/activate"
python scripts/extract_embeddings.py \
  --embedder scimilarity \
  "${EXTRACT_COMMON[@]}"
deactivate

# ---------- TranscriptFormer ----------
export SCFM_TRANSCRIPTFORMER_WORKDIR="$REPO_ROOT/cache/tf_workdir"
test -d "$SCFM_TRANSCRIPTFORMER_WORKDIR/checkpoints/tf_sapiens" || {
  echo "[error] TranscriptFormer checkpoint missing; run download_transcriptformer_assets first"
  exit 1
}

module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-transcriptformer/bin/activate"
python scripts/extract_embeddings.py \
  --embedder transcriptformer \
  "${EXTRACT_COMMON[@]}"
deactivate

# ---------- append to immune v5 + pooled ranking ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

for model in scimilarity transcriptformer; do
  if ! compgen -G "cache/embeddings/$model/immune_human/*.npz" > /dev/null; then
    echo "[error] missing cache for $model/immune_human"
    exit 1
  fi
done

PYTHONPATH=. python scripts/append_responses_to_run.py \
  --run runs/ranking_immune_human_v5 \
  --config configs/ranking_immune_human_scim_tf_v1.yaml \
  --backup

PYTHONPATH=. python scripts/pooled_ranking.py --out runs/pooled_3dataset_v2

echo "[immune-scim-tf] done"
echo "  appended -> runs/ranking_immune_human_v5/responses.parquet"
echo "  pooled   -> runs/pooled_3dataset_v2/ability_ranking.csv"
