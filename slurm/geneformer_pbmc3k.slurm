#!/bin/bash
#SBATCH --job-name=scfm-irt-gf
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/geneformer_pbmc3k_%j.out
#SBATCH --error=logs/geneformer_pbmc3k_%j.err

# Extract Geneformer embeddings on PBMC3k, then fit IRT ranking vs
# DevEmbedder + UCE + scGPT. Chain ONLY on asset download (not on scGPT —
# afterok:scgpt leaves this job DependencyNeverSatisfied if scGPT fails):
#
#   ASSET=$(sbatch --parsable slurm/download_geneformer_assets.slurm)
#   sbatch --dependency=afterok:$ASSET slurm/geneformer_pbmc3k.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings cache/geneformer_workdir data/raw

echo "[geneformer-job] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

export SCFM_GENEFORMER_WORKDIR="$REPO_ROOT/cache/geneformer_workdir"
GF_REPO="$SCFM_GENEFORMER_WORKDIR/Geneformer"
MODEL_DIR="$GF_REPO/Geneformer-V1-10M"

# ---------- ensure checkpoint ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
python scripts/download_geneformer_assets.py --out "$SCFM_GENEFORMER_WORKDIR"
test -d "$MODEL_DIR"
echo "[ckpt] $MODEL_DIR ready"

# ---------- analysis half: data + cached baselines ----------
python scripts/download_data.py --dataset pbmc3k
if [[ ! -d cache/embeddings/dev_pca/pbmc3k ]]; then
  python scripts/extract_embeddings.py \
    --embedder dev_pca --dataset pbmc3k --split stratified --seed 0 --device cpu
fi

# ---------- Geneformer isolated venv ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" geneformer
# shellcheck disable=SC1091
source "${GF_VENV:-$REPO_ROOT/.venv-geneformer}/bin/activate"

python - <<'PY'
import sys, torch
print("[geneformer-env]", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
from geneformer.emb_extractor import EmbExtractor
from geneformer.tokenizer import TranscriptomeTokenizer
print("[geneformer-api]", EmbExtractor, TranscriptomeTokenizer)
PY

python "$REPO_ROOT/scripts/extract_embeddings.py" \
  --embedder geneformer \
  --dataset pbmc3k \
  --split stratified \
  --seed 0 \
  --cache-dir "$REPO_ROOT/cache/embeddings" \
  --device cuda \
  --force

# ---------- IRT ranking in analysis env ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

missing=()
for m in uce scgpt; do
  if [[ ! -d "cache/embeddings/$m/pbmc3k" ]]; then
    missing+=("$m")
  fi
done
if [[ ! -d cache/embeddings/geneformer/pbmc3k ]]; then
  echo "[error] geneformer cache missing after extract step"
  exit 1
fi
if (( ${#missing[@]} > 0 )); then
  echo "[warn] missing embedding caches: ${missing[*]}"
fi
if [[ ! -d cache/embeddings/scgpt/pbmc3k ]]; then
  echo "[warn] waiting up to 3h for scgpt cache (job may still be running)..."
  for _ in $(seq 1 36); do
    if [[ -d cache/embeddings/scgpt/pbmc3k ]]; then
      echo "[ok] scgpt cache appeared"
      break
    fi
    sleep 300
  done
fi

if [[ ! -d cache/embeddings/uce/pbmc3k ]]; then
  echo "[error] UCE cache missing — cannot rank. Re-run UCE job."
  exit 1
fi
if [[ ! -d cache/embeddings/scgpt/pbmc3k ]]; then
  echo "[warn] scGPT still missing — ranking without scgpt (dev_pca, uce, geneformer)"
  python scripts/run_pipeline.py --config configs/ranking_pbmc3k_geneformer_interim.yaml
  echo "[geneformer-job] done -> runs/ranking_pbmc3k_geneformer_interim/report.md"
else
  python scripts/run_pipeline.py --config configs/ranking_pbmc3k_v2.yaml
  echo "[geneformer-job] done -> runs/ranking_pbmc3k_v2/report.md"
fi
