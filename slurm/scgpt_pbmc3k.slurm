#!/bin/bash
#SBATCH --job-name=scfm-irt-scgpt
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/scgpt_pbmc3k_%j.out
#SBATCH --error=logs/scgpt_pbmc3k_%j.err

# Extract scGPT embeddings on PBMC3k, then fit IRT ranking vs UCE + DevEmbedder.
# Prefers existing caches for uce/dev_pca. Chain after asset download:
#
#   ASSET=$(sbatch --parsable slurm/download_scgpt_assets.slurm)
#   sbatch --dependency=afterok:$ASSET slurm/scgpt_pbmc3k.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings cache/scgpt_workdir data/raw

echo "[scgpt-job] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

export SCFM_SCGPT_WORKDIR="$REPO_ROOT/cache/scgpt_workdir"
MODEL_DIR="$SCFM_SCGPT_WORKDIR/scGPT_human"

# ---------- ensure checkpoint ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q gdown
python scripts/download_scgpt_assets.py --out "$MODEL_DIR"
python - <<'PY'
from pathlib import Path
p = Path("cache/scgpt_workdir/scGPT_human")
for f in ("args.json", "best_model.pt", "vocab.json"):
    fp = p / f
    assert fp.exists() and fp.stat().st_size > 0, (f, fp)
    print("[ckpt]", f, fp.stat().st_size)
PY

# ---------- analysis half: data + cached baselines ----------
python scripts/download_data.py --dataset pbmc3k
# Only recompute if missing
if [[ ! -d cache/embeddings/dev_pca/pbmc3k ]]; then
  python scripts/extract_embeddings.py \
    --embedder dev_pca --dataset pbmc3k --split stratified --seed 0 --device cpu
fi

# ---------- scGPT isolated venv ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" scgpt
# shellcheck disable=SC1091
source "${SCGPT_VENV:-$REPO_ROOT/.venv-scgpt}/bin/activate"

python - <<'PY'
import sys, torch, IPython, scgpt, torchtext
import scgpt.tokenizer.gene_tokenizer  # noqa: F401
print("[scgpt-env]", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
print("[scgpt-pkg]", getattr(scgpt, "__version__", "unknown"))
PY

python "$REPO_ROOT/scripts/extract_embeddings.py" \
  --embedder scgpt \
  --dataset pbmc3k \
  --split stratified \
  --seed 0 \
  --cache-dir "$REPO_ROOT/cache/embeddings" \
  --device cuda \
  --force

# ---------- IRT ranking in analysis env (needs uce cache too) ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

if [[ ! -d cache/embeddings/uce/pbmc3k ]]; then
  echo "[warn] UCE cache missing — ranking config expects it. Re-run UCE job or drop uce from config."
  exit 1
fi

python scripts/run_pipeline.py --config configs/ranking_pbmc3k_v1.yaml
echo "[scgpt-job] done -> runs/ranking_pbmc3k_v1/report.md"
