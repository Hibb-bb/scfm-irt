#!/bin/bash
#SBATCH --job-name=state-pbmc3k
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/state_pbmc3k_%j.out
#SBATCH --error=logs/state_pbmc3k_%j.err

# Extract State (SE-600M) embeddings on pbmc3k, then run IRT probe + ranking.
# Chain after asset download:
#
#   ASSET=$(sbatch --parsable slurm/download_state_assets.slurm)
#   sbatch --dependency=afterok:$ASSET slurm/state_pbmc3k.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings cache/state_workdir data/raw

export SCFM_STATE_WORKDIR="$REPO_ROOT/cache/state_workdir"
MODEL_DIR="$SCFM_STATE_WORKDIR/SE-600M"

echo "[state-job] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# ---------- dataset + baseline cache (analysis env) ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/download_data.py --dataset pbmc3k
if [[ ! -d cache/embeddings/dev_pca/pbmc3k ]]; then
  python scripts/extract_embeddings.py \
    --embedder dev_pca --dataset pbmc3k --split stratified --seed 0 --device cpu
fi
deactivate

# ---------- verify SE-600M weights ----------
if [[ ! -f "$MODEL_DIR/se600m_epoch16.ckpt" ]]; then
  echo "[error] SE-600M checkpoint missing at $MODEL_DIR"
  echo "  submit: sbatch slurm/download_state_assets.slurm"
  exit 1
fi

# ---------- State isolated venv ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
bash "$REPO_ROOT/scripts/setup_embedder_venv.sh" state
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv-state/bin/activate"

python - <<'PY'
import sys
import torch

print("[state-env]", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

python "$REPO_ROOT/scripts/extract_embeddings.py" \
  --embedder state \
  --dataset pbmc3k \
  --split stratified \
  --seed 0 \
  --cache-dir "$REPO_ROOT/cache/embeddings" \
  --device cuda \
  --force

deactivate

# ---------- IRT evaluation (analysis env) ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

python scripts/run_pipeline.py --config configs/ranking_pbmc3k_state.yaml
echo "[state-job] done -> runs/ranking_pbmc3k_state/report.md"
