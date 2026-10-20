#!/bin/bash
#SBATCH --job-name=scfm-extract
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_%x_%j.out
#SBATCH --error=logs/extract_%x_%j.err

# One extraction job for any of the newer embedders, parameterized by EMBEDDER + DATASET.
# The per-model venv, Python module, and workdir env vars all differ, so they are resolved
# here rather than duplicated across a dozen near-identical scripts.
#
#   EMBEDDER=scprint DATASET=pbmc3k SPLIT=stratified \
#     sbatch -J scprint-pbmc3k slurm/extract_new_model.slurm
#
# Use scripts/submit_extractions.sh to submit the whole matrix at once.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings data/raw

EMBEDDER="${EMBEDDER:?set EMBEDDER, e.g. scprint}"
DATASET="${DATASET:?set DATASET, e.g. pbmc3k}"
SPLIT="${SPLIT:-cross_donor}"
SEED="${SEED:-0}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/cache/embeddings}"

echo "[extract] embedder=$EMBEDDER dataset=$DATASET split=$SPLIT seed=$SEED"
echo "[extract] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# ---------- dataset must exist before the heavy env is loaded ----------
module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/download_data.py --dataset "$DATASET"
deactivate

# ---------- resolve the model's isolated env ----------
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
export SCFM_GENEFORMER_WORKDIR="$REPO_ROOT/cache/geneformer_workdir"

case "$EMBEDDER" in
  geneformer_v2|geneformer_v2_316m)
    VENV="$REPO_ROOT/.venv-geneformer"
    PY_MODULE="python/3.10.1"
    ;;
  cellplm)
    VENV="$REPO_ROOT/.venv-cellplm"
    PY_MODULE="python/3.10.1"
    export SCFM_CELLPLM_WORKDIR="$REPO_ROOT/cache/cellplm_workdir"
    ;;
  scfoundation)
    VENV="$REPO_ROOT/.venv-scfoundation"
    PY_MODULE="python/3.10.1"
    export SCFM_SCFOUNDATION_WORKDIR="$REPO_ROOT/cache/scfoundation_workdir"
    ;;
  scprint|scprint2|scprint_small_v1|scprint_medium_v1|scprint_large_v1)
    VENV="$REPO_ROOT/.venv-scprint"
    PY_MODULE="python/3.10.1"
    export SCFM_SCPRINT_WORKDIR="$REPO_ROOT/cache/scprint_workdir"
    export LAMINDB_STORAGE="$REPO_ROOT/cache/scprint_workdir/lamindb"
    export LAMIN_CACHE_DIR="$REPO_ROOT/cache/scprint_workdir/lamin_cache"
    ;;
  transcriptformer)
    VENV="$REPO_ROOT/.venv-transcriptformer"
    PY_MODULE="python/3.12.10"
    export SCFM_TRANSCRIPTFORMER_WORKDIR="$REPO_ROOT/cache/tf_workdir"
    ;;
  langcell)
    VENV="$REPO_ROOT/.venv-langcell"
    PY_MODULE="python/3.10.1"
    export SCFM_LANGCELL_WORKDIR="$REPO_ROOT/cache/langcell_workdir"
    ;;
  state)
    VENV="$REPO_ROOT/.venv-state"
    PY_MODULE="python/3.12.10"
    export SCFM_STATE_WORKDIR="$REPO_ROOT/cache/state_workdir"
    ;;
  *)
    echo "[error] unknown embedder for this job: $EMBEDDER"
    echo "  supported: geneformer_v2 geneformer_v2_316m cellplm scfoundation transcriptformer"
    echo "             scprint scprint2 scprint_{small,medium,large}_v1 langcell state"
    exit 1
    ;;
esac

module load "$PY_MODULE" 2>/dev/null || true
case "$PY_MODULE" in
  python/3.10.1) export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}" ;;
  python/3.12.10) export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}" ;;
esac

if [[ ! -x "$VENV/bin/python" ]]; then
  # The setup target is named after the venv, not the embedder — variants share a parent env.
  echo "[error] $VENV missing. Build it on a login node (it needs network):"
  echo "  bash scripts/setup_embedder_venv.sh ${VENV##*/.venv-}"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import sys; print('[env]', sys.version.split()[0], sys.prefix)"

# ---------- extract ----------
python "$REPO_ROOT/scripts/extract_embeddings.py" \
  --embedder "$EMBEDDER" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --seed "$SEED" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --force

# The cache directory is the contract with the analysis half; verify it actually landed.
if ! compgen -G "$CACHE_DIR/$EMBEDDER/$DATASET/*.npz" > /dev/null; then
  echo "[error] no artifact under $CACHE_DIR/$EMBEDDER/$DATASET after extraction"
  exit 1
fi

echo "[extract] done -> $CACHE_DIR/$EMBEDDER/$DATASET"
ls -lh "$CACHE_DIR/$EMBEDDER/$DATASET"
