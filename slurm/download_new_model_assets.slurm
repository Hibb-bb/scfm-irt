#!/bin/bash
#SBATCH --job-name=scfm-new-assets
#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/new_model_assets_%j.out
#SBATCH --error=logs/new_model_assets_%j.err

# Fetch checkpoints for the newly added embedders. Kept off the GPU partition on purpose:
# these are slow network pulls and should never sit on a GPU allocation.
#
#   sbatch slurm/download_new_model_assets.slurm
#
# Each download is independent, so one failure reports and moves on rather than sinking the
# whole batch. scFoundation is expected to "fail" here until its checkpoint is staged by hand
# (SharePoint only — see scripts/download_scfoundation_assets.py).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs cache

failed=()
ok=()

try_step() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  if "$@"; then
    ok+=("$label")
  else
    echo "[warn] $label failed"
    failed+=("$label")
  fi
}

module load python/3.12.10 2>/dev/null || true
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Geneformer V2-104M lands beside the existing V1-10M in the same workdir.
try_step "geneformer-v2" python scripts/download_geneformer_assets.py \
  --out cache/geneformer_workdir --model Geneformer-V2-104M

try_step "scprint" python scripts/download_scprint_assets.py --out cache/scprint_workdir

try_step "cellplm" python scripts/download_cellplm_assets.py \
  --out cache/cellplm_workdir/ckpt

# Clones the repo and tells you where to put the SharePoint checkpoint.
try_step "scfoundation" python scripts/download_scfoundation_assets.py \
  --out cache/scfoundation_workdir

deactivate

# LangCell needs gdown, which lives in its own venv.
if [[ -x "$REPO_ROOT/.venv-langcell/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-langcell/bin/activate"
  try_step "langcell" python scripts/download_langcell_assets.py --out cache/langcell_workdir
  deactivate
else
  echo "[warn] .venv-langcell missing; run scripts/setup_embedder_venv.sh langcell first"
  failed+=("langcell")
fi

# TranscriptFormer downloads through its own CLI, so it needs its own venv.
if [[ -x "$REPO_ROOT/.venv-transcriptformer/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-transcriptformer/bin/activate"
  try_step "transcriptformer" python scripts/download_transcriptformer_assets.py \
    --out cache/tf_workdir --model tf_sapiens
  deactivate
else
  echo "[warn] .venv-transcriptformer missing; run scripts/setup_embedder_venv.sh transcriptformer"
  failed+=("transcriptformer")
fi

if [[ -x "$REPO_ROOT/.venv-state/bin/python" ]]; then
  try_step "state" python scripts/download_state_assets.py --out cache/state_workdir/SE-600M
else
  echo "[warn] .venv-state missing; run scripts/setup_embedder_venv.sh state first (optional for HF download)"
  try_step "state" python scripts/download_state_assets.py --out cache/state_workdir/SE-600M
fi

echo
echo "=== summary ==="
echo "ok:     ${ok[*]:-none}"
echo "failed: ${failed[*]:-none}"

# Exit non-zero only if everything failed; partial success still unblocks some extractions.
if (( ${#ok[@]} == 0 )); then
  exit 1
fi
