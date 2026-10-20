#!/bin/bash
#SBATCH --job-name=scfm-irt-uce
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/uce_pbmc3k_%j.out
#SBATCH --error=logs/uce_pbmc3k_%j.err

# Phase 5 end-to-end: PBMC3k + DevEmbedder + UCE + Bayesian IRT.
#
# Prerequisites that used to fail silently are now explicit steps:
#   - python/3.10.1 + LD_LIBRARY_PATH for the UCE venv
#   - scripts/download_uce_assets.py (size-checked Figshare downloads)
#   - UCEEmbedder fails fast if assets are empty / missing
#
#   mkdir -p logs && sbatch slurm/uce_pbmc3k.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs runs cache/embeddings cache/uce_workdir data/raw

echo "[uce-job] host=$(hostname)  cuda=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# ---------- analysis half (DevEmbedder + later IRT) ----------
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

python scripts/download_data.py --dataset pbmc3k
python scripts/extract_embeddings.py \
  --embedder dev_pca --dataset pbmc3k --split stratified --seed 0 --device cpu --force

# ---------- UCE half ----------
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"

UCE_VENV="${UCE_VENV:-$REPO_ROOT/.venv-uce}"
UCE_DIR="$REPO_ROOT/cache/uce_workdir/UCE"
export SCFM_UCE_WORKDIR="$REPO_ROOT/cache/uce_workdir"

if [[ ! -x "$UCE_VENV/bin/python" ]] || ! "$UCE_VENV/bin/python" -c "import sys" 2>/dev/null; then
  rm -rf "$UCE_VENV"
  /software/python/3.10.1/bin/python -m venv "$UCE_VENV"
  # shellcheck disable=SC1091
  source "$UCE_VENV/bin/activate"
  pip install -U pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch
  pip install numpy pandas scanpy anndata accelerate tqdm requests pyyaml
  pip install -e "$REPO_ROOT[sc]"
else
  # shellcheck disable=SC1091
  source "$UCE_VENV/bin/activate"
fi
python -c "import sys, torch; print('[uce-env]', sys.version.split()[0], 'cuda', torch.cuda.is_available())"

# Fetch UCE source if needed (no bare `git` dependency).
if [[ ! -f "$UCE_DIR/eval_single_anndata.py" ]]; then
  module load git 2>/dev/null || true
  mkdir -p "$REPO_ROOT/cache/uce_workdir"
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/snap-stanford/UCE.git "$UCE_DIR"
  else
    TMP_TGZ="$REPO_ROOT/cache/uce_workdir/uce_main.tar.gz"
    curl -L --fail -o "$TMP_TGZ" https://github.com/snap-stanford/UCE/archive/refs/heads/main.tar.gz
    rm -rf "$UCE_DIR" "$REPO_ROOT/cache/uce_workdir/UCE-main"
    tar -xzf "$TMP_TGZ" -C "$REPO_ROOT/cache/uce_workdir"
    mv "$REPO_ROOT/cache/uce_workdir/UCE-main" "$UCE_DIR"
  fi
  pip install -r "$UCE_DIR/requirements.txt" || true
fi

# Purge empty Figshare leftovers, then download with size checks
# (ndownloader.figshare.com — figshare.com URLs are WAF-blocked / empty).
find "$UCE_DIR/model_files" -type f -size 0 -delete 2>/dev/null || true
python "$REPO_ROOT/scripts/download_uce_assets.py" --uce-dir "$UCE_DIR"


# Fail fast before GPU work if assets still bad.
python - <<'PY'
from pathlib import Path
root = Path("cache/uce_workdir/UCE/model_files")
need = {
    "species_chrom.csv": 1_000_000,
    "species_offsets.pkl": 100,  # tiny pickle (~139 B) — do not over-threshold
    "all_tokens.torch": 2_000_000_000,
    "4layer_model.torch": 2_000_000_000,
}
for name, mn in need.items():
    p = root / name
    assert p.exists() and p.stat().st_size >= mn, (name, p.exists(), p.stat().st_size if p.exists() else 0)
pe = root / "protein_embeddings"
assert pe.is_dir() and any(pe.rglob("*.pt")), "protein_embeddings missing"
print("[assets-ok]", {k: (root / k).stat().st_size for k in need})
PY

python "$REPO_ROOT/scripts/extract_embeddings.py" \
  --embedder uce --dataset pbmc3k --split stratified --seed 0 \
  --cache-dir "$REPO_ROOT/cache/embeddings" --device cuda --force

# ---------- IRT in analysis env ----------
module load python/3.12.10 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.12.10/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python scripts/run_pipeline.py --config configs/uce_pbmc3k_v1.yaml

echo "[uce-job] done -> runs/uce_pbmc3k_v1/report.md"
