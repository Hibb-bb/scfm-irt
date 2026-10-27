#!/usr/bin/env bash
# Build .venv-aidocell for AIDO.Cell (renamed GB.Cell on Hugging Face: genbio-ai/GB.Cell-{3M,10M,100M}).
# Uses the standalone gb_cell package from genbio-ai/ModelGenerator (huggingface/gb.cell). Without
# flash-attn the model falls back to torch scaled_dot_product_attention, which runs on this cluster's
# glibc; flash-attn's prebuilt wheels need glibc >= 2.32.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SCFM_AIDOCELL_WORKDIR:-$REPO_ROOT/cache/aidocell_workdir}/ModelGenerator/huggingface/gb.cell"
VENV="$REPO_ROOT/.venv-aidocell"
UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/cache/uv}"
module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
[ -d "$SRC" ] || { echo "missing $SRC (clone genbio-ai/ModelGenerator, sparse path huggingface/gb.cell)"; exit 1; }
rm -rf "$VENV"
"$UV" venv -q -p /software/python/3.10.1/bin/python "$VENV"
P="$VENV/bin/python"
"$UV" pip install -q -p "$P" "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
"$UV" pip install -q -p "$P" "transformers>=4.40,<4.50" "numpy<2" anndata scanpy pandas tqdm safetensors huggingface_hub
"$UV" pip install -q -p "$P" --no-deps -e "$SRC"
"$UV" pip install -q -p "$P" --no-deps -e "$REPO_ROOT"
"$P" -c "import torch, transformers, gb_cell; from gb_cell.models import CellFoundationModel; print('[aidocell] ok torch', torch.__version__, 'transformers', transformers.__version__)"
