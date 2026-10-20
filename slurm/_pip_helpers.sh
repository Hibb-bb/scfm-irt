# Shared pip helpers for isolated embedder venvs on Quest.
# Source from slurm/*.slurm or scripts/setup_embedder_venv.sh after set -euo pipefail.

pip_retry() {
  # Usage: pip_retry install [pip args...]
  local attempt max=8
  for attempt in $(seq 1 "$max"); do
    if pip --default-timeout=300 --retries 5 "$@"; then
      return 0
    fi
    echo "[warn] pip $* failed (attempt ${attempt}/${max})"
    sleep $((attempt * 20))
  done
  echo "[error] pip failed after ${max} attempts: pip $*"
  return 1
}

pip_install_each() {
  # Install packages one at a time to survive flaky PyPI downloads on Quest.
  local pkg
  for pkg in "$@"; do
    pip_retry install "$pkg" || return 1
  done
}

install_numpy_first() {
  pip_retry install "numpy<2"
}

install_torch_stack() {
  # Pin torch/torchvision/torchtext together for scGPT + Geneformer.
  install_numpy_first
  pip_retry install torch==2.1.2 torchvision==0.16.2 torchtext==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121 \
    || pip_retry install torch==2.1.2 torchvision==0.16.2 torchtext==0.16.2
  # torchvision may pull numpy 2.x; keep the stack on numpy 1.x.
  pip_retry install "numpy<2" --force-reinstall --no-deps
}

repin_torch_stack() {
  pip_retry install --force-reinstall --no-deps \
    torch==2.1.2 torchvision==0.16.2 torchtext==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121 \
    || pip_retry install --force-reinstall --no-deps \
    torch==2.1.2 torchvision==0.16.2 torchtext==0.16.2
  pip_retry install "numpy<2" --force-reinstall --no-deps
}

install_torch_modern() {
  # For the newer stacks (scPRINT, TranscriptFormer, CellPLM) that cannot run on the
  # torch 2.1.2 / numpy<2 pins above. torchtext is deliberately omitted: it was archived and
  # has no build for these versions, and only scGPT needs it.
  #
  # Ceiling is torch 2.5.1 — TranscriptFormer warns 2.6.0+ raises pickle errors loading its
  # checkpoints. Quest provides cuda-12.1, so cu121 wheels are the right match.
  local ver="${1:-2.5.1}"
  local vision="${2:-0.20.1}"
  pip_retry install "torch==${ver}" "torchvision==${vision}" \
    --index-url https://download.pytorch.org/whl/cu121 \
    || pip_retry install "torch==${ver}" "torchvision==${vision}"
}

assert_torch_version() {
  # Usage: assert_torch_version 2.5   (prefix match, so patch releases pass)
  local want="$1"
  python - "$want" <<'PY'
import sys
import torch

want = sys.argv[1]
got = torch.__version__
if not got.startswith(want):
    raise SystemExit(f"[error] torch {got} does not match required prefix {want}")
print(f"[ok] torch {got} cuda={torch.cuda.is_available()}")
PY
}
