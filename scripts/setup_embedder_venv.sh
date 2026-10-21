#!/usr/bin/env bash
# Build or verify the isolated embedder venvs, one per model (plan section 2.1).
#
# Run on a login or GPU node with network access BEFORE submitting SLURM jobs — several of
# these steps (checkpoint pulls, lamindb ontology population) need the internet, which compute
# nodes may not have:
#   module load python/3.10.1
#   export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
#   bash scripts/setup_embedder_venv.sh all
#
# Usage:
#   bash scripts/setup_embedder_venv.sh <target> [--force]
#
# Targets: scgpt geneformer cellplm scfoundation scprint transcriptformer langcell state all

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/slurm/_pip_helpers.sh"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

TARGET="${1:-all}"
FORCE="${2:-}"

module load python/3.10.1 2>/dev/null || true
export LD_LIBRARY_PATH="/software/python/3.10.1/lib:${LD_LIBRARY_PATH:-}"
PY310="/software/python/3.10.1/bin/python"
# TranscriptFormer requires >=3.11. Prefer the module-managed 3.12; fall back to spack 3.11.
PY312="/software/2025/python/3.12.10/bin/python3.12"
PY311="/hpc/software/spack_v20d1/spack/opt/spack/linux-rhel7-x86_64/gcc-12.3.0/python-3.11.11-aweofbtlcw2nhelffvllbn5tl3m4pql7/bin/python3.11"

pick_modern_python() {
  if [[ -x "$PY312" ]]; then
    echo "$PY312"
  elif [[ -x "$PY311" ]]; then
    echo "$PY311"
  else
    echo "[error] no Python >=3.11 found for this target" >&2
    return 1
  fi
}

scgpt_healthy() {
  [[ -x "$REPO_ROOT/.venv-scgpt/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-scgpt/bin/python" - <<'PY' >/dev/null 2>&1
import numpy as np
assert np.__version__.startswith("1."), np.__version__
import IPython, scgpt, torch, torchtext
import scgpt.tokenizer.gene_tokenizer  # noqa: F401
assert torch.__version__.startswith("2.1"), torch.__version__
import scfm_irt  # noqa: F401
PY
}

geneformer_healthy() {
  [[ -x "$REPO_ROOT/.venv-geneformer/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-geneformer/bin/python" - <<'PY' >/dev/null 2>&1
import numpy as np
assert np.__version__.startswith("1."), np.__version__
import torch
assert torch.__version__.startswith("2.1"), torch.__version__
from geneformer.emb_extractor import EmbExtractor
from geneformer.tokenizer import TranscriptomeTokenizer
import tdigest
import scfm_irt  # noqa: F401
PY
}

setup_scgpt() {
  local venv="$REPO_ROOT/.venv-scgpt"
  if [[ "$FORCE" == "--force" ]] || ! scgpt_healthy; then
    echo "[setup-scgpt] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    install_torch_stack
    pip_install_each \
      "numpy<2" \
      pandas \
      scipy \
      scanpy \
      anndata \
      tqdm \
      pyyaml \
      gdown \
      ipython
    pip_retry install scgpt || pip_retry install scgpt --no-deps
    pip_install_each datasets "transformers==4.40.2" tiktoken
    repin_torch_stack
    pip_retry install -e "$REPO_ROOT[sc]"
    deactivate
  else
    echo "[setup-scgpt] already healthy"
  fi
  scgpt_healthy
  date -Iseconds > "$REPO_ROOT/.venv-scgpt/.setup_complete"
  "$REPO_ROOT/.venv-scgpt/bin/python" - <<'PY'
import sys, torch, scgpt
print("[setup-scgpt] ok", sys.version.split()[0], "torch", torch.__version__)
PY
}

setup_geneformer() {
  local venv="$REPO_ROOT/.venv-geneformer"
  local gf_repo="${SCFM_GENEFORMER_WORKDIR:-$REPO_ROOT/cache/geneformer_workdir}/Geneformer"
  if [[ ! -d "$gf_repo/geneformer" ]]; then
    echo "[setup-geneformer] Geneformer repo missing at $gf_repo"
    echo "  run: python scripts/download_geneformer_assets.py"
    return 1
  fi

  if [[ "$FORCE" == "--force" ]] || ! geneformer_healthy; then
    echo "[setup-geneformer] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    install_torch_stack
    pip_install_each \
      "numpy<2" \
      pandas \
      scipy \
      scanpy \
      anndata \
      tqdm \
      pyyaml \
      packaging
    pip_install_each \
      "transformers==4.40.2" \
      datasets \
      pyarrow \
      loompy \
      tdigest \
      "peft==0.10.0" \
      optuna \
      optuna-integration \
      tensorboard
    pip_retry install "numpy<2" --force-reinstall --no-deps
    pip_retry install -e "$gf_repo" --no-deps
    pip_retry install -e "$REPO_ROOT[sc]"
    repin_torch_stack
    deactivate
  else
    echo "[setup-geneformer] already healthy"
  fi
  geneformer_healthy
  date -Iseconds > "$REPO_ROOT/.venv-geneformer/.setup_complete"
  "$REPO_ROOT/.venv-geneformer/bin/python" - <<'PY'
import sys, torch
from geneformer.emb_extractor import EmbExtractor
print("[setup-geneformer] ok", sys.version.split()[0], "torch", torch.__version__)
PY
}

cellplm_healthy() {
  [[ -x "$REPO_ROOT/.venv-cellplm/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-cellplm/bin/python" - <<'PY' >/dev/null 2>&1
import numpy as np

# torch 2.1.2 is built against the numpy 1.x C API. With numpy 2 installed it still imports,
# but every tensor <-> array conversion fails at runtime ("_ARRAY_API not found"), which is
# what kept CellPLM from ever producing embeddings.
assert np.__version__.startswith("1."), f"numpy {np.__version__} breaks torch 2.1.2"
import torch

torch.from_numpy(np.zeros(4, dtype=np.float32)).numpy()
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline  # noqa: F401
import scfm_irt  # noqa: F401
PY
}

setup_cellplm() {
  local venv="$REPO_ROOT/.venv-cellplm"
  if [[ "$FORCE" == "--force" ]] || ! cellplm_healthy; then
    echo "[setup-cellplm] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    install_torch_stack
    pip_install_each scanpy anndata tqdm pyyaml hdf5plugin
    # CellPLM pins old deps that fight the torch stack; --no-deps keeps ours intact.
    pip_retry install cellplm || pip_retry install cellplm --no-deps
    pip_retry install -e "$REPO_ROOT[sc]"
    # Transitive deps drag in numpy 2, which silently breaks torch 2.1.2 at runtime.
    repin_torch_stack
    deactivate
  else
    echo "[setup-cellplm] already healthy"
  fi
  cellplm_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-cellplm] ok"
}

scfoundation_healthy() {
  [[ -x "$REPO_ROOT/.venv-scfoundation/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-scfoundation/bin/python" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
import einops  # noqa: F401
import local_attention  # noqa: F401
import scanpy  # noqa: F401
import scfm_irt  # noqa: F401
PY
}

setup_scfoundation() {
  local venv="$REPO_ROOT/.venv-scfoundation"
  local work="${SCFM_SCFOUNDATION_WORKDIR:-$REPO_ROOT/cache/scfoundation_workdir}"
  if [[ ! -d "$work/scFoundation/model" ]]; then
    echo "[setup-scfoundation] repo missing at $work/scFoundation"
    echo "  run: python scripts/download_scfoundation_assets.py"
    return 1
  fi
  if [[ "$FORCE" == "--force" ]] || ! scfoundation_healthy; then
    echo "[setup-scfoundation] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    install_torch_stack
    pip_install_each \
      "numpy<2" pandas scipy scanpy anndata tqdm pyyaml \
      einops local_attention
    pip_retry install -e "$REPO_ROOT[sc]"
    repin_torch_stack
    deactivate
  else
    echo "[setup-scfoundation] already healthy"
  fi
  scfoundation_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-scfoundation] ok"
}

scprint_healthy() {
  [[ -x "$REPO_ROOT/.venv-scprint/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-scprint/bin/python" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
from scprint import scPrint  # noqa: F401
from scprint.tasks import Embedder  # noqa: F401
import scfm_irt  # noqa: F401
PY
}

setup_scprint() {
  local venv="$REPO_ROOT/.venv-scprint"
  local lamin_db="${SCFM_LAMIN_STORAGE:-$REPO_ROOT/cache/scprint_workdir/lamindb}"
  if [[ "$FORCE" == "--force" ]] || ! scprint_healthy; then
    echo "[setup-scprint] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    install_torch_modern
    pip_install_each scanpy anndata pandas scipy tqdm pyyaml huggingface_hub
    pip_retry install scprint
    # scPRINT-2 ships separately; optional so a failure here does not sink scPRINT v1.
    pip_retry install scprint2 || echo "[warn] scprint2 unavailable; scprint2 embedder disabled"
    # lamindb's stdlib sqlite3 is too old on RHEL7 (documented scPRINT FAQ workaround).
    pip_retry install "pysqlite3-binary>=0.5.2" || true
    local sitedir
    sitedir="$(python -c 'import site; print(site.getsitepackages()[0])')"
    cat > "$sitedir/sitecustomize.py" <<'PY'
try:
    import sys

    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass
PY
    pip_retry install -e "$REPO_ROOT[sc]"
    deactivate
  else
    echo "[setup-scprint] already healthy"
  fi

  # lamindb instance + ontologies: network-bound, 2-10 minutes, must exist before any job runs.
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  mkdir -p "$(dirname "$lamin_db")"
  if [[ ! -d "$lamin_db" ]]; then
    echo "[setup-scprint] lamin init -> $lamin_db"
    lamin init --storage "$lamin_db" --name scfm_irt --modules bionty
    python -c 'from scdataloader.utils import populate_my_ontology; populate_my_ontology()' \
      || echo "[warn] ontology population incomplete; scPRINT inference may fail"
  else
    echo "[setup-scprint] lamin instance present at $lamin_db"
  fi
  deactivate

  scprint_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-scprint] ok"
}

transcriptformer_healthy() {
  [[ -x "$REPO_ROOT/.venv-transcriptformer/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-transcriptformer/bin/python" - <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__ < "2.6", torch.__version__
import zarr
assert zarr.__version__ < "3", zarr.__version__
import anndata  # noqa: F401
import transcriptformer  # noqa: F401
import scfm_irt  # noqa: F401
PY
  [[ -x "$REPO_ROOT/.venv-transcriptformer/bin/transcriptformer" ]]
}

setup_transcriptformer() {
  local venv="$REPO_ROOT/.venv-transcriptformer"
  local py
  py="$(pick_modern_python)"
  if [[ "$FORCE" == "--force" ]] || ! transcriptformer_healthy; then
    echo "[setup-transcriptformer] rebuilding $venv with $py"
    rm -rf "$venv"
    "$py" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    # 2.6.0+ breaks checkpoint unpickling per the upstream README.
    install_torch_modern 2.5.1 0.20.1
    pip_install_each scanpy anndata pandas h5py hydra-core tqdm
    pip_retry install transcriptformer
    pip_retry install -e "$REPO_ROOT[sc]"
    # anndata refuses to import against zarr 3 ("major version > 2 is not supported"), and
    # transcriptformer's own deps pull zarr 3 in.
    pip_retry install "zarr<3"
    assert_torch_version 2.5
    deactivate
  else
    echo "[setup-transcriptformer] already healthy"
  fi
  transcriptformer_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-transcriptformer] ok"
}

langcell_healthy() {
  [[ -x "$REPO_ROOT/.venv-langcell/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-langcell/bin/python" - <<'PY' >/dev/null 2>&1
import numpy as np
assert np.__version__.startswith("1."), np.__version__
import torch
from transformers import BertModel  # noqa: F401
from geneformer.tokenizer import TranscriptomeTokenizer  # noqa: F401
import scfm_irt  # noqa: F401
PY
}

state_healthy() {
  [[ -x "$REPO_ROOT/.venv-state/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-state/bin/python" - <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__ >= "2.7"
import scfm_irt  # noqa: F401
PY
  [[ -x "$REPO_ROOT/.venv-state/bin/state" ]]
}

setup_langcell() {
  local venv="$REPO_ROOT/.venv-langcell"
  local gf_repo="${SCFM_GENEFORMER_WORKDIR:-$REPO_ROOT/cache/geneformer_workdir}/Geneformer"
  # LangCell's cell encoder is Geneformer-initialized, so it reuses that tokenizer verbatim.
  if [[ ! -d "$gf_repo/geneformer" ]]; then
    echo "[setup-langcell] Geneformer repo missing at $gf_repo (needed for the tokenizer)"
    echo "  run: python scripts/download_geneformer_assets.py"
    return 1
  fi
  if [[ "$FORCE" == "--force" ]] || ! langcell_healthy; then
    echo "[setup-langcell] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    install_torch_stack
    # geneformer/__init__.py eagerly imports its fine-tuning and hyperparameter-search
    # modules, so peft/optuna/tensorboard are required even though we only want the tokenizer.
    # peft is pinned to 0.10.0 to match transformers 4.40.2; newer peft imports symbols that
    # release does not have.
    pip_install_each \
      "numpy<2" pandas scipy scanpy anndata tqdm pyyaml packaging \
      "transformers==4.40.2" "peft==0.10.0" optuna optuna-integration tensorboard \
      datasets pyarrow loompy tdigest gdown
    pip_retry install "numpy<2" --force-reinstall --no-deps
    pip_retry install -e "$gf_repo" --no-deps
    pip_retry install -e "$REPO_ROOT[sc]"
    repin_torch_stack
    deactivate
  else
    echo "[setup-langcell] already healthy"
  fi
  langcell_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-langcell] ok"
}

uce_healthy() {
  [[ -x "$REPO_ROOT/.venv-uce/bin/python" ]] || return 1
  "$REPO_ROOT/.venv-uce/bin/python" - <<'PY' >/dev/null 2>&1
import torch, scanpy, accelerate  # noqa: F401  (the UCE driver imports accelerate at startup)
import scfm_irt  # noqa: F401
PY
}

setup_uce() {
  local venv="$REPO_ROOT/.venv-uce"
  if [[ "$FORCE" == "--force" ]] || ! uce_healthy; then
    echo "[setup-uce] rebuilding $venv"
    rm -rf "$venv"
    "$PY310" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    pip_retry install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip_retry install torch
    pip_install_each numpy pandas scanpy anndata accelerate tqdm requests pyyaml
    pip_retry install -e "$REPO_ROOT[sc]"
    deactivate
  else
    echo "[setup-uce] already healthy"
  fi
  uce_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-uce] ok"
}

setup_state() {
  local venv="$REPO_ROOT/.venv-state"
  local py
  py="$(pick_modern_python)"
  if [[ "$FORCE" == "--force" ]] || ! state_healthy; then
    echo "[setup-state] rebuilding $venv with $py"
    rm -rf "$venv"
    "$py" -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip_retry install -U pip
    pip_install_each scanpy anndata pandas scipy h5py tqdm pyyaml
    # Matches the Colab workflow (arc-state from PyPI; same CLI as git@main).
    pip_retry install arc-state
    pip_retry install -e "$REPO_ROOT[sc]"
    deactivate
  else
    echo "[setup-state] already healthy"
  fi
  state_healthy
  date -Iseconds > "$venv/.setup_complete"
  echo "[setup-state] ok"
}

case "$TARGET" in
  scgpt) setup_scgpt ;;
  geneformer) setup_geneformer ;;
  cellplm) setup_cellplm ;;
  scfoundation) setup_scfoundation ;;
  scprint) setup_scprint ;;
  transcriptformer) setup_transcriptformer ;;
  langcell) setup_langcell ;;
  state) setup_state ;;
  uce) setup_uce ;;
  all)
    setup_scgpt
    setup_geneformer
    setup_cellplm
    setup_transcriptformer
    setup_scprint
    setup_uce
    # These two need assets fetched first, so a miss is a warning, not a hard failure.
    setup_langcell || echo "[warn] langcell setup skipped"
    setup_scfoundation || echo "[warn] scfoundation setup skipped"
    setup_state || echo "[warn] state setup skipped"
    ;;
  *)
    echo "usage: $0 scgpt|geneformer|cellplm|scfoundation|scprint|transcriptformer|langcell|state|uce|all [--force]"
    exit 1
    ;;
esac

echo "[setup] done"
