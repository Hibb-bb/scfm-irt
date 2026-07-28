"""Stubs for the real foundation models — one class each, all in their own environments.

They live in one file only while they are stubs; split into `uce.py`, `scgpt.py`,
`geneformer.py`, `cell_jepa.py`, ... as each is actually implemented (Phase 5 onward).

Three rules, each corresponding to a way this silently goes wrong:

1. **Use the model's official preprocessing/tokenization.** Substituting a generic scanpy
   pipeline yields embeddings that look fine and mean nothing (plan section 10.10).
2. **Import inside methods, never at module scope.** These stacks conflict with each other and
   with the analysis env; a module-scope import breaks the whole registry.
3. **`requires_dataset_fit=True` means train cells only.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


class _FoundationEmbedder:
    """Shared shape for the zero-shot foundation models."""

    name: str = "unset"
    version: str = "unset"
    requires_dataset_fit: bool = False
    label_supervised: bool = False  # True if cell-type labels were used in pretraining
    env: str = "unset"  # the envs/*.yml this model must run under

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device

    def preprocess(self, adata: AnnData):  # noqa: ANN201
        raise NotImplementedError(f"TODO(phase-5): {self.name} official preprocessing")

    def fit(self, adata: AnnData) -> None:
        return None  # zero-shot

    def embed(self, adata: AnnData) -> np.ndarray:
        raise NotImplementedError(f"TODO(phase-5): {self.name} forward pass -> (n_cells, d)")


class UCEEmbedder(_FoundationEmbedder):
    name = "uce"
    env = "envs/env-uce.yml"


class SCimilarityEmbedder(_FoundationEmbedder):
    name = "scimilarity"
    env = "envs/env-scimilarity.yml"
    # Pretrained with cell-type labels, so its ability estimate is not comparable to the
    # zero-shot models on equal footing. Surface this in the report.
    label_supervised = True


class SCGPTEmbedder(_FoundationEmbedder):
    name = "scgpt"
    env = "envs/env-scgpt.yml"


class GeneformerEmbedder(_FoundationEmbedder):
    name = "geneformer"
    env = "envs/env-geneformer.yml"


class CellPLMEmbedder(_FoundationEmbedder):
    name = "cellplm"
    env = "envs/env-cellplm.yml"


class CellJEPAEmbedder(_FoundationEmbedder):
    """The in-house model — the reason the rest of this exists."""

    name = "cell_jepa"
    env = "envs/env-cell-jepa.yml"
