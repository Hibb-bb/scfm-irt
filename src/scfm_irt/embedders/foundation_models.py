"""Stubs for foundation models that do not have a real implementation yet.

Every model that works lives in its own module (`uce.py`, `scgpt.py`, `geneformer.py`,
`scimilarity.py`, `cellplm.py`, `scfoundation.py`, `scprint.py`, `transcriptformer.py`,
`langcell.py`). What is left here is the shape a new one should take, plus the in-house model
we cannot wire until there is a checkpoint.

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
        raise NotImplementedError(f"{self.name}: official preprocessing not implemented")

    def fit(self, adata: AnnData) -> None:
        return None  # zero-shot

    def embed(self, adata: AnnData) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: forward pass -> (n_cells, d) not implemented")


class CellJEPAEmbedder(_FoundationEmbedder):
    """The in-house model — the reason the rest of this exists.

    Not registered: there is no checkpoint or inference path in the repo yet. See the
    TODO(phase-6) marker in `registry.py`.
    """

    name = "cell_jepa"
    env = "envs/env-cell-jepa.yml"
