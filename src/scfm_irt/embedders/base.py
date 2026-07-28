"""The embedder interface and the cached artifact format.

The artifact is a contract between two environments that never share an interpreter, so it
carries its own provenance: which embedder version, which dataset, which split, which
preprocessing. The downstream pipeline reads these and must never trigger a recompute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


@runtime_checkable
class CellEmbedder(Protocol):
    """One interface for every model, wrapping that model's *official* preprocessing."""

    name: str
    version: str
    # True for scVI-style models that train per dataset (fit on TRAIN CELLS ONLY — plan
    # section 10.8); False for zero-shot foundation models.
    requires_dataset_fit: bool

    def preprocess(self, adata: AnnData) -> Any:
        """Model-specific normalization/tokenization. Never substitute a generic pipeline."""
        ...

    def fit(self, adata: AnnData) -> None:
        """No-op for zero-shot models; train-split-only fitting for the rest."""
        ...

    def embed(self, adata: AnnData) -> np.ndarray:
        """Return (n_cells, d), rows aligned to `adata.obs_names` in order."""
        ...


@dataclass(frozen=True)
class EmbeddingArtifact:
    """A cached embedding plus the provenance needed to trust it."""

    embedder: str
    embedder_version: str
    dataset: str
    split_scheme: str
    split_seed: int
    preprocessing_hash: str
    n_cells: int
    dim: int

    def cache_key(self) -> str:
        """Hash of (embedder, version, dataset, split, preprocessing) — plan section 5.1."""
        import hashlib

        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def save_embedding(
    path: str | Path,
    embedding: np.ndarray,
    cell_ids: np.ndarray,
    meta: EmbeddingArtifact,
) -> None:
    """Write the embedding, the aligned cell ids, and a JSON provenance sidecar.

    Cell ids are stored *with* the matrix so row alignment survives caching — an embedding
    without its ids is unusable, not merely inconvenient.
    """
    raise NotImplementedError("TODO(phase-5): npz/h5 write + .json sidecar")


def load_embedding(path: str | Path) -> tuple[np.ndarray, np.ndarray, EmbeddingArtifact]:
    """Read a cached artifact as (embedding, cell_ids, meta).

    Must assert `embedding.shape[0] == len(cell_ids)` and that ids are unique.
    """
    raise NotImplementedError("TODO(phase-5): read + validate alignment")
