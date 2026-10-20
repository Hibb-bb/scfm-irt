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


def artifact_paths(cache_dir: str | Path, meta: EmbeddingArtifact) -> tuple[Path, Path]:
    """Return `(npz_path, sidecar_path)` for a cache entry."""
    root = Path(cache_dir) / meta.embedder / meta.dataset
    key = meta.cache_key()
    return root / f"{key}.npz", root / f"{key}.json"


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
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape {embedding.shape}")
    if embedding.shape[0] != len(cell_ids):
        raise ValueError(
            f"row mismatch: embedding has {embedding.shape[0]} rows, {len(cell_ids)} cell ids"
        )
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique")
    if embedding.shape != (meta.n_cells, meta.dim):
        raise ValueError(
            f"meta says ({meta.n_cells}, {meta.dim}) but embedding is {embedding.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".json")

    # dtype=U keeps ids as a unicode array so np.load(..., allow_pickle=False) works.
    np.savez_compressed(
        path,
        embedding=np.asarray(embedding, dtype=np.float32),
        cell_ids=np.asarray(cell_ids, dtype="U"),
    )
    sidecar.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n")


def align_embedding_rows(
    embedding: np.ndarray,
    got_ids: np.ndarray,
    want_ids: np.ndarray,
    model_name: str = "embedder",
) -> np.ndarray:
    """Reorder `embedding` from `got_ids` onto `want_ids`.

    Several models shuffle cells (batched dataloaders) or drop them (QC filters, empty
    tokenizations) and hand back rows in their own order. Silently accepting that order is the
    same class of bug as a transposed response matrix: everything still runs, and every number
    afterwards is wrong. So realign by id and refuse to guess when a cell is missing.
    """
    got = [str(c) for c in got_ids]
    want = [str(c) for c in want_ids]
    if embedding.shape[0] == len(want) and got == want:
        return embedding

    position = {cid: i for i, cid in enumerate(got)}
    missing = [cid for cid in want if cid not in position]
    if missing:
        raise RuntimeError(
            f"{model_name} returned no embedding for {len(missing)}/{len(want)} cells "
            f"(e.g. {missing[:5]}); cannot align to the response matrix."
        )
    return embedding[[position[cid] for cid in want]]


def load_embedding(path: str | Path) -> tuple[np.ndarray, np.ndarray, EmbeddingArtifact]:
    """Read a cached artifact as (embedding, cell_ids, meta).

    Must assert `embedding.shape[0] == len(cell_ids)` and that ids are unique.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing provenance sidecar: {sidecar}")

    raw = np.load(path, allow_pickle=False)
    embedding = np.asarray(raw["embedding"])
    cell_ids = np.asarray(raw["cell_ids"])
    meta = EmbeddingArtifact(**json.loads(sidecar.read_text()))

    if embedding.shape[0] != len(cell_ids):
        raise ValueError(
            f"corrupt artifact {path}: {embedding.shape[0]} rows vs {len(cell_ids)} ids"
        )
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError(f"corrupt artifact {path}: duplicate cell_ids")
    if embedding.shape != (meta.n_cells, meta.dim):
        raise ValueError(
            f"corrupt artifact {path}: meta ({meta.n_cells}, {meta.dim}) vs {embedding.shape}"
        )
    return embedding, cell_ids, meta
