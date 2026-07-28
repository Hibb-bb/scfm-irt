"""Cell embedders — the heavy side of the boundary.

Nothing in `probe`, `response`, `irt`, or `analysis` may import this package. Embedders run in
their own isolated environments (one per model) and communicate downstream only through cached
embedding artifacts. See plan section 2.1.
"""

from scfm_irt.embedders.base import CellEmbedder, EmbeddingArtifact
from scfm_irt.embedders.registry import get_embedder, list_embedders

__all__ = ["CellEmbedder", "EmbeddingArtifact", "get_embedder", "list_embedders"]
