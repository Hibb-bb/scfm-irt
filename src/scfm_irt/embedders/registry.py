"""Embedder registry.

Adding a respondent to the evaluation — including a raw-feature baseline (LR on HVGs or PCA) —
is a registry entry, not an architecture change: a baseline is just another subject in the
response matrix (plan section 12).

Entries are lazy: importing this module must not import any model's dependency stack.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scfm_irt.embedders.base import CellEmbedder

# name -> a zero-arg factory, so the heavy import happens only when that embedder is used.
_FACTORIES: dict[str, Callable[[], CellEmbedder]] = {}


def register_embedder(name: str, factory: Callable[[], CellEmbedder]) -> None:
    if name in _FACTORIES:
        raise ValueError(f"embedder {name!r} is already registered")
    _FACTORIES[name] = factory


def list_embedders() -> list[str]:
    return sorted(_FACTORIES)


def get_embedder(name: str) -> CellEmbedder:
    if name not in _FACTORIES:
        raise KeyError(f"unknown embedder {name!r}; registered: {list_embedders()}")
    return _FACTORIES[name]()


def _dev_pca() -> CellEmbedder:
    from scfm_irt.embedders.dev import DevEmbedder

    return DevEmbedder(method="pca")


def _dev_random() -> CellEmbedder:
    from scfm_irt.embedders.dev import DevEmbedder

    return DevEmbedder(method="random")


register_embedder("dev_pca", _dev_pca)
register_embedder("dev_random", _dev_random)

# TODO(phase-5/6): register the real models, each behind a lazy factory and an isolated env.
# Verify each loading API against its own repo at implementation time — these interfaces move.
#   uce, scimilarity (both zero-shot; SCimilarity was trained with cell-type labels, so flag it
#   as label-supervised when interpreting its ability), scgpt, geneformer, cellplm,
#   scfoundation, cellfm, cell_jepa.
