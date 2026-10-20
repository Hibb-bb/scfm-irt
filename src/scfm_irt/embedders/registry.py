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


def _uce() -> CellEmbedder:
    from scfm_irt.embedders.uce import UCEEmbedder

    return UCEEmbedder()


register_embedder("uce", _uce)


def _scgpt() -> CellEmbedder:
    from scfm_irt.embedders.scgpt import SCGPTEmbedder

    return SCGPTEmbedder()


register_embedder("scgpt", _scgpt)


def _geneformer() -> CellEmbedder:
    from scfm_irt.embedders.geneformer import GeneformerEmbedder

    return GeneformerEmbedder()


register_embedder("geneformer", _geneformer)


def _scimilarity() -> CellEmbedder:
    from scfm_irt.embedders.scimilarity import SCimilarityEmbedder

    return SCimilarityEmbedder()


register_embedder("scimilarity", _scimilarity)


def _cellplm() -> CellEmbedder:
    from scfm_irt.embedders.cellplm import CellPLMEmbedder

    return CellPLMEmbedder()


register_embedder("cellplm", _cellplm)


def _geneformer_v2() -> CellEmbedder:
    from scfm_irt.embedders.geneformer import GeneformerV2Embedder

    return GeneformerV2Embedder()


# Separate entry from `geneformer` (V1-10M) on purpose: the pipeline uses the registry name as
# the IRT model_id, so a shared name would merge two distinct models into one respondent.
register_embedder("geneformer_v2", _geneformer_v2)


def _geneformer_v2_316m() -> CellEmbedder:
    from scfm_irt.embedders.geneformer import GeneformerV2_316MEmbedder

    return GeneformerV2_316MEmbedder()


register_embedder("geneformer_v2_316m", _geneformer_v2_316m)


def _transcriptformer() -> CellEmbedder:
    from scfm_irt.embedders.transcriptformer import TranscriptFormerEmbedder

    return TranscriptFormerEmbedder()


register_embedder("transcriptformer", _transcriptformer)


def _scprint() -> CellEmbedder:
    from scfm_irt.embedders.scprint import SCPRINTEmbedder

    return SCPRINTEmbedder()


register_embedder("scprint", _scprint)


def _scprint2() -> CellEmbedder:
    from scfm_irt.embedders.scprint import SCPRINT2Embedder

    return SCPRINT2Embedder()


register_embedder("scprint2", _scprint2)


def _scprint_small_v1() -> CellEmbedder:
    from scfm_irt.embedders.scprint import SCPRINTSmallV1Embedder

    return SCPRINTSmallV1Embedder()


def _scprint_medium_v1() -> CellEmbedder:
    from scfm_irt.embedders.scprint import SCPRINTMediumV1Embedder

    return SCPRINTMediumV1Embedder()


def _scprint_large_v1() -> CellEmbedder:
    from scfm_irt.embedders.scprint import SCPRINTLargeV1Embedder

    return SCPRINTLargeV1Embedder()


# The v1 size ladder. Same corpus and tokenizer at three capacities, so these are strongly
# correlated respondents by construction — the point is the size comparison, not extra
# independent evidence about item difficulty.
register_embedder("scprint_small_v1", _scprint_small_v1)
register_embedder("scprint_medium_v1", _scprint_medium_v1)
register_embedder("scprint_large_v1", _scprint_large_v1)


def _scfoundation() -> CellEmbedder:
    from scfm_irt.embedders.scfoundation import SCFoundationEmbedder

    return SCFoundationEmbedder()


register_embedder("scfoundation", _scfoundation)


def _langcell() -> CellEmbedder:
    from scfm_irt.embedders.langcell import LangCellEmbedder

    return LangCellEmbedder()


# Pretrained against cell-identity text, so `label_supervised=True` — supplementary ranking
# only, alongside scimilarity.
register_embedder("langcell", _langcell)


def _state() -> CellEmbedder:
    from scfm_irt.embedders.state import StateEmbedder

    return StateEmbedder()


register_embedder("state", _state)

# cell_jepa: in-house model — not yet wired (no checkpoint / embedder code in repo).
# TODO(phase-6): register once Cell-JEPA inference path is available.
