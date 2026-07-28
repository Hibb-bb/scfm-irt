"""Loading datasets to AnnData.

`anndata`/`scanpy` are imported lazily so that the IRT and analysis modules stay importable in
an environment without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scfm_irt.data.registry import DatasetSpec

if TYPE_CHECKING:
    from anndata import AnnData


def load_dataset(spec: DatasetSpec) -> AnnData:
    """Load `spec.path` and validate it against the spec.

    Must check that `label_key` (and `batch_key`/`donor_key` when set) exist in `.obs`, that
    `obs_names` are unique, and that the orientation is cells x genes — the inventory has a few
    possibly-transposed entries and a silent transposition here poisons everything downstream
    (plan section 11).
    """
    raise NotImplementedError("TODO(phase-1): read h5ad, validate obs keys and orientation")


def preprocess_for_probe(adata: AnnData, n_top_genes: int = 2000) -> AnnData:
    """Generic log-normalization + HVG selection, for the DevEmbedder path only.

    Real embedders must use their own official preprocessing instead (plan section 10.10);
    substituting this one produces plausible-looking but meaningless embeddings.
    """
    raise NotImplementedError("TODO(phase-1): normalize_total, log1p, highly_variable_genes")
