"""Dataset inventory / sanity check.

`describe` reproducing the known inventory table (dims, #cell types, #donors) for a dataset is
the Phase 0 acceptance criterion, and it is also the cheapest way to catch a wrong download or
a transposed matrix before any GPU time is spent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scfm_irt.data.registry import DatasetSpec, get_spec

if TYPE_CHECKING:
    from anndata import AnnData


def describe(adata: AnnData, spec: DatasetSpec) -> dict[str, Any]:
    """Summarize a loaded dataset.

    Returns n_cells, n_genes, n_cell_types, n_donors, n_batches, the rarest and most common
    class with counts, the imbalance ratio, and median total counts per cell. Warns when
    `expected_n_cells`/`expected_n_types` are set on the spec and disagree.
    """
    raise NotImplementedError("TODO(phase-0): compute inventory fields from adata")


def describe_by_name(name: str) -> dict[str, Any]:
    """Convenience wrapper used by `scfm-irt describe <dataset>`."""
    from scfm_irt.data.loaders import load_dataset

    spec = get_spec(name)
    return describe(load_dataset(spec), spec)
