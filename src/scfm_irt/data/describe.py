"""Dataset inventory / sanity check.

`describe` reproducing the known inventory table (dims, #cell types, #donors) for a dataset is
the Phase 0 acceptance criterion, and it is also the cheapest way to catch a wrong download or
a transposed matrix before any GPU time is spent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scfm_irt.data.registry import DatasetSpec, get_spec

if TYPE_CHECKING:
    from anndata import AnnData


def describe(adata: AnnData, spec: DatasetSpec) -> dict[str, Any]:
    """Summarize a loaded dataset.

    Returns n_cells, n_genes, n_cell_types, n_donors, n_batches, the rarest and most common
    class with counts, the imbalance ratio, and median total counts per cell. Warns when
    `expected_n_cells`/`expected_n_types` are set on the spec and disagree.
    """
    labels = adata.obs[spec.label_key].astype(str)
    counts = labels.value_counts()
    most_common = str(counts.index[0])
    rarest = str(counts.index[-1])
    imbalance = float(counts.iloc[0] / max(counts.iloc[-1], 1))

    totals = adata.X.sum(axis=1)
    if hasattr(totals, "A1"):
        totals = totals.A1
    totals = np.asarray(totals).ravel()

    out: dict[str, Any] = {
        "name": spec.name,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_cell_types": int(labels.nunique()),
        "n_donors": int(adata.obs[spec.donor_key].nunique()) if spec.donor_key else None,
        "n_batches": int(adata.obs[spec.batch_key].nunique()) if spec.batch_key else None,
        "most_common_type": most_common,
        "most_common_count": int(counts.iloc[0]),
        "rarest_type": rarest,
        "rarest_count": int(counts.iloc[-1]),
        "imbalance_ratio": imbalance,
        "median_total_counts": float(np.median(totals)),
        "label_key": spec.label_key,
        "family": spec.family,
    }

    warnings: list[str] = []
    if spec.expected_n_cells is not None and out["n_cells"] != spec.expected_n_cells:
        warnings.append(f"expected_n_cells={spec.expected_n_cells} but found {out['n_cells']}")
    if spec.expected_n_types is not None and out["n_cell_types"] != spec.expected_n_types:
        warnings.append(f"expected_n_types={spec.expected_n_types} but found {out['n_cell_types']}")
    if warnings:
        out["warnings"] = warnings

    return out


def describe_by_name(name: str) -> dict[str, Any]:
    """Convenience wrapper used by `scfm-irt describe <dataset>`."""
    from scfm_irt.data.loaders import load_dataset

    spec = get_spec(name)
    return describe(load_dataset(spec), spec)
