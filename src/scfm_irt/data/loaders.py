"""Loading datasets to AnnData.

`anndata`/`scanpy` are imported lazily so that the IRT and analysis modules stay importable in
an environment without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scfm_irt.data.h5ad_compat import harmonize_var_names, read_h5ad_compat
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
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(
            f"{spec.name}: dataset not found at {path}. "
            "Run `python scripts/make_smoke_data.py` for the toy set, "
            "or `python scripts/download_data.py --dataset ...` for real data."
        )

    adata = read_h5ad_compat(path)
    adata = harmonize_var_names(adata)

    # Suspected transposition: far more obs than vars, with vars looking like a cell count
    # rather than a gene panel. (2k cells × 30k genes is the normal orientation.)
    if adata.n_obs > 5 * max(adata.n_vars, 1) and adata.n_vars < 2_000:
        raise ValueError(
            f"{spec.name}: matrix looks gene-oriented ({adata.n_obs} x {adata.n_vars}); "
            "expected cells x genes. Transpose or fix the download."
        )

    if not adata.obs_names.is_unique:
        raise ValueError(f"{spec.name}: obs_names are not unique")

    for key, role in (
        (spec.label_key, "label_key"),
        (spec.batch_key, "batch_key"),
        (spec.donor_key, "donor_key"),
    ):
        if key is not None and key not in adata.obs.columns:
            raise KeyError(f"{spec.name}: {role}={key!r} not in .obs columns {list(adata.obs)}")

    if spec.expected_n_cells is not None and adata.n_obs != spec.expected_n_cells:
        raise ValueError(
            f"{spec.name}: expected {spec.expected_n_cells} cells, found {adata.n_obs}"
        )
    if spec.expected_n_types is not None:
        n_types = int(adata.obs[spec.label_key].nunique())
        if n_types != spec.expected_n_types:
            raise ValueError(
                f"{spec.name}: expected {spec.expected_n_types} cell types, found {n_types}"
            )

    return adata


def preprocess_for_probe(adata: AnnData, n_top_genes: int = 2000) -> AnnData:
    """Generic log-normalization + HVG selection, for the DevEmbedder path only.

    Real embedders must use their own official preprocessing instead (plan section 10.10);
    substituting this one produces plausible-looking but meaningless embeddings.
    """
    import scanpy as sc

    out = adata.copy()
    # Counts may already be normalized; only normalize when totals look like raw counts.
    totals = out.X.sum(axis=1)
    if hasattr(totals, "A1"):
        totals = totals.A1
    median_total = float(totals.mean()) if len(totals) else 0.0
    if median_total > 50:  # heuristic: raw UMI/count matrices sit well above this
        sc.pp.normalize_total(out, target_sum=1e4)
        sc.pp.log1p(out)

    n_genes = out.n_vars
    if n_genes > n_top_genes:
        sc.pp.highly_variable_genes(out, n_top_genes=n_top_genes, subset=True, flavor="seurat")
    return out
