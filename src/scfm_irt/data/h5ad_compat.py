"""Read/write h5ad across embedder venvs (mixed anndata versions)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def sanitize_adata(adata: AnnData, *, strip_uns: bool = True) -> AnnData:
    """Drop metadata that breaks older anndata readers; normalize string dtypes."""
    import pandas as pd

    out = adata.copy()
    out.obs_names = out.obs_names.astype(str)
    out.var_names = out.var_names.astype(str)

    for frame in (out.obs, out.var):
        for col in frame.columns:
            series = frame[col]
            if isinstance(series.dtype, pd.StringDtype) or str(series.dtype) == "string":
                frame[col] = series.astype(str)

    # scanpy >=1.10 writes nullable IOSpec fields (log1p, pca params, …) that anndata 0.10.x
    # cannot read. Embedders only need expression + obs/var + counts layer.
    if strip_uns:
        out.uns = {}
    out.obsm = {}

    return out


def harmonize_var_names(adata: AnnData) -> AnnData:
    """Map CELLxGENE / census var index to gene symbols embedders expect."""
    import re

    out = adata.copy()
    names = out.var_names.astype(str)
    numeric_index = len(names) > 0 and all(re.fullmatch(r"\d+", n) for n in names[: min(50, len(names))])

    if "feature_name" in out.var.columns:
        symbols = out.var["feature_name"].astype(str)
        if "feature_id" in out.var.columns:
            fid = out.var["feature_id"].astype(str)
            symbols = symbols.where(~symbols.str.startswith("ENSG"), fid)
        out.var_names = symbols
        out.var["gene_name"] = out.var_names.astype(str)
        if "feature_id" in out.var.columns:
            out.var["ensembl_id"] = out.var["feature_id"].astype(str)
        out.var_names_make_unique()
        out.var.index.name = None
        return out

    if numeric_index:
        raise ValueError(
            "var_names look like numeric census indices but no feature_name column; "
            "cannot map genes for foundation models"
        )

    if "gene_symbols" in out.var.columns and not out.var_names.str.match(r"^[A-Z0-9]", na=False).mean() > 0.5:
        out.var_names = out.var["gene_symbols"].astype(str)
        out.var_names_make_unique()

    if "gene_name" not in out.var.columns:
        out.var["gene_name"] = out.var_names.astype(str)
    return out


def write_h5ad_compat(adata: AnnData, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_adata(adata).write_h5ad(path)


def read_h5ad_compat(path: Path | str) -> AnnData:
    """Load h5ad; on legacy IOSpec errors, rebuild a minimal AnnData without uns/obsm."""
    import anndata as ad

    path = Path(path)
    try:
        return ad.read_h5ad(path)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        if "IORegistryError" not in err and "null" not in err:
            raise

    # Last resort: load matrix + obs/var/layers only (skip uns/obsm with bad encodings).
    import h5py

    with h5py.File(path, "r") as _:
        backed = ad.read_h5ad(path, backed="r")
        try:
            out = ad.AnnData(
                X=backed.X[:],
                obs=backed.obs.copy(),
                var=backed.var.copy(),
            )
            for key in backed.layers.keys():
                layer = backed.layers[key]
                out.layers[key] = layer[:] if hasattr(layer, "__getitem__") else layer
            return sanitize_adata(out)
        finally:
            backed.file.close()
