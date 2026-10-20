"""Gene identifier helpers shared by the embedders that tokenize on Ensembl IDs.

Geneformer, TranscriptFormer, scFoundation, and LangCell all need `var` keyed by Ensembl
gene ID, and all four fail in the same quiet way when the mapping is wrong: genes drop out of
the model's vocabulary, the cell is tokenized from whatever survived, and the embedding looks
plausible while meaning nothing (plan section 10.10). One implementation, one place to fix.

Lives outside any model module because it must be importable from every isolated env, so it
depends on nothing heavier than numpy plus an optional `mygene` lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

# Below this many mapped genes the tokenized cells are too sparse to mean anything.
MIN_MAPPED_GENES = 100


def resolve_ensembl_ids(adata: AnnData, mapping_pkl: Path | None = None) -> np.ndarray:
    """Resolve Ensembl gene IDs from `var` metadata, falling back to symbol lookup."""
    for col in ("ensembl_id", "gene_ids", "ensembl"):
        if col in adata.var.columns:
            vals = adata.var[col].astype(str)
            if vals.str.startswith("ENSG").fillna(False).sum() >= MIN_MAPPED_GENES:
                return vals.to_numpy(dtype=object)

    return symbols_to_ensembl(
        adata.var_names.astype(str).tolist(),
        mapping_pkl=mapping_pkl,
    )


def symbols_to_ensembl(
    symbols: list[str],
    mapping_pkl: Path | None = None,
) -> np.ndarray:
    """Map gene symbols -> Ensembl IDs using a bundled pickle dict, else mygene.

    `mapping_pkl` is an offline dict (Geneformer ships one); mygene is the online fallback for
    whatever it misses. Unmapped symbols come back as NaN for the caller to drop.
    """
    by_symbol: dict[str, str] = {}
    if mapping_pkl is not None and mapping_pkl.exists():
        import pickle

        with open(mapping_pkl, "rb") as fh:
            raw = pickle.load(fh)
        if isinstance(raw, dict):
            for key, val in raw.items():
                if val is None:
                    continue
                k = str(key).upper()
                v = str(val)
                if v.startswith("ENSG"):
                    by_symbol[k] = v
                elif k.startswith("ENSG"):
                    by_symbol[k] = k

    missing = [s for s in symbols if s.upper() not in by_symbol]
    if missing:
        try:
            import mygene
        except ImportError:
            mygene = None  # type: ignore[assignment]
        if mygene is not None:
            mg = mygene.MyGeneInfo()
            hits = mg.querymany(
                missing,
                scopes="symbol,alias",
                fields="ensembl.gene",
                species="human",
                as_dataframe=False,
                returnall=False,
                verbose=False,
            )
            for h in hits:
                q = str(h.get("query", "")).upper()
                if h.get("notfound"):
                    continue
                ens = h.get("ensembl")
                if ens is None:
                    continue
                if isinstance(ens, list):
                    gene = ens[0].get("gene") if isinstance(ens[0], dict) else None
                elif isinstance(ens, dict):
                    gene = ens.get("gene")
                else:
                    gene = None
                if gene:
                    by_symbol[q] = str(gene)

    return np.array([by_symbol.get(s.upper(), np.nan) for s in symbols], dtype=object)


def add_ensembl_var(
    adata: AnnData,
    mapping_pkl: Path | None = None,
    col: str = "ensembl_id",
    model_name: str = "model",
) -> AnnData:
    """Return `adata` restricted to genes with a resolved Ensembl ID in `var[col]`.

    Raises rather than returning a near-empty matrix, because a silently sparse gene set is the
    failure mode that produces meaningless-but-plausible embeddings.
    """
    out = adata
    if col not in out.var.columns:
        out = out.copy()
        out.var[col] = resolve_ensembl_ids(out, mapping_pkl=mapping_pkl)

    mapped = out.var[col].notna() & (out.var[col].astype(str) != "nan")
    n_mapped = int(mapped.sum())
    if n_mapped < MIN_MAPPED_GENES:
        raise RuntimeError(
            f"{model_name}: only {n_mapped} of {out.n_vars} genes mapped to Ensembl IDs; "
            "check gene symbols / mygene connectivity."
        )
    out = out[:, mapped].copy()
    out.var[col] = out.var[col].astype(str)
    return out


def write_h5ad_compat(adata: AnnData, path: Path) -> None:
    """Write h5ad without nullable pandas StringDtype, which older readers choke on."""
    import pandas as pd

    out = adata.copy()
    out.obs_names = out.obs_names.astype(str)
    out.var_names = out.var_names.astype(str)
    for frame in (out.obs, out.var):
        for col in frame.columns:
            series = frame[col]
            if isinstance(series.dtype, pd.StringDtype) or str(series.dtype) == "string":
                frame[col] = series.astype(str)
    out.write_h5ad(path)
