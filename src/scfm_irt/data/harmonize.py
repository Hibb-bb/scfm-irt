"""Cell-type label harmonization across datasets.

Datasets use disparate cell-type vocabularies ("CD8 T cell" / "CD8+ T" / "T cell CD8"). An
inconsistent mapping poisons both the probe labels and the difficulty aggregation, so all of it
is centralized here and every applied mapping is logged (plan section 10.6).

Target vocabulary: Cell Ontology (CL) terms.

For the smoke / per-dataset-fit path, harmonization is a no-op identity map: v1 can fit each
dataset separately with its own labels and pool only at the response-matrix level (roadmap
open question). Cross-dataset difficulty comparisons then require a populated `_LABEL_MAP`.
"""

from __future__ import annotations

import pandas as pd

# Populate from the datasets actually used; keep one entry per source label so the mapping is
# auditable rather than inferred by fuzzy matching at runtime.
_LABEL_MAP: dict[str, dict[str, str]] = {
    # "zheng68k": {"CD8+ Cytotoxic T": "CL:0000794", ...},
}


def harmonize_labels(labels: pd.Series, dataset: str, strict: bool = True) -> pd.Series:
    """Map dataset-specific labels onto the common ontology.

    With `strict`, an unmapped label is an error — silently passing it through means the same
    biological type appears as two items in the difficulty aggregation.

    If no map is registered for `dataset`, returns labels unchanged (identity). That is the
    intentional v1 default for per-dataset probes.
    """
    mapping = _LABEL_MAP.get(dataset)
    if mapping is None:
        return labels.astype(str)

    out = labels.astype(str).map(mapping)
    unmapped = labels.astype(str)[out.isna()].unique().tolist()
    if unmapped and strict:
        raise KeyError(f"{dataset}: unmapped labels: {unmapped}")
    if unmapped:
        out = out.fillna(labels.astype(str))
    return out


def mapping_report(dataset: str) -> pd.DataFrame:
    """The applied mapping as a table, written into every run directory for auditability."""
    mapping = _LABEL_MAP.get(dataset, {})
    return pd.DataFrame(
        {"source_label": list(mapping.keys()), "target_label": list(mapping.values())}
    )
