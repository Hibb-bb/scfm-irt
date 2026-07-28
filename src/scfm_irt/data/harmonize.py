"""Cell-type label harmonization across datasets.

Datasets use disparate cell-type vocabularies ("CD8 T cell" / "CD8+ T" / "T cell CD8"). An
inconsistent mapping poisons both the probe labels and the difficulty aggregation, so all of it
is centralized here and every applied mapping is logged (plan section 10.6).

Target vocabulary: Cell Ontology (CL) terms.
"""

from __future__ import annotations

import pandas as pd

# TODO(phase-1): populate from the datasets actually used; keep one entry per source label so
# the mapping is auditable rather than inferred by fuzzy matching at runtime.
_LABEL_MAP: dict[str, dict[str, str]] = {
    # "zheng68k": {"CD8+ Cytotoxic T": "CL:0000794", ...},
}


def harmonize_labels(labels: pd.Series, dataset: str, strict: bool = True) -> pd.Series:
    """Map dataset-specific labels onto the common ontology.

    With `strict`, an unmapped label is an error — silently passing it through means the same
    biological type appears as two items in the difficulty aggregation.
    """
    raise NotImplementedError("TODO(phase-1): apply _LABEL_MAP with strict unmapped handling")


def mapping_report(dataset: str) -> pd.DataFrame:
    """The applied mapping as a table, written into every run directory for auditability."""
    raise NotImplementedError("TODO(phase-1)")
