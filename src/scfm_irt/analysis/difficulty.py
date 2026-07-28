"""Item difficulty and discrimination summaries.

Difficulty is estimated *relative to the respondent pool*. "A hard item" here means "hard for
these embedding models" — not hard in any absolute sense, and not evidence that the cell
requires more than raw gene features. Every table this module emits should be read that way,
and the report says so (MATH.md section 1.1).
"""

from __future__ import annotations

import pandas as pd

from scfm_irt.irt.models import IRTFit


def difficulty_by_group(
    fit: IRTFit,
    resp: pd.DataFrame,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-item difficulty by cell type and/or dataset.

    Per-cell difficulties are noisy (~P responses each); the aggregate is the interpretable
    unit. Report the group mean, its spread, and the group size.
    """
    raise NotImplementedError("TODO(phase-8): join item params onto covariates, group, summarize")


def hardest_items(fit: IRTFit, resp: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """The n highest-difficulty items with their cell type and dataset."""
    raise NotImplementedError("TODO(phase-8)")


def difficulty_histogram(fit: IRTFit, path: str) -> None:
    """Plot the difficulty distribution — a first-class diagnostic, not decoration.

    A distribution piled up at the bottom of the scale means the eval is at the ceiling: every
    model gets nearly everything right, items carry almost no information, and abilities cannot
    separate. Seeing that here is the signal to add harder items — rare types, fine-grained
    subtypes, cross-donor and cross-dataset cells (MATH.md section 8.3).
    """
    raise NotImplementedError("TODO(phase-8)")


def discrimination_extremes(fit: IRTFit, resp: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Highest- and lowest-discrimination items (2PL only).

    High-`a` cells are the ones that best separate strong from weak models — the most
    informative evaluation items, which formalizes "find cells that spread models apart"
    (MATH.md section 8.2). Interpret cautiously: per-item `a` is the noisiest thing estimated
    here.
    """
    raise NotImplementedError("TODO(phase-8)")
