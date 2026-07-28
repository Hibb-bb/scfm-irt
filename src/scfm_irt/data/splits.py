"""Train/test splits.

Scheme choice is not cosmetic: it is the main lever on item difficulty spread.

- `stratified` — random, stratified by cell type. In-distribution, so common types land near
  the ceiling and separate models poorly.
- `cross_donor` / `cross_batch` — hold out whole donors or batches. Harder, and the source of
  the difficulty range IRT needs (MATH.md section 8.3).

Cross-dataset transfer (train on A, test on harmonized B) is a v2 extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from scfm_irt.data.registry import DatasetSpec

if TYPE_CHECKING:
    from anndata import AnnData

SplitScheme = Literal["stratified", "cross_donor", "cross_batch"]


@dataclass(frozen=True)
class Split:
    """Train/test cell indices plus enough provenance to reproduce them."""

    dataset: str
    scheme: SplitScheme
    seed: int
    train_idx: np.ndarray  # positional indices into adata
    test_idx: np.ndarray
    train_ids: np.ndarray  # obs_names, so the split survives caching and reordering
    test_ids: np.ndarray
    held_out_groups: list[str] | None = None  # donors/batches held out, for cross_* schemes

    def __post_init__(self) -> None:
        overlap = set(self.train_ids) & set(self.test_ids)
        if overlap:
            raise ValueError(f"{len(overlap)} cell ids appear in both train and test")


def make_split(
    adata: AnnData,
    spec: DatasetSpec,
    scheme: SplitScheme = "stratified",
    seed: int = 0,
    test_frac: float = 0.2,
) -> Split:
    """Build a leakage-free split.

    For `cross_donor`/`cross_batch`, whole groups go to one side — hold out donors until the
    test fraction is approximately reached, and raise if the spec has no donor/batch key rather
    than silently falling back to a stratified split.
    """
    raise NotImplementedError("TODO(phase-1): implement stratified and group-holdout splits")


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Per-class counts, logged for every split (imbalance affects probe interpretation)."""
    values, counts = np.unique(labels, return_counts=True)
    return dict(zip(map(str, values), map(int, counts), strict=True))
