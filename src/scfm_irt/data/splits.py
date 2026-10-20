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
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    obs_names = np.asarray(adata.obs_names)

    if scheme == "stratified":
        train_idx, test_idx = _stratified_indices(
            labels=np.asarray(adata.obs[spec.label_key].astype(str)),
            test_frac=test_frac,
            seed=seed,
        )
        held_out = None
    elif scheme in ("cross_donor", "cross_batch"):
        group_key = spec.donor_key if scheme == "cross_donor" else spec.batch_key
        role = "donor_key" if scheme == "cross_donor" else "batch_key"
        if group_key is None:
            raise ValueError(
                f"{spec.name}: {scheme} requires {role} on the DatasetSpec; "
                "refusing to fall back to a stratified split"
            )
        train_idx, test_idx, held_out = _group_holdout_indices(
            groups=np.asarray(adata.obs[group_key].astype(str)),
            test_frac=test_frac,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")

    return Split(
        dataset=spec.name,
        scheme=scheme,
        seed=seed,
        train_idx=train_idx,
        test_idx=test_idx,
        train_ids=obs_names[train_idx],
        test_ids=obs_names[test_idx],
        held_out_groups=held_out,
    )


def _stratified_indices(
    labels: np.ndarray,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) > 1 else 0
        # Keep at least one train cell when the class has ≥2 cells.
        if len(idx) > 1 and n_test >= len(idx):
            n_test = len(idx) - 1
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    if len(test_idx) == 0 or len(train_idx) == 0:
        raise ValueError("stratified split produced an empty train or test set")
    return train_idx.astype(int), test_idx.astype(int)


def _group_holdout_indices(
    groups: np.ndarray,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(unique))
    unique = unique[order]
    counts = counts[order]

    n = len(groups)
    target = test_frac * n
    held: list[str] = []
    n_held = 0
    for g, c in zip(unique, counts, strict=True):
        # Always leave at least one group for train when possible.
        if len(held) and (n_held >= target or n_held + c == n):
            break
        if n_held + c == n and not held:
            # Single group — cannot do a true group holdout.
            raise ValueError("cannot build a group-holdout split with only one group")
        held.append(str(g))
        n_held += int(c)
        if n_held >= target:
            break

    if not held or n_held == n:
        raise ValueError("group holdout left train or test empty")

    held_set = set(held)
    test_idx = np.flatnonzero(np.isin(groups, list(held_set)))
    train_idx = np.flatnonzero(~np.isin(groups, list(held_set)))
    return train_idx.astype(int), test_idx.astype(int), held


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Per-class counts, logged for every split (imbalance affects probe interpretation)."""
    values, counts = np.unique(labels, return_counts=True)
    return dict(zip(map(str, values), map(int, counts), strict=True))
