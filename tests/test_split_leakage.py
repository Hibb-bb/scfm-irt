"""Splits must not leak.

Two distinct failure modes, both silent and both inflating every ability estimate:

1. A cell appears in train and test.
2. For `cross_donor`/`cross_batch`, a *donor* spans both sides — the cells differ but the
   biological source does not, so the split is not measuring generalization.

Plus the third, one layer up: an embedder with `requires_dataset_fit=True` fitted on all cells
rather than train cells only (plan section 10.8).
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_irt.data.splits import Split, class_distribution

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_split_rejects_overlapping_cells():
    """Split refuses to exist in a leaked state, so nothing downstream can inherit one."""
    with pytest.raises(ValueError, match="both train and test"):
        Split(
            dataset="toy",
            scheme="stratified",
            seed=0,
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([2, 3]),
            train_ids=np.array(["c0", "c1", "c2"]),
            test_ids=np.array(["c2", "c3"]),
        )


def test_clean_split_is_accepted():
    split = Split(
        dataset="toy",
        scheme="stratified",
        seed=0,
        train_idx=np.array([0, 1]),
        test_idx=np.array([2, 3]),
        train_ids=np.array(["c0", "c1"]),
        test_ids=np.array(["c2", "c3"]),
    )
    assert len(split.train_ids) == 2
    assert not set(split.train_ids) & set(split.test_ids)


def test_class_distribution_counts_labels():
    labels = np.array(["A", "A", "B", "C", "C", "C"])
    assert class_distribution(labels) == {"A": 2, "B": 1, "C": 3}


@pytest.mark.xfail(reason="TODO(phase-1): make_split not implemented", raises=NotImplementedError)
def test_stratified_split_preserves_class_proportions():
    raise NotImplementedError


@pytest.mark.xfail(reason="TODO(phase-1): make_split not implemented", raises=NotImplementedError)
def test_cross_donor_split_holds_out_whole_donors():
    """No donor may appear on both sides — the point of the scheme."""
    raise NotImplementedError


@pytest.mark.xfail(reason="TODO(phase-1): make_split not implemented", raises=NotImplementedError)
def test_cross_donor_raises_when_dataset_has_no_donor_key():
    """Must fail loudly rather than quietly degrading to a stratified split."""
    raise NotImplementedError


@pytest.mark.xfail(reason="TODO(phase-2): DevEmbedder not implemented", raises=NotImplementedError)
def test_dataset_fit_embedder_never_sees_test_cells():
    raise NotImplementedError
