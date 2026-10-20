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

from scfm_irt.data.splits import Split, class_distribution, make_split

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


def _toy_adata(n_per_type: int = 40, n_donors: int = 6, seed: int = 0):
    ad = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.default_rng(seed)
    types = np.repeat(["A", "B", "C"], n_per_type)
    n = len(types)
    donors = rng.choice([f"d{i}" for i in range(n_donors)], size=n)
    X = rng.poisson(2.0, size=(n, 30)).astype(np.float32)
    obs = pd.DataFrame(
        {"celltype": types, "donor": donors},
        index=[f"c{i}" for i in range(n)],
    )
    return ad.AnnData(X=X, obs=obs)


def test_stratified_split_preserves_class_proportions():
    from scfm_irt.data.registry import DatasetSpec

    adata = _toy_adata()
    spec = DatasetSpec(name="toy", path="unused", label_key="celltype", donor_key="donor")
    split = make_split(adata, spec, scheme="stratified", seed=0, test_frac=0.25)

    assert not set(split.train_ids) & set(split.test_ids)
    train_labels = adata.obs.loc[split.train_ids, "celltype"]
    test_labels = adata.obs.loc[split.test_ids, "celltype"]
    # Every class appears on both sides.
    assert set(train_labels) == set(test_labels) == {"A", "B", "C"}
    # Test fraction roughly respected.
    assert 0.15 <= len(split.test_idx) / adata.n_obs <= 0.35


def test_cross_donor_split_holds_out_whole_donors():
    """No donor may appear on both sides — the point of the scheme."""
    from scfm_irt.data.registry import DatasetSpec

    adata = _toy_adata()
    spec = DatasetSpec(name="toy", path="unused", label_key="celltype", donor_key="donor")
    split = make_split(adata, spec, scheme="cross_donor", seed=0, test_frac=0.3)

    train_donors = set(adata.obs.loc[split.train_ids, "donor"])
    test_donors = set(adata.obs.loc[split.test_ids, "donor"])
    assert train_donors.isdisjoint(test_donors)
    assert split.held_out_groups is not None
    assert set(split.held_out_groups) == test_donors


def test_cross_donor_raises_when_dataset_has_no_donor_key():
    """Must fail loudly rather than quietly degrading to a stratified split."""
    from scfm_irt.data.registry import DatasetSpec

    adata = _toy_adata()
    spec = DatasetSpec(name="toy", path="unused", label_key="celltype", donor_key=None)
    with pytest.raises(ValueError, match="donor_key"):
        make_split(adata, spec, scheme="cross_donor", seed=0)


def test_dataset_fit_embedder_never_sees_test_cells():
    from scfm_irt.data.registry import DatasetSpec
    from scfm_irt.embedders.dev import DevEmbedder

    adata = _toy_adata(n_per_type=50)
    spec = DatasetSpec(name="toy", path="unused", label_key="celltype", donor_key="donor")
    split = make_split(adata, spec, scheme="stratified", seed=0, test_frac=0.2)

    emb = DevEmbedder(method="pca", dim=10, seed=0)
    train = emb.preprocess(adata[split.train_idx].copy())
    genes = list(train.var_names)
    full = emb.preprocess(adata[:, genes].copy())
    full = full[:, genes].copy()

    # Spy: fit must only see train obs names.
    seen = {}

    def _fit(self, ad):
        seen["ids"] = set(map(str, ad.obs_names))
        DevEmbedder.fit(self, ad)

    emb.fit = _fit.__get__(emb, DevEmbedder)  # type: ignore[method-assign]
    emb.fit(full[split.train_idx].copy())
    assert seen["ids"].isdisjoint(set(map(str, split.test_ids)))
    assert seen["ids"] <= set(map(str, split.train_ids))
