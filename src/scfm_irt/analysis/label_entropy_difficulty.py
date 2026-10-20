"""Label-entropy item difficulty (model-independent).

For each test cell, measure Shannon entropy of cell-type labels among k nearest
train neighbors in expression (HVG) space. High entropy ⇒ locally ambiguous label
context ⇒ harder annotation item.

This difficulty bank does **not** use embedding-model probe outcomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scfm_irt.response.build import make_cell_id

if TYPE_CHECKING:
    from anndata import AnnData

    from scfm_irt.data.splits import Split


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _shannon_entropy(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def knn_label_entropy_for_split(
    adata: AnnData,
    split: Split,
    spec_label_key: str,
    dataset_name: str,
    *,
    k: int = 30,
    n_top_genes: int = 1000,
    n_pcs: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute kNN label entropy for each test cell in `split`. See module docstring."""
    train = adata[split.train_idx].copy()
    test = adata[split.test_idx].copy()
    return knn_label_entropy_train_test(
        train,
        test,
        spec_label_key,
        dataset_name,
        k=k,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        seed=seed,
    )


def knn_label_entropy_train_test(
    train: AnnData,
    test: AnnData,
    label_key: str,
    dataset_name: str,
    *,
    k: int = 30,
    n_top_genes: int = 1000,
    n_pcs: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    train = train.copy()
    test = test.copy()
    for obj in (train, test):
        totals = obj.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        if float(np.asarray(totals).mean()) > 50:
            sc.pp.normalize_total(obj, target_sum=1e4)
            sc.pp.log1p(obj)
    sc.pp.highly_variable_genes(train, n_top_genes=min(n_top_genes, train.n_vars - 1), subset=True)
    genes = train.var_names
    test = test[:, genes].copy()

    X_train = _to_dense(train.X)
    X_test = _to_dense(test.X)
    n_comp = min(n_pcs, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)

    labels_train = train.obs[label_key].astype(str).to_numpy()
    labels_test = test.obs[label_key].astype(str).to_numpy()
    test_names = test.obs_names.astype(str).to_numpy()

    k_eff = min(k, len(labels_train))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="brute", n_jobs=4)
    nn.fit(X_train_p)
    nbr_idx = nn.kneighbors(X_test_p, return_distance=False)

    entropies = np.zeros(len(test_names), dtype=float)
    for i, neighbors in enumerate(nbr_idx):
        neigh_labels = labels_train[neighbors]
        _, counts = np.unique(neigh_labels, return_counts=True)
        entropies[i] = _shannon_entropy(counts)

    b = entropies - float(np.mean(entropies))
    cell_ids = [make_cell_id(dataset_name, name) for name in test_names]
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": labels_test,
            "label_entropy": entropies,
            "difficulty_entropy": b,
        }
    )


def align_difficulty_to_items(item_ids: list[str], table: pd.DataFrame) -> np.ndarray:
    """Map a cell_id → difficulty table onto the IRT item order."""
    lookup = table.set_index("cell_id")["difficulty_entropy"].astype(float)
    missing = [i for i in item_ids if i not in lookup.index]
    if missing:
        raise KeyError(f"missing label-entropy difficulty for {len(missing)} items; e.g. {missing[:3]}")
    return lookup.loc[item_ids].to_numpy(dtype=float)
