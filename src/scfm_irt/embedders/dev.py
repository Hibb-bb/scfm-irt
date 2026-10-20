"""Dependency-free embedder for bringing the pipeline up.

PCA or a random projection of log-normalized counts. Its purpose is to exercise the *entire*
downstream path — probe, response table, IRT, reporting — before anyone wrestles with real
checkpoints (Phase 2). It is a smoke-test fixture, not an eval competitor; whether a
raw-feature embedder belongs in the final evaluation is a separate question (plan section 12).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


class DevEmbedder:
    requires_dataset_fit = True  # PCA basis is fit on train cells only
    version = "0.0.1"

    def __init__(self, method: Literal["pca", "random"] = "pca", dim: int = 50, seed: int = 0):
        self.name = f"dev_{method}"
        self.method = method
        self.dim = dim
        self.seed = seed
        self._basis: np.ndarray | None = None  # (n_genes, dim)
        self._mean: np.ndarray | None = None

    def preprocess(self, adata: AnnData) -> AnnData:
        """Generic log-normalization + HVGs. Real embedders must not reuse this."""
        from scfm_irt.data.loaders import preprocess_for_probe

        return preprocess_for_probe(adata)

    def fit(self, adata: AnnData) -> None:
        """Fit the PCA basis on train cells only — fitting on all cells leaks."""
        X = _to_dense(adata.X)
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        k = min(self.dim, X.shape[0] - 1, X.shape[1])
        if k < 1:
            raise ValueError(f"cannot fit DevEmbedder: need more cells/genes, got {X.shape}")

        if self.method == "pca":
            # Economy SVD; right singular vectors are the gene loadings.
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            self._basis = vt[:k].T  # (n_genes, k)
        elif self.method == "random":
            rng = np.random.default_rng(self.seed)
            basis = rng.normal(size=(X.shape[1], k))
            basis, _ = np.linalg.qr(basis)
            self._basis = basis
        else:
            raise ValueError(f"unknown DevEmbedder method {self.method!r}")

        self.dim = k

    def embed(self, adata: AnnData) -> np.ndarray:
        """Project onto the fitted basis; rows stay aligned to `adata.obs_names`."""
        if self._basis is None or self._mean is None:
            raise RuntimeError("DevEmbedder.fit must be called before embed")
        X = _to_dense(adata.X)
        if X.shape[1] != self._basis.shape[0]:
            raise ValueError(
                f"gene dim mismatch: data has {X.shape[1]} genes, basis expects "
                f"{self._basis.shape[0]} (preprocess train and test with the same genes)"
            )
        return (X - self._mean) @ self._basis
