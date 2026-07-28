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


class DevEmbedder:
    requires_dataset_fit = True  # PCA basis is fit on train cells only
    version = "0.0.1"

    def __init__(self, method: Literal["pca", "random"] = "pca", dim: int = 50, seed: int = 0):
        self.name = f"dev_{method}"
        self.method = method
        self.dim = dim
        self.seed = seed
        self._basis: np.ndarray | None = None

    def preprocess(self, adata: AnnData) -> AnnData:
        """Generic log-normalization + HVGs. Real embedders must not reuse this."""
        raise NotImplementedError("TODO(phase-2): delegate to data.loaders.preprocess_for_probe")

    def fit(self, adata: AnnData) -> None:
        """Fit the PCA basis on train cells only — fitting on all cells leaks."""
        raise NotImplementedError("TODO(phase-2): fit PCA/random projection basis")

    def embed(self, adata: AnnData) -> np.ndarray:
        """Project onto the fitted basis; rows stay aligned to `adata.obs_names`."""
        raise NotImplementedError("TODO(phase-2): project onto self._basis")
