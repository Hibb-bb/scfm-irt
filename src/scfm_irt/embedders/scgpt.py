"""scGPT — zero-shot cell foundation model (bowang-lab).

Runs inside `envs/env-scgpt.yml` only. Uses the official
`scgpt.tasks.embed_data` entrypoint, which writes `obsm["X_scGPT"]`.

Checkpoint: whole-human (recommended default). Download with
`scripts/download_scgpt_assets.py` (Google Drive via gdown).

See https://github.com/bowang-lab/scGPT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


class SCGPTEmbedder:
    name = "scgpt"
    version = "whole-human"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-scgpt.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 1200,
        gene_col: str = "gene_name",
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.gene_col = gene_col
        self.work_dir = Path(
            work_dir
            or os.environ.get("SCFM_SCGPT_WORKDIR", "cache/scgpt_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """Keep counts; ensure a gene-name column for scGPT's vocab matching."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]
        # scGPT matches gene *symbols* against its vocab via adata.var[gene_col].
        if self.gene_col not in out.var.columns:
            out.var[self.gene_col] = out.var_names.astype(str)
        return out

    def fit(self, adata: AnnData) -> None:
        return None

    def _model_dir(self) -> Path:
        if self.checkpoint:
            path = Path(self.checkpoint).resolve()
        else:
            path = self.work_dir / "scGPT_human"
        required = ["args.json", "best_model.pt", "vocab.json"]
        missing = [f for f in required if not (path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"scGPT checkpoint incomplete at {path}; missing {missing}. "
                f"Run: python scripts/download_scgpt_assets.py --out {path}"
            )
        return path

    def embed(self, adata: AnnData) -> np.ndarray:
        """Return (n_cells, d) cell embeddings via scgpt.tasks.embed_data."""
        # Heavy imports stay inside the method (isolated env only).
        import torch
        from scgpt.tasks import embed_data

        model_dir = self._model_dir()
        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        # flash-attn is fragile on clusters; disable unless explicitly available.
        use_fast = False
        try:
            import flash_attn  # noqa: F401

            use_fast = True
        except ImportError:
            pass

        to_embed = self.preprocess(adata)
        # embed_data mutates / returns adata with X_scGPT; keep row order.
        result = embed_data(
            to_embed,
            model_dir=str(model_dir),
            gene_col=self.gene_col,
            max_length=self.max_length,
            batch_size=self.batch_size,
            device=device,
            use_fast_transformer=use_fast,
            return_new_adata=False,
        )
        if "X_scGPT" not in result.obsm:
            raise KeyError(
                f"scGPT did not write obsm['X_scGPT']; keys={list(result.obsm)}"
            )
        emb = np.asarray(result.obsm["X_scGPT"], dtype=np.float32)
        if emb.shape[0] != adata.n_obs:
            raise ValueError(
                f"scGPT returned {emb.shape[0]} rows for {adata.n_obs} cells"
            )
        # Align to input obs order if anything reordered.
        if list(map(str, result.obs_names)) != list(map(str, adata.obs_names)):
            idx = {str(n): i for i, n in enumerate(result.obs_names)}
            aligned = np.empty((adata.n_obs, emb.shape[1]), dtype=np.float32)
            for i, name in enumerate(map(str, adata.obs_names)):
                if name not in idx:
                    raise RuntimeError(f"scGPT dropped cell {name!r}")
                aligned[i] = emb[idx[name]]
            emb = aligned
        return emb
