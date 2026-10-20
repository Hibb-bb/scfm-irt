"""CellPLM — cell-centric foundation model (ICLR 2024).

Runs in an isolated Python 3.9+GPU env. Zero-shot embedding via
`CellPLM.pipeline.cell_embedding.CellEmbeddingPipeline`.

Checkpoint: download with `scripts/download_cellplm_assets.py`
(pretrain_prefix `20230926_85M` from the official Dropbox).

See https://github.com/OmicsML/CellPLM
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


class CellPLMEmbedder:
    name = "cellplm"
    version = "20230926_85M"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-cellplm.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        pretrain_prefix: str = "20230926_85M",
        device: str = "cuda",
        batch_size: int = 128,
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.pretrain_prefix = pretrain_prefix
        self.device = device
        self.batch_size = batch_size
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_CELLPLM_WORKDIR", "cache/cellplm_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """Keep raw counts; CellPLM handles filtering/tokenization internally."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]
        if "gene_name" not in out.var.columns:
            out.var["gene_name"] = out.var_names.astype(str)
        return out

    def fit(self, adata: AnnData) -> None:
        return None

    def _ckpt_dir(self) -> Path:
        if self.checkpoint:
            path = Path(self.checkpoint).resolve()
        else:
            path = self.work_dir / "ckpt"
        prefix = self.pretrain_prefix
        required = [
            path / f"{prefix}.config.json",
            path / f"{prefix}.best.ckpt",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"CellPLM checkpoint incomplete under {path}; missing {missing}. "
                f"Run: python scripts/download_cellplm_assets.py --out {path}"
            )
        return path

    def embed(self, adata: AnnData) -> np.ndarray:
        import torch
        from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline

        ckpt_dir = self._ckpt_dir()
        pipeline = CellEmbeddingPipeline(
            pretrain_prefix=self.pretrain_prefix,
            pretrain_directory=str(ckpt_dir),
        )
        adata_pp = self.preprocess(adata)
        pred = pipeline.predict(
            adata_pp,
            inference_config={"batch_size": self.batch_size},
            ensembl_auto_conversion=True,
            device=self.device,
        )
        if isinstance(pred, torch.Tensor):
            return pred.detach().cpu().numpy()
        return np.asarray(pred, dtype=np.float64)
