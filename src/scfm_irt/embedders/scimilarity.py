"""SCimilarity — label-supervised cell embedding search model (Genentech).

Runs inside `envs/env-scimilarity.yml` only. Uses official gene alignment +
tp10k log-normalization before `CellEmbedding.get_embeddings`.

Model weights (~30 GB) must be downloaded first:
    python scripts/download_scimilarity_assets.py

Note: SCimilarity was pretrained with cell-type labels — treat its IRT ability as
**supplementary**, not directly comparable to zero-shot foundation models.

See https://github.com/Genentech/scimilarity
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


class SCimilarityEmbedder:
    name = "scimilarity"
    version = "v1.1"
    requires_dataset_fit = False
    label_supervised = True
    env = "envs/env-scimilarity.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_SCIMILARITY_WORKDIR", "cache/scimilarity_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """Keep count-like expression; SCimilarity applies its own alignment + lognorm."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]
        if "counts" not in out.layers:
            totals = out.X.sum(axis=1)
            if hasattr(totals, "A1"):
                totals = totals.A1
            median_total = float(np.asarray(totals).mean()) if len(totals) else 0.0
            if median_total > 50:
                out.layers["counts"] = out.X.copy()
            else:
                raise ValueError(
                    "SCimilarity needs raw counts in layers['counts'] or count-scale .X"
                )
        out.X = out.layers["counts"]
        return out

    def _model_dir(self) -> Path:
        if self.checkpoint:
            path = Path(self.checkpoint).resolve()
        else:
            path = self.work_dir / "model_v1.1"
        if not path.exists():
            raise FileNotFoundError(
                f"SCimilarity model not found at {path}. "
                "Run: python scripts/download_scimilarity_assets.py"
            )
        return path

    def fit(self, adata: AnnData) -> None:
        return None

    def embed(self, adata: AnnData) -> np.ndarray:
        from scimilarity import CellEmbedding
        from scimilarity.utils import align_dataset, lognorm_counts

        prepared = self.preprocess(adata)
        use_gpu = self.device.startswith("cuda")
        ce = CellEmbedding(model_path=str(self._model_dir()), use_gpu=use_gpu)

        overlap = len(set(prepared.var_names) & set(ce.gene_order))
        if overlap < 5000:
            raise RuntimeError(
                f"SCimilarity needs ≥5000 gene symbols overlapping its vocabulary; "
                f"got {overlap} from {prepared.n_vars} vars in {getattr(prepared, 'filename', 'dataset')}. "
                "HVG-subset atlases (e.g. immune_human with 2k genes) are incompatible — "
                "use a full-transcriptome dataset (pbmc3k, lung_atlas)."
            )

        aligned = align_dataset(prepared, ce.gene_order)
        aligned = lognorm_counts(aligned)
        x = aligned.X
        if hasattr(x, "toarray"):
            x = x.tocsr()
        embeddings = ce.get_embeddings(x)
        return np.asarray(embeddings, dtype=np.float32)
