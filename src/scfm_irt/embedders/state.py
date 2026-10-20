"""Arc Institute State Embedding model (SE-600M).

Runs inside `.venv-state` only. Inference goes through the official CLI:

    state emb transform --model-folder SE-600M --input in.h5ad --output out.h5ad

Input `.X` must be **library-size normalized + log1p** expression (scanpy convention,
``target_sum=1e4`` then ``log1p``). The State dataloader treats matrices as log counts
when individual values are below its raw-count heuristic.

Download weights with ``scripts/download_state_assets.py``.

See https://github.com/ArcInstitute/state and the SE Colab:
https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, issparse

from scfm_irt.data.h5ad_compat import harmonize_var_names, write_h5ad_compat
from scfm_irt.embedders.base import align_embedding_rows

if TYPE_CHECKING:
    from anndata import AnnData


class StateEmbedder:
    name = "state"
    version = "se600m_lognorm"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-state.yml"

    def __init__(
        self,
        model_folder: str | None = None,
        checkpoint: str | None = None,
        device: str = "cuda",
        embed_key: str = "X_state",
        batch_size: int | None = None,
        gene_col: str = "gene_name",
        work_dir: str | None = None,
    ):
        self.model_folder = model_folder
        self.checkpoint = checkpoint
        self.device = device
        self.embed_key = embed_key
        self.batch_size = batch_size
        self.gene_col = gene_col
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_STATE_WORKDIR", "cache/state_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """Library-size normalize + log1p counts; CSR matrix with gene-name column for SE."""
        import scanpy as sc

        out = harmonize_var_names(adata.copy())
        for key in list(out.obsm.keys()):
            del out.obsm[key]

        if self.gene_col not in out.var.columns:
            out.var[self.gene_col] = out.var_names.astype(str)

        if not issparse(out.X):
            out.X = csr_matrix(out.X)
        elif not isinstance(out.X, csr_matrix):
            out.X = csr_matrix(out.X)

        totals = out.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        median_total = float(np.asarray(totals).mean() or 0.0)
        if median_total > 50:
            sc.pp.normalize_total(out, target_sum=1e4)
            sc.pp.log1p(out)
        elif float(out.X.max()) > 20:
            # Normalized but not log-transformed — apply log1p only.
            sc.pp.log1p(out)

        if not issparse(out.X):
            out.X = csr_matrix(out.X)
        elif not isinstance(out.X, csr_matrix):
            out.X = csr_matrix(out.X)

        return out

    def fit(self, adata: AnnData) -> None:
        return None

    def _model_dir(self) -> Path:
        path = (
            Path(self.model_folder).resolve()
            if self.model_folder
            else self.work_dir / "SE-600M"
        )
        ckpts = sorted(path.glob("*.ckpt"))
        if not ckpts and self.checkpoint is None:
            raise FileNotFoundError(
                f"State SE-600M checkpoint not found under {path}. "
                "Run: python scripts/download_state_assets.py"
            )
        return path

    def _checkpoint_path(self) -> Path | None:
        if self.checkpoint:
            return Path(self.checkpoint).resolve()
        ckpts = sorted(self._model_dir().glob("*.ckpt"))
        return ckpts[-1] if ckpts else None

    def embed(self, adata: AnnData) -> np.ndarray:
        """Run ``state emb transform`` and return (n_cells, d) aligned to ``obs_names``."""
        import anndata as ad

        model_dir = self._model_dir()
        prepared = self.preprocess(adata)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="state_", dir=str(self.work_dir)) as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "processed.h5ad"
            out_path = tmp_path / "processed_emb.h5ad"
            write_h5ad_compat(prepared, in_path)

            cmd = [
                "state",
                "emb",
                "transform",
                "--model-folder",
                str(model_dir),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--embed-key",
                self.embed_key,
            ]
            ckpt = self._checkpoint_path()
            if ckpt is not None:
                cmd.extend(["--checkpoint", str(ckpt)])
            if self.batch_size is not None:
                cmd.extend(["--batch-size", str(int(self.batch_size))])

            env = os.environ.copy()
            if self.device.startswith("cuda"):
                env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            if proc.returncode != 0:
                raise RuntimeError(
                    "State embedding transform failed.\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout[-4000:]}\n"
                    f"stderr:\n{proc.stderr[-4000:]}"
                )

            if not out_path.exists():
                raise FileNotFoundError(f"State transform wrote no output at {out_path}")

            result = ad.read_h5ad(out_path)
            if self.embed_key not in result.obsm:
                raise KeyError(
                    f"expected obsm[{self.embed_key!r}] in {out_path}; got {list(result.obsm)}"
                )
            emb = np.asarray(result.obsm[self.embed_key], dtype=np.float32)

            if emb.shape[0] == prepared.n_obs:
                return emb
            return align_embedding_rows(
                emb,
                result.obs_names.to_numpy(),
                prepared.obs_names.to_numpy(),
                model_name="State",
            )
