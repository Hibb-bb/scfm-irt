"""Tahoe-x1 (Tahoe Therapeutics), Apache-2.0, https://huggingface.co/tahoebio/Tahoe-x1.

Tahoe-x1 only supports flash attention, and flash-attn's prebuilt wheels need glibc >= 2.32, newer
than RHEL 8 on Quest. So the model runs from the upstream container image through Apptainer, and
this adapter (which runs in a light venv with anndata only) shells out to
`scripts/tahoex1_embed_inner.py` inside it, the same pattern as the TranscriptFormer CLI adapter.

Environment: SCFM_TAHOEX1_WORKDIR (default cache/tahoex1_workdir) holds `sif/tahoe-x1_1.0.0.sif`,
the cloned `tahoe-x1` source (put on PYTHONPATH inside the container) and the Hugging Face cache
`hf/`, filled on a login node beforehand because compute nodes may be offline.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scfm_irt.embedders._genes import add_ensembl_var, write_h5ad_compat

if TYPE_CHECKING:
    from anndata import AnnData

REPO = Path(__file__).resolve().parents[3]


class TahoeX1Embedder:
    requires_dataset_fit = False
    label_supervised = False

    def __init__(self, size: str = "70m", batch_size: int | None = None, work_dir: str | None = None):
        if size not in ("70m", "1b", "3b"):
            raise ValueError(f"unknown Tahoe-x1 size {size!r}")
        self.size = size
        self.name = f"tahoex1_{size}"
        self.version = f"tahoebio/Tahoe-x1:{size}"
        self.batch_size = batch_size or {"70m": 64, "1b": 16, "3b": 8}[size]
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_TAHOEX1_WORKDIR", REPO / "cache" / "tahoex1_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """Raw counts plus `var['ensembl_id']`; the model bins counts itself."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]
        totals = np.asarray(out.X.sum(axis=1)).ravel()
        if float(totals.mean() or 0.0) <= 50:
            raise ValueError("Tahoe-x1 expects raw counts in .X; this matrix looks normalized.")
        if "ensembl_id" in out.var.columns and out.var["ensembl_id"].astype(str).str.startswith("ENSG").sum() >= 1000:
            return out
        return add_ensembl_var(out, model_name="Tahoe-x1")

    def fit(self, adata: AnnData) -> None:
        return None

    def embed(self, adata: AnnData) -> np.ndarray:
        sif = self.work_dir / "sif" / "tahoe-x1_1.0.0.sif"
        src = self.work_dir / "tahoe-x1"
        if not sif.exists():
            raise FileNotFoundError(f"{sif} missing: sbatch slurm/pull_tahoex1_sif.slurm")
        prepared = self.preprocess(adata)
        with tempfile.TemporaryDirectory(prefix="tx1_", dir=str(self.work_dir)) as tmp:
            tmp = Path(tmp)
            write_h5ad_compat(prepared, tmp / "in.h5ad")
            env = dict(os.environ)
            env.update(
                APPTAINERENV_PYTHONPATH=str(src),
                APPTAINERENV_HF_HOME=str(self.work_dir / "hf"),
                APPTAINERENV_HF_HUB_OFFLINE="1",
            )
            cmd = [
                "apptainer", "exec", "--nv", "--cleanenv",
                "--bind", f"{self.work_dir}:{self.work_dir}",
                "--bind", f"{REPO}:{REPO}",
                str(sif), "python", str(REPO / "scripts" / "tahoex1_embed_inner.py"),
                "--h5ad", str(tmp / "in.h5ad"), "--size", self.size,
                "--out", str(tmp / "emb.npy"), "--batch-size", str(self.batch_size),
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            print(proc.stdout[-4000:], flush=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Tahoe-x1 container run failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
            emb = np.load(tmp / "emb.npy")
        if emb.shape[0] != adata.n_obs:
            raise RuntimeError(f"Tahoe-x1 returned {emb.shape[0]} rows for {adata.n_obs} cells")
        return emb
