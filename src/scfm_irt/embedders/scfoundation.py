"""scFoundation — 100M-parameter xTrimoGene MAE (BioMap).

Runs inside `envs/env-scfoundation.yml` only. Drives the repo's own `model/get_embedding.py`
as a subprocess, the way `uce.py` drives UCE, because the released code is a script with a
hard-coded module layout rather than an importable package.

Unlike Geneformer/TranscriptFormer, scFoundation keys on **HUGO gene symbols**, not Ensembl
IDs: it uses a fixed 19,264-gene vocabulary (protein-coding + common mitochondrial), and the
official preprocessing zero-pads whatever the dataset is missing.

Assets (repo + checkpoint) come from `scripts/download_scfoundation_assets.py`. The checkpoint
is distributed only via the authors' SharePoint link and must be staged by hand.

See https://github.com/biomap-research/scFoundation
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

N_GENES = 19264
GENE_INDEX_FILENAME = "OS_scRNA_gene_index.19264.tsv"


class SCFoundationEmbedder:
    name = "scfoundation"
    version = "01B-resolution"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-scfoundation.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        tgthighres: str = "a5",
        pool_type: str = "all",
        model_version: str = "rde",
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        # "a5" adds 5 to the log10 total count as the target resolution — the value the repo
        # uses for single-cell (as opposed to bulk) cell embeddings.
        self.tgthighres = tgthighres
        self.pool_type = pool_type
        self.model_version = model_version
        self.work_dir = Path(
            work_dir
            or os.environ.get("SCFM_SCFOUNDATION_WORKDIR", "cache/scfoundation_workdir")
        ).resolve()

    def _repo_root(self) -> Path:
        root = self.work_dir / "scFoundation"
        if not (root / "model" / "get_embedding.py").exists():
            raise FileNotFoundError(
                f"scFoundation repo missing at {root}. "
                "Run: python scripts/download_scfoundation_assets.py"
            )
        return root

    def _checkpoint(self) -> Path:
        if self.checkpoint:
            path = Path(self.checkpoint).resolve()
        else:
            path = self._repo_root() / "model" / "models" / "models.ckpt"
        if not path.exists() or path.stat().st_size < 100_000_000:
            raise FileNotFoundError(
                f"scFoundation checkpoint missing or truncated at {path}. "
                "Run: python scripts/download_scfoundation_assets.py (it prints the "
                "SharePoint URL; the file must be staged by hand)."
            )
        return path

    def _gene_list(self) -> list[str]:
        matches = sorted(self._repo_root().rglob(GENE_INDEX_FILENAME))
        if not matches:
            raise FileNotFoundError(
                f"{GENE_INDEX_FILENAME} not found under {self._repo_root()}; "
                "the repo checkout is incomplete."
            )
        import pandas as pd

        table = pd.read_csv(matches[0], header=0, delimiter="\t")
        genes = [str(g) for g in table["gene_name"].tolist()]
        if len(genes) != N_GENES:
            raise RuntimeError(f"expected {N_GENES} genes in {matches[0]}, got {len(genes)}")
        return genes

    def preprocess(self, adata: AnnData) -> AnnData:
        """Align raw counts onto the fixed 19,264-symbol vocabulary, zero-padding the rest.

        This is the official `main_gene_selection` behaviour: the model's gene axis is fixed,
        so genes it has never heard of are dropped and genes the dataset lacks are zeros.
        """
        import anndata as ad
        import pandas as pd
        from scipy import sparse

        totals = adata.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        if float(np.asarray(totals).mean() or 0.0) <= 50:
            raise ValueError(
                "scFoundation expects raw counts in .X; this matrix looks normalized "
                "(mean total per cell <= 50)."
            )

        genes = self._gene_list()
        symbols = [str(s).upper() for s in adata.var_names]
        wanted = {g.upper(): i for i, g in enumerate(genes)}

        overlap = sum(1 for s in symbols if s in wanted)
        if overlap < 1000:
            raise RuntimeError(
                f"scFoundation: only {overlap} of {adata.n_vars} gene symbols matched its "
                f"{N_GENES}-gene vocabulary. This dataset's var_names are probably Ensembl "
                "IDs or a different nomenclature."
            )

        source = adata.X
        dense = np.zeros((adata.n_obs, N_GENES), dtype=np.float32)
        for src_idx, sym in enumerate(symbols):
            dst = wanted.get(sym)
            if dst is None:
                continue
            col = source[:, src_idx]
            if sparse.issparse(col):
                col = col.toarray()
            dense[:, dst] = np.asarray(col, dtype=np.float32).ravel()

        out = ad.AnnData(
            X=dense,
            obs=pd.DataFrame(index=adata.obs_names.astype(str)),
            var=pd.DataFrame(index=pd.Index(genes, name=None)),
        )
        return out

    def fit(self, adata: AnnData) -> None:
        return None

    def embed(self, adata: AnnData) -> np.ndarray:
        """Run get_embedding.py on the aligned matrix; return (n_cells, d)."""
        repo = self._repo_root()
        ckpt = self._checkpoint()
        prepared = self.preprocess(adata)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="scf_", dir=str(self.work_dir)) as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "input.npy"
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            np.save(in_path, np.asarray(prepared.X, dtype=np.float32))

            cmd = [
                sys.executable,
                "get_embedding.py",
                "--task_name",
                "scfm_irt",
                "--input_type",
                "singlecell",
                "--output_type",
                "cell",
                "--pool_type",
                self.pool_type,
                "--tgthighres",
                self.tgthighres,
                "--data_path",
                str(in_path),
                "--save_path",
                str(out_dir) + "/",
                "--pre_normalized",
                "F",
                "--version",
                self.model_version,
            ]
            env = dict(os.environ)
            env.setdefault("CUDA_VISIBLE_DEVICES", "0" if self.device.startswith("cuda") else "")
            env["SCFOUNDATION_CKPT"] = str(ckpt)

            proc = subprocess.run(
                cmd,
                cwd=str(repo / "model"),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "scFoundation embedding failed.\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout[-4000:]}\n"
                    f"stderr:\n{proc.stderr[-4000:]}"
                )

            # Output filename encodes task/version/resolution and has drifted between
            # releases, so take whatever .npy landed rather than reconstructing the name.
            outs = sorted(out_dir.rglob("*.npy"), key=lambda p: p.stat().st_mtime)
            if not outs:
                raise FileNotFoundError(
                    f"no .npy embedding written under {out_dir}\n"
                    f"stdout:\n{proc.stdout[-2000:]}"
                )
            emb = np.load(outs[-1])

        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != adata.n_obs:
            raise RuntimeError(
                f"scFoundation returned shape {emb.shape}; expected ({adata.n_obs}, d). "
                "Row order is positional here, so a mismatch means cells were dropped."
            )
        return emb
