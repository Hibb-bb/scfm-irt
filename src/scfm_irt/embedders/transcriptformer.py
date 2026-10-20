"""TranscriptFormer — cross-species generative cell atlas model (Chan Zuckerberg Initiative).

Runs inside `envs/env-transcriptformer.yml` only. Two constraints from the upstream repo drive
that env: Python >= 3.11 (the only embedder here that cannot use 3.10) and torch <= 2.5.1,
since 2.6.0+ raises pickle errors loading the released checkpoints.

Embedding goes through the official CLI rather than the internals, because inference is wired
together with hydra configs and Lightning; shelling out keeps us on the supported path:

    transcriptformer inference --checkpoint-path <ckpt> --data-file in.h5ad --emb-type cell

Default checkpoint: TF-Sapiens (human-only, 57M cells). Download with
`scripts/download_transcriptformer_assets.py`.

See https://github.com/czi-ai/transcriptformer
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scfm_irt.embedders._genes import add_ensembl_var, write_h5ad_compat
from scfm_irt.embedders.base import align_embedding_rows

if TYPE_CHECKING:
    from anndata import AnnData


class TranscriptFormerEmbedder:
    name = "transcriptformer"
    version = "tf-sapiens"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-transcriptformer.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        model: str = "tf_sapiens",
        batch_size: int = 8,
        precision: str = "16-mixed",
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.precision = precision
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_TRANSCRIPTFORMER_WORKDIR", "cache/tf_workdir")
        ).resolve()

    def _aux_vocab_keys(self) -> list[str]:
        """`obs` columns the checkpoint's auxiliary tokenizer will index into.

        Read from the shipped `vocabs/<key>_vocab.json` filenames rather than hard-coded, so a
        checkpoint with a different auxiliary set does not fail deep inside the dataloader.
        """
        vocab_dir = self._checkpoint_dir() / "vocabs"
        return sorted(
            path.name[: -len("_vocab.json")]
            for path in vocab_dir.glob("*_vocab.json")
        )

    def preprocess(self, adata: AnnData) -> AnnData:
        """Raw counts, `var['ensembl_id']`, and the auxiliary `obs` keys the tokenizer needs.

        Upstream is explicit that no further preprocessing is needed — it clips counts and
        normalizes internally — so anything extra here would double-apply.
        """
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]

        totals = out.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        if float(np.asarray(totals).mean() or 0.0) <= 50:
            raise ValueError(
                "TranscriptFormer expects raw counts in .X; this matrix looks normalized "
                "(mean total per cell <= 50)."
            )

        # The auxiliary tokenizer does obs[key] directly and raises KeyError on a missing
        # column. Every vocab carries an "unknown" entry, so filling in that sentinel is the
        # honest default — guessing an assay we do not actually know would be worse.
        for key in self._aux_vocab_keys():
            if key not in out.obs.columns:
                out.obs[key] = "unknown"
            else:
                out.obs[key] = out.obs[key].astype(str).fillna("unknown")

        return add_ensembl_var(out, model_name="TranscriptFormer")

    def fit(self, adata: AnnData) -> None:
        return None

    def _checkpoint_dir(self) -> Path:
        path = (
            Path(self.checkpoint).resolve()
            if self.checkpoint
            else self.work_dir / "checkpoints" / self.model
        )
        if not path.is_dir():
            raise FileNotFoundError(
                f"TranscriptFormer checkpoint not found at {path}. "
                f"Run: python scripts/download_transcriptformer_assets.py --model {self.model}"
            )
        return path

    def embed(self, adata: AnnData) -> np.ndarray:
        """Run CLI inference and return (n_cells, d) aligned to `adata.obs_names`."""
        import anndata as ad

        ckpt = self._checkpoint_dir()
        prepared = self.preprocess(adata)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="tf_", dir=str(self.work_dir)) as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "input.h5ad"
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            write_h5ad_compat(prepared, in_path)

            cmd = [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                str(ckpt),
                "--data-file",
                str(in_path),
                "--output-path",
                str(out_dir),
                "--output-filename",
                "embeddings.h5ad",
                "--emb-type",
                "cell",
                "--batch-size",
                str(int(self.batch_size)),
                "--gene-col-name",
                "ensembl_id",
                "--precision",
                self.precision,
                "--device",
                "cuda" if self.device.startswith("cuda") else "cpu",
                # We already chose .X deliberately in preprocess; auto-detect could silently
                # pick up a stale .raw with a different gene space.
                "--use-raw",
                "False",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    "TranscriptFormer inference failed.\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout[-4000:]}\n"
                    f"stderr:\n{proc.stderr[-4000:]}"
                )

            out_path = out_dir / "embeddings.h5ad"
            if not out_path.exists():
                candidates = sorted(out_dir.rglob("*.h5ad"))
                if not candidates:
                    raise FileNotFoundError(f"no embeddings h5ad written under {out_dir}")
                out_path = candidates[0]

            result = ad.read_h5ad(out_path)
            if "embeddings" not in result.obsm:
                raise KeyError(
                    f"expected obsm['embeddings'] in {out_path}; got {list(result.obsm)}"
                )
            emb = np.asarray(result.obsm["embeddings"], dtype=np.float32)

            # The CLI rebuilds obs from auxiliary columns only, so obs_names come back as
            # 0..n-1 even though row order matches the input. Positional return when counts
            # agree; fall back to name-based alignment only when cells were actually dropped.
            if emb.shape[0] == prepared.n_obs:
                return emb
            return align_embedding_rows(
                emb,
                result.obs_names.to_numpy(),
                prepared.obs_names.to_numpy(),
                model_name="TranscriptFormer",
            )
