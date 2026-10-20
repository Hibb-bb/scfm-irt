"""Geneformer — zero-shot transcriptome transformer (Theodoris lab).

Runs inside `envs/env-geneformer.yml` only. Pipeline:
  1. map gene symbols -> Ensembl IDs, add n_counts
  2. TranscriptomeTokenizer -> HuggingFace .dataset
  3. EmbExtractor (pretrained) -> cell embeddings

Assets: clone `ctheodoris/Geneformer` via `scripts/download_geneformer_assets.py`.
Default model: Geneformer-V1-10M (smaller, input 2048) for faster first ranking runs.

See https://huggingface.co/ctheodoris/Geneformer
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scfm_irt.embedders._genes import (
    resolve_ensembl_ids as _resolve_ensembl_ids,
)
from scfm_irt.embedders._genes import (
    write_h5ad_compat as _write_h5ad_compat,
)

if TYPE_CHECKING:
    from anndata import AnnData


class GeneformerEmbedder:
    name = "geneformer"
    version = "V1-10M"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-geneformer.yml"
    # Checkpoint directory inside the downloaded HF repo; subclasses override to pick a size.
    model_dirname = "Geneformer-V1-10M"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        nproc: int = 4,
        model_version: str = "V1",
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.nproc = nproc
        self.model_version = model_version
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_GENEFORMER_WORKDIR", "cache/geneformer_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """Add ensembl_id + n_counts required by Geneformer's tokenizer."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]

        # Total counts for rank-value normalization.
        totals = out.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        out.obs["n_counts"] = np.asarray(totals).ravel().astype(np.float64)

        # Preserve original cell id for re-alignment after EmbExtractor shuffles.
        out.obs["joinid"] = np.arange(out.n_obs, dtype=np.int64)
        out.obs["cell_barcode"] = out.obs_names.astype(str)

        if "ensembl_id" not in out.var.columns:
            out.var["ensembl_id"] = _resolve_ensembl_ids(
                out,
                mapping_pkl=self._symbol_mapping_pkl(),
            )

        # Drop genes without a mapping — tokenizer needs Ensembl IDs in its vocab.
        mapped = out.var["ensembl_id"].notna() & (out.var["ensembl_id"].astype(str) != "nan")
        if int(mapped.sum()) < 100:
            raise RuntimeError(
                f"Geneformer: only {int(mapped.sum())} genes mapped to Ensembl IDs; "
                "check gene symbols / mygene connectivity."
            )
        out = out[:, mapped].copy()
        out.var["ensembl_id"] = out.var["ensembl_id"].astype(str)
        return out

    def fit(self, adata: AnnData) -> None:
        return None

    def _repo_root(self) -> Path:
        root = self.work_dir / "Geneformer"
        if not (root / "geneformer").exists() and not (root / "setup.py").exists():
            # Also accept a pip-installed layout where only model weights live here.
            if not (self.work_dir / "Geneformer-V1-10M").exists() and self.checkpoint is None:
                raise FileNotFoundError(
                    f"Geneformer assets missing under {self.work_dir}. "
                    "Run: python scripts/download_geneformer_assets.py"
                )
        return root

    def _symbol_mapping_pkl(self) -> Path | None:
        """Bundled Geneformer symbol -> Ensembl map (offline, no mygene).

        V2 ships its own 95M-corpus dictionaries; fall back to the 30M ones since the
        symbol->Ensembl direction is corpus-independent.
        """
        dicts = self.work_dir / "Geneformer" / "geneformer"
        candidates: tuple[Path, ...] = (
            dicts / "gene_dictionaries_30m" / "gene_name_id_dict_gc30M.pkl",
            dicts / "gene_dictionaries_30m" / "ensembl_mapping_dict_gc30M.pkl",
        )
        if self.model_version.upper() == "V2":
            candidates = (
                dicts / "gene_name_id_dict_gc95M.pkl",
                dicts / "ensembl_mapping_dict_gc95M.pkl",
                *candidates,
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _ensembl_mapping_pkl(self) -> Path | None:
        """Alias kept for callers; prefer symbol map."""
        return self._symbol_mapping_pkl()

    def _model_dir(self) -> Path:
        if self.checkpoint:
            path = Path(self.checkpoint).resolve()
        else:
            dirname = self.model_dirname
            path = self.work_dir / "Geneformer" / dirname
            if not path.exists():
                path = self.work_dir / dirname
        if not path.exists():
            raise FileNotFoundError(
                f"Geneformer model directory not found: {path}. "
                f"Run: python scripts/download_geneformer_assets.py --model {self.model_dirname}"
            )
        return path

    def embed(self, adata: AnnData) -> np.ndarray:
        """Tokenize + extract cell embeddings; return (n_cells, d) aligned to adata."""
        # Import submodules only — package __init__ pulls classifier/pretrainer stacks.
        from geneformer.emb_extractor import EmbExtractor
        from geneformer.tokenizer import TranscriptomeTokenizer

        prepared = self.preprocess(adata)
        model_dir = self._model_dir()

        with tempfile.TemporaryDirectory(prefix="gf_", dir=str(self.work_dir)) as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "h5ad"
            tok_dir = tmp_path / "tokenized"
            emb_dir = tmp_path / "embs"
            data_dir.mkdir()
            tok_dir.mkdir()
            emb_dir.mkdir()

            # Tokenizer expects a *directory* of h5ad files.
            h5_path = data_dir / "input.h5ad"
            _write_h5ad_compat(prepared, h5_path)

            # First positional arg is loom/h5ad attr -> dataset column mapping.
            # model_version="V1" auto-selects 2048 input, special_token=False, V1 dict.
            tk = TranscriptomeTokenizer(
                {"joinid": "joinid", "cell_barcode": "cell_barcode"},
                nproc=self.nproc,
                model_version=self.model_version.upper(),
            )
            tk.tokenize_data(
                str(data_dir),
                str(tok_dir),
                "pbmc",
                file_format="h5ad",
            )

            # tokenize_data writes tok_dir/pbmc.dataset (directory)
            dataset_path = tok_dir / "pbmc.dataset"
            if not dataset_path.exists():
                # Some versions append differently.
                candidates = list(tok_dir.glob("*.dataset"))
                if not candidates:
                    raise FileNotFoundError(f"no tokenized dataset under {tok_dir}")
                dataset_path = candidates[0]

            embex = EmbExtractor(
                model_type="Pretrained",
                num_classes=0,
                emb_mode="cell",
                cell_emb_style="mean_pool",
                max_ncells=None,
                emb_layer=-1,
                emb_label=["joinid", "cell_barcode"],
                forward_batch_size=self.batch_size,
                nproc=self.nproc,
                model_version=self.model_version.upper(),
            )
            embs = embex.extract_embs(
                model_directory=str(model_dir),
                input_data_file=str(dataset_path),
                output_directory=str(emb_dir),
                output_prefix="cell_emb",
            )

            # embs is a DataFrame: embedding columns + joinid / cell_barcode
            if hasattr(embs, "to_numpy"):
                meta_cols = [c for c in ("joinid", "cell_barcode") if c in embs.columns]
                feat_cols = [c for c in embs.columns if c not in meta_cols]
                joinids = embs["joinid"].to_numpy() if "joinid" in embs.columns else None
                mat = embs[feat_cols].to_numpy(dtype=np.float32)
            else:
                raise TypeError(f"unexpected EmbExtractor return type: {type(embs)}")

            if joinids is None:
                raise RuntimeError("Geneformer embeddings missing joinid for alignment")

            # Map back onto the *prepared* cell order (same as input after gene filter).
            # Input adata may have more genes but same cells — align by joinid on prepared.
            n = prepared.n_obs
            dim = mat.shape[1]
            aligned = np.full((n, dim), np.nan, dtype=np.float32)
            for row, jid in enumerate(joinids):
                j = int(jid)
                if 0 <= j < n:
                    aligned[j] = mat[row]

            if np.isnan(aligned).any():
                n_miss = int(np.isnan(aligned[:, 0]).sum())
                raise RuntimeError(
                    f"Geneformer failed to embed {n_miss}/{n} cells after tokenization"
                )

            # prepared may be gene-subset of adata but same n_obs / order.
            if prepared.n_obs != adata.n_obs:
                raise RuntimeError(
                    f"cell count changed during preprocess ({adata.n_obs} -> {prepared.n_obs})"
                )
            return aligned


class GeneformerV2Embedder(GeneformerEmbedder):
    """Geneformer V2-104M — 104M params, 4096-token input, ~104M-cell pretraining corpus.

    A distinct `name` rather than a `version` bump on the V1 class: `pipeline.py` keys the
    embedding cache and the IRT `model_id` off `name`, so sharing it would silently merge two
    different models into one respondent.

    Runs in the same venv as V1; `TranscriptomeTokenizer`/`EmbExtractor` switch input size,
    special tokens, and dictionaries off `model_version` on their own.
    """

    name = "geneformer_v2"
    version = "V2-104M"

    model_dirname = "Geneformer-V2-104M"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("model_version", "V2")
        # 4096-token context at 104M params needs a smaller forward batch than V1's 10M.
        kwargs.setdefault("batch_size", 16)
        super().__init__(*args, **kwargs)


class GeneformerV2_316MEmbedder(GeneformerV2Embedder):
    """Geneformer V2-316M — the largest released Geneformer.

    Same tokenizer and corpus as V2-104M, three times the parameters. Included to give the
    ranking a within-family size ladder: whether extra capacity buys anything specifically on
    the cells IRT scores as difficult is a question you can only ask if both are respondents.
    """

    name = "geneformer_v2_316m"
    version = "V2-316M"
    model_dirname = "Geneformer-V2-316M"

    def __init__(self, *args, **kwargs):
        # 316M params at 4096 tokens; halve the batch again to stay inside GPU memory.
        kwargs.setdefault("batch_size", 8)
        super().__init__(*args, **kwargs)

