"""LangCell — language-cell pretrained model (PharMolix / AIR).

Runs inside `envs/env-langcell.yml` only. LangCell's cell encoder is a standard HuggingFace
`BertModel` initialized from Geneformer, so the input pipeline is Geneformer's verbatim: V1
tokenizer, 2048-token input, `special_token=False`. Only the weights differ.

**Interpretation caveat.** LangCell was pretrained by contrasting cells against natural-language
descriptions of cell identity, so cell-type semantics were part of its supervision. Its ability
estimate is therefore not on equal footing with the zero-shot models, exactly as with
SCimilarity — hence `label_supervised = True`, and the headline ranking should treat it as
supplementary.

We take the cell encoder's `[CLS]` representation. The `cell_proj` head that maps into the
shared cell/text space is deliberately not applied: it exists to align with text embeddings for
retrieval, and using it would push the representation further toward the label semantics we are
trying to keep at arm's length.

Assets: `scripts/download_langcell_assets.py` (Google Drive).

See https://github.com/PharMolix/LangCell
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scfm_irt.embedders._genes import (
    add_ensembl_var,
    write_h5ad_compat,
)

if TYPE_CHECKING:
    from anndata import AnnData

# LangCell's cell encoder is Geneformer-V1 shaped: 2048 tokens, no special tokens.
MODEL_INPUT_SIZE = 2048


class LangCellEmbedder:
    name = "langcell"
    version = "cell_bert"
    requires_dataset_fit = False
    label_supervised = True
    env = "envs/env-langcell.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        nproc: int = 4,
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.nproc = nproc
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_LANGCELL_WORKDIR", "cache/langcell_workdir")
        ).resolve()
        self.geneformer_dir = Path(
            os.environ.get("SCFM_GENEFORMER_WORKDIR", "cache/geneformer_workdir")
        ).resolve()

    def _cell_bert_dir(self) -> Path:
        path = (
            Path(self.checkpoint).resolve()
            if self.checkpoint
            else self.work_dir / "langcell" / "cell_bert"
        )
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"LangCell cell_bert not found at {path}. "
                "Run: python scripts/download_langcell_assets.py"
            )
        return path

    def _symbol_mapping_pkl(self) -> Path | None:
        dicts = self.geneformer_dir / "Geneformer" / "geneformer" / "gene_dictionaries_30m"
        for name in ("gene_name_id_dict_gc30M.pkl", "ensembl_mapping_dict_gc30M.pkl"):
            if (dicts / name).exists():
                return dicts / name
        return None

    def preprocess(self, adata: AnnData) -> AnnData:
        """Geneformer's tokenizer inputs: `ensembl_id`, `n_counts`, and stable cell ids."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]

        totals = out.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        out.obs["n_counts"] = np.asarray(totals).ravel().astype(np.float64)

        # EmbExtractor-style pipelines reorder cells; carry an index through to realign.
        out.obs["joinid"] = np.arange(out.n_obs, dtype=np.int64)
        out.obs["cell_barcode"] = out.obs_names.astype(str)

        return add_ensembl_var(
            out,
            mapping_pkl=self._symbol_mapping_pkl(),
            model_name="LangCell",
        )

    def fit(self, adata: AnnData) -> None:
        return None

    def embed(self, adata: AnnData) -> np.ndarray:
        """Tokenize with Geneformer, run LangCell's cell encoder, return [CLS] per cell."""
        import torch
        from geneformer.tokenizer import TranscriptomeTokenizer
        from transformers import BertModel

        prepared = self.preprocess(adata)
        model_dir = self._cell_bert_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="lc_", dir=str(self.work_dir)) as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "h5ad"
            tok_dir = tmp_path / "tokenized"
            data_dir.mkdir()
            tok_dir.mkdir()
            write_h5ad_compat(prepared, data_dir / "input.h5ad")

            tk = TranscriptomeTokenizer(
                {"joinid": "joinid", "cell_barcode": "cell_barcode"},
                nproc=self.nproc,
                model_version="V1",
            )
            tk.tokenize_data(str(data_dir), str(tok_dir), "langcell", file_format="h5ad")

            dataset_path = tok_dir / "langcell.dataset"
            if not dataset_path.exists():
                candidates = list(tok_dir.glob("*.dataset"))
                if not candidates:
                    raise FileNotFoundError(f"no tokenized dataset under {tok_dir}")
                dataset_path = candidates[0]

            from datasets import load_from_disk

            tokenized = load_from_disk(str(dataset_path))

            device = torch.device(
                "cuda" if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
            )
            model = BertModel.from_pretrained(str(model_dir)).to(device).eval()

            embeddings = _forward_in_batches(
                model=model,
                tokenized=tokenized,
                device=device,
                batch_size=self.batch_size,
                torch_module=torch,
            )

        joinids = np.asarray(tokenized["joinid"], dtype=np.int64)
        return _scatter_by_joinid(embeddings, joinids, n_cells=adata.n_obs)


def _forward_in_batches(model, tokenized, device, batch_size: int, torch_module) -> np.ndarray:
    """Run the cell encoder over tokenized cells, returning the [CLS] vector for each."""
    torch = torch_module
    pad_id = int(getattr(model.config, "pad_token_id", 0) or 0)

    out: list[np.ndarray] = []
    n = len(tokenized)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = tokenized[start : start + batch_size]["input_ids"]
            width = min(max(len(ids) for ids in chunk), MODEL_INPUT_SIZE)

            input_ids = np.full((len(chunk), width), pad_id, dtype=np.int64)
            attention = np.zeros((len(chunk), width), dtype=np.int64)
            for row, ids in enumerate(chunk):
                ids = ids[:width]
                input_ids[row, : len(ids)] = ids
                attention[row, : len(ids)] = 1

            result = model(
                input_ids=torch.as_tensor(input_ids, device=device),
                attention_mask=torch.as_tensor(attention, device=device),
            )
            # [CLS] is position 0 of the last hidden state — LangCell's cell representation.
            cls = result.last_hidden_state[:, 0, :]
            out.append(cls.detach().cpu().float().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)


def _scatter_by_joinid(emb: np.ndarray, joinids: np.ndarray, n_cells: int) -> np.ndarray:
    """Place embedded rows back onto the original cell order using the carried index."""
    if emb.shape[0] != len(joinids):
        raise RuntimeError(
            f"LangCell produced {emb.shape[0]} embeddings for {len(joinids)} tokenized cells"
        )

    aligned = np.full((n_cells, emb.shape[1]), np.nan, dtype=np.float32)
    for row, jid in enumerate(joinids):
        j = int(jid)
        if 0 <= j < n_cells:
            aligned[j] = emb[row]

    if np.isnan(aligned).any():
        n_miss = int(np.isnan(aligned[:, 0]).sum())
        raise RuntimeError(
            f"LangCell failed to embed {n_miss}/{n_cells} cells after tokenization"
        )
    return aligned
