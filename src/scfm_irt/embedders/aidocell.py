"""AIDO.Cell (GenBio AI), published on Hugging Face as GB.Cell-{3M,10M,100M}.

Runs inside `.venv-aidocell` only (`scripts/setup_aidocell_venv.sh`). The model reads the full
19,264-gene human transcriptome per cell plus two read-depth tokens; there is no gene selection or
ranking step. Preprocessing and pooling follow the official `huggingface/gb.cell/embed.py` of
genbio-ai/ModelGenerator exactly:

1. `gb_cell.utils.align_adata` reorders genes to the model's list by *symbol* (`var_names`) and
   zero-fills missing genes; the returned mask marks which genes were present.
2. `gb_cell.utils.preprocess_counts` turns raw counts into log1p(counts per 10k), appends the two
   depth tokens and clips at 20.
3. The last hidden state is mean-pooled over the present genes and the two depth tokens.

Attention: upstream uses flash-attn when installed and torch `scaled_dot_product_attention`
otherwise. Both attend over every position without a padding mask (the mask only enters the
pooling), so the fallback computes the same function; we use it because flash-attn's prebuilt
wheels need glibc >= 2.32. The model runs in bfloat16 on CUDA as upstream does.

Licence: GenBio AI Community License (weights), Apache-2.0 (gb_cell code).
See https://huggingface.co/genbio-ai/GB.Cell-100M
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

MIN_PRESENT_GENES = 2000  # far fewer matched symbols means the gene names are not human symbols


class AIDOCellEmbedder:
    requires_dataset_fit = False
    label_supervised = False

    def __init__(self, size: str = "100M", device: str = "cuda", batch_size: int | None = None):
        if size not in ("3M", "10M", "100M"):
            raise ValueError(f"unknown AIDO.Cell size {size!r}")
        self.size = size
        self.name = f"aidocell_{size.lower()}"
        self.version = f"genbio-ai/GB.Cell-{size}"
        self.device = device
        # 19,266 tokens per cell: activation memory, not weights, sets the batch size
        self.batch_size = batch_size or {"3M": 32, "10M": 16, "100M": 4}[size]
        self._model = None

    def preprocess(self, adata: AnnData) -> AnnData:
        """Align to the model's gene list; the present-gene mask goes to `var['aido_present']`.

        Idempotent: an already aligned matrix (as handed back by the extractor) passes through.
        """
        from gb_cell.utils import align_adata

        if "aido_present" in adata.var.columns:
            return adata
        out = adata.copy()
        out.var_names = out.var_names.astype(str)
        out.var_names_make_unique()
        for key in list(out.obsm.keys()):
            del out.obsm[key]
        totals = np.asarray(out.X.sum(axis=1)).ravel()
        if float(totals.mean() or 0.0) <= 50:
            raise ValueError("AIDO.Cell expects raw counts in .X; this matrix looks normalized.")
        aligned, mask = align_adata(out)
        present = int(mask.sum())
        if present < MIN_PRESENT_GENES:
            raise ValueError(
                f"only {present} of {len(mask)} AIDO.Cell genes found in var_names; "
                "AIDO.Cell matches human gene symbols"
            )
        aligned = aligned.copy()
        aligned.var["aido_present"] = np.asarray(mask, dtype=bool)
        return aligned

    def fit(self, adata: AnnData) -> None:
        return None

    def _load(self):
        if self._model is None:
            import torch
            from gb_cell.models import CellFoundationConfig, CellFoundationModel

            config = CellFoundationConfig.from_pretrained(self.version)
            model = CellFoundationModel.from_pretrained(self.version, config=config).to(self.device)
            if str(self.device).startswith("cuda"):
                model = model.to(torch.bfloat16)
            self._model = model.eval()
        return self._model

    def embed(self, adata: AnnData) -> np.ndarray:
        """Return (n_cells, hidden) mean-pooled embeddings, rows in `adata.obs_names` order."""
        import torch
        from gb_cell.utils import preprocess_counts

        aligned = self.preprocess(adata)
        mask = aligned.var["aido_present"].to_numpy()
        model = self._load()
        gene_mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0).to(self.device)
        out = []
        with torch.no_grad():
            for start in range(0, aligned.n_obs, self.batch_size):
                block = aligned.X[start : start + self.batch_size]
                block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
                x = preprocess_counts(np.asarray(block, dtype=np.float32), device=self.device)
                attn = gene_mask.repeat(x.shape[0], 1)
                attn = torch.cat([attn, torch.ones((x.shape[0], 2), device=self.device)], dim=1)
                hidden = model(input_ids=x, attention_mask=attn, output_hidden_states=True).last_hidden_state
                m = attn.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * m).sum(dim=1) / m.sum(dim=1)
                out.append(pooled.float().cpu().numpy())
        emb = np.vstack(out)
        if emb.shape[0] != adata.n_obs:
            raise RuntimeError(f"AIDO.Cell returned {emb.shape[0]} rows for {adata.n_obs} cells")
        return emb
