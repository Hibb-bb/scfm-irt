#!/usr/bin/env python
"""Tahoe-x1 cell embeddings; runs INSIDE the upstream container (ghcr.io/tahoebio/tahoe-x1:1.0.0).

Called by `scfm_irt.embedders.tahoex1.TahoeX1Embedder` through `apptainer exec --nv`. Follows the
official `scripts/inference/predict_embeddings.py`: genes matched to the model vocabulary by the
Ensembl id in `var[gene_id_key]`, raw counts in `.X` (the collator bins them), sequence length 2048,
`get_batch_embeddings`, which L2-normalises the cell embeddings as upstream does.

    python tahoex1_embed_inner.py --h5ad in.h5ad --size 70m --out emb.npy [--batch-size 64]
"""
import argparse

import numpy as np
import scanpy as sc

from tahoe_x1.model import ComposerTX
from tahoe_x1.tasks import get_batch_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--size", required=True, choices=["70m", "1b", "3b"])
ap.add_argument("--out", required=True)
ap.add_argument("--gene-id-key", default="ensembl_id")
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--max-length", type=int, default=2048)
ap.add_argument("--num-workers", type=int, default=4)
a = ap.parse_args()

model, vocab, model_cfg, coll_cfg = ComposerTX.from_hf("tahoebio/Tahoe-x1", a.size)
adata = sc.read_h5ad(a.h5ad)
ids = np.array([vocab[g] if g in vocab else -1 for g in adata.var[a.gene_id_key].astype(str)], dtype=int)
print(f"[tx1] matched {int((ids >= 0).sum())}/{len(ids)} genes to a vocabulary of {len(vocab)}", flush=True)
if (ids >= 0).sum() < 1000:
    raise SystemExit("[tx1] fewer than 1000 genes matched the vocabulary; check var[gene_id_key]")
adata = adata[:, ids >= 0].copy()
emb = get_batch_embeddings(
    adata=adata,
    model=model.model,
    vocab=vocab,
    model_cfg=model_cfg,
    collator_cfg=coll_cfg,
    gene_ids=ids[ids >= 0],
    batch_size=a.batch_size,
    num_workers=a.num_workers,
    prefetch_factor=4,
    max_length=a.max_length,
    return_gene_embeddings=False,
)
emb = np.asarray(emb, dtype=np.float32)
if emb.shape[0] != adata.n_obs:
    raise SystemExit(f"[tx1] {emb.shape[0]} rows for {adata.n_obs} cells")
np.save(a.out, emb)
print(f"[tx1] wrote {a.out} {emb.shape}", flush=True)
