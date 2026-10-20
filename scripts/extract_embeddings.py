#!/usr/bin/env python
"""Extract and cache embeddings for ONE model on ONE dataset.

**This script runs inside that model's isolated environment**, not the analysis env:

    conda run -n scgpt python scripts/extract_embeddings.py \
        --embedder scgpt --dataset zheng68k --split cross_donor --seed 0

For DevEmbedder (`dev_pca` / `dev_random`) it is safe to run in the analysis env — that is
how the smoke path brings the pipeline up without a GPU.

It is the only half of the pipeline that imports an embedder. Its sole output is a cached
artifact under `cache/embeddings/` — the contract with everything downstream (plan section 2.1).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from scfm_irt.data.h5ad_compat import harmonize_var_names, read_h5ad_compat
from scfm_irt.data.loaders import load_dataset, preprocess_for_probe
from scfm_irt.data.registry import get_spec
from scfm_irt.data.splits import make_split
from scfm_irt.embedders.base import EmbeddingArtifact, artifact_paths, save_embedding
from scfm_irt.embedders.registry import get_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--adata-path",
        default=None,
        help="override h5ad path (same cells/obs as registry dataset; for full-transcriptome extraction)",
    )
    parser.add_argument("--split", default="cross_donor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    if args.adata_path:
        adata = harmonize_var_names(read_h5ad_compat(Path(args.adata_path)))
        if spec.label_key not in adata.obs.columns:
            raise KeyError(
                f"{args.adata_path}: missing label_key {spec.label_key!r} in .obs"
            )
    else:
        adata = load_dataset(spec)
    split = make_split(
        adata,
        spec,
        scheme=args.split,  # type: ignore[arg-type]
        seed=args.seed,
        test_frac=args.test_frac,
    )

    embedder = get_embedder(args.embedder)
    if hasattr(embedder, "device"):
        embedder.device = args.device  # type: ignore[attr-defined]

    # Select HVGs / preprocess recipe on TRAIN only, then apply the same gene set to all cells.
    if embedder.name.startswith("dev_"):
        train_for_genes = preprocess_for_probe(adata[split.train_idx].copy())
        genes = list(train_for_genes.var_names)
        aligned = adata[:, genes].copy()
        # Normalize/log1p without re-selecting HVGs.
        full_proc = preprocess_for_probe(aligned, n_top_genes=aligned.n_vars)
        full_proc = full_proc[:, genes].copy()
        train_proc = full_proc[split.train_idx].copy()
    else:
        # Real foundation models: their preprocess is model-specific; still fit on train only.
        full_proc = embedder.preprocess(adata)
        train_proc = full_proc[split.train_idx].copy()

    if embedder.requires_dataset_fit:
        embedder.fit(train_proc)

    embedding = np.asarray(embedder.embed(full_proc), dtype=np.float32)
    cell_ids = np.asarray(full_proc.obs_names)

    prep_hash = hashlib.sha256(
        f"{embedder.name}:{embedder.version}:{','.join(map(str, cell_ids[:5]))}".encode()
    ).hexdigest()[:12]
    meta = EmbeddingArtifact(
        embedder=embedder.name,
        embedder_version=getattr(embedder, "version", "unknown"),
        dataset=spec.name,
        split_scheme=split.scheme,
        split_seed=split.seed,
        preprocessing_hash=prep_hash,
        n_cells=int(embedding.shape[0]),
        dim=int(embedding.shape[1]),
    )
    npz_path, _ = artifact_paths(args.cache_dir, meta)
    if npz_path.exists() and not args.force:
        print(f"[extract] cache hit: {npz_path}")
        return 0

    save_embedding(npz_path, embedding, cell_ids, meta)
    print(f"[extract] wrote {npz_path}  shape={embedding.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
