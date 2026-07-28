#!/usr/bin/env python
"""Extract and cache embeddings for ONE model on ONE dataset.

**This script runs inside that model's isolated environment**, not the analysis env:

    conda run -n scgpt python scripts/extract_embeddings.py \
        --embedder scgpt --dataset zheng68k --split cross_donor --seed 0

It is the only half of the pipeline that imports an embedder. Its sole output is a cached
artifact under `cache/embeddings/` — the contract with everything downstream (plan section 2.1).

Embeddings are expensive, so they are computed once and reused across every probe and IRT
experiment. `run_pipeline.py` reads these and will fail loudly rather than recompute.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cross_donor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    parser.parse_args()

    # Steps:
    #   1. load dataset + build the split (data)
    #   2. instantiate the embedder from the registry (heavy import happens here and only here)
    #   3. run the model's OWN preprocessing — never a generic scanpy pipeline
    #   4. if requires_dataset_fit: fit on TRAIN CELLS ONLY, or the eval leaks
    #   5. embed all cells, keeping rows aligned to obs_names
    #   6. save the matrix, the aligned cell_ids, and the provenance sidecar
    raise NotImplementedError("TODO(phase-5): wire embedder -> save_embedding")


if __name__ == "__main__":
    raise SystemExit(main())
