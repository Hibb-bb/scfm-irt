"""End-to-end orchestration: config in, run directory out.

The only place the six modules are wired together. Note what this function does *not* do: it
never imports an embedder. It reads cached embedding artifacts produced by
`scripts/extract_embeddings.py`, which runs separately in each model's isolated environment
(plan section 2.1).
"""

from __future__ import annotations

from scfm_irt.config import RunConfig


def run_pipeline(cfg: RunConfig) -> None:
    """Run data -> cached embeddings -> probe -> response table -> IRT -> report.

    Steps, each delegating to its module:
      1. resolve datasets from the registry and build splits (`data`)
      2. load cached embeddings for every (embedder, dataset, split) — fail loudly if a cache
         entry is missing rather than recomputing, since recompute needs the heavy env
      3. fit a probe per (embedder, dataset) on train cells, score test cells (`probe`)
      4. assemble responses.parquet (`response`)
      5. fit IRT (`irt`)
      6. emit tables, figures, and the summary report (`analysis`)

    Writes the resolved config and library versions into `cfg.run_dir` for reproducibility.
    """
    raise NotImplementedError("TODO(phase-2): wire modules end-to-end on the DevEmbedder first")
