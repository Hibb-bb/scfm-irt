#!/usr/bin/env python
"""Run the light-side pipeline from a config: cached embeddings -> ranking.

    python scripts/run_pipeline.py --config configs/annotation_irt_v1.yaml

Equivalent to `scfm-irt run --config ...`. Runs in the analysis environment and never imports
an embedder; if an embedding artifact is missing it says which `extract_embeddings.py`
invocation would produce it rather than trying to compute it here.
"""

from __future__ import annotations

import argparse

from scfm_irt.config import load_config
from scfm_irt.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[run] {cfg.run_name}  hash={cfg.config_hash()}")
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
