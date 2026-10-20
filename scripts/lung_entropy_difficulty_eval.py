#!/usr/bin/env python
"""Label-entropy difficulty eval for a completed IRT run (lung atlas first).

    python scripts/lung_entropy_difficulty_eval.py --run runs/ranking_lung_atlas_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scfm_irt.analysis.entropy_difficulty_eval import write_entropy_difficulty_eval
from scfm_irt.analysis.run_io import load_fit_from_run, load_responses
from scfm_irt.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="runs/ranking_lung_atlas_v1",
        help="Completed IRT run directory",
    )
    parser.add_argument("--k", type=int, default=30, help="kNN neighbors for label entropy")
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    cfg = load_config(run_dir / "config.yaml")
    fit = load_fit_from_run(run_dir)
    resp = load_responses(run_dir, cfg)

    write_entropy_difficulty_eval(
        run_dir,
        cfg,
        fit,
        resp,
        k=args.k,
        n_replicates=args.replicates,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
