#!/usr/bin/env python
"""Subsample simulation: item count vs parameter recovery error.

Uses an existing IRT run as ground truth, subsamples items, re-fits 1PL, and plots
difficulty RMSE (and ability error) vs subset size.

    python scripts/subsample_sim.py --run runs/uce_pbmc3k_v1
    python scripts/subsample_sim.py --run runs/uce_pbmc3k_v1 --replicates 20 --steps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scfm_irt.analysis.subsample_sim import (
    run_subsample_simulation,
    summarize_simulation,
    write_subsample_report,
)
from scfm_irt.analysis.run_io import load_fit_from_run, load_responses
from scfm_irt.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run dir, e.g. runs/uce_pbmc3k_v1")
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--steps", type=int, default=400, help="VI steps per subsample re-fit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    cfg = load_config(run_dir / "config.yaml")
    fit = load_fit_from_run(run_dir)
    resp = load_responses(run_dir, cfg)

    print(f"[subsample] full items={len(fit.item_ids)} models={fit.model_ids}")
    sim = run_subsample_simulation(
        resp,
        fit,
        n_replicates=args.replicates,
        n_steps=args.steps,
        seed=args.seed,
    )
    summary = summarize_simulation(sim)
    report = write_subsample_report(run_dir, fit, sim, summary, source_run_name=cfg.run_name)
    print(f"[subsample] wrote {report}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
