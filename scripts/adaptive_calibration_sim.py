#!/usr/bin/env python
"""Test difficulty-bank stability and new-model adaptive ability estimation.

Leave-one-model-out calibration + fixed-b adaptive/random item selection.

    python scripts/adaptive_calibration_sim.py --run runs/ranking_pbmc3k_v2
    python scripts/adaptive_calibration_sim.py --run runs/ranking_pbmc3k_v2 --replicates 30 --steps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scfm_irt.analysis.adaptive_calibration_sim import (
    loo_difficulty_stability,
    run_new_model_adaptive_sim,
    summarize_adaptive,
    write_adaptive_calibration_report,
)
from scfm_irt.analysis.run_io import load_fit_from_run, load_responses
from scfm_irt.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Multi-model IRT run dir")
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--steps", type=int, default=400, help="VI steps per LOO calibration fit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    cfg = load_config(run_dir / "config.yaml")
    fit = load_fit_from_run(run_dir)
    resp = load_responses(run_dir, cfg)

    print(f"[adaptive-cal] models={fit.model_ids}  items={len(fit.item_ids)}")
    print("[adaptive-cal] LOO difficulty stability...")
    stability, loo_fits = loo_difficulty_stability(resp, fit, n_steps=args.steps, seed=args.seed)
    print(stability.to_string(index=False))

    print("[adaptive-cal] new-model adaptive simulation...")
    sim = run_new_model_adaptive_sim(
        resp,
        fit,
        loo_fits,
        n_replicates=args.replicates,
        seed=args.seed,
    )
    summary = summarize_adaptive(sim)
    report = write_adaptive_calibration_report(
        run_dir,
        fit,
        stability,
        sim,
        summary,
        source_run_name=cfg.run_name,
    )
    print(f"[adaptive-cal] wrote {report}")
    print(
        summary.loc[summary["b_source"] == "loo_calibrated"]
        .pivot_table(index="n_items", columns="strategy", values="ability_error_mean")
        .round(4)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
