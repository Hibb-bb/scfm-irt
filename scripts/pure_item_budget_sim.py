#!/usr/bin/env python
"""Pure synthetic simulation: item budget vs ability error for two selection policies.

    python scripts/pure_item_budget_sim.py
    python scripts/pure_item_budget_sim.py --bank-size 500 --bank-reps 30 --persons 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scfm_irt.analysis.pure_item_budget_sim import (
    PureSimConfig,
    _default_n_grid,
    run_pure_item_budget_sim,
    summarize_pure_sim,
    write_pure_sim_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/pure_item_budget_v1")
    parser.add_argument("--bank-size", type=int, default=500)
    parser.add_argument("--bank-reps", type=int, default=25)
    parser.add_argument("--persons", type=int, default=100)
    parser.add_argument("--difficulty-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = PureSimConfig(
        bank_size=args.bank_size,
        n_grid=_default_n_grid(args.bank_size),
        n_bank_replicates=args.bank_reps,
        n_persons=args.persons,
        difficulty_sd=args.difficulty_sd,
        seed=args.seed,
    )

    print(f"[pure-sim] bank={cfg.bank_size} reps={cfg.n_bank_replicates} persons={cfg.n_persons}")
    sim = run_pure_item_budget_sim(cfg)
    summary = summarize_pure_sim(sim)
    report = write_pure_sim_report(args.out, sim, summary, cfg)
    print(f"[pure-sim] wrote {report}")
    print(summary.pivot(index="n_items", columns="strategy", values="ability_error_mean").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
