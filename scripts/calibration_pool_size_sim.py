#!/usr/bin/env python
"""Study how difficulty calibration improves with more calibration models.

    python scripts/calibration_pool_size_sim.py --run runs/ranking_lung_atlas_v2
    python scripts/calibration_pool_size_sim.py --synthetic --n-models 12 --n-items 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scfm_irt.analysis.calibration_pool_size_sim import (
    replot_calibration_size_ability,
    run_calibration_size_ability_sim,
    run_calibration_size_study,
    summarize_calibration_size_ability,
    summarize_calibration_size_stability,
    synthetic_irt_responses,
    write_calibration_pool_size_report,
)
from scfm_irt.analysis.run_io import load_fit_from_run, load_responses
from scfm_irt.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Multi-model IRT run dir (real data)")
    parser.add_argument("--synthetic", action="store_true", help="Use simulated 1PL data")
    parser.add_argument("--n-models", type=int, default=12, help="Synthetic model count")
    parser.add_argument("--n-items", type=int, default=500, help="Synthetic item count")
    parser.add_argument("--subsamples", type=int, default=5, help="Random k-subsets per k")
    parser.add_argument("--ability-subsamples", type=int, default=3, help="Subsamples to average b for ability sim")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--steps", type=int, default=400, help="VI steps per calibration fit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-k", action="store_true", help="Ability sim at every k=2..P-1")
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip LOO difficulty stability (ability sim only)",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate figures from existing new_model_ability_by_k_summary.csv",
    )
    args = parser.parse_args()

    if args.replot_only:
        if not args.run:
            parser.error("--replot-only requires --run")
        run_dir = Path(args.run).resolve()
        summary_path = run_dir / "calibration_pool_size_sim" / "new_model_ability_by_k_summary.csv"
        if not summary_path.exists():
            parser.error(f"missing {summary_path}")
        summary = pd.read_csv(summary_path)
        out = run_dir / "calibration_pool_size_sim"
        paths = replot_calibration_size_ability(summary, out)
        print(f"[cal-pool] replotted {len(paths)} per-pct figures under {out / 'ability_error_vs_k'}")
        return 0

    if args.synthetic:
        print(f"[cal-pool] synthetic: P={args.n_models}  I={args.n_items}")
        resp, fit = synthetic_irt_responses(
            n_models=args.n_models,
            n_items=args.n_items,
            seed=args.seed,
        )
        run_dir = Path("runs") / f"calibration_pool_synthetic_p{args.n_models}_i{args.n_items}"
        source_label = f"synthetic P={args.n_models}, I={args.n_items}"
    elif args.run:
        run_dir = Path(args.run).resolve()
        cfg = load_config(run_dir / "config.yaml")
        fit = load_fit_from_run(run_dir)
        resp = load_responses(run_dir, cfg)
        source_label = cfg.run_name
        print(f"[cal-pool] run={run_dir.name}  models={len(fit.model_ids)}  items={len(fit.item_ids)}")
    else:
        parser.error("Provide --run or --synthetic")

    print("[cal-pool] difficulty stability vs k...")
    if args.skip_stability:
        stability = pd.DataFrame()
        stability_summary = pd.DataFrame()
        print("[cal-pool] skipped (--skip-stability)")
    else:
        stability = run_calibration_size_study(
            resp,
            fit,
            n_subsamples=args.subsamples,
            n_steps=args.steps,
            seed=args.seed,
        )
        stability_summary = summarize_calibration_size_stability(stability)
        print(stability_summary.to_string(index=False))

    k_values = None
    if args.all_k:
        k_values = list(range(2, len(fit.model_ids)))

    print("[cal-pool] new-model ability simulation vs k...")
    sim = run_calibration_size_ability_sim(
        resp,
        fit,
        k_values=k_values,
        n_subsamples=args.ability_subsamples,
        n_replicates=args.replicates,
        n_steps=args.steps,
        seed=args.seed,
    )
    sim_summary = summarize_calibration_size_ability(sim)

    out_dir = run_dir / "calibration_pool_size_sim"
    if args.skip_stability:
        out_dir.mkdir(parents=True, exist_ok=True)
        sim.to_csv(out_dir / "new_model_ability_by_k_raw.csv", index=False)
        sim_summary.to_csv(out_dir / "new_model_ability_by_k_summary.csv", index=False)
        replot_calibration_size_ability(sim_summary, out_dir)
        report = out_dir / "report.md"
        print(f"[cal-pool] wrote ability figures under {out_dir}")
    else:
        report = write_calibration_pool_size_report(
            out_dir,
            fit,
            stability,
            stability_summary,
            sim,
            sim_summary,
            source_label=source_label,
        )
    print(f"[cal-pool] wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
