#!/usr/bin/env python
"""Run online benchmark-maintenance simulation (Algorithm 1).

    PYTHONPATH=. python sim/run_sim.py
    PYTHONPATH=. python sim/run_sim.py --drift-mode real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.analysis.diagnostics import write_diagnostics_report
from sim.analysis.figures import plot_bank_budget_sweep, plot_diagnostics, plot_fig2, plot_fig3
from sim.data.build_tensor import build_correctness_tensor
from sim.engine.maintain import SimConfig, run_all_arms
from sim.irt.rasch import fit_joint_rasch, linking_summary, oracle_params


def _load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sim/conf/sim.yaml")
    parser.add_argument("--drift-mode", choices=("synthetic", "real"), default=None)
    parser.add_argument("--bank-budget", type=int, default=None)
    parser.add_argument("--bank-budget-sweep", type=str, default=None, help="comma-separated budgets, e.g. 50,80,100,120")
    parser.add_argument("--equal-bank-size", action="store_true", help="Seed all arms with same N items (controlled comparison)")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    cfg_raw = _load_cfg(REPO / args.config)
    out_dir = REPO / "sim" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # M0 + tensor
    print("[sim] building correctness tensor...")
    resp = build_correctness_tensor(
        lung_run=REPO / cfg_raw["lung_run"],
        pbmc_run=REPO / cfg_raw["pbmc_run"],
        out_path=REPO / "sim" / "data" / "correctness.parquet",
    )

    print("[sim] M0 linking (pbmc3k + lung, shared models)...")
    link = linking_summary(resp, datasets=["pbmc3k", "lung_atlas"], seed=cfg_raw.get("seed", 0))
    link.to_csv(out_dir / "m0_linking_summary.csv", index=False)
    print(link.to_string(index=False))

    print("[sim] M1 joint Rasch oracle fit (lung atlas)...")
    lung_resp = resp.loc[resp["dataset"] == "lung_atlas"]
    fit = fit_joint_rasch(lung_resp, seed=cfg_raw.get("seed", 0))
    theta_star, diff_star = oracle_params(fit)
    theta_star.to_csv(out_dir / "theta_star.csv", index=False)
    diff_star.to_csv(out_dir / "difficulty_star.csv", index=False)

    sim_cfg = SimConfig(
        drift_mode=args.drift_mode or cfg_raw.get("drift_mode", "synthetic"),
        n_rounds=int(cfg_raw.get("n_rounds", 10)),
        bank_budget=int(args.bank_budget or cfg_raw.get("bank_budget", 400)),
        k_ask=int(cfg_raw.get("k_ask", 40)),
        n_exp=int(cfg_raw.get("n_exp", 8)),
        kappa=int(cfg_raw.get("kappa", 15)),
        tau=float(cfg_raw.get("tau", 1.0)),
        q=float(cfg_raw.get("q", 0.05)),
        p0=float(cfg_raw.get("p0", 2.0)),
        p_star=float(cfg_raw.get("p_star", 0.25)),
        retire_v=float(cfg_raw.get("retire_v", 0.01)),
        n_anchors=int(cfg_raw.get("n_anchors", 15)),
        mu0=float(cfg_raw.get("mu0", 0.0)),
        r=float(cfg_raw.get("r", 0.12)),
        s=float(cfg_raw.get("s", 1.0)),
        n_synthetic=int(cfg_raw.get("n_synthetic", 10)),
        seed=int(cfg_raw.get("seed", 0)),
        equal_bank_size=bool(args.equal_bank_size or cfg_raw.get("equal_bank_size", False)),
    )

    mode_note = "equal-N" if sim_cfg.equal_bank_size else "uncontrolled"
    print(
        f"[sim] M2 running arms (drift_mode={sim_cfg.drift_mode}, "
        f"bank_budget={sim_cfg.bank_budget}, {mode_note})..."
    )
    metrics = run_all_arms(lung_resp, theta_star, diff_star, sim_cfg)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    try:
        metrics.to_parquet(out_dir / "metrics.parquet", index=False)
    except Exception as exc:
        print(f"[sim] warning: could not write parquet ({exc}); metrics.csv is canonical")
    plot_fig2(metrics, out_dir / "fig2_value_and_ranking.png")
    plot_diagnostics(metrics, out_dir)
    write_diagnostics_report(metrics, out_dir)

    if args.bank_budget_sweep:
        budgets = [int(x.strip()) for x in args.bank_budget_sweep.split(",") if x.strip()]
        print(f"[sim] bank budget sweep: {budgets}")
        sweep_frames: list[pd.DataFrame] = []
        summary_rows: list[dict] = []
        for bb in budgets:
            sc = SimConfig(**{**sim_cfg.__dict__, "bank_budget": bb})
            m = run_all_arms(lung_resp, theta_star, diff_star, sc)
            m["bank_budget"] = bb
            sweep_frames.append(m)
            final = m.loc[m["round"] == m["round"].max()]
            for arm in ("fixed", "recalibrate", "full"):
                row = final.loc[final["arm"] == arm].iloc[0]
                summary_rows.append(
                    {
                        "bank_budget": bb,
                        "arm": arm,
                        "bank_value": float(row["bank_value"]),
                        "ranking_spearman": float(row["ranking_spearman"]),
                        "bank_size": int(row["bank_size"]),
                        "total_retired": int(m.loc[m["arm"] == "full", "n_retired"].sum()) if arm == "full" else 0,
                    }
                )
        budget_metrics = pd.concat(sweep_frames, ignore_index=True)
        budget_metrics.to_csv(out_dir / "bank_budget_metrics.csv", index=False)
        budget_summary = pd.DataFrame(summary_rows)
        budget_summary.to_csv(out_dir / "bank_budget_summary.csv", index=False)
        plot_bank_budget_sweep(budget_summary, out_dir / "fig_bank_budget_sweep.png")
        print(budget_summary.to_string(index=False))

    if not args.skip_sweep:
        print("[sim] M3 κ vs r sweep...")
        sweep_rows: list[dict] = []
        s_hat = sim_cfg.s
        n_bank = sim_cfg.bank_budget
        for r_val in cfg_raw.get("sweep_r", [0.08, 0.12, 0.16]):
            for kappa in cfg_raw.get("sweep_kappa", [5, 10, 15, 25, 40]):
                sc = SimConfig(**{**sim_cfg.__dict__, "r": r_val, "kappa": kappa, "n_rounds": 8})
                m = run_all_arms(lung_resp, theta_star, diff_star, sc)
                tail = m.loc[m["arm"] == "full"].sort_values("round").tail(2)
                a_rate = 4.0 * s_hat * kappa / max(n_bank, 1)
                sweep_rows.append(
                    {
                        "r": r_val,
                        "kappa": kappa,
                        "a_over_r": a_rate / max(r_val, 1e-6),
                        "ranking_spearman": float(tail["ranking_spearman"].mean()),
                    }
                )
        sweep = pd.DataFrame(sweep_rows)
        sweep.to_csv(out_dir / "fig3_sweep.csv", index=False)
        plot_fig3(sweep, out_dir / "fig3_survival_threshold.png")

    report = out_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Online maintenance simulation",
                "",
                f"- drift_mode: **{sim_cfg.drift_mode}**",
                f"- models: {theta_star['model_id'].nunique()}",
                f"- lung items: {len(diff_star)}",
                "",
                "## M0 linking",
                "",
                "```",
                link.to_string(index=False),
                "```",
                "",
                "## Outputs",
                "",
                "- `fig2_value_and_ranking.png`",
                "- `fig_diagnostics.png` (μ drift, seed staleness, bank size)",
                "- `diagnostics.md` + `diagnostics_final_summary.csv`",
                "- `fig3_survival_threshold.png` (if sweep enabled)",
                "- `metrics.csv`",
                "- `fig_bank_budget_sweep.png` (if `--bank-budget-sweep` set)",
                "",
                f"- equal_bank_size: **{sim_cfg.equal_bank_size}**",
                "",
            ]
        )
    )
    print(f"[sim] done -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
