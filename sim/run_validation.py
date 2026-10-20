#!/usr/bin/env python
"""Synthetic validation + survival curve on real difficulty landscape.

    PYTHONPATH=. python sim/run_validation.py
    PYTHONPATH=. python sim/run_validation.py --skip-sweep
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
from sim.analysis.figures import plot_diagnostics, plot_fig2, plot_fig3
from sim.analysis.survival import tuned_drift_rate, write_survival_report
from sim.data.build_tensor import build_correctness_tensor
from sim.engine.maintain import SimConfig, run_all_arms
from sim.irt.rasch import fit_joint_rasch, oracle_params


def _load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _tail_rho(metrics: pd.DataFrame, arm: str, n: int = 3) -> float:
    sub = metrics.loc[metrics["arm"] == arm].sort_values("round").tail(n)
    return float(sub["ranking_spearman"].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sim/conf/sim.yaml")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    cfg_raw = _load_cfg(REPO / args.config)
    out_dir = REPO / "sim" / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = build_correctness_tensor(
        lung_run=REPO / cfg_raw["lung_run"],
        pbmc_run=REPO / cfg_raw["pbmc_run"],
        out_path=REPO / "sim" / "data" / "correctness.parquet",
    )
    lung_resp = resp.loc[resp["dataset"] == "lung_atlas"]
    fit = fit_joint_rasch(lung_resp, seed=cfg_raw.get("seed", 0))
    theta_star, diff_star = oracle_params(fit)

    fitted_s = float(theta_star["theta_star"].std(ddof=1))
    n_rounds = int(cfg_raw.get("n_rounds", 15))
    n_bank = int(cfg_raw.get("bank_budget", 400))
    kappa = int(cfg_raw.get("kappa", 15))

    r_empirical = 0.0167
    drift_path = REPO / "sim" / "results" / "diagnostics_drift.csv"
    if drift_path.exists():
        r_empirical = float(pd.read_csv(drift_path)["empirical_r"].iloc[0])

    r_tuned = tuned_drift_rate(fitted_s, n_rounds)
    write_survival_report(
        fitted_s=fitted_s,
        n_bank=n_bank,
        kappa=kappa,
        r_empirical=r_empirical,
        r_tuned=r_tuned,
        n_rounds=n_rounds,
        out_dir=out_dir,
    )
    print(f"[validation] fitted s={fitted_s:.3f}, r_tune={r_tuned:.3f} (r·T≈4s)")

    base_cfg = SimConfig(
        drift_mode="synthetic",
        n_rounds=n_rounds,
        bank_budget=n_bank,
        k_ask=int(cfg_raw.get("k_ask", 40)),
        n_exp=int(cfg_raw.get("n_exp", 8)),
        kappa=kappa,
        tau=float(cfg_raw.get("tau", 1.0)),
        q=float(cfg_raw.get("q", 0.05)),
        p0=float(cfg_raw.get("p0", 2.0)),
        p_star=float(cfg_raw.get("p_star", 0.25)),
        retire_v=float(cfg_raw.get("retire_v", 0.01)),
        n_anchors=int(cfg_raw.get("n_anchors", 15)),
        mu0=float(cfg_raw.get("mu0", 0.0)),
        r=r_tuned,
        s=fitted_s,
        n_synthetic=int(cfg_raw.get("n_synthetic", 10)),
        seed=int(cfg_raw.get("seed", 0)),
        equal_bank_size=True,
    )

    print("[validation] Fig 2 — synthetic drift on real b_i (equal N, r tuned)...")
    metrics = run_all_arms(lung_resp, theta_star, diff_star, base_cfg)
    metrics.to_csv(out_dir / "fig2_metrics.csv", index=False)
    plot_fig2(metrics, out_dir / "fig2_synthetic_real_landscape.png")
    plot_diagnostics(metrics, out_dir)
    write_diagnostics_report(metrics, out_dir)

    final = metrics.loc[metrics["round"] == metrics["round"].max()]
    for arm in ("fixed", "recalibrate", "full"):
        row = final.loc[final["arm"] == arm].iloc[0]
        print(
            f"  {arm:12s}  ρ={row['ranking_spearman']:.3f}  "
            f"V/|B|={row['bank_value_per_item']:.3f}  seed_v={row['seed_mean_value']:.4f}"
        )

    if not args.skip_sweep:
        print("[validation] Fig 3 — κ vs r sweep on real bank...")
        sweep_rows: list[dict] = []
        r_grid = [0.15, 0.22, 0.27, 0.33, 0.40]
        kappa_grid = cfg_raw.get("sweep_kappa", [5, 10, 15, 25, 40, 60])
        for r_val in r_grid:
            for k in kappa_grid:
                sc = SimConfig(**{**base_cfg.__dict__, "r": r_val, "kappa": k})
                m = run_all_arms(lung_resp, theta_star, diff_star, sc)
                a_rate = 4.0 * fitted_s * k / max(n_bank, 1)
                rho_full = _tail_rho(m, "full")
                rho_fixed = _tail_rho(m, "fixed")
                sweep_rows.append(
                    {
                        "r": r_val,
                        "kappa": k,
                        "a_over_r": a_rate / max(r_val, 1e-6),
                        "ranking_spearman_full": rho_full,
                        "ranking_spearman_fixed": rho_fixed,
                        "ranking_gap": rho_full - rho_fixed,
                    }
                )
        sweep = pd.DataFrame(sweep_rows)
        sweep.to_csv(out_dir / "fig3_sweep.csv", index=False)
        plot_fig3(sweep, out_dir / "fig3_survival_threshold.png")

    print(f"[validation] done -> {out_dir / 'survival_curve.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
