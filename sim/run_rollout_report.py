#!/usr/bin/env python
"""Scenario B rollout simulation + spec markdown for downstream Word generation.

Models AND cells enter over time; full maintenance arm only.

    PYTHONPATH=. python sim/run_rollout_report.py
    PYTHONPATH=. python sim/run_rollout_report.py --report-only
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

from sim.analysis.geneformer_upgrade_report import plot_geneformer_upgrade
from sim.analysis.rollout_report import generate_rollout_report
from sim.analysis.rollout_spec import write_rollout_spec
from sim.data.build_tensor import build_correctness_tensor
from sim.engine.maintain import SimConfig, run_arm
from sim.irt.rasch import fit_joint_rasch, oracle_params


def _load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sim/conf/sim.yaml")
    parser.add_argument("--exclude-models", default="cellplm")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    cfg_raw = _load_cfg(REPO / args.config)
    exclude = tuple(x.strip() for x in args.exclude_models.split(",") if x.strip())
    out_dir = REPO / "sim" / "results" / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    rankings_path = out_dir / "model_rankings.csv"
    spec_path = out_dir / "SIMULATION_SPEC.md"
    pdf_path = out_dir / "rollout_report.pdf"
    upgrade_path = out_dir / "geneformer_upgrade.csv"
    upgrade_fig_path = out_dir / "geneformer_upgrade.png"
    retire_v = float(cfg_raw.get("retire_v", 0.01))

    theta_star = pd.DataFrame()
    model_rankings = pd.DataFrame()
    geneformer_upgrade = pd.DataFrame()
    if args.report_only:
        if not metrics_path.exists():
            print(f"[rollout] missing {metrics_path}; run without --report-only first")
            return 1
        metrics = pd.read_csv(metrics_path)
        if rankings_path.exists():
            model_rankings = pd.read_csv(rankings_path)
        if upgrade_path.exists():
            geneformer_upgrade = pd.read_csv(upgrade_path)
        if (out_dir / "theta_star_rollout.csv").exists():
            theta_star = pd.read_csv(out_dir / "theta_star_rollout.csv")
    else:
        print("[rollout] building correctness tensor...")
        resp = build_correctness_tensor(
            lung_run=REPO / cfg_raw["lung_run"],
            pbmc_run=REPO / cfg_raw["pbmc_run"],
            immune_run=REPO / cfg_raw["immune_run"] if cfg_raw.get("immune_run") else None,
            oasis_run=REPO / cfg_raw["oasis_run"] if cfg_raw.get("oasis_run") else None,
            human_thymus_run=REPO / cfg_raw["human_thymus_run"]
            if cfg_raw.get("human_thymus_run")
            else None,
            out_path=REPO / "sim" / "data" / "correctness.parquet",
        )
        datasets = cfg_raw.get("rollout_datasets", ["lung_atlas"])
        sim_resp = resp.loc[resp["dataset"].isin(datasets)].copy()
        if exclude:
            sim_resp = sim_resp.loc[~sim_resp["model_id"].isin(exclude)].copy()
            print(f"[rollout] excluded models: {exclude}")
        print(f"[rollout] datasets: {datasets}, cells={sim_resp['cell_id'].nunique()}")

        print(f"[rollout] joint Rasch oracle fit ({'+'.join(datasets)})...")
        fit = fit_joint_rasch(sim_resp, seed=cfg_raw.get("seed", 0))
        theta_star, diff_star = oracle_params(fit)
        theta_star = theta_star.loc[~theta_star["model_id"].isin(exclude)]
        theta_star.to_csv(out_dir / "theta_star_rollout.csv", index=False)
        print(f"[rollout] models: {theta_star['model_id'].tolist()}")

        n_cells = sim_resp["cell_id"].nunique()
        k_ask = int(cfg_raw.get("k_ask", max(n_cells, 40)))
        if k_ask < n_cells:
            k_ask = n_cells
            print(f"[rollout] bumped k_ask to {k_ask} (full bank query)")

        cfg = SimConfig(
            drift_mode="real",
            rollout_mode="models_and_items",
            n_rounds=int(cfg_raw.get("rollout_n_rounds", cfg_raw.get("n_rounds", 10))),
            bank_budget=int(cfg_raw.get("bank_budget", 400)),
            k_ask=k_ask,
            n_exp=int(cfg_raw.get("n_exp", 8)),
            kappa=int(cfg_raw.get("kappa", 15)),
            tau=float(cfg_raw.get("tau", 1.0)),
            q=float(cfg_raw.get("q", 0.05)),
            p0=float(cfg_raw.get("p0", 2.0)),
            p_star=float(cfg_raw.get("p_star", 0.25)),
            retire_v=retire_v,
            n_anchors=int(cfg_raw.get("n_anchors", 15)),
            mu0=float(cfg_raw.get("mu0", 0.0)),
            r=float(cfg_raw.get("r", 0.12)),
            s=float(theta_star["theta_star"].std(ddof=1)),
            n_synthetic=int(cfg_raw.get("n_synthetic", 10)),
            seed=int(cfg_raw.get("seed", 0)),
            equal_bank_size=False,
            exclude_models=exclude,
            baseline_models=tuple(cfg_raw.get("baseline_models", ["dev_pca", "dev_random"])),
            start_at_first_model=bool(cfg_raw.get("start_at_first_model", True)),
            retire_cap=bool(cfg_raw.get("retire_cap", True)),
            field_window_top_k=cfg_raw.get("field_window_top_k"),
            field_window_top_frac=cfg_raw.get("field_window_top_frac"),
            retire_mode=str(cfg_raw.get("retire_mode", "v_threshold")),
            retire_top_model_frac=float(cfg_raw.get("retire_top_model_frac", 0.5)),
            retire_bottom_pct=float(cfg_raw.get("retire_bottom_pct", 0.05)),
            seed_full_pool=bool(cfg_raw.get("seed_full_pool", False)),
            geneformer_upgrade=bool(cfg_raw.get("geneformer_upgrade", False)),
            geneformer_v1_model=str(cfg_raw.get("geneformer_v1_model", "geneformer")),
            geneformer_v2_model=str(cfg_raw.get("geneformer_v2_model", "geneformer_v2")),
        )
        fw_frac = cfg.field_window_top_frac
        fw_k = cfg.field_window_top_k
        if fw_frac:
            print(f"[rollout] field_window_top_frac={fw_frac} (top {100*float(fw_frac):.0f}% of models)")
        elif fw_k:
            print(f"[rollout] field_window_top_k={fw_k!r} (top {int(fw_k)} models)")
        else:
            print("[rollout] field window: full active pool")
        print(
            f"[rollout] retire_mode={cfg.retire_mode}, "
            f"retire_cap={cfg.retire_cap}, k_ask={cfg.k_ask}, "
            f"seed_full_pool={cfg.seed_full_pool}, kappa={cfg.kappa}"
        )
        if cfg.geneformer_upgrade:
            print(
                f"[rollout] geneformer upgrade: {cfg.geneformer_v1_model} → {cfg.geneformer_v2_model}"
            )
        print("[rollout] running Scenario B (models + cells, full maintenance)...")
        metrics, model_rankings, geneformer_upgrade = run_arm(
            "full", sim_resp, theta_star, diff_star, cfg
        )
        metrics["scenario"] = "models_and_items"
        metrics.to_csv(metrics_path, index=False)
        model_rankings.to_csv(rankings_path, index=False)
        if not geneformer_upgrade.empty:
            geneformer_upgrade.to_csv(upgrade_path, index=False)
            plot_geneformer_upgrade(geneformer_upgrade, upgrade_fig_path)
            r0 = geneformer_upgrade.iloc[0]
            print(
                f"  geneformer upgrade round {int(r0['round'])}: "
                f"Δ_oracle={r0['delta_oracle']:.3f}, "
                f"Δ_disjoint={r0['delta_disjoint']:.3f}, "
                f"Δ_anchor={r0['delta_anchor']:.3f}, "
                f"Δ_full={r0['delta_full']:.3f}"
            )
        print(f"  rounds={metrics['round'].nunique()}, {metrics['round_date'].iloc[0]} → {metrics['round_date'].iloc[-1]}")

    write_rollout_spec(
        metrics,
        spec_path,
        cfg={**cfg_raw, "retire_v": retire_v},
        theta_star=theta_star if not theta_star.empty else pd.read_csv(out_dir / "theta_star_rollout.csv"),
        exclude_models=exclude,
    )
    generate_rollout_report(
        metrics,
        pdf_path,
        exclude_models=exclude,
        retire_v=retire_v,
        retire_cap=bool(cfg_raw.get("retire_cap", True)),
        model_rankings=model_rankings,
    )
    if geneformer_upgrade.empty and upgrade_path.exists():
        geneformer_upgrade = pd.read_csv(upgrade_path)
    if not geneformer_upgrade.empty:
        plot_geneformer_upgrade(geneformer_upgrade, upgrade_fig_path)

    print(f"[rollout] done:")
    print(f"  {metrics_path}")
    print(f"  {rankings_path}")
    print(f"  {spec_path}")
    print(f"  {pdf_path}")
    if upgrade_path.exists():
        print(f"  {upgrade_path}")
    if upgrade_fig_path.exists():
        print(f"  {upgrade_fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
