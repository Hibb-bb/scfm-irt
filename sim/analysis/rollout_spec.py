"""Machine-readable simulation spec + results for downstream Word/doc generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


def write_rollout_spec(
    metrics: pd.DataFrame,
    out_path: Path,
    *,
    cfg: dict,
    theta_star: pd.DataFrame,
    exclude_models: tuple[str, ...],
) -> None:
    """Write detailed markdown spec; primary data is metrics CSV (no figures required)."""
    m = metrics.sort_values("round")
    dates_cfg = yaml.safe_load((REPO / "sim/data/release_dates.yaml").read_text())

    retire_mode = cfg.get("retire_mode", "v_threshold")
    if retire_mode == "bottom_pct_top_half":
        retire_lines = [
            "6. **Retire:**",
            f"   - **PI rule:** score each non-anchor cell by mean Fisher information "
            f"among the top `{cfg.get('retire_top_model_frac', 0.5)}` fraction of models (by fitted θ); "
            f"drop the lowest `{100 * float(cfg.get('retire_bottom_pct', 0.05)):.0f}%` each round.",
            "   - Anchors are never retired.",
        ]
        if cfg.get("retire_cap", True):
            retire_lines.append(
                "   - **Cap rule (if enabled):** if `|B| > bank_budget`, drop lowest-value non-anchors until at cap."
            )
        retire_plain = (
            "**Retirement (plain language):** remove the worst-discriminating cells for the "
            "current top-half of models; each round culls the bottom X% by that score."
        )
    else:
        retire_lines = [
            "6. **Retire:**",
            "   - **Value rule:** drop non-anchor cell if expected Fisher information `v_t(b) < retire_v`.",
        ]
        if cfg.get("retire_cap", True):
            retire_lines.append(
                "   - **Cap rule (if enabled):** if `|B| > bank_budget`, drop lowest-value non-anchors until at cap."
            )
        retire_plain = (
            "**Retirement (plain language):** remove a cell when it no longer discriminates between models "
            "(expected information below threshold); optionally enforce a fixed bank size cap."
        )

    lines: list[str] = [
        "# Online benchmark maintenance simulation — specification & results",
        "",
        "This document describes a single-scenario rollout simulation intended for",
        "downstream formatting (e.g. Word). **Primary numerical results:** `metrics.csv`.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Simulate **Algorithm 1 (full maintenance)** on the real lung atlas benchmark as:",
        "- new **embedding models** enter by publication/release date, and",
        "- new **cells** enter by sub-study proxy release dates.",
        "",
        "Drift mode: **real** (oracle abilities from joint Rasch fit; no synthetic field rise).",
        "",
        "---",
        "",
        "## 2. Scenario (B only)",
        "",
        "**Rollout mode:** `models_and_items`",
        "",
        "- **Models:** non-baseline embedders activate when `model_release_date <= round_date`.",
        "  Baseline models (`dev_pca`, `dev_random`) are excluded from the active field.",
        "- **Cells:** a cell enters the available pool when its sub-study proxy `release_date <= round_date`.",
        "- **Simulation clock starts** at the first embedder release (geneformer), not at the first cell date.",
        "- **Round 0 bank:** "
        + (
            "all cells in the pool are seeded at once (`seed_full_pool=true`)."
            if cfg.get("seed_full_pool")
            else "seeded to `bank_budget` cells drawn from the pool accumulated by the start date."
        ),
        "",
        "---",
        "",
        "## 3. Maintenance algorithm (full arm)",
        "",
        "Each round:",
        "",
        f"1. **Ask:** each active model answers anchors + routed trusted/provisional items "
        f"(`k_ask={cfg.get('k_ask', 40)}`, `n_exp={cfg.get('n_exp', 8)}`; "
        f"when `k_ask >= |bank|`, all bank items are queried).",
        "2. **Fit abilities:** MLE on anchor + trusted answers only (provisional excluded).",
        "3. **Recalibrate:** Kalman filter update on difficulties for all asked non-anchor items.",
        "4. **Promote:** provisional → trusted when uncertainty `P <= p_star`.",
        "5. **Acquire:** up to `kappa` new cells/round — hardest real cells from pool, then synthetic frontier items.",
    ]
    lines.extend(retire_lines)
    lines.extend(["", retire_plain, "", "---", "", "## 4. Configuration", "", "| Parameter | Value | Meaning |", "|-----------|-------|---------|"])
    param_rows = [
        ("rollout_n_rounds", cfg.get("rollout_n_rounds", cfg.get("n_rounds")), "Number of simulation rounds"),
        ("bank_budget", cfg.get("bank_budget"), "Initial bank seed size (cap only if retire_cap)"),
        ("seed_full_pool", cfg.get("seed_full_pool", False), "Seed entire pool at round 0"),
        ("rollout_datasets", cfg.get("rollout_datasets"), "Datasets included in rollout"),
        ("kappa", cfg.get("kappa"), "Cells acquired per round"),
        ("k_ask", cfg.get("k_ask"), "Items asked per model per round"),
        ("n_exp", cfg.get("n_exp"), "Provisional items in each model's query"),
        ("n_anchors", cfg.get("n_anchors"), "Fixed anchor cells (difficulty pinned)"),
        ("retire_v", cfg.get("retire_v"), "Retire if v_t(b) below this (v_threshold mode)"),
        ("retire_mode", retire_mode, "Retirement policy"),
        ("retire_top_model_frac", cfg.get("retire_top_model_frac"), "Top model fraction for PI retirement score"),
        ("retire_bottom_pct", cfg.get("retire_bottom_pct"), "Bottom fraction of cells retired each round (PI mode)"),
        ("retire_cap", cfg.get("retire_cap", True), "Enforce bank_budget via cap drops"),
        ("field_window_top_frac", cfg.get("field_window_top_frac"), "Top fraction of models for μ/σ field window"),
        ("field_window_top_k", cfg.get("field_window_top_k"), "Top-k models for μ/σ field window (if no fraction)"),
        ("p_star", cfg.get("p_star"), "Promote provisional when P <= this"),
        ("start_at_first_model", cfg.get("start_at_first_model"), "Start timeline at first embedder"),
        ("baseline_models", cfg.get("baseline_models"), "Excluded from active model field"),
        ("lung_run", cfg.get("lung_run"), "Ranking run for responses"),
        ("seed", cfg.get("seed"), "RNG seed"),
    ]
    for key, val, desc in param_rows:
        lines.append(f"| `{key}` | {val} | {desc} |")

    lines.extend(
        [
            "",
            f"**Excluded models:** {', '.join(exclude_models) or 'none'}",
            "",
            "---",
            "",
            "## 5. Models in oracle fit",
            "",
            "| model_id | theta_star | theta_sd | release_date |",
            "|----------|------------|----------|--------------|",
        ]
    )
    model_dates = {k: v["release_date"] for k, v in dates_cfg.get("models", {}).items()}
    for _, row in theta_star.sort_values("theta_star", ascending=False).iterrows():
        mid = row["model_id"]
        rd = model_dates.get(mid, "n/a")
        lines.append(
            f"| {mid} | {row['theta_star']:.4f} | {row['theta_sd']:.4f} | {rd} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Output files",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `metrics.csv` | **Main results** — one row per round (full maintenance arm) |",
            "| `SIMULATION_SPEC.md` | This file |",
            "| `rollout_report.pdf` | Optional visual report (single scenario) |",
            "",
            "---",
            "",
            "## 7. metrics.csv column dictionary",
            "",
            "| Column | Description |",
            "|--------|-------------|",
            "| `round` | Round index (0-based) |",
            "| `round_date` | Simulated calendar date for this round |",
            "| `n_active_models` | Embedders active this round |",
            "| `n_new_models` | Models newly activated this round |",
            "| `new_models` | Comma-separated model IDs activated |",
            "| `n_items_available` | Cells in the release pool |",
            "| `n_items_new` | Cells newly entering pool this round |",
            "| `bank_size` | Cells currently in the maintained bank |",
            "| `n_acquired` | Cells added to bank this round |",
            "| `n_acquired_pool` | Real cells pulled from pool |",
            "| `n_acquired_synth` | Synthetic provisional cells created |",
            "| `n_retired` | Total cells removed this round |",
            "| `n_retired_value` | Removed for low v_t(b) |",
            "| `n_retired_cap` | Removed to enforce bank_budget |",
            "| `n_promoted` | Provisional → trusted promotions |",
            "| `mu_oracle` | Mean oracle θ* over active embedders |",
            "| `mu_hat` | Estimated field mean from fitted abilities |",
            "| `s_hat` | Estimated field spread |",
            "| `bank_value` | Sum of item values v_t(b) over bank |",
            "| `bank_value_per_item` | bank_value / bank_size |",
            "| `ranking_spearman` | Spearman ρ between θ̂ and oracle (active models) |",
            "| `ranking_spearman_incumbent` | ρ among models active previous round |",
            "| `seed_mean_value` | Mean v_t(b_seed) for initial bank difficulties |",
            "",
            "---",
            "",
            "## 8. Per-round results (from metrics.csv)",
            "",
        ]
    )

    cols = [
        "round",
        "round_date",
        "n_active_models",
        "n_new_models",
        "new_models",
        "n_items_available",
        "n_items_new",
        "bank_size",
        "n_acquired",
        "n_retired",
        "n_retired_value",
        "n_retired_cap",
        "mu_oracle",
        "ranking_spearman",
    ]
    display = m[cols].copy()
    display["round_date"] = display["round_date"].astype(str).str[:10]
    for c in ["mu_oracle", "ranking_spearman"]:
        if c in display.columns:
            display[c] = display[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in display.iterrows():
        cells = [str(row[c]).replace("|", "/").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 9. Summary statistics",
            "",
        ]
    )
    fr = m.iloc[-1]
    lines.extend(
        [
            f"- Simulation span: `{m['round_date'].iloc[0]}` → `{m['round_date'].iloc[-1]}` ({len(m)} rounds)",
            f"- Final active embedders: {int(fr['n_active_models'])}",
            f"- Final cell pool size: {int(fr['n_items_available'])}",
            f"- Final bank size: {int(fr['bank_size'])} (budget {int(fr['bank_budget'])})",
            f"- Total cells retired: {int(m['n_retired'].sum())} (value: {int(m['n_retired_value'].sum())}, cap: {int(m['n_retired_cap'].sum())})",
            f"- Total cells acquired: {int(m['n_acquired'].sum())}",
            f"- Field mean μ: {fr['mu_oracle']:.3f} (started {m['mu_oracle'].iloc[0]:.3f})",
        ]
    )
    rho = fr["ranking_spearman"]
    lines.append(
        f"- Final ranking Spearman ρ: {rho:.3f}" if pd.notna(rho) else "- Final ranking Spearman ρ: n/a (<3 models or insufficient overlap)"
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 10. Model launch rounds",
            "",
        ]
    )
    launches = m.loc[m["n_new_models"] > 0, ["round", "round_date", "new_models"]]
    if launches.empty:
        lines.append("(none recorded)")
    else:
        for _, row in launches.iterrows():
            lines.append(
                f"- Round {int(row['round'])} ({str(row['round_date'])[:10]}): {row['new_models']}"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 11. Notes for document generation",
            "",
            "- Do **not** compare to Scenario A (`models_only`); this run uses Scenario B only.",
            "- If all cells are in the pool by the sim start date, `n_items_new` is 0 after round 0;",
            "  model rollout is still the main temporal signal.",
            "- Ranking ρ requires ≥3 active embedders; early rounds may show `n/a`.",
            "- cellplm excluded from fit by default ( suspect calibration).",
            "",
            "## 12. Regenerate",
            "",
            "```bash",
            "PYTHONPATH=. python sim/run_rollout_report.py --exclude-models cellplm",
            "PYTHONPATH=. python sim/run_rollout_report.py --report-only  # from existing metrics.csv",
            "```",
            "",
        ]
    )

    out_path.write_text("\n".join(lines))
