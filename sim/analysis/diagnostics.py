"""Controlled-comparison diagnostics: drift slope, staleness, summary tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def empirical_drift_rate(metrics: pd.DataFrame) -> pd.DataFrame:
    """OLS slope of mu_oracle vs round (one row per arm)."""
    rows: list[dict] = []
    for arm, sub in metrics.groupby("arm"):
        sub = sub.sort_values("round")
        x = sub["round"].to_numpy(dtype=float)
        y = sub["mu_oracle"].to_numpy(dtype=float)
        if len(x) < 2:
            slope = float("nan")
        else:
            slope = float(np.polyfit(x, y, 1)[0])
        rows.append({"arm": arm, "empirical_r": slope, "mu_oracle_start": y[0], "mu_oracle_end": y[-1]})
    return pd.DataFrame(rows)


def final_round_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Size-normalized final-round comparison across arms."""
    final_round = metrics["round"].max()
    rows: list[dict] = []
    for arm in ("fixed", "recalibrate", "full"):
        sub = metrics.loc[(metrics["arm"] == arm) & (metrics["round"] == final_round)].iloc[0]
        rows.append(
            {
                "arm": arm,
                "round": int(final_round),
                "bank_size": int(sub["bank_size"]),
                "bank_value": float(sub["bank_value"]),
                "bank_value_per_item": float(sub["bank_value_per_item"]),
                "ranking_spearman": float(sub["ranking_spearman"]),
                "seed_mean_value": float(sub["seed_mean_value"]),
                "n_rankable_models": int(sub["n_rankable_models"]),
            }
        )
    return pd.DataFrame(rows)


def write_diagnostics_report(metrics: pd.DataFrame, out_dir: Path) -> None:
    drift = empirical_drift_rate(metrics)
    summary = final_round_summary(metrics)
    drift.to_csv(out_dir / "diagnostics_drift.csv", index=False)
    summary.to_csv(out_dir / "diagnostics_final_summary.csv", index=False)

    lines = [
        "# Simulation diagnostics",
        "",
        "## Empirical field drift (μ_oracle slope vs round)",
        "",
        "```",
        drift.to_string(index=False),
        "```",
        "",
        "If `empirical_r ≈ 0`, the real model rollout has little capability drift —",
        "fixed ≥ full on ranking is expected; the theorem's decay regime is not active.",
        "",
        "## Final round (controlled metrics)",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "Compare `bank_value_per_item` (not raw ΣV) across equal-sized banks.",
        "Compare `ranking_spearman` (size-free) for the paper's claim.",
        "Compare `seed_mean_value`: do round-0 items go stale under the final field?",
        "",
    ]
    (out_dir / "diagnostics.md").write_text("\n".join(lines))
