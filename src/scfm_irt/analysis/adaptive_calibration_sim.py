"""Calibrated-bank simulation: difficulty stability + adaptive ability for a new model.

Uses a completed multi-model IRT run as reference.

1. **Difficulty stability (leave-one-model-out)**
   Re-fit item difficulties on P−1 calibration models; compare to the full-fit bank.

2. **New-model adaptive ability estimation**
   Treat one held-out model as "new": fix difficulties at the calibration bank,
   administer a small item budget (random vs adaptive), MLE for θ only, and
   measure |θ̂ − θ_ref|.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.loo_calibration import align_difficulty, loo_calibrate_banks
from scfm_irt.analysis.pure_item_budget_sim import (
    select_adaptive_5050,
    select_random,
    select_stratified,
)
from scfm_irt.analysis.subsample_sim import mle_theta_1pl_fixed_b
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import to_matrix


def _default_n_grid(n_total: int) -> list[int]:
    if n_total <= 10:
        return list(range(2, n_total + 1))
    cands = sorted(
        {
            5,
            10,
            15,
            20,
            25,
            30,
            40,
            50,
            75,
            100,
            125,
            150,
            200,
            250,
            300,
            400,
            n_total,
        }
    )
    return [n for n in cands if 2 <= n <= n_total]


def loo_difficulty_stability(
    resp: pd.DataFrame,
    fit: IRTFit,
    *,
    n_steps: int = 400,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, IRTFit]]:
    """Leave-one-model-out IRT fits; compare each calibration bank to the full fit."""
    stability, loo_fits = loo_calibrate_banks(resp, fit, n_steps=n_steps, seed=seed)
    return stability.drop(columns=["calibration_models"], errors="ignore"), loo_fits


def run_new_model_adaptive_sim(
    resp: pd.DataFrame,
    fit: IRTFit,
    loo_fits: dict[str, IRTFit],
    *,
    n_replicates: int = 50,
    n_items_grid: list[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a new model with fixed calibration difficulties + small item budgets."""
    Y, model_ids, item_ids = to_matrix(resp)
    b_full = fit.difficulty_mean.astype(float)
    n_total = len(item_ids)
    grid = n_items_grid or _default_n_grid(n_total)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    strategies = ("random", "stratified", "adaptive_5050")
    b_sources = ("loo_calibrated", "full_oracle")

    for holdout_idx, holdout_id in enumerate(model_ids):
        y_holdout = Y[holdout_idx, :].astype(float)
        b_loo = loo_fits[holdout_id].difficulty_mean.astype(float)
        b_loo_aligned = align_difficulty(b_loo, b_full)
        # Reference θ: full-bank MLE with difficulties that exclude this model.
        theta_true = mle_theta_1pl_fixed_b(y_holdout, b_loo_aligned)

        for b_source in b_sources:
            b_bank = b_full if b_source == "full_oracle" else b_loo_aligned

            for n_items in grid:
                for rep in range(n_replicates):
                    for strat in strategies:
                        if strat == "random":
                            idx = select_random(b_bank, n_items, rng)
                        elif strat == "stratified":
                            idx = select_stratified(b_bank, n_items, rng)
                        elif strat == "adaptive_5050":
                            idx, _ = select_adaptive_5050(b_bank, n_items, y_holdout, rng)
                        else:
                            raise ValueError(strat)

                        theta_hat = mle_theta_1pl_fixed_b(y_holdout[idx], b_bank[idx])
                        rows.append(
                            {
                                "holdout_model": holdout_id,
                                "b_source": b_source,
                                "strategy": strat,
                                "n_items": n_items,
                                "pct_bank": 100.0 * n_items / n_total,
                                "replicate": rep,
                                "theta_ref": theta_true,
                                "theta_hat": theta_hat,
                                "ability_error": abs(theta_hat - theta_true),
                            }
                        )

    return pd.DataFrame(rows)


def summarize_loo(stability: pd.DataFrame) -> pd.DataFrame:
    return stability.copy()


def summarize_adaptive(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim.groupby(["b_source", "strategy", "n_items"], as_index=False)
        .agg(
            pct_bank=("pct_bank", "first"),
            ability_error_mean=("ability_error", "mean"),
            ability_error_std=("ability_error", "std"),
            ability_error_median=("ability_error", "median"),
            n_obs=("ability_error", "size"),
        )
        .sort_values(["b_source", "strategy", "n_items"])
    )


def write_adaptive_calibration_report(
    run_dir: str | Path,
    fit: IRTFit,
    stability: pd.DataFrame,
    sim: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    source_run_name: str,
) -> Path:
    out = Path(run_dir) / "adaptive_calibration_sim"
    out.mkdir(parents=True, exist_ok=True)

    stability.to_csv(out / "difficulty_stability_loo.csv", index=False)
    sim.to_csv(out / "new_model_adaptive_raw.csv", index=False)
    summary.to_csv(out / "new_model_adaptive_summary.csv", index=False)

    _plot_difficulty_stability(stability, out / "difficulty_stability_loo.png")
    _plot_adaptive_errors(summary, out / "new_model_ability_error.png", b_source="loo_calibrated")
    _plot_adaptive_ratio(summary, out / "adaptive_gain_vs_random.png", b_source="loo_calibrated")

    stab_mean = stability.mean(numeric_only=True)
    report = out / "report.md"
    lines = [
        "# Adaptive calibration simulation",
        "",
        f"- source IRT run: `{source_run_name}`",
        f"- models: **{', '.join(fit.model_ids)}**",
        f"- items (bank size): **{len(fit.item_ids)}**",
        "",
        "## 1. Are difficulty scores stable?",
        "",
        "Leave-one-model-out: calibrate difficulties on P−1 models, compare to the full-fit bank.",
        "High Pearson/Spearman r ⇒ the item bank transfers when a new model joins the pool.",
        "",
        _df_md(stability.round(4)),
        "",
        f"**Mean LOO difficulty correlation (Pearson): {stab_mean.get('difficulty_pearson_r', float('nan')):.3f}**",
        f"**Mean LOO difficulty RMSE: {stab_mean.get('difficulty_rmse', float('nan')):.3f}**",
        "",
        "## 2. New-model adaptive ability estimation",
        "",
        "Each model is held out as *new*. Difficulties fixed at the LOO calibration bank "
        "(fit without that model); reference θ is the full-bank MLE under those difficulties.",
        "only θ is estimated (MLE, 1PL). Compare random vs stratified vs adaptive item budgets.",
        "",
        "Oracle row (`full_oracle`) uses full-fit difficulties — optimistic upper bound.",
        "",
        _df_md(
            summary.loc[summary["b_source"] == "loo_calibrated"]
            .pivot_table(
                index="n_items",
                columns="strategy",
                values="ability_error_mean",
                aggfunc="first",
            )
            .round(4)
            .reset_index()
        ),
        "",
        "## Figures",
        "",
        "- `difficulty_stability_loo.png` — LOO vs full difficulty (shift-aligned)",
        "- `new_model_ability_error.png` — |θ̂−θ| vs % of bank (LOO-calibrated b)",
        "- `adaptive_gain_vs_random.png` — random / adaptive error (>1 ⇒ adaptive wins)",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def _df_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in cols]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def _plot_difficulty_stability(stability: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(stability))
    ax.bar(x, stability["difficulty_pearson_r"], color="#1f4e79", alpha=0.85)
    ax.axhline(0.9, color="#999", ls="--", lw=1, label="r = 0.9")
    ax.set_xticks(x)
    ax.set_xticklabels(stability["holdout_model"], rotation=30, ha="right")
    ax.set_ylabel("Pearson r (LOO b vs full b, shift-aligned)")
    ax.set_title("Difficulty bank stability (leave-one-model-out)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_adaptive_errors(summary: pd.DataFrame, path: Path, *, b_source: str) -> None:
    import matplotlib.pyplot as plt

    sub = summary.loc[summary["b_source"] == b_source].copy()
    styles = {
        "random": ("#888888", 0.18),
        "adaptive_5050": ("#2ca02c", 0.15),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for strat in ("random", "adaptive_5050"):
        s = sub.loc[sub["strategy"] == strat].sort_values("pct_bank")
        if s.empty:
            continue
        color, band_alpha = styles[strat]
        x = s["pct_bank"].to_numpy()
        y = s["ability_error_mean"].to_numpy()
        sd = s["ability_error_std"].fillna(0).to_numpy()
        ax.fill_between(x, np.maximum(y - sd, 0), y + sd, color=color, alpha=band_alpha, linewidth=0)
        ax.plot(x, y, "-", color=color, lw=2, label=strat.replace("_", " "))

    ax.set_xlabel("% of item bank administered")
    ax.set_ylabel("|θ̂ − θ_ref| for held-out (new) model")
    ax.set_title(f"New-model ability error ({b_source.replace('_', ' ')})")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_adaptive_ratio(summary: pd.DataFrame, path: Path, *, b_source: str) -> None:
    import matplotlib.pyplot as plt

    sub = summary.loc[summary["b_source"] == b_source]
    rand = sub.loc[sub["strategy"] == "random"].set_index("pct_bank")
    adapt = sub.loc[sub["strategy"] == "adaptive_5050"].set_index("pct_bank")
    common = rand.index.intersection(adapt.index)
    ratio = (rand.loc[common, "ability_error_mean"] / adapt.loc[common, "ability_error_mean"]).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(1.0, color="#999", ls="--", lw=1)
    ax.plot(ratio["pct_bank"], ratio["ability_error_mean"], "o-", color="#2ca02c", lw=2)
    ax.set_xlabel("% of item bank administered")
    ax.set_ylabel("random error / adaptive error")
    ax.set_title("Adaptive advantage for new model (>1 = adaptive better)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
