"""Plot Fig 2 / Fig 3 style outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fig2(metrics: pd.DataFrame, out_path: Path) -> None:
    has_per_item = "bank_value_per_item" in metrics.columns
    ncols = 3 if has_per_item else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 2:
        axes = list(axes)
    colors = {"fixed": "#888", "recalibrate": "#1f77b4", "full": "#2ca02c"}
    for arm in ("fixed", "recalibrate", "full"):
        sub = metrics.loc[metrics["arm"] == arm].sort_values("round")
        color = colors[arm]
        axes[0].plot(sub["round"], sub["bank_value"], "o-", color=color, label=arm, lw=2)
        if has_per_item:
            axes[1].plot(sub["round"], sub["bank_value_per_item"], "o-", color=color, label=arm, lw=2)
        ax_r = axes[2] if has_per_item else axes[1]
        ax_r.plot(sub["round"], sub["ranking_spearman"], "o-", color=color, label=arm, lw=2)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Bank value V_t (total)")
    axes[0].set_title("Total benchmark value")
    if has_per_item:
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("V_t / |B|")
        axes[1].set_title("Per-item value (size-normalized)")
    ax_r = axes[2] if has_per_item else axes[1]
    ax_r.set_xlabel("Round")
    ax_r.set_ylabel("Spearman ρ (θ̂ vs ref)")
    ax_r.set_title("Ranking quality")
    ax_r.set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_diagnostics(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Field drift, seed-item staleness, and bank-size trajectories."""
    colors = {"fixed": "#888", "recalibrate": "#1f77b4", "full": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    sub0 = metrics.loc[metrics["arm"] == "fixed"].sort_values("round")
    axes[0, 0].plot(sub0["round"], sub0["mu_oracle"], "k--", lw=2, label="μ_oracle (fixed arm)")
    axes[0, 0].plot(sub0["round"], sub0["mu_hat"], "k-", lw=1.5, alpha=0.7, label="μ̂ (fixed arm)")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Ability scale")
    axes[0, 0].set_title("Field drift check (μ_oracle, μ̂)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.25)

    for arm in ("fixed", "recalibrate", "full"):
        sub = metrics.loc[metrics["arm"] == arm].sort_values("round")
        axes[0, 1].plot(
            sub["round"],
            sub["seed_mean_value"],
            "o-",
            color=colors[arm],
            label=arm,
            lw=2,
        )
    axes[0, 1].axhline(0.01, color="#999", ls=":", lw=1, label="retire_v=0.01")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Mean v_t(b_seed)")
    axes[0, 1].set_title("Seed-item staleness (initial difficulties)")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.25)

    for arm in ("fixed", "recalibrate", "full"):
        sub = metrics.loc[metrics["arm"] == arm].sort_values("round")
        axes[1, 0].plot(sub["round"], sub["bank_size"], "o-", color=colors[arm], label=arm, lw=2)
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("|B|")
    axes[1, 0].set_title("Bank size over rounds")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.25)

    for arm in ("fixed", "recalibrate", "full"):
        sub = metrics.loc[metrics["arm"] == arm].sort_values("round")
        axes[1, 1].plot(
            sub["round"],
            sub["n_rankable_models"],
            "o-",
            color=colors[arm],
            label=arm,
            lw=2,
        )
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("n models")
    axes[1, 1].set_title("Rankable models (need ≥3 for ρ)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_diagnostics.png", dpi=150)
    plt.close(fig)


def plot_bank_budget_sweep(summary: pd.DataFrame, out_path: Path) -> None:
    """Compare arms at final round across bank budgets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    arms = ("fixed", "recalibrate", "full")
    colors = {"fixed": "#888", "recalibrate": "#1f77b4", "full": "#2ca02c"}
    budgets = sorted(summary["bank_budget"].unique())
    x = np.arange(len(budgets))
    width = 0.25

    for i, arm in enumerate(arms):
        sub = summary.loc[summary["arm"] == arm].set_index("bank_budget").reindex(budgets)
        axes[0].bar(x + (i - 1) * width, sub["bank_value"], width, label=arm, color=colors[arm])
        axes[1].bar(x + (i - 1) * width, sub["ranking_spearman"], width, label=arm, color=colors[arm])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(budgets)
    axes[0].set_xlabel("Bank budget")
    axes[0].set_ylabel("Final bank value V_t")
    axes[0].set_title("Benchmark value (final round)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(budgets)
    axes[1].set_xlabel("Bank budget")
    axes[1].set_ylabel("Spearman ρ (θ̂ vs ref)")
    axes[1].set_title("Ranking quality (final round)")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig3(sweep: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r_val in sorted(sweep["r"].unique()):
        sub = sweep.loc[sweep["r"] == r_val].sort_values("a_over_r")
        axes[0].plot(sub["a_over_r"], sub["ranking_spearman_full"], "o-", label=f"r={r_val}", lw=2)
        if "ranking_spearman_fixed" in sweep.columns:
            axes[1].plot(
                sub["a_over_r"],
                sub["ranking_gap"],
                "o-",
                label=f"r={r_val}",
                lw=2,
            )
    axes[0].axvline(1.0, color="#999", ls="--", lw=1)
    axes[0].set_xlabel("a / r")
    axes[0].set_ylabel("Spearman ρ (full arm)")
    axes[0].set_title("Survival threshold — full maintenance")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.25)
    axes[1].axvline(1.0, color="#999", ls="--", lw=1)
    axes[1].axhline(0.0, color="#ccc", ls="-", lw=0.5)
    axes[1].set_xlabel("a / r")
    axes[1].set_ylabel("ρ_full − ρ_fixed")
    axes[1].set_title("Maintenance advantage (ranking gap)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
