"""PDF + PNG report for single-scenario (B) rollout simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_FULL = "#2ca02c"


def _sort(metrics: pd.DataFrame) -> pd.DataFrame:
    if "arm" in metrics.columns:
        return metrics.loc[metrics["arm"] == "full"].sort_values("round").copy()
    return metrics.sort_values("round").copy()


def _style_axes(
    ax,
    *,
    title: str,
    xlabel: str = "Simulation round",
    ylabel: str | None = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    y = 0.88
    for line in lines:
        ax.text(0.05, y, line, transform=ax.transAxes, fontsize=11, va="top")
        y -= 0.04
    pdf.savefig(fig)
    plt.close(fig)


def plot_model_ranking(rankings: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of final-round estimated abilities, best at top."""
    out_path = Path(out_path)
    if rankings.empty:
        return

    final = rankings.loc[rankings["round"] == rankings["round"].max()].copy()
    final = final.sort_values("theta_hat", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(final) + 1.5)))
    y = range(len(final))
    ax.barh(
        y,
        final["theta_hat"],
        color=_FULL,
        alpha=0.85,
        height=0.6,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(final["model_id"])
    _style_axes(ax, title="Final model ranking", ylabel="Model", xlabel="Estimated ability (θ̂)")

    for i, (_, row) in enumerate(final.iterrows()):
        ax.text(
            row["theta_hat"] + 0.02,
            i,
            f"#{int(row['rank_hat'])}  ({row['theta_hat']:.2f})",
            va="center",
            fontsize=9,
        )

    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_rollout_report(
    metrics: pd.DataFrame,
    out_path: Path,
    *,
    exclude_models: tuple[str, ...] = (),
    retire_v: float = 0.01,
    retire_cap: bool = True,
    model_rankings: pd.DataFrame | None = None,
) -> None:
    """Single-scenario (B) PDF + PNG exports."""
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    m = _sort(metrics)

    mu_start, mu_end = float(m["mu_oracle"].iloc[0]), float(m["mu_oracle"].iloc[-1])
    ret_total = int(m["n_retired"].sum())
    ret_val = int(m["n_retired_value"].sum())
    rho = m["ranking_spearman"].iloc[-1]

    # PI summary PNG
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Benchmark maintenance rollout (Scenario B)", fontsize=13, fontweight="bold")

    axes[0, 0].plot(m["round"], m["mu_oracle"], "o-", color="k", lw=2)
    _style_axes(axes[0, 0], title="Oracle field drift", ylabel="Mean oracle ability (μ_oracle)")

    axes[0, 1].plot(m["round"], m["bank_size"], "o-", color=_FULL, lw=2)
    if retire_cap:
        axes[0, 1].axhline(m["bank_budget"].iloc[0], ls="--", color="#999", label="Budget")
        axes[0, 1].legend()
    _style_axes(axes[0, 1], title="Maintained cell bank", ylabel="Cells in bank")

    axes[1, 0].bar(m["round"], m["n_retired"], color="#e74c3c", alpha=0.8)
    _style_axes(axes[1, 0], title="Cell retirements", ylabel="Cells retired")

    axes[1, 1].plot(m["round"], m["ranking_spearman"], "o-", color=_FULL, lw=2)
    axes[1, 1].set_ylim(0, 1.05)
    _style_axes(axes[1, 1], title="Model ranking quality", ylabel="Spearman ρ")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "pi_executive_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Left-column panels only (ability + retirements)
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 8))
    axes[0].plot(m["round"], m["mu_oracle"], "o-", color="k", lw=2)
    _style_axes(axes[0], title="Oracle field drift", ylabel="Mean oracle ability (μ_oracle)")

    axes[1].bar(m["round"], m["n_retired"], color="#e74c3c", alpha=0.8)
    _style_axes(axes[1], title="Cell retirements", ylabel="Cells retired")

    fig.tight_layout()
    fig.savefig(out_dir / "pi_executive_summary_left.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m["round"], m["mu_oracle"], "o-", label="Oracle (μ_oracle)", lw=2)
    ax.plot(m["round"], m["mu_hat"], "s--", label="Estimated (μ̂)", lw=1.5, alpha=0.8)
    _style_axes(ax, title="Field drift", ylabel="Mean ability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "field_drift.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m["round"], m["bank_size"], "o-", color=_FULL, lw=2)
    if retire_cap:
        ax.axhline(m["bank_budget"].iloc[0], ls="--", color="#999", label="Budget")
        ax.legend()
    _style_axes(ax, title="Maintained cell bank", ylabel="Cells in bank")
    fig.tight_layout()
    fig.savefig(out_dir / "bank_size.png", dpi=150)
    plt.close(fig)

    if model_rankings is not None and not model_rankings.empty:
        plot_model_ranking(model_rankings, out_dir / "model_ranking.png")

    ds_plot = [
        ("bank_lung_atlas", "Lung"),
        ("bank_pbmc3k", "PBMC3k"),
        ("bank_immune_human", "Immune"),
    ]
    ds_cols = [(col, label) for col, label in ds_plot if col in m.columns]
    if ds_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        for col, label in ds_cols:
            ax.plot(m["round"], m[col], "o-", lw=2, label=label)
        _style_axes(ax, title="Bank survival by dataset", ylabel="Cells in bank")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dataset_survival.png", dpi=150)
        plt.close(fig)

    excl = ", ".join(exclude_models) if exclude_models else "(none)"
    intro = [
        "Scenario B: models + new cells enter over time",
        f"Full maintenance only | excluded: {excl}",
        f"Retire when v_t(b) < {retire_v}" + ("" if retire_cap else " | no bank cap"),
        f"μ_oracle: {mu_start:.2f} → {mu_end:.2f}",
        f"Total retirements: {ret_total} (value: {ret_val})",
        f"Final ρ: {rho:.3f}" if pd.notna(rho) else "Final ρ: n/a",
        "",
        "See SIMULATION_SPEC.md and metrics.csv for full details.",
    ]

    with PdfPages(out_path) as pdf:
        _text_page(pdf, "Rollout report (Scenario B)", intro)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("Benchmark maintenance rollout (Scenario B)", fontsize=13, fontweight="bold")

        axes[0, 0].plot(m["round"], m["mu_oracle"], "o-", color="k", lw=2)
        _style_axes(axes[0, 0], title="Oracle field drift", ylabel="Mean oracle ability (μ_oracle)")

        axes[0, 1].plot(m["round"], m["bank_size"], "o-", color=_FULL, lw=2)
        _style_axes(axes[0, 1], title="Maintained cell bank", ylabel="Cells in bank")

        axes[1, 0].bar(m["round"], m["n_retired"], color="#e74c3c")
        _style_axes(axes[1, 0], title="Cell retirements", ylabel="Cells retired")

        axes[1, 1].plot(m["round"], m["ranking_spearman"], "o-", color=_FULL)
        axes[1, 1].set_ylim(0, 1.05)
        _style_axes(axes[1, 1], title="Model ranking quality", ylabel="Spearman ρ")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)
