"""Figure for Geneformer v1→v2 upgrade case study (disjoint cell pools)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_METHODS = [
    ("delta_oracle", "Oracle Δθ", "#111111"),
    ("delta_disjoint", "Disjoint pools", "#2ca02c"),
    ("delta_anchor", "Anchors only", "#1f77b4"),
    ("delta_full", "Full bank", "#ff7f0e"),
]
_ERR_COLS = [("err_disjoint", "Disjoint"), ("err_anchor", "Anchors"), ("err_full", "Full bank")]


def plot_geneformer_upgrade(upgrade: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of estimated Δθ methods vs oracle at the v2 release round."""
    out_path = Path(out_path)
    if upgrade.empty:
        return

    row = upgrade.iloc[0]
    labels = [label for _, label, _ in _METHODS]
    vals = [float(row[col]) for col, _, _ in _METHODS]
    colors = [c for _, _, c in _METHODS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Geneformer upgrade (round {int(row['round'])}, {row['round_date']})",
        fontsize=12,
        fontweight="bold",
    )

    ax = axes[0]
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.88, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Estimated ability gain (Δθ)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title("Δθ by estimation method")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 if v >= 0 else -0.06),
            f"{v:.3f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )

    ax = axes[1]
    err_labels = [label for _, label in _ERR_COLS]
    err_vals = [float(row[col]) for col, _ in _ERR_COLS]
    ax.bar(range(len(err_labels)), err_vals, color="#e74c3c", alpha=0.75)
    ax.set_xticks(range(len(err_labels)))
    ax.set_xticklabels(err_labels)
    ax.set_ylabel("Error vs oracle (Δθ − Δθ_oracle)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title("Bias relative to oracle")
    ax.grid(True, axis="y", alpha=0.3)

    meta = (
        f"S1 legacy (GF1): {int(row['n_s1_legacy'])} cells  |  "
        f"S2 frontier (GF2): {int(row['n_s2_frontier'])} cells  |  "
        f"Anchors: {int(row['n_anchors'])}  |  "
        f"S1∩S2 overlap: {int(row['s1_s2_overlap'])}"
    )
    fig.text(0.5, 0.01, meta, ha="center", fontsize=9, color="#555")

    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
