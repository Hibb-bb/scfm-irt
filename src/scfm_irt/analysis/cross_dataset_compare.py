"""Compare IRT ability vs raw probe accuracy within and across runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.report import _accuracy_table, _plot_ability_and_accuracy_forest


def load_run_metrics(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    ability = pd.read_csv(run_dir / "ability_ranking.csv")
    resp = pd.read_parquet(run_dir / "responses.parquet")
    accuracy = _accuracy_table(resp)

    merged = ability.merge(
        accuracy[["model_id", "accuracy", "sd", "n_items"]],
        on="model_id",
        suffixes=("_irt", ""),
    )
    merged["run"] = run_dir.name
    merged["rank_irt"] = merged["ability"].rank(ascending=False, method="min").astype(int)
    merged["rank_accuracy"] = merged["accuracy"].rank(ascending=False, method="min").astype(int)
    merged["rank_shift"] = merged["rank_accuracy"] - merged["rank_irt"]
    return merged


def write_cross_dataset_comparison(
    run_dirs: dict[str, Path],
    out_dir: str | Path,
) -> Path:
    """Emit tables + figures comparing IRT θ against raw accuracy across runs."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = [load_run_metrics(path) for path in run_dirs.values()]
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out / "irt_vs_accuracy_table.csv", index=False)

    # --- scatter: IRT θ vs raw accuracy, one point per (run, model) ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    run_labels = {
        "ranking_pbmc3k_v2": "PBMC3k",
        "ranking_immune_human_v1": "immune_human",
        "ranking_lung_atlas_v1": "lung_atlas",
        "ranking_oasis_v1": "oasis",
        "ranking_human_thymus_v1": "human_thymus",
    }
    palette = ["#1f4e79", "#c44e52", "#2ca02c", "#9467bd", "#ff7f0e"]
    colors = {
        path.name: palette[i % len(palette)]
        for i, path in enumerate(run_dirs.values())
    }
    for run_name in table["run"].unique():
        colors.setdefault(run_name, palette[len(colors) % len(palette)])
    markers = {"uce": "o", "dev_pca": "s", "scgpt": "^", "geneformer": "D", "scimilarity": "P"}

    for run_name, grp in table.groupby("run"):
        for _, row in grp.iterrows():
            ax.scatter(
                row["accuracy"],
                row["ability"],
                s=70,
                c=colors.get(run_name, "#666666"),
                marker=markers.get(row["model_id"], "o"),
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
            short = run_labels.get(run_name, run_name.removeprefix("ranking_"))
            ax.annotate(
                f"{row['model_id']} ({short})",
                (row["accuracy"], row["ability"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.9,
            )

    ax.axhline(0.0, color="#bbbbbb", lw=1)
    ax.set_xlabel("raw probe accuracy")
    ax.set_ylabel("IRT ability (θ)")
    ax.set_title("IRT ability vs raw accuracy by dataset")
    legend_runs = [r for r in table["run"].unique() if r in colors]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[r],
            markersize=8,
            label=run_labels.get(r, r),
        )
        for r in legend_runs
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "irt_vs_accuracy_scatter.png", dpi=150)
    fig.savefig(out / "irt_vs_accuracy_scatter.svg")
    plt.close(fig)

    # --- paired bars: rank under IRT vs rank under raw accuracy ---
    fig, axes = plt.subplots(1, len(run_dirs), figsize=(4.2 * len(run_dirs), 4.2), sharey=True)
    if len(run_dirs) == 1:
        axes = [axes]

    for ax, (label, run_path) in zip(axes, run_dirs.items()):
        run_name = run_path.name
        grp = table.loc[table["run"] == run_name].sort_values("rank_irt")
        y = np.arange(len(grp))
        width = 0.35
        ax.barh(y - width / 2, grp["rank_irt"], height=width, color="#1f4e79", label="IRT rank")
        ax.barh(y + width / 2, grp["rank_accuracy"], height=width, color="#888888", label="accuracy rank")
        ax.set_yticks(y)
        ax.set_yticklabels(grp["model_id"])
        ax.invert_yaxis()
        ax.set_xlabel("rank (1 = best)")
        ax.set_title(label)
        ax.set_xlim(0.5, len(grp) + 0.5)
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle("Ranking under IRT vs raw accuracy", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "irt_vs_accuracy_ranks.png", dpi=150)
    fig.savefig(out / "irt_vs_accuracy_ranks.svg")
    plt.close(fig)

    report = out / "report.md"
    lines = [
        "# IRT vs raw accuracy comparison",
        "",
        "Compares latent IRT ability (θ) against plain probe accuracy on the same test items.",
        "",
        "## Summary table",
        "",
        _df_md(table.round(4)),
        "",
        "## Figures",
        "",
        "- `irt_vs_accuracy_scatter.png` — θ vs accuracy per model × dataset",
        "- `irt_vs_accuracy_ranks.png` — rank under IRT vs rank under raw accuracy",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def _df_md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])
