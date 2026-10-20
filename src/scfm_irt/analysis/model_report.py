"""Single-model item analysis: accuracy vs difficulty, calibration, cell-type breakdown."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.irt.models import IRTFit, icc


def join_responses_items(resp: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Attach item difficulty to each response row."""
    out = resp.merge(items, left_on="cell_id", right_on="item_id", how="inner")
    if out.empty:
        raise ValueError("no rows after joining responses to item_params on cell_id/item_id")
    return out


def model_accuracy_by_difficulty_bin(
    resp: pd.DataFrame,
    items: pd.DataFrame,
    model_id: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bin test items by difficulty quantile; report accuracy per bin for one model."""
    sub = join_responses_items(
        resp.loc[resp["model_id"] == model_id].copy(),
        items,
    )
    sub["difficulty_bin"] = pd.qcut(
        sub["difficulty"],
        q=n_bins,
        duplicates="drop",
        labels=False,
    )
    rows = []
    for bin_id, grp in sub.groupby("difficulty_bin", observed=True):
        lo = float(grp["difficulty"].min())
        hi = float(grp["difficulty"].max())
        rows.append(
            {
                "model_id": model_id,
                "difficulty_bin": int(bin_id),
                "difficulty_lo": lo,
                "difficulty_hi": hi,
                "difficulty_mean": float(grp["difficulty"].mean()),
                "n_items": int(len(grp)),
                "accuracy": float(grp["correct"].mean()),
                "n_correct": int(grp["correct"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("difficulty_mean").reset_index(drop=True)


def model_accuracy_by_cell_type(
    resp: pd.DataFrame,
    items: pd.DataFrame,
    model_id: str,
    n_difficulty_bins: int = 3,
) -> pd.DataFrame:
    """Accuracy by cell type × difficulty tertile for one model."""
    sub = join_responses_items(
        resp.loc[resp["model_id"] == model_id].copy(),
        items,
    )
    sub["difficulty_bin"] = pd.qcut(
        sub["difficulty"],
        q=n_difficulty_bins,
        duplicates="drop",
        labels=["easy", "medium", "hard"][: n_difficulty_bins],
    )
    agg = (
        sub.groupby(["cell_type", "difficulty_bin"], observed=True)
        .agg(
            n_items=("correct", "size"),
            accuracy=("correct", "mean"),
            difficulty_mean=("difficulty", "mean"),
        )
        .reset_index()
    )
    agg.insert(0, "model_id", model_id)
    return agg.sort_values(["cell_type", "difficulty_mean"])


def irt_expected_vs_observed(
    fit: IRTFit,
    resp: pd.DataFrame,
    items: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    """Per-item observed correctness vs 1PL predicted P(correct | theta, b)."""
    if model_id not in fit.model_ids:
        raise KeyError(f"{model_id!r} not in fit.model_ids: {fit.model_ids}")
    p_idx = fit.model_ids.index(model_id)
    theta = float(fit.ability_mean[p_idx])

    sub = join_responses_items(
        resp.loc[resp["model_id"] == model_id].copy(),
        items,
    )
    b = sub["difficulty"].to_numpy(dtype=float)
    p_exp = icc(theta, b).ravel()
    sub = sub.assign(
        theta=theta,
        p_correct_expected=p_exp,
        residual=sub["correct"].to_numpy(dtype=float) - p_exp,
    )
    return sub[
        [
            "cell_id",
            "cell_type",
            "correct",
            "difficulty",
            "difficulty_sd",
            "p_correct_expected",
            "residual",
        ]
    ].sort_values("difficulty")


def calibration_by_expected_prob(
    resp: pd.DataFrame,
    items: pd.DataFrame,
    fit: IRTFit,
    model_id: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin items by IRT-expected P(correct); compare mean predicted vs observed rate."""
    detail = irt_expected_vs_observed(fit, resp, items, model_id)
    detail["prob_bin"] = pd.qcut(
        detail["p_correct_expected"],
        q=n_bins,
        duplicates="drop",
        labels=False,
    )
    rows = []
    for bin_id, grp in detail.groupby("prob_bin", observed=True):
        rows.append(
            {
                "model_id": model_id,
                "prob_bin": int(bin_id),
                "p_expected_mean": float(grp["p_correct_expected"].mean()),
                "p_observed": float(grp["correct"].mean()),
                "n_items": int(len(grp)),
                "calibration_error": float(grp["correct"].mean() - grp["p_correct_expected"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("p_expected_mean")


def hardest_and_easiest_items(
    resp: pd.DataFrame,
    items: pd.DataFrame,
    fit: IRTFit,
    model_id: str,
    n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Items the model got wrong/right among the hardest/easiest by IRT difficulty."""
    sub = irt_expected_vs_observed(fit, resp, items, model_id)
    sub = sub.sort_values("difficulty")
    wrong = sub.loc[sub["correct"] == 0].tail(n)
    right = sub.loc[sub["correct"] == 1].head(n)
    return wrong, right


def write_single_model_report(
    run_dir: str | Path,
    model_id: str,
    fit: IRTFit,
    resp: pd.DataFrame,
    *,
    source_run_name: str,
    n_difficulty_bins: int = 5,
) -> Path:
    """Write CSV tables, figures, and Markdown for one model's item-level performance."""
    out = Path(run_dir) / "model_reports" / model_id
    out.mkdir(parents=True, exist_ok=True)

    items = fit.item_table()
    sub = resp.loc[resp["model_id"] == model_id]
    if sub.empty:
        raise ValueError(f"no responses for model_id={model_id!r}")

    by_bin = model_accuracy_by_difficulty_bin(resp, items, model_id, n_bins=n_difficulty_bins)
    by_type = model_accuracy_by_cell_type(resp, items, model_id)
    calib = calibration_by_expected_prob(resp, items, fit, model_id)
    detail = irt_expected_vs_observed(fit, resp, items, model_id)
    wrong_hard, right_easy = hardest_and_easiest_items(resp, items, fit, model_id)

    by_bin.to_csv(out / "accuracy_by_difficulty_bin.csv", index=False)
    by_type.to_csv(out / "accuracy_by_cell_type_and_difficulty.csv", index=False)
    calib.to_csv(out / "calibration_by_expected_prob.csv", index=False)
    detail.to_csv(out / "item_expected_vs_observed.csv", index=False)
    wrong_hard.to_csv(out / "errors_on_hard_items.csv", index=False)
    right_easy.to_csv(out / "correct_on_easy_items.csv", index=False)

    ability_row = fit.ability_table()
    ability_row = ability_row.loc[ability_row["model_id"] == model_id].iloc[0]
    overall_acc = float(sub["correct"].mean())

    _plot_accuracy_by_difficulty(by_bin, out / "accuracy_by_difficulty.png", model_id)
    _plot_calibration(calib, out / "calibration.png", model_id)
    _plot_scatter_difficulty(detail, out / "accuracy_vs_difficulty.png", model_id)

    report = out / "report.md"
    lines = [
        f"# {model_id} — item difficulty report",
        "",
        f"- source IRT run: `{source_run_name}`",
        f"- test items: **{len(sub)}**",
        f"- overall probe accuracy: **{overall_acc:.1%}**",
        f"- IRT ability θ: **{ability_row['ability']:.3f}** "
        f"(95% CI {ability_row['lower']:.3f} … {ability_row['upper']:.3f})",
        "",
        "> Difficulty is relative to the respondent pool in the source IRT fit, not absolute biology.",
        "",
        "## Accuracy by item difficulty (IRT bins)",
        "",
        "Items binned by estimated difficulty (easy → hard). "
        "Under a well-calibrated strong model, accuracy should **decrease** as difficulty increases.",
        "",
        _df_to_md(by_bin),
        "",
        "## IRT calibration (expected vs observed)",
        "",
        "Items grouped by model-predicted P(correct) = σ(θ − b). "
        "Points near the diagonal mean the 1PL model matches probe outcomes for this respondent.",
        "",
        _df_to_md(calib.round(4)),
        "",
        "## Hardest items answered incorrectly",
        "",
        _df_to_md(
            wrong_hard[["cell_id", "cell_type", "difficulty", "correct", "p_correct_expected"]].round(4)
        ),
        "",
        "## Easiest items answered correctly",
        "",
        _df_to_md(
            right_easy[["cell_id", "cell_type", "difficulty", "correct", "p_correct_expected"]].round(4)
        ),
        "",
        "## Figures",
        "",
        "- `accuracy_by_difficulty.png` — accuracy vs difficulty bin",
        "- `calibration.png` — expected vs observed P(correct)",
        "- `accuracy_vs_difficulty.png` — per-item scatter + rolling mean",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def _plot_accuracy_by_difficulty(df: pd.DataFrame, path: Path, model_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = df["difficulty_mean"]
    y = df["accuracy"]
    ax.plot(x, y, "o-", color="#1f4e79", lw=2, markersize=8)
    for _, row in df.iterrows():
        ax.annotate(
            f"n={int(row['n_items'])}",
            (row["difficulty_mean"], row["accuracy"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("item difficulty (b)")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_id}: accuracy vs item difficulty")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration(df: pd.DataFrame, path: Path, model_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.plot(
        df["p_expected_mean"],
        df["p_observed"],
        "o-",
        color="#c44e52",
        lw=2,
        markersize=7,
    )
    ax.set_xlabel("expected P(correct) from IRT")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_id}: IRT calibration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_scatter_difficulty(detail: pd.DataFrame, path: Path, model_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(detail["difficulty"].to_numpy())
    x = detail["difficulty"].to_numpy()[order]
    y = detail["correct"].to_numpy()[order]
    ax.scatter(x, y + np.random.default_rng(0).uniform(-0.03, 0.03, size=len(x)), alpha=0.35, s=12)
    # rolling mean over sorted difficulties
    window = max(15, len(x) // 20)
    roll = pd.Series(y).rolling(window, center=True, min_periods=max(5, window // 3)).mean()
    ax.plot(x, roll, color="#1f4e79", lw=2, label=f"rolling accuracy (w={window})")
    ax.set_xlabel("item difficulty (b)")
    ax.set_ylabel("correct (jittered)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"{model_id}: per-item outcomes vs difficulty")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
