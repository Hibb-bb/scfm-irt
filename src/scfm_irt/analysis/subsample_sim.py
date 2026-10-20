"""Subsample simulation: how many items are needed to recover abilities/difficulties?

Uses a completed IRT run as reference (full test = ground truth), repeatedly subsamples
items, re-fits 1PL IRT, and measures recovery error vs subset size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.irt.fit_pyirt import _fit_mfvi
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import to_matrix


def mle_theta_1pl_fixed_b(y: np.ndarray, b: np.ndarray) -> float:
    """MLE ability under 1PL with fixed item difficulties."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if y.shape != b.shape:
        raise ValueError("y and b must match")

    def neg_ll(theta: float) -> float:
        p = 1.0 / (1.0 + np.exp(-np.clip(theta - b, -60, 60)))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(neg_ll, bounds=(-12.0, 12.0), method="bounded")
    return float(res.x)


def mle_theta_1pl_fixed_b_se(y: np.ndarray, b: np.ndarray, theta: float | None = None) -> float:
    """Asymptotic SE for MLE ability under 1PL with fixed difficulties."""
    y = np.asarray(y, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if theta is None:
        theta = mle_theta_1pl_fixed_b(y, b)
    p = 1.0 / (1.0 + np.exp(-np.clip(theta - b, -60, 60)))
    info = float(np.sum(p * (1.0 - p)))
    if info <= 1e-9:
        return float("nan")
    return float(1.0 / np.sqrt(info))


def _default_n_grid(n_total: int) -> list[int]:
    """Log-ish grid of subset sizes, always including n_total."""
    if n_total <= 10:
        return list(range(2, n_total + 1))
    candidates = sorted(
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
            350,
            400,
            450,
            500,
            n_total,
        }
    )
    return [n for n in candidates if 2 <= n <= n_total]


def run_subsample_simulation(
    resp: pd.DataFrame,
    fit: IRTFit,
    *,
    n_replicates: int = 40,
    n_steps: int = 400,
    seed: int = 0,
    n_items_grid: list[int] | None = None,
) -> pd.DataFrame:
    """Monte Carlo subsample simulation.

    Reference (ground truth): full-test variational fit `fit`.
    For each subset size n and replicate:
      - sample n items without replacement
      - re-fit 1PL IRT on the subsampled response matrix
      - record ability error (|θ̂−θ_ref|) and difficulty RMSE on subsampled items

    Also records **fixed-b ability MLE** error (θ MLE with difficulties fixed at reference),
    which is cheaper and assumes item params are known from the full calibration.
    """
    Y, model_ids, item_ids = to_matrix(resp)
    if list(model_ids) != list(fit.model_ids):
        raise ValueError("model_ids mismatch between responses and fit")
    if list(item_ids) != list(fit.item_ids):
        raise ValueError("item_ids mismatch between responses and fit")

    theta_ref = fit.ability_mean.astype(float)
    b_ref = fit.difficulty_mean.astype(float)
    n_models, n_total = Y.shape
    if n_models < 2:
        raise ValueError("subsample simulation needs ≥2 respondents for IRT re-fit")

    grid = n_items_grid or _default_n_grid(n_total)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for n_items in grid:
        for rep in range(n_replicates):
            idx = rng.choice(n_total, size=n_items, replace=False)
            idx.sort()
            Y_sub = Y[:, idx]
            b_sub = b_ref[idx]

            # --- Re-fit IRT on subset (joint θ and b) ---
            try:
                fit_sub = _fit_mfvi(
                    Y_sub.astype(float),
                    list(model_ids),
                    [item_ids[i] for i in idx],
                    model=fit.model,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    n_steps=n_steps,
                    hierarchical=True,
                    weights=None,
                    item_granularity=fit.item_granularity,
                    engine="mfvi_subsample",
                )
                theta_hat = fit_sub.ability_mean
                b_hat = fit_sub.difficulty_mean
                ability_err_refit = np.abs(theta_hat - theta_ref)
                diff_rmse_refit = float(np.sqrt(np.mean((b_hat - b_sub) ** 2)))
                diff_mae_refit = float(np.mean(np.abs(b_hat - b_sub)))
            except Exception:
                ability_err_refit = np.full(n_models, np.nan)
                diff_rmse_refit = np.nan
                diff_mae_refit = np.nan

            # --- Fixed-b MLE for ability only (fast lower bound) ---
            ability_err_mle = np.array(
                [abs(mle_theta_1pl_fixed_b(Y_sub[p], b_sub) - theta_ref[p]) for p in range(n_models)]
            )

            for p, mid in enumerate(model_ids):
                rows.append(
                    {
                        "n_items": n_items,
                        "replicate": rep,
                        "model_id": mid,
                        "ability_error_refit": float(ability_err_refit[p]),
                        "ability_error_fixed_b_mle": float(ability_err_mle[p]),
                    }
                )
            rows.append(
                {
                    "n_items": n_items,
                    "replicate": rep,
                    "model_id": "__aggregate__",
                    "ability_error_refit": float(np.nanmean(ability_err_refit)),
                    "ability_error_fixed_b_mle": float(np.mean(ability_err_mle)),
                    "difficulty_rmse_refit": diff_rmse_refit,
                    "difficulty_mae_refit": diff_mae_refit,
                }
            )

    return pd.DataFrame(rows)


def summarize_simulation(sim: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of errors by n_items (aggregate across models/replicates)."""
    agg = sim.loc[sim["model_id"] == "__aggregate__"].copy()
    summary = (
        agg.groupby("n_items", as_index=False)
        .agg(
            difficulty_rmse_mean=("difficulty_rmse_refit", "mean"),
            difficulty_rmse_std=("difficulty_rmse_refit", "std"),
            difficulty_mae_mean=("difficulty_mae_refit", "mean"),
            ability_error_refit_mean=("ability_error_refit", "mean"),
            ability_error_refit_std=("ability_error_refit", "std"),
            ability_error_mle_mean=("ability_error_fixed_b_mle", "mean"),
            ability_error_mle_std=("ability_error_fixed_b_mle", "std"),
            n_replicates=("replicate", "nunique"),
        )
        .sort_values("n_items")
    )
    return summary


def write_subsample_report(
    run_dir: str | Path,
    fit: IRTFit,
    sim: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    source_run_name: str,
) -> Path:
    """Write CSV summaries, plots, and markdown."""
    out = Path(run_dir) / "subsample_sim"
    out.mkdir(parents=True, exist_ok=True)

    sim.to_csv(out / "subsample_sim_raw.csv", index=False)
    summary.to_csv(out / "subsample_sim_summary.csv", index=False)

    _plot_subsample(summary, out / "difficulty_error_vs_n_items.png", metric="difficulty")
    _plot_subsample(summary, out / "ability_error_vs_n_items.png", metric="ability")

    n_full = int(summary["n_items"].max())
    row_full = summary.loc[summary["n_items"] == n_full].iloc[0]
    n_50 = summary.loc[summary["n_items"] <= 50, "n_items"].max() if (summary["n_items"] <= 50).any() else None

    lines = [
        f"# Subsample simulation — `{source_run_name}`",
        "",
        "Ground truth: full-test 1PL variational fit on all items and respondents.",
        "For each subset size `n`, we randomly sample `n` items, re-fit IRT, and compare",
        "to the full-fit parameters.",
        "",
        f"- respondents (models): **{len(fit.model_ids)}** ({', '.join(fit.model_ids)})",
        f"- full items: **{len(fit.item_ids)}**",
        f"- replicates per n: **{int(summary['n_replicates'].iloc[0])}**",
        "",
        "## Difficulty recovery (y-axis you asked for)",
        "",
        "RMSE between re-fitted item difficulties on the subsample vs full-fit difficulties",
        "for those same items.",
        "",
        _summary_md(summary, ["n_items", "difficulty_rmse_mean", "difficulty_rmse_std", "difficulty_mae_mean"]),
        "",
        "## Ability recovery (related question)",
        "",
        "Mean |θ̂ − θ_ref| after re-fitting on subsampled items (vs full-fit θ).",
        "Dashed reference: fixed-b MLE using known full-fit difficulties (optimistic bound).",
        "",
        _summary_md(
            summary,
            ["n_items", "ability_error_refit_mean", "ability_error_refit_std", "ability_error_mle_mean"],
        ),
        "",
        "## Figures",
        "",
        "- `difficulty_error_vs_n_items.png` — difficulty RMSE vs number of items",
        "- `ability_error_vs_n_items.png` — ability error vs number of items",
        "",
    ]
    if n_50 is not None:
        r50 = summary.loc[summary["n_items"] == n_50].iloc[0]
        lines.extend(
            [
                "## Quick read",
                "",
                f"- At **n={n_50}** items: difficulty RMSE ≈ **{r50['difficulty_rmse_mean']:.3f}** "
                f"(vs ≈ **{row_full['difficulty_rmse_mean']:.3f}** at n={n_full}).",
                f"- At **n={n_50}** items: ability error ≈ **{r50['ability_error_refit_mean']:.3f}** "
                f"(vs ≈ **{row_full['ability_error_refit_mean']:.3f}** at n={n_full}).",
                "",
            ]
        )

    report = out / "report.md"
    report.write_text("\n".join(lines))
    return report


def _summary_md(summary: pd.DataFrame, cols: list[str]) -> str:
    sub = summary[cols].round(4)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in sub.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *body])


def _plot_subsample(summary: pd.DataFrame, path: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = summary["n_items"].to_numpy()

    if metric == "difficulty":
        y = summary["difficulty_rmse_mean"].to_numpy()
        yerr = summary["difficulty_rmse_std"].fillna(0).to_numpy()
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#c44e52", capsize=3, lw=2)
        ax.set_ylabel("difficulty RMSE vs full fit")
        ax.set_title("Item difficulty recovery vs subset size")
    else:
        y = summary["ability_error_refit_mean"].to_numpy()
        yerr = summary["ability_error_refit_std"].fillna(0).to_numpy()
        ax.errorbar(
            x, y, yerr=yerr, fmt="o-", color="#1f4e79", capsize=3, lw=2, label="re-fit IRT"
        )
        ax.plot(
            x,
            summary["ability_error_mle_mean"],
            "s--",
            color="#4c72b0",
            alpha=0.8,
            label="fixed-b MLE (known difficulties)",
        )
        ax.set_ylabel("|θ̂ − θ_ref| (ability error)")
        ax.set_title("Ability recovery vs subset size")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("number of items in subsample")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
