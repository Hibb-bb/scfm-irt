"""How calibration quality scales with the number of calibration models.

Hold out one model as *new*, fit item difficulties using k randomly chosen
calibration models (k = 2 … P−1), and compare:

1. Difficulty stability vs the full-fit bank (Pearson r, RMSE).
2. New-model ability error under fixed calibration banks (random vs adaptive).

When k = P−1 this reduces to leave-one-model-out calibration.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.loo_calibration import align_difficulty
from scfm_irt.analysis.pure_item_budget_sim import select_adaptive_5050, select_random
from scfm_irt.analysis.subsample_sim import mle_theta_1pl_fixed_b
from scfm_irt.irt.fit_pyirt import _fit_mfvi
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import build_response_table, to_matrix


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


def _calibration_subsets(
    cal_ids: list[str],
    k: int,
    n_subsamples: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    all_subs = list(combinations(cal_ids, k))
    if not all_subs:
        return []
    if len(all_subs) <= n_subsamples:
        return all_subs
    pick = rng.choice(len(all_subs), size=n_subsamples, replace=False)
    return [all_subs[int(i)] for i in pick]


def _difficulty_metrics(b_aligned: np.ndarray, b_ref: np.ndarray) -> dict[str, float]:
    diff = b_aligned - b_ref
    return {
        "difficulty_pearson_r": float(np.corrcoef(b_ref, b_aligned)[0, 1]),
        "difficulty_spearman_r": float(
            pd.Series(b_ref).corr(pd.Series(b_aligned), method="spearman")
        ),
        "difficulty_rmse": float(np.sqrt(np.mean(diff**2))),
        "difficulty_mae": float(np.mean(np.abs(diff))),
    }


def fit_calibration_bank(
    resp: pd.DataFrame,
    fit: IRTFit,
    *,
    holdout_id: str,
    cal_model_ids: list[str] | tuple[str, ...],
    n_steps: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Fit difficulties on ``cal_model_ids``; return shift-aligned bank vs full fit."""
    Y, model_ids, item_ids = to_matrix(resp)
    if holdout_id in cal_model_ids:
        raise ValueError(f"holdout {holdout_id!r} cannot be in calibration pool")
    cal_idx = [model_ids.index(m) for m in cal_model_ids]
    Y_cal = Y[np.asarray(cal_idx), :]
    fit_cal = _fit_mfvi(
        Y_cal.astype(float),
        list(cal_model_ids),
        list(item_ids),
        model=fit.model,
        seed=seed,
        n_steps=n_steps,
        hierarchical=True,
        weights=None,
        item_granularity=fit.item_granularity,
        engine="mfvi_cal_pool",
    )
    b_full = fit.difficulty_mean.astype(float)
    return align_difficulty(fit_cal.difficulty_mean.astype(float), b_full)


def run_calibration_size_study(
    resp: pd.DataFrame,
    fit: IRTFit,
    *,
    n_subsamples: int = 5,
    n_steps: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Difficulty stability vs number of calibration models (per holdout, per subsample)."""
    _, model_ids, _ = to_matrix(resp)
    if len(model_ids) < 3:
        raise ValueError("calibration pool study needs ≥3 models")

    b_full = fit.difficulty_mean.astype(float)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for holdout_id in model_ids:
        cal_ids = [m for m in model_ids if m != holdout_id]
        for k in range(2, len(model_ids)):
            subs = _calibration_subsets(cal_ids, k, n_subsamples, rng)
            for sub_idx, sub in enumerate(subs):
                b_aligned = fit_calibration_bank(
                    resp,
                    fit,
                    holdout_id=holdout_id,
                    cal_model_ids=list(sub),
                    n_steps=n_steps,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                metrics = _difficulty_metrics(b_aligned, b_full)
                rows.append(
                    {
                        "holdout_model": holdout_id,
                        "n_calibration_models": k,
                        "subsample": sub_idx,
                        "calibration_models": ",".join(sub),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def summarize_calibration_size_stability(stability: pd.DataFrame) -> pd.DataFrame:
    return (
        stability.groupby("n_calibration_models", as_index=False)
        .agg(
            difficulty_pearson_r_mean=("difficulty_pearson_r", "mean"),
            difficulty_pearson_r_std=("difficulty_pearson_r", "std"),
            difficulty_rmse_mean=("difficulty_rmse", "mean"),
            difficulty_rmse_std=("difficulty_rmse", "std"),
            n_obs=("difficulty_pearson_r", "size"),
        )
        .sort_values("n_calibration_models")
    )


def _mean_bank_for_k(
    resp: pd.DataFrame,
    fit: IRTFit,
    holdout_id: str,
    k: int,
    *,
    n_subsamples: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    _, model_ids, _ = to_matrix(resp)
    cal_ids = [m for m in model_ids if m != holdout_id]
    subs = _calibration_subsets(cal_ids, k, n_subsamples, rng)
    banks = [
        fit_calibration_bank(
            resp,
            fit,
            holdout_id=holdout_id,
            cal_model_ids=list(sub),
            n_steps=n_steps,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        for sub in subs
    ]
    return np.mean(np.stack(banks, axis=0), axis=0)


def run_calibration_size_ability_sim(
    resp: pd.DataFrame,
    fit: IRTFit,
    *,
    k_values: list[int] | None = None,
    n_subsamples: int = 3,
    n_replicates: int = 30,
    n_items_grid: list[int] | None = None,
    n_steps: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """New-model ability error vs calibration pool size (mean bank over subsamples)."""
    Y, model_ids, item_ids = to_matrix(resp)
    n_total = len(item_ids)
    grid = n_items_grid or _default_n_grid(n_total)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    if k_values is None:
        p = len(model_ids)
        k_values = list(range(2, p)) if p <= 8 else sorted({2, 3, max(2, p // 2), p - 1})

    strategies = ("random", "adaptive_5050")

    for holdout_idx, holdout_id in enumerate(model_ids):
        y_holdout = Y[holdout_idx, :].astype(float)
        b_ref = _mean_bank_for_k(
            resp,
            fit,
            holdout_id,
            len(model_ids) - 1,
            n_subsamples=max(n_subsamples, 1),
            n_steps=n_steps,
            rng=rng,
        )
        theta_ref = mle_theta_1pl_fixed_b(y_holdout, b_ref)

        for k in k_values:
            if k >= len(model_ids) or k < 2:
                continue
            b_bank = _mean_bank_for_k(
                resp,
                fit,
                holdout_id,
                k,
                n_subsamples=n_subsamples,
                n_steps=n_steps,
                rng=rng,
            )

            for n_items in grid:
                for rep in range(n_replicates):
                    for strat in strategies:
                        if strat == "random":
                            idx = select_random(b_bank, n_items, rng)
                        elif strat == "adaptive_5050":
                            idx, _ = select_adaptive_5050(b_bank, n_items, y_holdout, rng)
                        else:
                            raise ValueError(strat)

                        theta_hat = mle_theta_1pl_fixed_b(y_holdout[idx], b_bank[idx])
                        rows.append(
                            {
                                "holdout_model": holdout_id,
                                "n_calibration_models": k,
                                "strategy": strat,
                                "n_items": n_items,
                                "pct_bank": 100.0 * n_items / n_total,
                                "replicate": rep,
                                "theta_ref": theta_ref,
                                "theta_hat": theta_hat,
                                "ability_error": abs(theta_hat - theta_ref),
                            }
                        )

    return pd.DataFrame(rows)


def summarize_calibration_size_ability(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim.groupby(["n_calibration_models", "strategy", "n_items"], as_index=False)
        .agg(
            pct_bank=("pct_bank", "first"),
            ability_error_mean=("ability_error", "mean"),
            ability_error_std=("ability_error", "std"),
            n_obs=("ability_error", "size"),
        )
        .sort_values(["n_calibration_models", "strategy", "n_items"])
    )


def synthetic_irt_responses(
    *,
    n_models: int = 12,
    n_items: int = 500,
    seed: int = 0,
) -> tuple[pd.DataFrame, IRTFit]:
    """Simulate 1PL responses from known θ, b for calibration-size experiments."""
    rng = np.random.default_rng(seed)
    model_ids = [f"model_{i:02d}" for i in range(n_models)]
    item_ids = [f"item_{i:04d}" for i in range(n_items)]
    theta = rng.normal(size=n_models)
    b = rng.normal(size=n_items)
    Y = np.zeros((n_models, n_items), dtype=np.int8)
    for p, th in enumerate(theta):
        p_corr = 1.0 / (1.0 + np.exp(-(th - b)))
        Y[p] = (rng.random(n_items) < p_corr).astype(np.int8)

    rows = []
    for pi, mid in enumerate(model_ids):
        for ii, iid in enumerate(item_ids):
            rows.append(
                {
                    "model_id": mid,
                    "dataset": "synthetic",
                    "split_scheme": "stratified",
                    "cell_id": iid,
                    "cell_type": "syn",
                    "correct": int(Y[pi, ii]),
                }
            )
    resp = build_response_table([pd.DataFrame(rows)])
    fit = _fit_mfvi(
        Y.astype(float),
        model_ids,
        item_ids,
        model="1pl",
        seed=seed,
        n_steps=800,
        hierarchical=True,
        weights=None,
        item_granularity="cell",
        engine="mfvi_synthetic",
    )
    return resp, fit


def write_calibration_pool_size_report(
    out_dir: str | Path,
    fit: IRTFit,
    stability: pd.DataFrame,
    stability_summary: pd.DataFrame,
    sim: pd.DataFrame,
    sim_summary: pd.DataFrame,
    *,
    source_label: str,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stability.to_csv(out / "difficulty_stability_by_k.csv", index=False)
    stability_summary.to_csv(out / "difficulty_stability_by_k_summary.csv", index=False)
    sim.to_csv(out / "new_model_ability_by_k_raw.csv", index=False)
    sim_summary.to_csv(out / "new_model_ability_by_k_summary.csv", index=False)

    _plot_difficulty_vs_k(stability_summary, out / "difficulty_stability_vs_k.png")
    _plot_ability_vs_k(sim_summary, out / "new_model_ability_error_vs_k.png")
    _plot_ability_curves_by_k(sim_summary, out / "new_model_ability_error_by_k_curves.png")
    pct_figs = _plot_ability_error_vs_k_per_pct_bank(sim_summary, out / "ability_error_vs_k")

    stab = stability_summary.copy()
    report = out / "report.md"
    lines = [
        "# Calibration pool size simulation",
        "",
        f"- source: `{source_label}`",
        f"- models: **{len(fit.model_ids)}** ({', '.join(fit.model_ids)})",
        f"- items: **{len(fit.item_ids)}**",
        "",
        "## Difficulty stability vs calibration pool size",
        "",
        "Hold out one model; fit difficulties on k other models (random subsets when k < P−1).",
        "Compare shift-aligned difficulties to the full-fit bank.",
        "",
        _df_md(stab.round(4)),
        "",
        "## New-model ability error vs k",
        "",
        "Reference θ uses the k = P−1 calibration bank. Item budgets: random vs adaptive_5050.",
        "",
        "## Figures",
        "",
        "- `difficulty_stability_vs_k.png` — Pearson r / RMSE vs number of calibration models",
        "- `new_model_ability_error_vs_k.png` — ability error vs k at selected % bank",
        "- `new_model_ability_error_by_k_curves.png` — error curves for each k (like LOO sim)",
        f"- `ability_error_vs_k/` — one panel per % bank: error (y) vs k calibration models (x)",
        "",
    ]
    if pct_figs:
        lines.append("### Per–% bank panels")
        lines.append("")
        for p in pct_figs:
            lines.append(f"- `{p.name}`")
        lines.append("")
    report.write_text("\n".join(lines))
    return report


def replot_calibration_size_ability(
    summary: pd.DataFrame,
    out_dir: str | Path,
    *,
    pct_targets: list[float] | None = None,
) -> list[Path]:
    """Regenerate ability-error vs k figures from an existing summary table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _plot_ability_vs_k(summary, out / "new_model_ability_error_vs_k.png")
    _plot_ability_curves_by_k(summary, out / "new_model_ability_error_by_k_curves.png")
    return _plot_ability_error_vs_k_per_pct_bank(
        summary, out / "ability_error_vs_k", pct_targets=pct_targets
    )


def _df_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        cells = [
            f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
            for c in cols
        ]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def _plot_difficulty_vs_k(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = summary["n_calibration_models"].to_numpy()

    ax = axes[0]
    ym = summary["difficulty_pearson_r_mean"].to_numpy()
    ys = summary["difficulty_pearson_r_std"].fillna(0).to_numpy()
    ax.errorbar(x, ym, yerr=ys, fmt="o-", color="#1f4e79", capsize=3, lw=2)
    ax.axhline(0.9, color="#999", ls="--", lw=1)
    ax.set_xlabel("Number of calibration models (k)")
    ax.set_ylabel("Pearson r vs full bank")
    ax.set_title("Difficulty stability")
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ym = summary["difficulty_rmse_mean"].to_numpy()
    ys = summary["difficulty_rmse_std"].fillna(0).to_numpy()
    ax.errorbar(x, ym, yerr=ys, fmt="o-", color="#c0392b", capsize=3, lw=2)
    ax.set_xlabel("Number of calibration models (k)")
    ax.set_ylabel("RMSE vs full bank")
    ax.set_title("Difficulty error")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ability_vs_k(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    # Pick a few representative bank fractions.
    pct_targets = [0.5, 1.0, 2.0, 5.0]
    sub = summary.loc[summary["strategy"] == "adaptive_5050"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pct_targets)))

    for pct, color in zip(pct_targets, colors):
        chunk = sub.iloc[(sub["pct_bank"] - pct).abs().argsort()[:1]]
        if chunk.empty:
            continue
        row = chunk.iloc[0]
        k_vals = summary.loc[
            (summary["strategy"] == "adaptive_5050")
            & (summary["n_items"] == row["n_items"])
        ].sort_values("n_calibration_models")
        if k_vals.empty:
            continue
        ax.plot(
            k_vals["n_calibration_models"],
            k_vals["ability_error_mean"],
            "o-",
            color=color,
            lw=2,
            label=f"~{row['pct_bank']:.1f}% bank",
        )

    ax.set_xlabel("Number of calibration models (k)")
    ax.set_ylabel("|θ̂ − θ_ref| (adaptive_5050)")
    ax.set_title("New-model ability error vs calibration pool size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ability_curves_by_k(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = summary.loc[summary["strategy"] == "adaptive_5050"].copy()
    if sub.empty:
        return

    ks = sorted(sub["n_calibration_models"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ks)))

    for k, color in zip(ks, colors):
        s = sub.loc[sub["n_calibration_models"] == k].sort_values("pct_bank")
        ax.plot(
            s["pct_bank"],
            s["ability_error_mean"],
            "-",
            color=color,
            lw=2,
            label=f"k={k}",
        )

    ax.set_xlabel("% of item bank administered")
    ax.set_ylabel("|θ̂ − θ_ref| (adaptive_5050)")
    ax.set_title("Ability error curves by calibration pool size")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.legend(title="cal models", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _default_pct_targets(summary: pd.DataFrame) -> list[float]:
    """Representative bank fractions present in the simulation grid."""
    present = summary["pct_bank"].unique()
    if len(present) == 0:
        return []
    cands = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    picked: list[float] = []
    for target in cands:
        if (np.abs(present - target)).min() > 0.35:
            continue
        picked.append(float(present[np.abs(present - target).argmin()]))
    return sorted(set(picked))


def _pct_label(pct: float) -> str:
    if pct >= 10:
        return f"{pct:.0f}pct"
    if pct >= 1:
        return f"{pct:.1f}pct".replace(".", "p")
    return f"{pct:.2f}pct".replace(".", "p")


def _plot_ability_error_vs_k_per_pct_bank(
    summary: pd.DataFrame,
    out_dir: str | Path,
    *,
    pct_targets: list[float] | None = None,
    strategies: tuple[str, ...] = ("random", "adaptive_5050"),
) -> list[Path]:
    """One figure per % bank: |θ̂−θ_ref| (y) vs number of calibration models k (x)."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = pct_targets or _default_pct_targets(summary)
    if not targets:
        return []

    styles = {
        "random": ("#888888", "random"),
        "adaptive_5050": ("#2ca02c", "adaptive 50/50"),
    }
    paths: list[Path] = []

    for pct in targets:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for strat in strategies:
            sub = summary.loc[summary["strategy"] == strat].copy()
            if sub.empty:
                continue
            sub["dp"] = (sub["pct_bank"] - pct).abs()
            n_items = int(sub.nsmallest(1, "dp").iloc[0]["n_items"])
            chunk = sub.loc[sub["n_items"] == n_items].sort_values("n_calibration_models")
            if chunk.empty:
                continue
            color, label = styles.get(strat, ("#333333", strat))
            x = chunk["n_calibration_models"].to_numpy()
            y = chunk["ability_error_mean"].to_numpy()
            yerr = chunk["ability_error_std"].fillna(0).to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                color=color,
                lw=2,
                capsize=3,
                label=label,
            )

        actual_pct = float(
            summary.loc[(summary["pct_bank"] - pct).abs().idxmin(), "pct_bank"]
        )
        ax.set_xlabel("Number of calibration models (k)")
        ax.set_ylabel("|θ̂ − θ_ref| for held-out (new) model")
        ax.set_title(f"Ability estimation error at ~{actual_pct:.2f}% of item bank")
        ax.set_xticks(sorted(summary["n_calibration_models"].unique()))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out / f"ability_error_vs_k_{_pct_label(actual_pct)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths
