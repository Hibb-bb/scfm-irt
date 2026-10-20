"""Evaluate lung (or any) IRT run using label-entropy difficulties vs model-based IRT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.adaptive_calibration_sim import (
    _default_n_grid,
    _plot_adaptive_errors,
    _plot_adaptive_ratio,
)
from scfm_irt.analysis.label_entropy_difficulty import (
    align_difficulty_to_items,
    knn_label_entropy_train_test,
)
from scfm_irt.analysis.pure_item_budget_sim import (
    select_adaptive_5050,
    select_random,
    select_stratified,
)
from scfm_irt.analysis.report import _accuracy_table, _plot_ability_and_accuracy_forest
from scfm_irt.analysis.subsample_sim import mle_theta_1pl_fixed_b, mle_theta_1pl_fixed_b_se
from scfm_irt.config import RunConfig
from scfm_irt.data.loaders import load_dataset
from scfm_irt.data.registry import get_spec
from scfm_irt.data.splits import make_split
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import to_matrix


def entropy_ability_table(
    resp: pd.DataFrame,
    b_entropy: np.ndarray,
    *,
    model_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Ability per model: MLE under fixed label-entropy difficulties."""
    Y, mids, _ = to_matrix(resp)
    if model_ids is not None and list(mids) != list(model_ids):
        raise ValueError("model_ids mismatch")
    rows = []
    for idx, mid in enumerate(mids):
        y = Y[idx].astype(float)
        theta = mle_theta_1pl_fixed_b(y, b_entropy)
        se = mle_theta_1pl_fixed_b_se(y, b_entropy, theta)
        rows.append(
            {
                "model_id": mid,
                "ability": theta,
                "sd": se,
                "lower": theta - 1.96 * se,
                "upper": theta + 1.96 * se,
                "accuracy": float(y.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["rank"] = out["ability"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def run_entropy_adaptive_sim(
    resp: pd.DataFrame,
    b_entropy: np.ndarray,
    *,
    n_replicates: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Adaptive item-budget sim with fixed label-entropy difficulty bank."""
    Y, model_ids, item_ids = to_matrix(resp)
    n_total = len(item_ids)
    grid = _default_n_grid(n_total)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for holdout_idx, holdout_id in enumerate(model_ids):
        y_holdout = Y[holdout_idx, :].astype(float)
        theta_ref = mle_theta_1pl_fixed_b(y_holdout, b_entropy)

        for n_items in grid:
            for rep in range(n_replicates):
                for strat in ("random", "adaptive_5050"):
                    if strat == "random":
                        idx = select_random(b_entropy, n_items, rng)
                    elif strat == "adaptive_5050":
                        idx, _ = select_adaptive_5050(b_entropy, n_items, y_holdout, rng)
                    else:
                        raise ValueError(strat)
                    theta_hat = mle_theta_1pl_fixed_b(y_holdout[idx], b_entropy[idx])
                    rows.append(
                        {
                            "holdout_model": holdout_id,
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


def summarize_entropy_adaptive(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim.groupby(["strategy", "n_items"], as_index=False)
        .agg(
            pct_bank=("pct_bank", "first"),
            ability_error_mean=("ability_error", "mean"),
            ability_error_std=("ability_error", "std"),
            n_obs=("ability_error", "size"),
        )
        .sort_values(["strategy", "n_items"])
    )


def _load_model_difficulty(run_dir: Path, fit: IRTFit) -> pd.DataFrame:
    loo_path = run_dir / "item_params_loo_summary.csv"
    if loo_path.exists():
        tab = pd.read_csv(loo_path)
        b_col = "difficulty_loo_mean"
    else:
        tab = pd.read_csv(run_dir / "item_params.csv")
        b_col = "difficulty_mean" if "difficulty_mean" in tab.columns else "difficulty"
    if "item_id" not in tab.columns:
        tab = tab.copy()
        tab["item_id"] = fit.item_ids
    out = pd.DataFrame(
        {"item_id": tab["item_id"].astype(str), "difficulty_model": tab[b_col].astype(float).values}
    )
    return out


def _plot_difficulty_comparison(
    entropy_table: pd.DataFrame,
    model_b: pd.DataFrame,
    path: Path,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    merged = entropy_table.merge(model_b, left_on="cell_id", right_on="item_id")
    x = merged["difficulty_entropy"].to_numpy()
    y = merged["difficulty_model"].to_numpy()
    # Shift model b to same mean as entropy b for visual comparison.
    y_aligned = y - float(np.mean(y - x))
    pearson = float(np.corrcoef(x, y_aligned)[0, 1])
    spearman = float(pd.Series(x).corr(pd.Series(y_aligned), method="spearman"))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(x, y_aligned, s=6, alpha=0.25, c="#1f4e79", edgecolors="none")
    ax.axhline(0, color="#ddd", lw=1)
    ax.axvline(0, color="#ddd", lw=1)
    ax.set_xlabel("label-entropy difficulty (kNN train neighbors)")
    ax.set_ylabel("model-based difficulty (LOO mean, shift-aligned)")
    ax.set_title(f"Difficulty banks (r={pearson:.3f}, ρ={spearman:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"pearson_r": pearson, "spearman_r": spearman}


def write_entropy_difficulty_eval(
    run_dir: str | Path,
    cfg: RunConfig,
    fit: IRTFit,
    resp: pd.DataFrame,
    *,
    k: int = 30,
    n_replicates: int = 40,
    seed: int = 0,
) -> Path:
    """Full lung-style eval: entropy bank, abilities, adaptive sim, figures."""
    run_dir = Path(run_dir)
    out = run_dir / "entropy_difficulty_eval"
    out.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.datasets[0]
    spec = get_spec(dataset_name)
    adata = load_dataset(spec)
    split = make_split(
        adata,
        spec,
        scheme=cfg.split.scheme,
        seed=cfg.seed,
        test_frac=cfg.split.test_frac,
    )
    train = adata[split.train_idx].copy()
    test = adata[split.test_idx].copy()
    del adata

    print(f"[entropy-b] computing kNN label entropy (k={k}) for {dataset_name}...")
    entropy_table = knn_label_entropy_train_test(
        train,
        test,
        spec.label_key,
        dataset_name,
        k=k,
        seed=seed,
    )
    del train, test
    entropy_table.to_csv(out / "label_entropy_items.csv", index=False)

    b_entropy = align_difficulty_to_items(list(fit.item_ids), entropy_table)
    pd.DataFrame({"item_id": fit.item_ids, "difficulty_entropy": b_entropy}).to_csv(
        out / "item_params_entropy.csv", index=False
    )

    model_b = _load_model_difficulty(run_dir, fit)
    cmp_stats = _plot_difficulty_comparison(
        entropy_table,
        model_b,
        out / "difficulty_entropy_vs_model.png",
    )
    pd.Series(cmp_stats).to_csv(out / "difficulty_comparison_stats.csv")

    # --- abilities under entropy bank ---
    ability = entropy_ability_table(resp, b_entropy, model_ids=list(fit.model_ids))
    ability.to_csv(out / "ability_ranking_entropy.csv", index=False)

    # Joint / LOO abilities for comparison if present.
    joint_path = run_dir / "ability_ranking_joint.csv"
    if not joint_path.exists():
        joint_path = run_dir / "ability_ranking.csv"
    loo_path = run_dir / "ability_ranking_loo.csv"
    if joint_path.exists():
        ability = ability.merge(
            pd.read_csv(joint_path)[["model_id", "ability"]].rename(
                columns={"ability": "ability_joint"}
            ),
            on="model_id",
            how="left",
        )
    if loo_path.exists():
        ability = ability.merge(
            pd.read_csv(loo_path)[["model_id", "ability"]].rename(
                columns={"ability": "ability_loo"}
            ),
            on="model_id",
            how="left",
        )
    ability.to_csv(out / "ability_ranking_comparison.csv", index=False)

    plot_df = ability.rename(columns={"ability": "ability"})[
        ["model_id", "ability", "sd", "lower", "upper"]
    ]
    _plot_ability_and_accuracy_forest(
        plot_df,
        resp,
        f"{cfg.run_name} (label-entropy θ)",
        out,
        stem="ability_forest_entropy",
    )

    # Model-based forest copy for side-by-side (reuse LOO if available).
    if loo_path.exists():
        loo_ab = pd.read_csv(loo_path)
        if "sd" not in loo_ab.columns:
            loo_ab["sd"] = np.nan
            loo_ab["lower"] = loo_ab["ability"]
            loo_ab["upper"] = loo_ab["ability"]
        _plot_ability_and_accuracy_forest(
            loo_ab[["model_id", "ability", "sd", "lower", "upper"]],
            resp,
            f"{cfg.run_name} (LOO model-based θ)",
            out,
            stem="ability_forest_model_loo",
        )

    # --- adaptive sim ---
    print("[entropy-b] adaptive simulation with label-entropy bank...")
    sim = run_entropy_adaptive_sim(resp, b_entropy, n_replicates=n_replicates, seed=seed)
    summary = summarize_entropy_adaptive(sim)
    sim.to_csv(out / "adaptive_sim_raw.csv", index=False)
    summary.to_csv(out / "adaptive_sim_summary.csv", index=False)

    summary_plot = summary.copy()
    summary_plot["b_source"] = "label_entropy"
    _plot_adaptive_errors(summary_plot, out / "new_model_ability_error.png", b_source="label_entropy")
    _plot_adaptive_ratio(summary_plot, out / "adaptive_gain_vs_random.png", b_source="label_entropy")

    acc = _accuracy_table(resp)
    report = out / "report.md"
    lines = [
        "# Label-entropy difficulty evaluation",
        "",
        f"- source run: `{cfg.run_name}`",
        f"- dataset: `{dataset_name}`",
        f"- kNN neighbors: **{k}** (train cells only, HVG expression space)",
        f"- items: **{len(fit.item_ids)}**",
        "",
        "## Difficulty vs model-based IRT bank",
        "",
        f"- Pearson r (entropy vs LOO model b): **{cmp_stats['pearson_r']:.4f}**",
        f"- Spearman ρ: **{cmp_stats['spearman_r']:.4f}**",
        "",
        "Low correlation ⇒ label entropy and model-pool difficulty rank cells differently.",
        "",
        "## Ability ranking (fixed entropy difficulties)",
        "",
        _md_table(ability.round(4)),
        "",
        "## Adaptive ability estimation (entropy bank)",
        "",
        _md_table(
            summary.pivot_table(index="n_items", columns="strategy", values="ability_error_mean")
            .round(4)
            .reset_index()
        ),
        "",
        "## Figures",
        "",
        "- `difficulty_entropy_vs_model.png`",
        "- `ability_forest_entropy.png` — θ from entropy bank vs raw accuracy",
        "- `ability_forest_model_loo.png` — LOO model-based θ (comparison)",
        "- `new_model_ability_error.png` — adaptive vs random under entropy bank",
        "- `adaptive_gain_vs_random.png`",
        "",
    ]
    report.write_text("\n".join(lines))
    print(f"[entropy-b] wrote {report}")
    return report


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            cells.append(f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])
