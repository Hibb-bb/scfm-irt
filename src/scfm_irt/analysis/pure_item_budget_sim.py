"""Pure 1PL simulation: item budget vs ability estimation error.

No real cells or embeddings — synthetic item bank with known difficulties,
simulated binary responses, and two selection policies:

  1. random         — uniform random items
  2. adaptive_5050  — 50% budget stratified calibration, 50% max-information at θ̂

Item difficulties are treated as **known** (calibrated bank). Only ability θ is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.subsample_sim import mle_theta_1pl_fixed_b
from scfm_irt.irt.models import item_information, sigmoid


@dataclass(frozen=True)
class PureSimConfig:
    bank_size: int = 500
    n_grid: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500)
    n_persons: int = 100  # θ draws per bank replicate
    n_bank_replicates: int = 20  # different random banks
    difficulty_sd: float = 1.0  # b ~ N(0, sd)
    seed: int = 0


def _default_n_grid(bank_size: int) -> tuple[int, ...]:
    cands = sorted({5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400, bank_size})
    return tuple(n for n in cands if 2 <= n <= bank_size)


def select_stratified(bank_b: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Static stratified sample across the bank difficulty range (non-adaptive)."""
    if n >= len(bank_b):
        return np.arange(len(bank_b), dtype=int)
    order = np.argsort(bank_b)
    # n evenly spaced ranks across the sorted bank
    ranks = np.linspace(0, len(bank_b) - 1, n)
    idx = order[np.round(ranks).astype(int)]
    # dedupe while preserving count (rare when n << I)
    idx = np.unique(idx)
    if len(idx) < n:
        remaining = np.setdiff1d(np.arange(len(bank_b)), idx)
        extra = rng.choice(remaining, size=n - len(idx), replace=False)
        idx = np.concatenate([idx, extra])
    return np.sort(idx[:n])


def select_random(bank_b: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(len(bank_b), size=n, replace=False))


def select_adaptive_5050(
    bank_b: np.ndarray,
    n: int,
    y_bank: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """50% stratified calibration → θ̂, then 50% max-information items from the remainder."""
    n_cal = max(1, n // 2)
    n_fine = n - n_cal
    cal_idx = select_stratified(bank_b, n_cal, rng)
    theta_hat = mle_theta_1pl_fixed_b(y_bank[cal_idx], bank_b[cal_idx])

    if n_fine <= 0:
        return cal_idx, theta_hat

    remaining = np.setdiff1d(np.arange(len(bank_b)), cal_idx)
    info = item_information(theta_hat, bank_b[remaining], a=1.0).ravel()
    # break ties randomly
    order = np.lexsort((rng.uniform(size=len(remaining)), -info))
    fine_idx = remaining[order[:n_fine]]
    return np.sort(np.concatenate([cal_idx, fine_idx])), theta_hat


def select_full_bank(bank_b: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Use the full bank when n equals bank size; otherwise best static n-item design."""
    if n >= len(bank_b):
        return np.arange(len(bank_b), dtype=int)
    return select_stratified(bank_b, n, rng)


def simulate_bank(cfg: PureSimConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, cfg.difficulty_sd, size=cfg.bank_size)


def simulate_person_items(
    theta: float,
    b: np.ndarray,
    idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    p = sigmoid(theta - b[idx])
    return rng.binomial(1, p).astype(float)


def run_pure_item_budget_sim(cfg: PureSimConfig | None = None) -> pd.DataFrame:
    """Monte Carlo over banks, persons, budgets, and selection policies."""
    cfg = cfg or PureSimConfig()
    grid = cfg.n_grid if cfg.n_grid else _default_n_grid(cfg.bank_size)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []

    strategies = ("random", "adaptive_5050")

    for bank_rep in range(cfg.n_bank_replicates):
        bank_b = simulate_bank(cfg, rng)
        thetas = rng.normal(0.0, 1.0, size=cfg.n_persons)

        for theta in thetas:
            # Pre-generate full response vector for this person (same latent truth).
            p_full = sigmoid(theta - bank_b)
            y_full = rng.binomial(1, p_full).astype(float)

            for n_items in grid:
                for strat in strategies:
                    for _ in range(1):  # one policy draw per (person, n, bank)
                        if strat == "full_bank":
                            idx = select_full_bank(bank_b, n_items, rng)
                        elif strat == "random":
                            idx = select_random(bank_b, n_items, rng)
                        elif strat == "adaptive_5050":
                            idx, _ = select_adaptive_5050(bank_b, n_items, y_full, rng)
                        else:
                            raise ValueError(strat)

                        y = y_full[idx]
                        b = bank_b[idx]
                        theta_hat = mle_theta_1pl_fixed_b(y, b)
                        err = abs(theta_hat - theta)
                        # analytic SE from Fisher information at θ_hat
                        info = float(item_information(theta_hat, b, a=1.0).sum())
                        se = 1.0 / np.sqrt(max(info, 1e-9))

                        rows.append(
                            {
                                "bank_replicate": bank_rep,
                                "theta_true": theta,
                                "n_items": n_items,
                                "strategy": strat,
                                "ability_error": err,
                                "theta_hat": theta_hat,
                                "fisher_se": se,
                            }
                        )

    return pd.DataFrame(rows)


def summarize_pure_sim(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim.groupby(["strategy", "n_items"], as_index=False)
        .agg(
            ability_error_mean=("ability_error", "mean"),
            ability_error_std=("ability_error", "std"),
            ability_error_median=("ability_error", "median"),
            fisher_se_mean=("fisher_se", "mean"),
            n_obs=("ability_error", "size"),
        )
        .sort_values(["strategy", "n_items"])
    )


def write_pure_sim_report(
    out_dir: str | Path,
    sim: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: PureSimConfig,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim.to_csv(out / "pure_sim_raw.csv", index=False)
    summary.to_csv(out / "pure_sim_summary.csv", index=False)

    _plot_pure_sim(summary, out / "ability_error_vs_n_items.png", bank_size=cfg.bank_size)
    _plot_pure_sim_ratio(summary, out / "adaptive_gain_vs_random.png", bank_size=cfg.bank_size)

    report = out / "report.md"
    lines = [
        "# Pure item-budget simulation (synthetic 1PL)",
        "",
        "No embedding models or real cells. A calibrated item bank with known difficulties",
        "`b ~ N(0, σ)` generates Bernoulli responses; only **ability θ** is estimated (MLE, fixed b).",
        "",
        f"- bank size: **{cfg.bank_size}**",
        f"- difficulty SD: **{cfg.difficulty_sd}**",
        f"- θ ~ N(0,1): **{cfg.n_persons}** draws × **{cfg.n_bank_replicates}** bank replicates",
        "",
        "## Strategies",
        "",
        "1. **random** — uniform random subset of items.",
        "2. **adaptive_5050** — `⌊n/2⌋` stratified calibration items → θ̂, then remaining budget on",
        "   items with highest Fisher information `p(1−p)` at θ̂.",
        "",
        "## Summary (mean |θ̂ − θ|)",
        "",
        _pivot_md(summary),
        "",
        "## Figures",
        "",
        "- `ability_error_vs_n_items.png` — random vs adaptive (x-axis = % of bank administered)",
        "- `adaptive_gain_vs_random.png` — random error / adaptive error (>1 = adaptive wins)",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def _pivot_md(summary: pd.DataFrame) -> str:
    wide = summary.pivot(index="n_items", columns="strategy", values="ability_error_mean").round(4)
    cols = list(wide.columns)
    header = "| n_items | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    body = []
    for n, row in wide.iterrows():
        body.append("| " + " | ".join([str(n), *[str(row[c]) for c in cols]]) + " |")
    return "\n".join([header, sep, *body])


def _pct_of_bank(n_items: np.ndarray | pd.Series, bank_size: int) -> np.ndarray:
    return 100.0 * np.asarray(n_items, dtype=float) / bank_size


def _plot_pure_sim(
    summary: pd.DataFrame,
    path: Path,
    *,
    bank_size: int,
    x_max_pct: float = 100.0,
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "random": {
            "color": "#888888",
            "label": "Random",
            "line_alpha": 0.85,
            "band_alpha": 0.18,
            "lw": 2.0,
            "zorder": 1,
        },
        "adaptive_5050": {
            "color": "#2ca02c",
            "label": "Adaptive 50% cal + 50% max-info",
            "line_alpha": 0.9,
            "band_alpha": 0.15,
            "lw": 2.0,
            "zorder": 2,
        },
    }

    plot_df = summary.loc[summary["strategy"] != "full_bank"].copy()
    plot_df["pct_bank"] = _pct_of_bank(plot_df["n_items"], bank_size)
    plot_df = plot_df.loc[plot_df["pct_bank"] <= x_max_pct]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for strat in ("random", "adaptive_5050"):
        sub = plot_df.loc[plot_df["strategy"] == strat].sort_values("pct_bank")
        if sub.empty:
            continue
        sty = styles.get(strat, {"color": "#333", "label": strat, "line_alpha": 0.8, "band_alpha": 0.15, "lw": 2, "zorder": 1})
        x = sub["pct_bank"].to_numpy(dtype=float)
        y = sub["ability_error_mean"].to_numpy(dtype=float)
        sd = sub["ability_error_std"].fillna(0).to_numpy(dtype=float)
        color = sty["color"]

        lo = np.maximum(y - sd, 0.0)
        hi = y + sd
        ax.fill_between(
            x,
            lo,
            hi,
            color=color,
            alpha=sty["band_alpha"],
            linewidth=0,
            zorder=sty["zorder"] - 0.1,
        )
        ax.plot(
            x,
            y,
            "-",
            color=color,
            lw=sty["lw"],
            alpha=sty["line_alpha"],
            label=sty["label"],
            zorder=sty["zorder"],
        )

    ax.set_xlabel("% of item bank administered")
    ax.set_ylabel("ability estimation error |θ̂ − θ|")
    ax.set_title("Pure 1PL simulation: item selection strategies")
    ax.text(
        0.02,
        0.02,
        "shaded band = mean ± 1 SD",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, which="both")
    ax.set_xscale("log")
    min_pct = float(plot_df["pct_bank"].min()) if not plot_df.empty else 1.0
    ax.set_xlim(max(0.5, min_pct * 0.85), x_max_pct)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pure_sim_ratio(summary: pd.DataFrame, path: Path, *, bank_size: int) -> None:
    import matplotlib.pyplot as plt

    rand = summary.loc[summary["strategy"] == "random"].set_index("n_items")
    adapt = summary.loc[summary["strategy"] == "adaptive_5050"].set_index("n_items")
    common = rand.index.intersection(adapt.index)
    ratio = (rand.loc[common, "ability_error_mean"] / adapt.loc[common, "ability_error_mean"]).reset_index()
    ratio["pct_bank"] = _pct_of_bank(ratio["n_items"], bank_size)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(1.0, color="#999", ls="--", lw=1)
    ax.plot(ratio["pct_bank"], ratio["ability_error_mean"], "o-", color="#2ca02c", lw=2)
    ax.set_xlabel("% of item bank administered")
    ax.set_ylabel("random error / adaptive error")
    ax.set_title("Adaptive advantage (>1 = adaptive better than random)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
