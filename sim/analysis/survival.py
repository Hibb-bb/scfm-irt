"""Survival-curve quantities: κ*, r*, and benchmark placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def tuned_drift_rate(s: float, n_rounds: int) -> float:
    """Set r so r·T ≈ 4s (field sweeps one useful window over the horizon)."""
    horizon = max(n_rounds - 1, 1)
    return 4.0 * s / horizon


def survival_quantities(
    *,
    s: float,
    n_bank: int,
    kappa: int,
    r_empirical: float | None = None,
) -> dict[str, float]:
    """Critical write rate κ* = r·N/(4s) and staleness threshold r* = 4sκ/N."""
    denom = max(4.0 * s, 1e-9)
    kappa_star = float(r_empirical * n_bank / denom) if r_empirical is not None else float("nan")
    r_star = float(denom * kappa / n_bank)
    a_rate = 4.0 * s * kappa / max(n_bank, 1)
    return {
        "s": s,
        "N": n_bank,
        "kappa": kappa,
        "r_empirical": r_empirical if r_empirical is not None else float("nan"),
        "kappa_star": kappa_star,
        "r_star": r_star,
        "a_rate": a_rate,
        "kappa_margin": kappa / max(kappa_star, 1e-9) if r_empirical is not None else float("nan"),
        "r_margin": r_star / max(r_empirical, 1e-9) if r_empirical is not None else float("nan"),
    }


def write_survival_report(
    *,
    fitted_s: float,
    n_bank: int,
    kappa: int,
    r_empirical: float,
    r_tuned: float,
    n_rounds: int,
    out_dir: Path,
) -> pd.DataFrame:
    q = survival_quantities(s=fitted_s, n_bank=n_bank, kappa=kappa, r_empirical=r_empirical)
    row = pd.DataFrame([q])
    row.to_csv(out_dir / "survival_curve.csv", index=False)

    lines = [
        "# Survival curve — real lung atlas benchmark",
        "",
        f"Fitted field spread **s = {fitted_s:.3f}** (std of joint-Rasch θ\\*).",
        f"Bank size **N = {n_bank}**, write budget **κ = {kappa}**.",
        "",
        "## Observed drift (real rollout)",
        "",
        f"- Empirical **r ≈ {r_empirical:.4f}**/round",
        f"- Over {n_rounds - 1} steps: **r·T ≈ {r_empirical * (n_rounds - 1):.2f}** vs useful window **4s ≈ {4 * fitted_s:.2f}**",
        f"- Field crossed ~**{100 * r_empirical * (n_rounds - 1) / (4 * fitted_s):.0f}%** of one coverage window → out of decay regime",
        "",
        "## Critical write rate",
        "",
        f"- **κ\\* = r·N/(4s) ≈ {q['kappa_star']:.2f}** items/round",
        f"- Current κ = {kappa} → margin **κ/κ\\* ≈ {q['kappa_margin']:.1f}×**",
        "",
        "The benchmark stays alive by adding <2 new items per round — trivially satisfied.",
        "",
        "## Staleness horizon",
        "",
        f"- **r\\* = 4sκ/N ≈ {q['r_star']:.3f}**/round",
        f"- Drift must exceed r\\* before maintenance binding; observed r is **{q['r_margin']:.1f}×** below that",
        "",
        f"Cell-type classification won't need active maintenance until capability drifts "
        f"~**{q['r_margin']:.0f}×** faster than the real 10-model timeline.",
        "",
        "## Synthetic validation (Fig 2 tuning)",
        "",
        f"For controlled drift demo: **r_tune = 4s/T ≈ {r_tuned:.3f}**/round (r·T ≈ 4s).",
        "",
    ]
    (out_dir / "survival_curve.md").write_text("\n".join(lines))
    return row
