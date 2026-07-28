"""Synthetic data generation for the parameter-recovery test (MATH.md section 9).

This is the acceptance gate: before any real embedding model is fit, each estimator must
recover known parameters from simulated responses. It is also what catches a persons/items
transposition, which otherwise shows up as scrambled recovery and nothing else.

Implemented here (not stubbed) because the tests depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scfm_irt.irt.models import icc


@dataclass(frozen=True)
class SyntheticData:
    """Simulated responses together with the ground truth that generated them."""

    Y: np.ndarray  # (P, I) — models x cells, the house orientation
    theta: np.ndarray  # (P,) true abilities
    b: np.ndarray  # (I,) true difficulties
    a: np.ndarray  # (I,) true discriminations (all ones for 1PL)
    model_ids: list[str]
    item_ids: list[str]

    @property
    def n_models(self) -> int:
        return self.Y.shape[0]

    @property
    def n_items(self) -> int:
        return self.Y.shape[1]


def simulate_responses(
    n_models: int = 10,
    n_items: int = 1000,
    model: str = "1pl",
    theta_sd: float = 1.0,
    b_mean: float = 0.0,
    b_sd: float = 1.5,
    a_log_sd: float = 0.3,
    seed: int = 0,
) -> SyntheticData:
    """Draw true parameters from the priors of MATH.md section 6.3 and simulate responses.

    Defaults mirror the real regime: few respondents, many items, and a wide difficulty spread
    so items actually carry information across the ability range.
    """
    rng = np.random.default_rng(seed)

    theta = rng.normal(0.0, theta_sd, size=n_models)
    b = rng.normal(b_mean, b_sd, size=n_items)
    a = np.ones(n_items) if model == "1pl" else rng.lognormal(0.0, a_log_sd, size=n_items)

    # (P, 1) against (1, I) -> (P, I), rows are models. Orientation is set here and nowhere else.
    p = icc(theta[:, None], b[None, :], a[None, :])
    Y = (rng.random(p.shape) < p).astype(np.int8)

    return SyntheticData(
        Y=Y,
        theta=theta,
        b=b,
        a=a,
        model_ids=[f"model_{i:02d}" for i in range(n_models)],
        item_ids=[f"item_{i:06d}" for i in range(n_items)],
    )


def to_long(data: SyntheticData, dataset: str = "synthetic") -> pd.DataFrame:
    """Convert simulated responses into the canonical long response table.

    Lets the synthetic path exercise the same `fit_irt` entry point as real data, so the test
    covers the pivot as well as the estimator.
    """
    n_models, n_items = data.Y.shape
    model_col = np.repeat(data.model_ids, n_items)
    item_col = np.tile(data.item_ids, n_models)

    return pd.DataFrame(
        {
            "model_id": model_col,
            "dataset": dataset,
            "split_scheme": "synthetic",
            "cell_id": item_col,
            "cell_type": np.tile([f"type_{i % 10}" for i in range(n_items)], n_models),
            "correct": data.Y.reshape(-1).astype(np.int8),
        }
    )


def recovery_metrics(
    truth: SyntheticData,
    theta_hat: np.ndarray,
    b_hat: np.ndarray,
    theta_sd: np.ndarray | None = None,
) -> dict[str, float]:
    """Correlations with truth, plus interval coverage when posterior SDs are given.

    Expected behavior in this regime: ability correlation very high even at small `I` (each
    theta is informed by ~I responses), difficulty correlation modest and *rising with I*
    (each b is informed by only ~P responses).
    """
    out = {
        "ability_corr": float(np.corrcoef(truth.theta, theta_hat)[0, 1]),
        "difficulty_corr": float(np.corrcoef(truth.b, b_hat)[0, 1]),
        "ability_rmse": float(np.sqrt(np.mean((truth.theta - theta_hat) ** 2))),
    }
    if theta_sd is not None:
        covered = np.abs(truth.theta - theta_hat) <= 1.96 * theta_sd
        out["ability_coverage_95"] = float(np.mean(covered))
    return out
