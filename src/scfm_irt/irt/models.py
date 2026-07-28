"""Response models, Fisher information, and the fitted-object contract.

The item characteristic curves of MATH.md section 3 and the information function of section
8.1. These are pure numpy and carry no estimator dependencies, so they are usable for
simulation, diagnostics, and plotting without torch/pyro installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

IRTModelName = Literal["1pl", "2pl", "3pl", "4pl"]


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x)))
    )


def icc(
    theta: np.ndarray | float,
    b: np.ndarray | float,
    a: np.ndarray | float = 1.0,
    c: np.ndarray | float = 0.0,
    d: np.ndarray | float = 1.0,
) -> np.ndarray:
    """P(correct) under the 4PL, which specializes to the others.

        1PL: a=1, c=0, d=1     ->  sigma(theta - b)
        2PL: c=0, d=1          ->  sigma(a (theta - b))
        3PL: d=1               ->  c + (1-c) sigma(a (theta - b))
        4PL: all four          ->  c + (d-c) sigma(a (theta - b))

    Broadcasting is the caller's business: pass `theta` as (P, 1) and item params as (1, I) to
    get the full (P, I) probability matrix in the house orientation.
    """
    return np.asarray(c) + (np.asarray(d) - np.asarray(c)) * sigmoid(
        np.asarray(a) * (np.asarray(theta) - np.asarray(b))
    )


def item_information(
    theta: np.ndarray | float,
    b: np.ndarray | float,
    a: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Fisher information a 2PL item contributes at ability `theta`: a^2 P (1-P).

    Peaks at theta = b, which is the whole point of the difficulty-spread argument: items far
    below the ability range contribute almost nothing (MATH.md section 8.3).
    """
    p = icc(theta, b, a)
    return np.asarray(a) ** 2 * p * (1.0 - p)


def test_information(
    theta_grid: np.ndarray,
    b: np.ndarray,
    a: np.ndarray | None = None,
) -> np.ndarray:
    """Total information across items at each point of `theta_grid`.

    SE(theta) = 1 / sqrt(I(theta)); plotting this shows where on the ability scale the
    evaluation actually resolves differences between models.
    """
    a_arr = np.ones_like(b, dtype=float) if a is None else np.asarray(a, dtype=float)
    grid = np.asarray(theta_grid, dtype=float).reshape(-1, 1)
    return item_information(grid, np.asarray(b).reshape(1, -1), a_arr.reshape(1, -1)).sum(axis=1)


@dataclass
class IRTFit:
    """What every estimator returns, regardless of engine.

    Posterior standard deviations are not optional decoration: with P around 10 respondents,
    the between-model comparison is what carries uncertainty, so a ranking without intervals is
    not reportable (MATH.md section 7).
    """

    model: IRTModelName
    method: str
    item_granularity: str
    model_ids: list[str]
    item_ids: list[str]
    ability_mean: np.ndarray  # (P,)
    ability_sd: np.ndarray  # (P,)
    difficulty_mean: np.ndarray  # (I,)
    difficulty_sd: np.ndarray  # (I,)
    discrimination_mean: np.ndarray | None = None  # (I,), 2PL+
    discrimination_sd: np.ndarray | None = None
    anchoring: str = ""  # which identification constraint was used (MATH.md section 5)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.ability_mean) != len(self.model_ids):
            raise ValueError("ability vector does not match model_ids")
        if len(self.difficulty_mean) != len(self.item_ids):
            raise ValueError("difficulty vector does not match item_ids")

    def ability_table(self, z: float = 1.96) -> pd.DataFrame:
        """The headline deliverable: model ranking with intervals, best first."""
        out = pd.DataFrame(
            {
                "model_id": self.model_ids,
                "ability": self.ability_mean,
                "sd": self.ability_sd,
                "lower": self.ability_mean - z * self.ability_sd,
                "upper": self.ability_mean + z * self.ability_sd,
            }
        ).sort_values("ability", ascending=False, ignore_index=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def item_table(self) -> pd.DataFrame:
        cols = {
            "item_id": self.item_ids,
            "difficulty": self.difficulty_mean,
            "difficulty_sd": self.difficulty_sd,
        }
        if self.discrimination_mean is not None:
            cols["discrimination"] = self.discrimination_mean
            cols["discrimination_sd"] = self.discrimination_sd
        return pd.DataFrame(cols)
