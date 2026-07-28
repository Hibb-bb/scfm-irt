"""Multinomial logistic-regression probe.

No fine-tuning: embeddings are frozen and only the linear head is trained. The output that
matters is one bit per test cell — correct or not — which becomes a single IRT response.

Deterministic given the seed: same embeddings + same seed must produce the same response table
(plan section 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass
class ProbeModel:
    """A fitted probe plus the feature scaler and the chosen regularization strength."""

    classifier: Any  # sklearn LogisticRegression
    scaler: Any  # sklearn StandardScaler
    classes: np.ndarray
    C: float
    penalty: str
    cv_scores: dict[float, float] | None = None  # macro-F1 per candidate C, when C was selected


def fit_probe(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    C: float | None = None,
    penalty: Literal["l2", "l1"] = "l2",
    seed: int = 0,
    max_iter: int = 1000,
) -> ProbeModel:
    """Standardize features and fit multinomial LR.

    `C=None` selects it by stratified CV on the train split, scored by macro-F1 — accuracy
    rewards ignoring rare types, and rare types are exactly the hard items IRT needs
    (plan section 10.9).
    """
    raise NotImplementedError("TODO(phase-3): StandardScaler + LogisticRegression, CV over C")


def score_cells(probe: ProbeModel, emb_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Score test cells: one row per cell, with the `correct` bit that IRT consumes.

    Columns: y_true, y_pred, correct (int8, 0/1), p_true (predicted probability of the true
    class, kept for diagnostics but not used as the IRT response).
    """
    raise NotImplementedError("TODO(phase-3): predict, compare, return the tidy frame")


def probe_summary(scores: pd.DataFrame) -> dict[str, float]:
    """Accuracy and macro-F1 for the run log. Report both — they diverge under imbalance."""
    raise NotImplementedError("TODO(phase-3)")
