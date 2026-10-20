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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeModel:
    """A fitted probe plus the feature scaler and the chosen regularization strength."""

    classifier: Any  # sklearn LogisticRegression
    scaler: Any  # sklearn StandardScaler
    classes: np.ndarray
    C: float
    penalty: str
    cv_scores: dict[float, float] | None = None  # macro-F1 per candidate C, when C was selected


_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


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
    y_train = np.asarray(y_train)
    scaler = StandardScaler()
    X = scaler.fit_transform(emb_train)

    cv_scores: dict[float, float] | None = None
    chosen = C
    if chosen is None:
        cv_scores = {}
        # Need enough samples per class for stratified CV; fall back to fixed C otherwise.
        classes, counts = np.unique(y_train, return_counts=True)
        n_splits = min(5, int(counts.min()))
        if n_splits < 2 or len(classes) < 2:
            chosen = 1.0
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            best_score = -np.inf
            for cand in _C_GRID:
                clf = _make_clf(cand, penalty, seed, max_iter)
                scores = cross_val_score(clf, X, y_train, cv=cv, scoring="f1_macro")
                mean = float(scores.mean())
                cv_scores[cand] = mean
                if mean > best_score:
                    best_score = mean
                    chosen = cand

    assert chosen is not None
    clf = _make_clf(chosen, penalty, seed, max_iter)
    clf.fit(X, y_train)
    return ProbeModel(
        classifier=clf,
        scaler=scaler,
        classes=clf.classes_,
        C=float(chosen),
        penalty=penalty,
        cv_scores=cv_scores,
    )


def _make_clf(C: float, penalty: str, seed: int, max_iter: int) -> LogisticRegression:
    solver = "saga" if penalty == "l1" else "lbfgs"
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
    )


def score_cells(probe: ProbeModel, emb_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Score test cells: one row per cell, with the `correct` bit that IRT consumes.

    Columns: y_true, y_pred, correct (int8, 0/1), p_true (predicted probability of the true
    class, kept for diagnostics but not used as the IRT response).
    """
    y_test = np.asarray(y_test)
    X = probe.scaler.transform(emb_test)
    y_pred = probe.classifier.predict(X)
    proba = probe.classifier.predict_proba(X)
    class_to_col = {c: i for i, c in enumerate(probe.classifier.classes_)}

    p_true = np.zeros(len(y_test), dtype=np.float64)
    for i, yt in enumerate(y_test):
        col = class_to_col.get(yt)
        p_true[i] = float(proba[i, col]) if col is not None else 0.0

    correct = (y_pred == y_test).astype(np.int8)
    return pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "correct": correct,
            "p_true": p_true,
        }
    )


def probe_summary(scores: pd.DataFrame) -> dict[str, float]:
    """Accuracy and macro-F1 for the run log. Report both — they diverge under imbalance."""
    y_true = scores["y_true"].to_numpy()
    y_pred = scores["y_pred"].to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test": float(len(scores)),
        "mean_correct": float(scores["correct"].mean()),
    }
