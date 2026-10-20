"""Round grid and item arrival by release date."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RoundSchedule:
    round_idx: int
    round_date: pd.Timestamp


def build_rounds(
    item_release: pd.Series,
    *,
    n_rounds: int = 12,
    end_date: pd.Timestamp | None = None,
    start_date: pd.Timestamp | None = None,
) -> list[RoundSchedule]:
    """Evenly spaced rounds from ``start_date`` (or first item) to ``end_date``."""
    t0 = pd.Timestamp(start_date if start_date is not None else item_release.min())
    t1 = pd.Timestamp(end_date if end_date is not None else item_release.max())
    if t1 < t0:
        t1 = t0
    if n_rounds <= 1:
        return [RoundSchedule(0, t1)]
    dates = pd.date_range(t0, t1, periods=n_rounds)
    return [RoundSchedule(i, pd.Timestamp(d)) for i, d in enumerate(dates)]


def first_embedder_release(
    model_release: pd.Series,
    baseline_models: set[str],
) -> pd.Timestamp:
    """Earliest release date among non-baseline (embedder) models."""
    dates = [
        pd.Timestamp(model_release[m])
        for m in model_release.index
        if m not in baseline_models
    ]
    if not dates:
        return pd.Timestamp(model_release.min())
    return min(dates)


def items_available_as_of(
    item_ids: np.ndarray,
    item_release: pd.Series,
    as_of: pd.Timestamp,
) -> list[str]:
    return [iid for iid in item_ids if pd.Timestamp(item_release[iid]) <= as_of]


def items_available_by_round(
    item_ids: np.ndarray,
    item_release: pd.Series,
    rounds: list[RoundSchedule],
) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for rd in rounds:
        avail = [iid for iid in item_ids if pd.Timestamp(item_release[iid]) <= rd.round_date]
        out[rd.round_idx] = avail
    return out


def select_anchors(
    item_ids: np.ndarray,
    b_star: np.ndarray,
    *,
    n_anchors: int,
    rng: np.random.Generator,
) -> list[str]:
    """Pick anchors spanning the difficulty range (stratified quantiles)."""
    n_anchors = min(n_anchors, len(item_ids))
    if n_anchors <= 0:
        return []
    order = np.argsort(b_star)
    qs = np.linspace(0, len(order) - 1, n_anchors).astype(int)
    pick = order[qs]
    return [str(item_ids[i]) for i in pick]
