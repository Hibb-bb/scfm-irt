"""Field drift: real model release order vs synthetic rising ability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FieldState:
    mu: float
    s: float
    active_model_ids: list[str]
    thetas: dict[str, float]


def _top_k_thetas(thetas: dict[str, float], top_k: int | None) -> dict[str, float]:
    """Keep the highest-ability models when ``top_k`` is set (frontier field window)."""
    if not thetas or top_k is None or top_k <= 0 or len(thetas) <= top_k:
        return thetas
    ranked = sorted(thetas.items(), key=lambda kv: kv[1], reverse=True)
    return dict(ranked[:top_k])


def _window_thetas(
    thetas: dict[str, float],
    *,
    top_k: int | None = None,
    top_frac: float | None = None,
) -> dict[str, float]:
    """Frontier window: top fraction and/or top-k cap (fraction takes precedence)."""
    if not thetas:
        return thetas
    ranked = sorted(thetas.items(), key=lambda kv: kv[1], reverse=True)
    if top_frac is not None and 0 < top_frac < 1:
        k = max(1, int(np.ceil(len(ranked) * top_frac)))
        return dict(ranked[:k])
    return _top_k_thetas(thetas, top_k)


def estimate_field(
    thetas: dict[str, float],
    *,
    top_k: int | None = None,
    top_frac: float | None = None,
) -> tuple[float, float]:
    if not thetas:
        return 0.0, 1.0
    window = _window_thetas(thetas, top_k=top_k, top_frac=top_frac)
    vals = np.asarray(list(window.values()), dtype=float)
    return float(vals.mean()), float(max(vals.std(ddof=1) if len(vals) > 1 else 1.0, 0.25))


def active_models_real(
    all_model_ids: list[str],
    model_release: pd.Series,
    round_date: pd.Timestamp,
    *,
    baseline_models: set[str] | None = None,
) -> list[str]:
    baseline_models = baseline_models or set()
    active = [
        m
        for m in all_model_ids
        if m not in baseline_models and model_release.get(m, round_date) <= round_date
    ]
    return active or list(all_model_ids[:1])


def synthetic_thetas(
    rng: np.random.Generator,
    *,
    n_models: int,
    mu: float,
    s: float,
    offsets: dict[str, float] | None = None,
) -> dict[str, float]:
    """Persistent synthetic models: θ_j = μ_t + offset_j (offset fixed across rounds)."""
    if offsets is None:
        return {f"synth_{i:02d}": float(rng.normal(mu, s)) for i in range(n_models)}
    return {m: mu + offsets[m] for m in offsets}


def init_synthetic_offsets(rng: np.random.Generator, n_models: int, s: float) -> dict[str, float]:
    return {f"synth_{i:02d}": float(rng.normal(0.0, s)) for i in range(n_models)}


def field_at_round(
    *,
    drift_mode: str,
    round_idx: int,
    round_date: pd.Timestamp,
    mu0: float,
    r: float,
    s: float,
    all_model_ids: list[str],
    model_release: pd.Series,
    oracle_theta: dict[str, float],
    rng: np.random.Generator,
    n_synthetic: int = 10,
    synth_offsets: dict[str, float] | None = None,
    baseline_models: set[str] | None = None,
    field_window_top_k: int | None = None,
    field_window_top_frac: float | None = None,
) -> FieldState:
    mu = mu0 + r * round_idx
    if drift_mode == "real":
        active = active_models_real(
            all_model_ids, model_release, round_date, baseline_models=baseline_models
        )
        thetas = {m: oracle_theta[m] for m in active if m in oracle_theta}
    else:
        active = [f"synth_{i:02d}" for i in range(n_synthetic)]
        thetas = synthetic_thetas(
            rng, n_models=n_synthetic, mu=mu, s=s, offsets=synth_offsets
        )
    mu_field, s_field = estimate_field(
        thetas, top_k=field_window_top_k, top_frac=field_window_top_frac
    )
    return FieldState(mu=mu_field, s=s_field, active_model_ids=active, thetas=thetas)
