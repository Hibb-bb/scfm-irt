"""Reference estimator: joint maximum likelihood via gradient ascent.

Transparent and easy to audit — it optimizes exactly the log-likelihood written in MATH.md
section 4 — which makes it the right thing to cross-check the Bayesian engine against.

Uses analytical gradients in numpy so the analysis env does not need torch for the smoke path.

It is *not* the headline estimator. Joint ML item estimates are inconsistent as I grows with P
fixed, which is precisely this project's regime, and uncertainty comes from the observed-
information diagonal (asymptotic SEs), not posteriors. Use it for validation and small runs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from scfm_irt.irt.models import IRTFit, IRTModelName, icc
from scfm_irt.response.build import aggregate_items, to_matrix


def fit_autodiff(
    resp: pd.DataFrame,
    model: IRTModelName = "1pl",
    item_granularity: Literal["cell", "type_dataset"] = "cell",
    seed: int = 0,
    n_steps: int = 2000,
    lr: float = 0.05,
    device: str = "cpu",  # kept for API compatibility; numpy path ignores it
) -> IRTFit:
    """Maximize the joint log-likelihood over abilities and item parameters.

    Implementation notes:
      - Pivot with `response.to_matrix`; keep Y as (P, I) throughout and never transpose.
      - Parameterize `a` as `exp(log_a)` for 2PL to enforce positivity.
      - Anchor with `mean(theta) = 0` for 1PL, plus `mean(log_a)=0` for 2PL.
      - Report `ability_sd` from the observed-information diagonal, and say plainly in the
        diagnostics that these are asymptotic standard errors, not posterior SDs.
    """
    del device  # numpy path
    if model not in ("1pl", "2pl"):
        raise NotImplementedError(f"mle_autodiff currently supports 1pl/2pl, not {model!r}")

    if item_granularity == "type_dataset":
        agg = aggregate_items(resp)
        wide_p = agg.pivot_table(
            index="model_id", columns="group_id", values="n_correct", aggfunc="sum"
        )
        wide_n = agg.pivot_table(
            index="model_id", columns="group_id", values="n_total", aggfunc="sum"
        )
        model_ids = sorted(wide_p.index.astype(str))
        item_ids = sorted(wide_p.columns.astype(str))
        Y = wide_p.reindex(index=model_ids, columns=item_ids).to_numpy(dtype=float)
        N = wide_n.reindex(index=model_ids, columns=item_ids).to_numpy(dtype=float)
        Y_frac = np.divide(Y, N, out=np.zeros_like(Y, dtype=float), where=N > 0)
        return _fit_matrix(
            Y_frac,
            model_ids,
            item_ids,
            model=model,
            seed=seed,
            n_steps=n_steps,
            lr=lr,
            weights=N,
            item_granularity=item_granularity,
        )

    Y, model_ids, item_ids = to_matrix(resp)
    return _fit_matrix(
        Y.astype(float),
        model_ids,
        item_ids,
        model=model,
        seed=seed,
        n_steps=n_steps,
        lr=lr,
        weights=None,
        item_granularity=item_granularity,
    )


def _fit_matrix(
    Y: np.ndarray,
    model_ids: list[str],
    item_ids: list[str],
    *,
    model: IRTModelName,
    seed: int,
    n_steps: int,
    lr: float,
    weights: np.ndarray | None,
    item_granularity: str,
) -> IRTFit:
    rng = np.random.default_rng(seed)
    n_models, n_items = Y.shape
    if n_models < 2 or n_items < 2:
        raise ValueError(f"need at least 2 models and 2 items, got {(n_models, n_items)}")

    # Initialize from observed margins (Rasch-style logits).
    eps = 1e-4
    row_mean = np.clip(Y.mean(axis=1), eps, 1 - eps)
    col_mean = np.clip(Y.mean(axis=0), eps, 1 - eps)
    theta = np.log(row_mean / (1 - row_mean))
    b = -np.log(col_mean / (1 - col_mean))
    theta = theta - theta.mean()
    b = b - b.mean()
    log_a = rng.normal(0.0, 0.01, size=n_items) if model == "2pl" else np.zeros(n_items)

    W = np.ones_like(Y) if weights is None else weights.astype(float)

    # Adam — raw GD with sum-gradients blows up when I is large (~sum of I residuals).
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    m_theta = np.zeros_like(theta)
    v_theta = np.zeros_like(theta)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)
    m_la = np.zeros_like(log_a)
    v_la = np.zeros_like(log_a)

    # Scale step so gradients averaged over responses stay O(1).
    step = lr
    history: list[float] = []

    for t in range(1, n_steps + 1):
        a = np.exp(log_a) if model == "2pl" else np.ones(n_items)
        logits = a[None, :] * (theta[:, None] - b[None, :])
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        resid = W * (Y - p)

        g_theta = resid.sum(axis=1) / n_items
        g_b = -resid.sum(axis=0) / n_models

        m_theta = beta1 * m_theta + (1 - beta1) * g_theta
        v_theta = beta2 * v_theta + (1 - beta2) * g_theta**2
        m_b = beta1 * m_b + (1 - beta1) * g_b
        v_b = beta2 * v_b + (1 - beta2) * g_b**2

        mh_t = m_theta / (1 - beta1**t)
        vh_t = v_theta / (1 - beta2**t)
        mh_b = m_b / (1 - beta1**t)
        vh_b = v_b / (1 - beta2**t)

        theta = theta + step * mh_t / (np.sqrt(vh_t) + eps_adam)
        b = b + step * mh_b / (np.sqrt(vh_b) + eps_adam)

        if model == "2pl":
            g_log_a = (resid * (theta[:, None] - b[None, :]) * a[None, :]).sum(axis=0) / n_models
            m_la = beta1 * m_la + (1 - beta1) * g_log_a
            v_la = beta2 * v_la + (1 - beta2) * g_log_a**2
            mh_la = m_la / (1 - beta1**t)
            vh_la = v_la / (1 - beta2**t)
            log_a = log_a + step * mh_la / (np.sqrt(vh_la) + eps_adam)
            log_a = log_a - log_a.mean()

        theta = theta - theta.mean()

        if t % max(1, n_steps // 20) == 0 or t == n_steps:
            ll = float(np.sum(W * (Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12))))
            history.append(ll)

    a = np.exp(log_a) if model == "2pl" else np.ones(n_items)
    p = icc(theta[:, None], b[None, :], a[None, :])

    info_theta = (W * (a[None, :] ** 2) * p * (1 - p)).sum(axis=1)
    info_b = (W * (a[None, :] ** 2) * p * (1 - p)).sum(axis=0)
    ability_sd = 1.0 / np.sqrt(np.maximum(info_theta, 1e-8))
    difficulty_sd = 1.0 / np.sqrt(np.maximum(info_b, 1e-8))

    anchoring = "mean(theta)=0"
    if model == "2pl":
        anchoring += ", mean(log_a)=0"

    return IRTFit(
        model=model,
        method="mle_autodiff",
        item_granularity=item_granularity,
        model_ids=list(model_ids),
        item_ids=list(item_ids),
        ability_mean=theta,
        ability_sd=ability_sd,
        difficulty_mean=b,
        difficulty_sd=difficulty_sd,
        discrimination_mean=a if model == "2pl" else None,
        discrimination_sd=None,
        anchoring=anchoring,
        diagnostics={
            "n_steps": n_steps,
            "final_ll": history[-1] if history else None,
            "ll_history": history,
            "uncertainty": "observed-information asymptotic SE (not posterior SD)",
            "seed": seed,
        },
    )
