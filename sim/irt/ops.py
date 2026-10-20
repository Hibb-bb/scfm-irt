"""Algorithm 1 operations: Fit, Route, Filter, Value, Acquire."""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar

from scfm_irt.irt.models import item_information, sigmoid


def fit_ability(
    u: np.ndarray,
    b: np.ndarray,
    *,
    tau: float = 1.0,
    theta0: float = 0.0,
) -> float:
    """Newton/RMS ability MLE: argmax sum [u log σ(θ−b) + (1−u) log(1−σ(θ−b))] − θ²/(2τ²)."""
    u = np.asarray(u, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if u.size == 0:
        return float(theta0)
    theta = float(theta0)
    inv_tau2 = 1.0 / (tau * tau)
    for _ in range(40):
        z = np.clip(theta - b, -30.0, 30.0)
        p = sigmoid(z)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        grad = float(np.sum(u - p) - theta * inv_tau2)
        hess = float(-np.sum(p * (1.0 - p)) - inv_tau2)
        if abs(hess) < 1e-9:
            break
        step = grad / hess
        theta = float(np.clip(theta - step, -12.0, 12.0))
        if abs(step) < 1e-6:
            break
    return theta


def route_items(
    theta: float,
    item_ids: np.ndarray,
    b_hat: np.ndarray,
    m: int,
) -> np.ndarray:
    """Return indices of m items with difficulty nearest to theta."""
    if m <= 0 or len(item_ids) == 0:
        return np.array([], dtype=int)
    m = min(m, len(item_ids))
    dist = np.abs(b_hat - theta)
    return np.argsort(dist)[:m]


def filter_update(
    b_hat: float,
    p_var: float,
    *,
    theta_hat: float,
    u: float,
    q: float,
) -> tuple[float, float]:
    """Kalman logistic update for one item difficulty observation."""
    p_hat = float(sigmoid(theta_hat - b_hat))
    p_tilde = p_var + q
    gain = p_tilde / (1.0 + p_tilde * p_hat * (1.0 - p_hat))
    b_new = b_hat + gain * (p_hat - u)
    p_new = 1.0 / (1.0 / p_tilde + p_hat * (1.0 - p_hat))
    return float(b_new), float(p_new)


def item_value(b: float, mu: float, s: float) -> float:
    """Expected Fisher information E_{θ~N(μ,s²)}[I_b(θ)] via Gauss-Hermite quadrature."""
    if s <= 1e-6:
        return float(item_information(mu, b))

    def integrand(theta: float) -> float:
        return float(item_information(theta, b) * _normal_pdf(theta, mu, s))

    val, _ = integrate.quad(integrand, mu - 6 * s, mu + 6 * s, limit=50)
    return float(val)


def top_model_thetas(theta_hat: dict[str, float], top_frac: float = 0.5) -> list[float]:
    """Fitted abilities for the top ``top_frac`` of models (at least one)."""
    if not theta_hat:
        return []
    ordered = sorted(theta_hat.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, int(np.ceil(len(ordered) * top_frac)))
    return [theta for _, theta in ordered[:k]]


def item_discrimination_top_models(b: float, thetas: list[float]) -> float:
    """Mean Rasch Fisher information I_b(θ) over a set of model abilities."""
    if not thetas:
        return 0.0
    return float(np.mean([item_information(float(t), b) for t in thetas]))


def retire_bottom_pct_ids(
    scored: list[tuple[str, float]],
    bottom_pct: float,
) -> list[str]:
    """Item ids in the lowest ``bottom_pct`` fraction by discrimination score."""
    if not scored or bottom_pct <= 0:
        return []
    n_retire = min(len(scored), max(1, int(np.ceil(len(scored) * bottom_pct))))
    ranked = sorted(scored, key=lambda x: x[1])
    return [iid for iid, _ in ranked[:n_retire]]


def bank_value(b_vals: np.ndarray, mu: float, s: float) -> float:
    return float(sum(item_value(float(b), mu, s) for b in b_vals))


def acquire_items(
    kappa: int,
    ceiling: float,
    *,
    band_width: float,
    p0: float,
    start_id: int = 0,
) -> list[dict]:
    """Commission kappa provisional items evenly spaced on [ceiling, ceiling + band_width]."""
    if kappa <= 0:
        return []
    out: list[dict] = []
    for ell in range(kappa):
        frac = (ell + 0.5) / kappa
        b_init = ceiling + frac * band_width
        out.append(
            {
                "item_id": f"acq_{start_id + ell}",
                "b_hat": float(b_init),
                "p_var": float(p0),
                "provisional": True,
                "anchor": False,
            }
        )
    return out


def _normal_pdf(x: float, mu: float, s: float) -> float:
    z = (x - mu) / s
    return float(np.exp(-0.5 * z * z) / (s * np.sqrt(2 * np.pi)))
