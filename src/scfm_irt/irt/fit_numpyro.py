"""Custom estimators in numpyro: explanatory, aggregated, and multidimensional variants.

The escape hatch for structure py-irt does not express. Everything here is easiest to write in
the GLMM view of MATH.md section 6.4 — Rasch is a logistic mixed model with crossed random
effects, a model effect crossed with a cell effect:

    y_pi ~ Bernoulli(sigma(theta_p - b_i))
    theta_p ~ N(0, sigma_theta^2)      # the model effect
    b_i     ~ N(mu_b, sigma_b^2)       # the cell effect, crossed with it

Once it is written that way, the extensions are drop-in.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from scfm_irt.irt.models import IRTFit, IRTModelName


def fit_numpyro(
    resp: pd.DataFrame,
    model: IRTModelName = "1pl",
    item_granularity: Literal["cell", "type_dataset"] = "cell",
    item_covariates: list[str] | None = None,
    seed: int = 0,
    inference: Literal["svi", "nuts"] = "svi",
    n_steps: int = 2000,
) -> IRTFit:
    """Fit the crossed-random-effects Rasch model, optionally explanatory or aggregated.

    With `item_covariates`, difficulty becomes `b_i = w' x_i + eps_i`, `eps_i ~ N(0, sigma^2)` —
    partial pooling rather than pure LLTM, so unexplained item variation still has somewhere to
    go. This collapses I free difficulties into a handful of coefficients (huge variance
    reduction in this regime) and answers "what makes a cell hard to annotate", which is a
    result in its own right rather than a means to one.

    With `item_granularity="type_dataset"`, fit the binomial form
    `k_pg ~ Binomial(n_g, sigma(theta_p - b_g))` over (dataset, cell_type) groups.

    Build design matrices for categorical covariates (cell_type, dataset, batch) with an
    explicit, recorded reference level; store coefficient posteriors in `diagnostics` under
    `covariate_effects` so the report can surface difficulty-vs-covariate effects.

    NUTS is feasible only on the aggregated granularity or a subsample; SVI is the default at
    per-cell scale.
    """
    raise NotImplementedError("TODO(phase-7): numpyro GLMM, explanatory + aggregated variants")


def fit_mirt(
    resp: pd.DataFrame,
    n_dims: int = 2,
    seed: int = 0,
) -> IRTFit:
    """Multidimensional IRT as a dimensionality check: P = sigma(a_i' theta_p - b_i).

    Unidimensionality assumes a single "embedding quality" axis. If a model is strong on immune
    cells and weak on neurons, that assumption fails and the ranking is hiding structure. Report
    whether the second dimension explains meaningful variance as a robustness check, not as a
    headline model (MATH.md section 11.3).
    """
    raise NotImplementedError("TODO(phase-7): 2D MIRT + residual PCA comparison")
