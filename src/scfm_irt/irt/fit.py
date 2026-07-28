"""The single entry point for fitting IRT — dispatches to one of three engines.

Callers use `fit_irt` and never import an engine module directly, so swapping estimators is a
config change and cross-checking one against another is trivial.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from scfm_irt.irt.models import IRTFit, IRTModelName

Method = Literal["mle_autodiff", "bayes_variational", "numpyro"]


def fit_irt(
    resp: pd.DataFrame,
    model: IRTModelName = "1pl",
    method: Method = "bayes_variational",
    item_granularity: Literal["cell", "type_dataset"] = "cell",
    item_covariates: list[str] | None = None,
    seed: int = 0,
    **kwargs,
) -> IRTFit:
    """Fit an IRT model to the long response table.

    Args:
        resp: canonical long table (one row per model x cell).
        model: 1PL is the headline; per-item discrimination is weakly identified with only
            ~P responses per item, so 2PL is exploratory (MATH.md section 10).
        method: `bayes_variational` (py-irt) is the default and the only one returning
            calibrated posteriors at scale; `mle_autodiff` is the transparent reference
            implementation; `numpyro` handles the explanatory and aggregated variants.
        item_granularity: `cell` for per-cell items, `type_dataset` for the aggregated
            binomial form that mitigates local independence (MATH.md section 11.2).
        item_covariates: enables the explanatory model — difficulty is predicted from cell
            covariates instead of being freely estimated per cell (MATH.md section 11.1).
            Requires `method="numpyro"`.

    Whichever engine runs, the returned `IRTFit` records its anchoring, since abilities are
    only comparable within a single anchored fit (MATH.md section 5).
    """
    if item_covariates and method != "numpyro":
        raise ValueError(f"explanatory item covariates require method='numpyro', got {method!r}")

    if method == "bayes_variational":
        from scfm_irt.irt.fit_pyirt import fit_pyirt

        return fit_pyirt(resp, model=model, item_granularity=item_granularity, seed=seed, **kwargs)

    if method == "mle_autodiff":
        from scfm_irt.irt.fit_autodiff import fit_autodiff

        return fit_autodiff(
            resp, model=model, item_granularity=item_granularity, seed=seed, **kwargs
        )

    if method == "numpyro":
        from scfm_irt.irt.fit_numpyro import fit_numpyro

        return fit_numpyro(
            resp,
            model=model,
            item_granularity=item_granularity,
            item_covariates=item_covariates,
            seed=seed,
            **kwargs,
        )

    raise ValueError(f"unknown method {method!r}")
