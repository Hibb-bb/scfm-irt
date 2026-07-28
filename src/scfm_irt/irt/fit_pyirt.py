"""Default estimator: scalable Bayesian variational inference via py-irt (Pyro).

Chosen because it gives posterior means *and variances* for every parameter, which is what the
few-respondents regime demands: with ~10 models, the point ranking is cheap and the question
is whether any two models are actually distinguishable.

Hierarchical priors on item difficulty do the other necessary job — shrinking thousands of
per-cell difficulties, each informed by only ~P binary responses, toward the population
(MATH.md sections 6.3 and 10).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from scfm_irt.irt.models import IRTFit, IRTModelName


def fit_pyirt(
    resp: pd.DataFrame,
    model: IRTModelName = "1pl",
    item_granularity: Literal["cell", "type_dataset"] = "cell",
    seed: int = 0,
    n_steps: int = 2000,
    hierarchical: bool = True,
    device: str = "cpu",
) -> IRTFit:
    """Fit via py-irt's SVI and return posterior summaries.

    Implementation notes:
      - py-irt consumes its own JSONL-ish format keyed by subject and item ids; build it from
        the long table directly rather than from a pivoted matrix, so there is no orientation
        decision to get wrong here.
      - Priors: theta ~ N(0,1); b ~ N(mu_b, sigma_b^2) with hyperpriors when `hierarchical`;
        a ~ LogNormal for 2PL. Priors are what fix location and scale, so record
        `anchoring="priors (theta ~ N(0,1), hierarchical item priors)"`.
      - Map py-irt's subject/item ordering back onto `model_ids`/`item_ids` explicitly. Do not
        assume its internal order matches the sorted ids.
      - Keep the ELBO trace in `diagnostics` — a non-converged SVI run looks like a
        confident answer otherwise.
    """
    raise NotImplementedError("TODO(phase-4): py-irt SVI fit + posterior extraction")
