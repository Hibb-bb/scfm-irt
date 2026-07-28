"""Reference estimator: direct maximum likelihood by autodiff (PyTorch).

Transparent and easy to audit — it optimizes exactly the log-likelihood written in MATH.md
section 4 — which makes it the right thing to cross-check the Bayesian engine against.

It is *not* the headline estimator. Joint ML item estimates are inconsistent as I grows with P
fixed, which is precisely this project's regime, and it returns no uncertainty. Use it for
validation and small runs.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from scfm_irt.irt.models import IRTFit, IRTModelName


def fit_autodiff(
    resp: pd.DataFrame,
    model: IRTModelName = "1pl",
    item_granularity: Literal["cell", "type_dataset"] = "cell",
    seed: int = 0,
    n_steps: int = 2000,
    lr: float = 0.05,
    device: str = "cpu",
) -> IRTFit:
    """Maximize the joint log-likelihood over abilities and item parameters.

    Implementation notes:
      - Pivot with `response.to_matrix`; keep Y as (P, I) throughout and never transpose.
      - Parameterize `a` as `exp(log_a)` for 2PL to enforce positivity.
      - Anchor with `mean(theta) = 0` for 1PL, plus `var(theta) = 1` for 2PL (MATH.md
        section 5); record the choice in `IRTFit.anchoring`.
      - Report `ability_sd` from the observed-information diagonal, and say plainly in the
        diagnostics that these are asymptotic standard errors, not posterior SDs.
    """
    raise NotImplementedError("TODO(phase-4): torch JML/MML reference implementation")
