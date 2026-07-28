"""Model ranking — the primary deliverable.

The temptation here is to print a sorted list and stop. Resist it: with ~10 respondents, two
models one rank apart are often statistically indistinguishable, and a bare ordering implies a
precision the data does not have.
"""

from __future__ import annotations

import pandas as pd

from scfm_irt.irt.models import IRTFit


def ability_ranking(fit: IRTFit, z: float = 1.96) -> pd.DataFrame:
    """Ranked abilities with credible intervals."""
    return fit.ability_table(z=z)


def pairwise_separation(fit: IRTFit) -> pd.DataFrame:
    """For each model pair, the ability gap and the probability that the ordering is real.

    Approximated as P(theta_i > theta_j) under independent normals, which is what turns a
    ranking into a defensible claim about which differences the evaluation can actually resolve.
    """
    raise NotImplementedError("TODO(phase-8): pairwise posterior comparison table")


def plot_abilities(fit: IRTFit, path: str) -> None:
    """Forest plot: ability with intervals, one row per model."""
    raise NotImplementedError("TODO(phase-8)")
