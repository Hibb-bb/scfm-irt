"""Model-fit diagnostics and the assumption checks of MATH.md section 13.

Local independence is the assumption most at risk here and the one worth checking hardest:
cells of the same (dataset, cell_type) are correlated, because a model that fails a rare type
tends to fail *every* cell of it. That inflates the effective sample size and makes intervals
look tighter than they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scfm_irt.irt.models import IRTFit


def standardized_residuals(fit: IRTFit, Y: np.ndarray) -> np.ndarray:
    """(P, I) standardized residuals: (y - P_hat) / sqrt(P_hat (1 - P_hat))."""
    raise NotImplementedError("TODO(phase-8)")


def infit_outfit(fit: IRTFit, Y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infit/outfit mean-square statistics per item and per person.

    Values far from 1 flag items or models the model does not describe well — for a *person*,
    an outlying statistic often means that model is strong on some tissues and weak on others,
    which is the unidimensionality assumption failing rather than noise.
    """
    raise NotImplementedError("TODO(phase-8)")


def residual_pca(fit: IRTFit, Y: np.ndarray, n_components: int = 5) -> pd.DataFrame:
    """PCA of standardized residuals — the cheap unidimensionality check.

    Meaningful variance in the first residual component means one ability axis is not enough;
    follow up with the 2D MIRT fit (MATH.md section 11.3).
    """
    raise NotImplementedError("TODO(phase-8)")


def local_dependence_check(resp: pd.DataFrame, fit: IRTFit) -> pd.DataFrame:
    """Within-(dataset, cell_type) residual correlation, per group.

    High values are the direct evidence for running the aggregated `type_dataset` item mode as
    the robustness comparison (MATH.md section 11.2).
    """
    raise NotImplementedError("TODO(phase-8)")
