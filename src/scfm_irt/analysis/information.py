"""Test information curve — where on the ability scale this evaluation actually resolves."""

from __future__ import annotations

import numpy as np

from scfm_irt.irt.models import IRTFit, test_information


def information_curve(
    fit: IRTFit,
    theta_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_grid, I(theta)) over the ability range.

    SE(theta) = 1 / sqrt(I(theta)). If the fitted abilities sit where information is low, the
    ranking is being read off a part of the scale the item pool does not measure well.
    """
    grid = np.linspace(-4, 4, 201) if theta_grid is None else np.asarray(theta_grid)
    info = test_information(grid, fit.difficulty_mean, fit.discrimination_mean)
    return grid, info


def plot_information(fit: IRTFit, path: str) -> None:
    """Plot I(theta) with the fitted model abilities marked on the same axis."""
    raise NotImplementedError("TODO(phase-8)")


def plot_iccs(fit: IRTFit, item_ids: list[str], path: str) -> None:
    """Item characteristic curves for representative items, with empirical proportions overlaid.

    The overlay is the monotonicity check: P(correct) should increase with ability.
    """
    raise NotImplementedError("TODO(phase-8)")
