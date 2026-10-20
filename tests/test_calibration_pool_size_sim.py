"""Tests for calibration pool size simulation."""

from __future__ import annotations

import numpy as np

from scfm_irt.analysis.calibration_pool_size_sim import (
    run_calibration_size_study,
    summarize_calibration_size_stability,
    synthetic_irt_responses,
)


def test_calibration_size_study_improves_with_k():
    resp, fit = synthetic_irt_responses(n_models=6, n_items=80, seed=1)
    stability = run_calibration_size_study(
        resp, fit, n_subsamples=2, n_steps=200, seed=0
    )
    summary = summarize_calibration_size_stability(stability)
    ks = summary["n_calibration_models"].to_numpy()
    rs = summary["difficulty_pearson_r_mean"].to_numpy()
    assert len(ks) >= 2
    assert rs[-1] >= rs[0] - 0.05
    assert np.all(np.isfinite(rs))
