"""Tests for leave-one-model-out calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scfm_irt.analysis.loo_calibration import (
    align_difficulty,
    loo_ability_table,
    loo_calibrate_banks,
)
from scfm_irt.irt.fit_pyirt import _fit_mfvi
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import build_response_table, to_matrix


def _synthetic_resp(seed: int = 0) -> tuple[pd.DataFrame, IRTFit]:
    rng = np.random.default_rng(seed)
    model_ids = ["m_a", "m_b", "m_c"]
    item_ids = [f"cell_{i}" for i in range(40)]
    theta = np.array([1.0, 0.0, -0.8])
    b = rng.normal(size=len(item_ids))
    Y = np.zeros((3, len(item_ids)), dtype=np.int8)
    for p, th in enumerate(theta):
        p_corr = 1.0 / (1.0 + np.exp(-(th - b)))
        Y[p] = (rng.random(len(item_ids)) < p_corr).astype(np.int8)

    rows = []
    for pi, mid in enumerate(model_ids):
        for ii, iid in enumerate(item_ids):
            rows.append(
                {
                    "model_id": mid,
                    "dataset": "syn",
                    "split_scheme": "stratified",
                    "cell_id": iid,
                    "cell_type": "T",
                    "correct": int(Y[pi, ii]),
                }
            )
    resp = build_response_table([pd.DataFrame(rows)])
    fit = _fit_mfvi(
        Y.astype(float),
        model_ids,
        item_ids,
        model="1pl",
        seed=seed,
        n_steps=300,
        hierarchical=True,
        weights=None,
        item_granularity="cell",
        engine="test",
    )
    return resp, fit


def test_loo_calibration_excludes_holdout_model():
    resp, fit = _synthetic_resp()
    stability, loo_fits = loo_calibrate_banks(resp, fit, n_steps=200, seed=0)
    assert len(stability) == 3
    for holdout, fit_loo in loo_fits.items():
        assert holdout not in fit_loo.model_ids
        assert len(fit_loo.model_ids) == 2


def test_loo_ability_uses_holdout_excluded_bank():
    resp, fit = _synthetic_resp()
    _, loo_fits = loo_calibrate_banks(resp, fit, n_steps=200, seed=0)
    table = loo_ability_table(resp, fit, loo_fits)
    _, model_ids, _ = to_matrix(resp)
    for row in table.itertuples():
        _, cal_ids, _ = to_matrix(resp.loc[resp["model_id"] != row.model_id])
        assert row.model_id not in cal_ids
        assert len(cal_ids) == len(model_ids) - 1
        assert np.isfinite(row.ability)
