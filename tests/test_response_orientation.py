"""Guard the (P models, I cells) orientation convention.

A persons/items transposition runs cleanly and corrupts every downstream number, so it gets its
own test with a hand-built case whose answer is obvious by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scfm_irt.response.build import aggregate_items, make_cell_id, to_matrix
from scfm_irt.response.schema import validate_response_table


def _toy_table() -> pd.DataFrame:
    """3 models x 4 cells. `strong` gets everything right, `weak` everything wrong."""
    rows = []
    per_model_correct = {
        "strong": [1, 1, 1, 1],
        "middle": [1, 1, 0, 0],
        "weak": [0, 0, 0, 0],
    }
    for model_id, corrects in per_model_correct.items():
        for j, correct in enumerate(corrects):
            rows.append(
                {
                    "model_id": model_id,
                    "dataset": "toy",
                    "split_scheme": "stratified",
                    "cell_id": make_cell_id("toy", f"cell{j}"),
                    "cell_type": "A" if j < 2 else "B",
                    "correct": correct,
                }
            )
    return pd.DataFrame(rows)


def test_matrix_is_models_by_cells():
    Y, model_ids, cell_ids = to_matrix(validate_response_table(_toy_table()))

    assert Y.shape == (3, 4), "Y must be (n_models, n_cells)"
    assert len(model_ids) == 3
    assert len(cell_ids) == 4


def test_row_sums_track_model_strength():
    """Row sums are per-model scores. If Y were transposed, these would be per-cell counts."""
    Y, model_ids, _ = to_matrix(validate_response_table(_toy_table()))
    scores = dict(zip(model_ids, Y.sum(axis=1), strict=True))

    assert scores["strong"] == 4
    assert scores["middle"] == 2
    assert scores["weak"] == 0
    assert scores["strong"] > scores["middle"] > scores["weak"]


def test_values_land_in_the_right_cells():
    """Check one specific entry, not just the marginals — marginals can match under a flip."""
    resp = validate_response_table(_toy_table())
    Y, model_ids, cell_ids = to_matrix(resp)

    p = model_ids.index("middle")
    i = cell_ids.index(make_cell_id("toy", "cell2"))
    assert Y[p, i] == 0

    i_easy = cell_ids.index(make_cell_id("toy", "cell0"))
    assert Y[p, i_easy] == 1


def test_synthetic_simulation_keeps_orientation_through_the_pivot():
    """Round-trip simulate -> long table -> matrix and require the array back unchanged."""
    from scfm_irt.irt.synthetic import simulate_responses, to_long

    data = simulate_responses(n_models=6, n_items=50, seed=0)
    Y, model_ids, cell_ids = to_matrix(validate_response_table(to_long(data)))

    assert Y.shape == data.Y.shape == (6, 50)
    assert model_ids == data.model_ids
    np.testing.assert_array_equal(Y, data.Y)


def test_validation_rejects_duplicate_responses():
    resp = _toy_table()
    with pytest.raises(ValueError, match="duplicated"):
        validate_response_table(pd.concat([resp, resp.head(1)], ignore_index=True), strict=False)


def test_validation_rejects_non_binary_correct():
    resp = _toy_table()
    resp.loc[0, "correct"] = 2
    with pytest.raises(ValueError, match="0/1"):
        validate_response_table(resp)


def test_validation_rejects_ragged_table():
    resp = _toy_table().iloc[:-1]
    with pytest.raises(ValueError, match="ragged"):
        validate_response_table(resp)


def test_aggregate_items_collapses_to_type_dataset_groups():
    agg = aggregate_items(validate_response_table(_toy_table()))

    assert set(agg["group_id"]) == {"toy|A", "toy|B"}
    assert len(agg) == 6  # 3 models x 2 groups

    middle_b = agg[(agg.model_id == "middle") & (agg.cell_type == "B")].iloc[0]
    assert middle_b["n_total"] == 2
    assert middle_b["n_correct"] == 0
