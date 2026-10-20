"""Unit tests for PI-style bottom-percent retirement scoring and scPRINT load patch."""

from sim.irt.ops import (
    item_discrimination_top_models,
    retire_bottom_pct_ids,
    top_model_thetas,
)


def test_top_model_thetas_takes_top_half():
    theta_hat = {"a": 1.0, "b": 0.5, "c": 0.0, "d": -1.0}
    assert top_model_thetas(theta_hat, 0.5) == [1.0, 0.5]


def test_item_discrimination_highest_at_matching_difficulty():
    thetas = [0.0, 1.0]
    assert item_discrimination_top_models(0.5, thetas) > item_discrimination_top_models(3.0, thetas)


def test_retire_bottom_pct_ids():
    scored = [("easy", 0.01), ("mid", 0.1), ("hard", 0.2), ("best", 0.25)]
    dropped = retire_bottom_pct_ids(scored, 0.5)
    assert dropped == ["easy", "mid"]
