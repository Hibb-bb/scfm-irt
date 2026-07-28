"""Parameter recovery — the acceptance gate for every estimator (MATH.md section 9).

No real embedding model gets fit until this passes. The simulator side is implemented and
tested now; the recovery tests are marked xfail until the estimators land in Phase 4, so they
flip to passing exactly when the gate is met.
"""

from __future__ import annotations

import numpy as np
import pytest

# `test_information` is the IRT term for total information across items; import it under an
# alias so pytest does not collect it as a test case.
from scfm_irt.irt.models import icc, item_information
from scfm_irt.irt.models import test_information as total_information
from scfm_irt.irt.synthetic import recovery_metrics, simulate_responses


class TestResponseModels:
    def test_1pl_is_half_at_equal_ability_and_difficulty(self):
        assert icc(0.0, 0.0) == pytest.approx(0.5)
        assert icc(2.0, 2.0) == pytest.approx(0.5)

    def test_probability_increases_with_ability(self):
        thetas = np.linspace(-3, 3, 50)
        p = icc(thetas, b=0.0)
        assert np.all(np.diff(p) > 0)

    def test_2pl_discrimination_steepens_the_curve(self):
        low = icc(0.5, b=0.0, a=0.5)
        high = icc(0.5, b=0.0, a=3.0)
        assert high > low > 0.5

    def test_guessing_floor_raises_the_lower_asymptote(self):
        assert icc(-10.0, b=0.0, a=1.0, c=0.25) == pytest.approx(0.25, abs=1e-3)

    def test_sigmoid_stable_in_the_tails(self):
        assert np.isfinite(icc(-800.0, 0.0))
        assert icc(-800.0, 0.0) == pytest.approx(0.0, abs=1e-12)
        assert icc(800.0, 0.0) == pytest.approx(1.0, abs=1e-12)

    def test_information_peaks_where_ability_equals_difficulty(self):
        thetas = np.linspace(-4, 4, 401)
        info = item_information(thetas, b=1.0)
        assert thetas[int(np.argmax(info))] == pytest.approx(1.0, abs=0.05)

    def test_information_is_negligible_for_far_too_easy_items(self):
        """The ceiling-effect argument, quantified: trivial items carry almost no information."""
        at_level = item_information(0.0, b=0.0)
        way_too_easy = item_information(0.0, b=-6.0)
        assert way_too_easy < 0.01 * at_level

    def test_test_information_adds_across_items(self):
        b = np.zeros(10)
        grid = np.array([0.0])
        assert total_information(grid, b)[0] == pytest.approx(10 * 0.25)


class TestSimulator:
    def test_shape_and_orientation(self):
        data = simulate_responses(n_models=8, n_items=200, seed=0)
        assert data.Y.shape == (8, 200)
        assert data.n_models == 8
        assert data.n_items == 200

    def test_is_deterministic_given_seed(self):
        a = simulate_responses(n_models=5, n_items=100, seed=7)
        b = simulate_responses(n_models=5, n_items=100, seed=7)
        np.testing.assert_array_equal(a.Y, b.Y)
        np.testing.assert_allclose(a.theta, b.theta)

    def test_higher_ability_scores_higher(self):
        """The sanity check the whole method rests on: ability must track observed score."""
        data = simulate_responses(n_models=12, n_items=4000, seed=1)
        scores = data.Y.mean(axis=1)
        assert np.corrcoef(data.theta, scores)[0, 1] > 0.95

    def test_harder_items_are_answered_correctly_less_often(self):
        data = simulate_responses(n_models=30, n_items=800, seed=2)
        per_item = data.Y.mean(axis=0)
        assert np.corrcoef(data.b, per_item)[0, 1] < -0.9

    def test_2pl_draws_non_unit_discriminations(self):
        data = simulate_responses(n_models=6, n_items=100, model="2pl", seed=3)
        assert data.a.std() > 0
        assert np.all(data.a > 0)

    def test_long_table_round_trips(self):
        from scfm_irt.irt.synthetic import to_long
        from scfm_irt.response.schema import validate_response_table

        data = simulate_responses(n_models=4, n_items=25, seed=0)
        long = validate_response_table(to_long(data))
        assert len(long) == 4 * 25
        assert long["correct"].isin([0, 1]).all()


# --- The gate itself -------------------------------------------------------------------
# Recovery must hold for BOTH estimators before any real model is fit (Phase 4).

RECOVERY_METHODS = ["mle_autodiff", "bayes_variational"]


@pytest.mark.xfail(reason="TODO(phase-4): estimators not implemented", raises=NotImplementedError)
@pytest.mark.parametrize("method", RECOVERY_METHODS)
def test_recovers_abilities(method):
    """Abilities are informed by ~I responses each, so recovery should be near-perfect."""
    from scfm_irt.irt.fit import fit_irt
    from scfm_irt.irt.synthetic import to_long

    data = simulate_responses(n_models=10, n_items=5000, seed=0)
    fit = fit_irt(to_long(data), model="1pl", method=method, seed=0)

    m = recovery_metrics(data, fit.ability_mean, fit.difficulty_mean, fit.ability_sd)
    assert m["ability_corr"] > 0.95
    assert m["ability_rmse"] < 0.25


@pytest.mark.xfail(reason="TODO(phase-4): estimators not implemented", raises=NotImplementedError)
@pytest.mark.parametrize("method", RECOVERY_METHODS)
def test_difficulty_recovery_improves_with_more_items(method):
    """Each b is informed by only ~P responses, so it is noisy — but it must still improve."""
    from scfm_irt.irt.fit import fit_irt
    from scfm_irt.irt.synthetic import to_long

    corrs = []
    for n_items in (1_000, 10_000):
        data = simulate_responses(n_models=10, n_items=n_items, seed=0)
        fit = fit_irt(to_long(data), model="1pl", method=method, seed=0)
        corrs.append(
            recovery_metrics(data, fit.ability_mean, fit.difficulty_mean)["difficulty_corr"]
        )

    assert corrs[-1] >= corrs[0] - 0.02
    assert corrs[-1] > 0.5


@pytest.mark.xfail(reason="TODO(phase-4): estimators not implemented", raises=NotImplementedError)
def test_credible_intervals_are_calibrated():
    """Intervals that do not cover at their nominal rate make the ranking unreportable."""
    from scfm_irt.irt.fit import fit_irt
    from scfm_irt.irt.synthetic import to_long

    coverage = []
    for seed in range(5):
        data = simulate_responses(n_models=10, n_items=2000, seed=seed)
        fit = fit_irt(to_long(data), model="1pl", method="bayes_variational", seed=seed)
        m = recovery_metrics(data, fit.ability_mean, fit.difficulty_mean, fit.ability_sd)
        coverage.append(m["ability_coverage_95"])

    assert 0.85 <= float(np.mean(coverage)) <= 1.0


@pytest.mark.xfail(reason="TODO(phase-4): estimators not implemented", raises=NotImplementedError)
def test_estimators_agree_with_each_other():
    """A cross-check: the transparent reference and the Bayesian engine should rank alike."""
    from scipy.stats import spearmanr

    from scfm_irt.irt.fit import fit_irt
    from scfm_irt.irt.synthetic import to_long

    long = to_long(simulate_responses(n_models=10, n_items=2000, seed=0))
    a = fit_irt(long, model="1pl", method="mle_autodiff", seed=0)
    b = fit_irt(long, model="1pl", method="bayes_variational", seed=0)

    assert spearmanr(a.ability_mean, b.ability_mean).statistic > 0.9
