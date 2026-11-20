# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import numpy as np
import pytest

from qbalance.diagnostics.distribution import cvm_1d, emd_1d, ks_1d, weighted_cdf
from qbalance.objectives import Objective
from qbalance.search.bandit import BanditSearcher
from qbalance.search.pareto import pareto_front
from qbalance.strategies import StrategySpec


def test_finite_distances_can_span_more_than_float_max():
    limit = float(np.finfo(float).max)
    sample = [-limit, limit]
    assert emd_1d(sample, [limit]) == limit
    assert cvm_1d(sample, [limit]) == limit / 2.0
    with pytest.raises(ValueError, match="not finite"):
        emd_1d([-limit], [limit])


def test_wide_intervals_keep_small_cvm_contributions():
    # Two distributions differ by 1e-200 across an interval of width 1e308.
    distance = cvm_1d([0.0, 1e308], [1e308], w1=[1e-200, 1.0])
    assert distance == pytest.approx(1e-92, rel=1e-12, abs=0.0)


def test_zero_weight_tail_cannot_change_or_reverse_the_cdf():
    _, cdf = weighted_cdf(range(10), [1.0] * 9 + [0.0])
    assert np.all(np.diff(cdf) >= 0.0)
    assert cdf[-2] == cdf[-1] == 1.0


@pytest.mark.parametrize(
    "sample", [np.array([1 + 2j]), (v for v in [np.complex128(2j)]), [10**400]]
)
def test_non_real_or_unrepresentable_samples_raise_value_error(sample):
    with pytest.raises(ValueError, match="finite real numbers"):
        weighted_cdf(sample)


def test_advertised_objective_weights_agree_with_scoring():
    objective = Objective({"depth": 2.0})
    with pytest.raises(TypeError):
        objective.weights["depth"] = 9.0
    restored = pickle.loads(pickle.dumps(objective))
    assert restored.weights == objective.weights
    assert restored.score({"depth": 3.0}) == objective.score({"depth": 3.0}) == 6.0


def test_pareto_treats_unrepresentable_numeric_metrics_as_missing():
    items = [("huge", {"depth": 10**400}), ("small", {"depth": 1})]
    assert pareto_front(items, ["depth"]) == [1]


def test_every_valid_strategy_has_serializable_fold_factors():
    with pytest.raises(ValueError, match="finite"):
        StrategySpec(zne=False, zne_factors=(float("nan"),))
    with pytest.raises(ValueError, match="boolean"):
        StrategySpec(num_twirls=np.bool_(True))


def test_bandit_posterior_with_finite_extreme_observations():
    # Each observation uses just the constant feature. With fixed or constant
    # adaptive noise the intercept is (2 / (sigma2 + 2)) * observation.
    large = 1e308
    constant = StrategySpec(optimization_level=0)
    for variance, expected_noise in [(1.0, 1.0), (None, 1e-6)]:
        model = BanditSearcher(sigma2=variance)
        model.observe(constant, large)
        model.observe(constant, large)
        mean, _ = model._posterior()
        assert mean[0] == pytest.approx(large * (2.0 / (expected_noise + 2.0)))
    # A sample variance beyond the float range still supplies finite
    # precision and mean, without overflow while computing the variance.
    model = BanditSearcher(sigma2=None)
    model.observe(constant, -large)
    model.observe(StrategySpec(optimization_level=1), large)
    mean, precision = model._posterior()
    assert mean[0] == 0.0
    assert mean[1] == pytest.approx(5e-309, rel=1e-12, abs=0.0)
    assert np.all(np.isfinite(precision))


@pytest.mark.parametrize(
    "distance, expected", [(emd_1d, 1.0), (ks_1d, 1e-16), (cvm_1d, 1e-16)]
)
def test_tiny_probability_in_either_tail_contributes_to_distance(distance, expected):
    for endpoint in (-1e16, 1e16):
        observed = distance([0, endpoint], [0], w1=[1.0, 1e-16])
        assert observed == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("complex_value", [np.complex64(2 + 1j), np.complex128(2 + 1j)])
def test_complex_metrics_and_weights_cannot_silently_contribute(complex_value):
    obj = Objective({"depth": 1.0, "invalid_weight": complex_value})
    assert obj.score({"depth": complex_value, "invalid_weight": 3}) == 0.0
    assert pareto_front(
        [(0, {"depth": complex_value}), (1, {"depth": 3})], ["depth"]
    ) == [1]
    with pytest.raises(ValueError, match="score must be finite"):
        BanditSearcher().observe(StrategySpec(), complex_value)
    with pytest.raises(ValueError, match="alpha"):
        BanditSearcher(alpha=complex_value)


@pytest.mark.parametrize(
    "metric, expected", [(ks_1d, 2.5e-17), (emd_1d, 0.5), (cvm_1d, 1.25e-17)]
)
def test_interior_weight_changes_nearly_identical_cdfs(metric, expected):
    # A small additional mass at the midpoint changes both halves of the CDF,
    # even though both distributions round to probability 0.5 on each side.
    result = metric(
        iter([0.0, 2e16]),
        iter([0.0, 1e16, 2e16]),
        iter([1.0, 1.0]),
        iter([1.0, 1e-16, 1.0]),
    )
    assert result == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_representable_distance_when_normalized_weight_underflows():
    result = emd_1d([-1e300, 0.0], [0.0], w1=[1e-200, 1e200])
    assert result == pytest.approx(1e-100, rel=1e-12, abs=0.0)
