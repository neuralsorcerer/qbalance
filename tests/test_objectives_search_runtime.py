# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np

from qbalance.metrics.circuit_metrics import extract_circuit_metrics
from qbalance.mitigation import runtime_options
from qbalance.objectives import Objective, default_objective
from qbalance.search.bandit import BanditSearcher, _featurize
from qbalance.search.candidates import default_candidate_strategies
from qbalance.search.pareto import pareto_front
from qbalance.strategies import StrategySpec
from tests.system_stubs import _Circ


def test_metrics_objective_search_and_runtime_options():

    m = extract_circuit_metrics(_Circ())
    assert m["two_qubit_ops"] == 1.0
    obj = Objective({"depth": 2.0, "x": 1.0})
    assert obj.score({"depth": 3, "x": "bad"}) == 6.0
    assert default_objective().weights["estimated_error"] == 10.0

    s0 = StrategySpec()
    s1 = StrategySpec(optimization_level=3, pauli_twirling=True, num_twirls=8)
    assert _featurize(s0).shape[0] == _featurize(s1).shape[0]
    b = BanditSearcher()
    mean, precision = b._posterior()
    assert mean.shape[0] == precision.shape[0] == precision.shape[1]
    b.observe(s0, 2.0)
    b.observe(s1, 1.0)
    pick = b.propose([s0, s1], rng=np.random.default_rng(0))
    assert pick in [s0, s1]

    cands = default_candidate_strategies(max_candidates=5)
    assert len(cands) == 5
    front = pareto_front(
        [("a", {"x": 2.0}), ("b", {"x": 1.0}), ("c", {"x": 3.0})], keys=["x"]
    )
    assert front == [1]

    opts = runtime_options.build_runtime_estimator_options(
        resilience_level=2,
        enable_gate_twirling=True,
        enable_measurement_mitigation=False,
        enable_zne=True,
        layer_noise_model={"l": 1},
    )
    assert opts["resilience_level"] == 2
    assert opts["twirling"]["enable_gates"] is True


def test_bandit_searcher_validates_hyperparameters():

    for bad_alpha in (0.0, -1.0, float("inf"), float("nan")):
        try:
            BanditSearcher(alpha=bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid alpha")

    for bad_sigma2 in (0.0, -1.0, float("inf"), float("nan")):
        try:
            BanditSearcher(sigma2=bad_sigma2)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid sigma2")


def test_bandit_searcher_observe_rejects_non_finite_scores():

    b = BanditSearcher()
    s = StrategySpec()
    for bad_score in (float("nan"), float("inf"), float("-inf")):
        try:
            b.observe(s, bad_score)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for non-finite score")


def test_bandit_searcher_propose_requires_nonempty_candidates():

    b = BanditSearcher()
    try:
        b.propose([], rng=np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty candidate list")


def test_pareto_front_handles_non_finite_metrics_robustly():

    front = pareto_front(
        [
            ("good", {"x": 1.0, "y": 1.0}),
            ("nan", {"x": float("nan"), "y": 0.5}),
            ("inf", {"x": float("inf"), "y": 0.0}),
            ("bad_type", {"x": "oops", "y": 0.0}),
        ],
        keys=["x", "y"],
    )
    assert front == [0, 2, 3]


def test_pareto_front_empty_items():

    assert pareto_front([], keys=["x"]) == []


def test_pareto_front_empty_keys_returns_all_items():

    items = [("a", {"x": 2.0}), ("b", {"x": 1.0}), ("c", None)]
    assert pareto_front(items, keys=[]) == [0, 1, 2]


def test_pareto_front_single_objective_returns_all_minima():

    front = pareto_front(
        [("a", {"x": 2.0}), ("b", {"x": 1.0}), ("c", {"x": 1.0})],
        keys=["x"],
    )
    assert front == [1, 2]


def test_pareto_front_preserves_duplicate_nondominated_items():

    front = pareto_front(
        [
            ("a", {"x": 1.0, "y": 1.0}),
            ("b", {"x": 1.0, "y": 1.0}),
            ("c", {"x": 2.0, "y": 2.0}),
        ],
        keys=["x", "y"],
    )
    assert front == [0, 1]


def test_pareto_front_matches_naive_reference_on_randomized_inputs():

    rng = np.random.default_rng(7)

    def normalize(metrics, key):

        if metrics is None:
            return float("inf")
        try:
            value = float(metrics.get(key, float("inf")))
        except (AttributeError, TypeError, ValueError):
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    def naive(items, keys):

        vectors = [
            tuple(normalize(metrics, key) for key in keys) for _, metrics in items
        ]
        out = []
        for i, mi in enumerate(vectors):
            dominated = False
            for j, mj in enumerate(vectors):
                if i == j:
                    continue
                if all(vj <= vi for vi, vj in zip(mi, mj)) and any(
                    vj < vi for vi, vj in zip(mi, mj)
                ):
                    dominated = True
                    break
            if not dominated:
                out.append(i)
        return out

    keys = ["x", "y", "z"]
    for _ in range(100):
        n = int(rng.integers(1, 40))
        items = []
        for i in range(n):
            draw = rng.random()
            if draw < 0.10:
                metrics = None
            elif draw < 0.15:
                metrics = "not-a-mapping"
            else:
                metrics = {}
                for key in keys:
                    value_draw = rng.random()
                    if value_draw < 0.10:
                        metrics[key] = float("inf")
                    elif value_draw < 0.20:
                        metrics[key] = float("nan")
                    elif value_draw < 0.25:
                        metrics[key] = "oops"
                    else:
                        metrics[key] = float(rng.integers(0, 8))
            items.append((str(i), metrics))

        assert pareto_front(items, keys) == naive(items, keys)


def test_pareto_front_non_mapping_metrics_are_treated_as_invalid():

    front = pareto_front(
        [("good", {"x": 1.0, "y": 1.0}), ("bad", None), ("also_bad", 3.14)],
        keys=["x", "y"],
    )
    assert front == [0]


def test_default_candidate_strategies_non_positive_limit_returns_empty():

    assert default_candidate_strategies(max_candidates=0) == []
    assert default_candidate_strategies(max_candidates=-3) == []


def test_default_candidate_strategies_single_limit_returns_first_strategy():

    assert default_candidate_strategies(max_candidates=1) == [
        StrategySpec(optimization_level=0)
    ]


def test_objective_score_ignores_non_finite_and_invalid_values():

    obj = Objective(
        {"depth": 2.0, "two_qubit_ops": float("nan"), "estimated_error": 3.0}
    )
    score = obj.score(
        {
            "depth": 4,
            "two_qubit_ops": 1,
            "estimated_error": float("inf"),
            "compile_time_s": "bad",
        }
    )
    assert score == 8.0


def test_objective_precomputes_only_finite_numeric_weights():

    obj = Objective({"depth": 2.0, "bad": "oops", "nan": float("nan")})
    assert obj.score({"depth": 3, "bad": 10, "nan": 10}) == 6.0


def test_objective_copies_input_weights_at_init():

    weights = {"depth": 2.0}
    obj = Objective(weights)
    weights["depth"] = 10.0
    assert obj.score({"depth": 3}) == 6.0


def test_objective_ignores_overflowing_weight_values():

    huge = 10**400
    obj = Objective({"depth": 2.0, "huge": huge})
    assert obj.score({"depth": 3, "huge": 1.0}) == 6.0


def test_objective_ignores_overflowing_metric_values():

    huge = 10**400
    obj = Objective({"depth": 2.0, "huge_metric": 1.0})
    assert obj.score({"depth": 3, "huge_metric": huge}) == 6.0


def test_objective_skips_non_finite_product_terms():

    obj = Objective({"huge": 1e308, "small": 2.0})
    score = obj.score({"huge": 1e308, "small": 3.0})
    assert score == 6.0
