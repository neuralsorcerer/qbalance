# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import types

import pytest

from qbalance.objectives import default_objective
from qbalance.strategies import StrategySpec
from qbalance.workflow import workload as wl
from tests.system_stubs import _Circ, as_ensemble


def test_no_regression_guard_keeps_baseline_when_candidates_are_worse(
    monkeypatch, tmp_path
):
    qc = _Circ()
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "guard_ds"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])
    monkeypatch.setattr(
        wl, "resolve_backend", lambda b: types.SimpleNamespace(name=lambda: "bk")
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    def fake_compile(circuit, backend, spec, profile):
        del circuit, backend, profile
        is_baseline = spec.routing_method == "sabre" and spec.optimization_level == 1
        depth = 1 if is_baseline else 10
        return qc, {"depth": depth, "two_qubit_ops": 0, "estimated_error": 0.0}

    monkeypatch.setattr(wl, "compile_ensemble", as_ensemble(fake_compile))
    balanced = (
        wl.Workload.from_dataset(ds)
        .set_target("b")
        .adjust(
            strategies=[StrategySpec(optimization_level=3)],
            allow_regression=False,
        )
    )

    selected = balanced.selections["c0"]
    assert selected.spec == StrategySpec(optimization_level=1, routing_method="sabre")
    assert selected.metrics["selected_by_regression_guard"] is True
    assert selected.metrics["depth"] == 1
    assert selected.metrics["objective_score"] == 1.0
    assert selected.metrics["rejected_candidate_objective_score"] == 10.0
    assert balanced.evaluation_history["c0"][0].metrics["depth"] == 10

    ranking = balanced.candidate_rankings()["c0"]
    selected_rows = [row for row in ranking if row["selected"]]
    assert len(selected_rows) == 1
    assert selected_rows[0]["original_index"] is None
    assert selected_rows[0]["spec"] == selected.spec.model_dump()


def test_no_regression_guard_allows_improvements():
    baseline = {"depth": 5, "two_qubit_ops": 0, "estimated_error": 0.0}
    chosen = {
        "depth": 3,
        "two_qubit_ops": 0,
        "estimated_error": 0.0,
        "objective_score": 3.0,
    }
    spec = StrategySpec(optimization_level=2)

    kept_spec, kept_metrics = wl._guard_against_regression(
        StrategySpec(optimization_level=1, routing_method="sabre"),
        baseline,
        spec,
        chosen,
        default_objective(),
    )

    assert kept_spec == spec
    assert kept_metrics is chosen


@pytest.mark.parametrize(
    ("baseline", "chosen", "falls_back"),
    [
        ({"depth": 5}, {"depth": 5, "objective_score": 5.0}, False),
        ({"depth": 5}, {"depth": 6, "objective_score": 6.0}, True),
        ({"depth": 5}, {"objective_score": float("nan")}, True),
        ({"depth": "bad"}, {"depth": 6, "objective_score": 6.0}, False),
    ],
)
def test_regression_guard_score_edge_cases(baseline, chosen, falls_back):
    baseline_spec = StrategySpec(optimization_level=1, routing_method="sabre")
    chosen_spec = StrategySpec(optimization_level=2)

    kept_spec, kept_metrics = wl._guard_against_regression(
        baseline_spec,
        baseline,
        chosen_spec,
        chosen,
        default_objective(),
    )

    if falls_back:
        assert kept_spec == baseline_spec
        assert kept_metrics is not baseline
        assert kept_metrics["selected_by_regression_guard"] is True
        assert kept_metrics["rejected_candidate_spec"] == chosen_spec.model_dump()
        assert math.isfinite(kept_metrics["objective_score"])
    else:
        assert kept_spec == chosen_spec
        assert kept_metrics is chosen


def test_allow_regression_requires_boolean(monkeypatch, tmp_path):
    qc = _Circ()
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "guard_bool_ds"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])

    with pytest.raises(ValueError, match="allow_regression must be a boolean"):
        wl.Workload.from_dataset(ds).set_target("b").adjust(
            strategies=[StrategySpec()], allow_regression=0
        )


@pytest.mark.parametrize("weight", [2.0, -2.0])
def test_regression_guard_ignores_terms_the_baseline_cannot_have(weight):
    """Execution-only metrics must not decide the guard.

    The baseline is only compiled; a candidate may also have been executed.
    Scoring each side on its own terms billed ``raw_top_prob`` as a regression
    (positive weight) or an improvement (negative weight) even for a candidate
    whose shared metrics are identical to the baseline's.
    """
    from qbalance.objectives import Objective

    objective = Objective({"depth": 1.0, "raw_top_prob": weight})
    baseline_spec = StrategySpec(optimization_level=1, routing_method="sabre")
    baseline = {"depth": 5}
    same = {"depth": 5, "raw_top_prob": 0.9, "objective_score": 5.0 + weight * 0.9}
    worse = {"depth": 6, "raw_top_prob": 0.9, "objective_score": 6.0 + weight * 0.9}
    chosen_spec = StrategySpec(optimization_level=2)

    kept_spec, kept = wl._guard_against_regression(
        baseline_spec, baseline, chosen_spec, same, objective
    )
    assert kept_spec == chosen_spec and kept is same

    kept_spec, kept = wl._guard_against_regression(
        baseline_spec, baseline, chosen_spec, worse, objective
    )
    assert kept_spec == baseline_spec
    assert kept["selected_by_regression_guard"] is True


def test_regression_guard_keeps_a_candidate_sharing_no_terms_with_the_baseline():
    from qbalance.objectives import Objective

    objective = Objective({"depth": 1.0, "raw_top_prob": 1.0})
    chosen = {"raw_top_prob": 0.9, "objective_score": 0.9}
    chosen_spec = StrategySpec(optimization_level=2)

    kept_spec, kept = wl._guard_against_regression(
        StrategySpec(optimization_level=1, routing_method="sabre"),
        {"depth": 0.1},
        chosen_spec,
        chosen,
        objective,
    )

    assert kept_spec == chosen_spec and kept is chosen


def test_regression_guard_prefers_a_lower_ranked_candidate_to_the_baseline():
    """Regression: the guard only ever tested the top candidate.

    The top candidate can win on a term the baseline cannot have (here the
    M3-mitigated parity) while regressing on the terms both share; the guard
    then fell back to the baseline even though a lower-ranked candidate beat
    it on every shared term.
    """
    from qbalance.objectives import Objective

    objective = Objective(
        {"raw_parity_expval": -100.0, "mitigated_parity_expval": -100.0}
    )
    baseline_spec = StrategySpec(optimization_level=1, routing_method="sabre")
    baseline = {"raw_parity_expval": 0.949}
    mitigated = StrategySpec(optimization_level=2, mthree=True)
    better = StrategySpec(optimization_level=2, layout_method="sabre")
    worse = StrategySpec(optimization_level=0)
    ranked = [
        (mitigated, {"raw_parity_expval": 0.942, "mitigated_parity_expval": 0.967}),
        (worse, {"raw_parity_expval": 0.940}),
        (better, {"raw_parity_expval": 0.962}),
    ]

    kept_spec, kept = wl._guard_against_regression(
        baseline_spec,
        baseline,
        *ranked[0],
        objective,
        alternatives=ranked[1:],
    )

    assert kept_spec == better
    assert kept["raw_parity_expval"] == 0.962
    assert kept["selected_by_regression_guard"] is True
    assert kept["rejected_candidate_spec"] == mitigated.model_dump()
    assert ranked[2][1] == {"raw_parity_expval": 0.962}  # not mutated

    # Only when every candidate regresses does the baseline come back.
    kept_spec, kept = wl._guard_against_regression(
        baseline_spec,
        baseline,
        *ranked[0],
        objective,
        alternatives=ranked[1:2],
    )
    assert kept_spec == baseline_spec
    assert kept["raw_parity_expval"] == 0.949


def test_adjust_guard_selects_the_non_regressing_candidate(monkeypatch, tmp_path):
    from qbalance.objectives import Objective

    qc = _Circ()
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "guard_alt_ds"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])
    monkeypatch.setattr(
        wl, "resolve_backend", lambda b: types.SimpleNamespace(name=lambda: "bk")
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    def fake_compile(circuit, backend, spec, profile):
        del circuit, backend, profile
        depth = {1: 5, 2: 4, 3: 6}[spec.optimization_level]
        metrics = {"depth": depth}
        if spec.dynamical_decoupling:
            # A term the baseline never reports, worth a lot to the objective.
            metrics["bonus"] = 1.0
        return qc, metrics

    monkeypatch.setattr(wl, "compile_ensemble", as_ensemble(fake_compile))
    balanced = (
        wl.Workload.from_dataset(ds)
        .set_target("b")
        .adjust(
            strategies=[
                StrategySpec(optimization_level=3, dynamical_decoupling=True),
                StrategySpec(optimization_level=2),
            ],
            objective=Objective({"depth": 1.0, "bonus": -10.0}),
            allow_regression=False,
        )
    )

    selected = balanced.selections["c0"]
    assert selected.spec == StrategySpec(optimization_level=2)
    assert selected.metrics["selected_by_regression_guard"] is True
    rows = [row for row in balanced.candidate_rankings()["c0"] if row["selected"]]
    # The guard-picked candidate is marked on its own row, not duplicated.
    assert len(rows) == 1 and rows[0]["original_index"] == 1


def test_equal_scores_select_the_earlier_candidate_in_every_search_mode(
    monkeypatch, tmp_path
):
    """Regression: with every candidate evaluated, the mode changed the pick.

    Equal scores went to whichever candidate the search evaluated first, so
    bandit search (which shuffles) and grid search selected different
    strategies for one circuit although both evaluated all of them -- and the
    documentation says the mode changes only the order then.  Ties now go to
    the earlier candidate, and the selection ranks first among equal scores.
    """
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    ds = wl.CircuitDataset(tmp_path, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(wl, "resolve_backend", lambda spec: object())
    monkeypatch.setattr(
        wl,
        "_evaluate_candidate",
        lambda circuit, backend, spec, **kwargs: {"depth": 5.0, "objective_score": 5.0},
    )
    strategies = [StrategySpec(optimization_level=level) for level in (2, 0, 3)]
    workload = wl.Workload(dataset=ds, backend_spec="stub")

    for search in ("grid", "bandit"):
        for seed in range(4):
            balanced = workload.adjust(
                objective={"depth": 1.0},
                search=search,
                strategies=strategies,
                seed=seed,
                warmup=1,
            )
            assert balanced.selections["c0"].spec == strategies[0]
            rankings = balanced.candidate_rankings()["c0"]
            assert rankings[0]["selected"] is True
            assert rankings[0]["spec"] == strategies[0].model_dump()
