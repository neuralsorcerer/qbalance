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
from tests.system_stubs import _Circ


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
    monkeypatch.setattr(wl, "load_compiled", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    def fake_compile(circuit, backend, spec, profile):
        del circuit, backend, profile
        is_baseline = spec.routing_method == "sabre" and spec.optimization_level == 1
        depth = 1 if is_baseline else 10
        return qc, {"depth": depth, "two_qubit_ops": 0, "estimated_error": 0.0}

    monkeypatch.setattr(wl, "compile_one", fake_compile)
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
