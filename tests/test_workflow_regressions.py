# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.transpiler import Target

from qbalance.benchmarking import matrix
from qbalance.dataset import DATASET_INDEX, CircuitDataset, save_dataset
from qbalance.errors import QBalanceError
from qbalance.objectives import Objective
from qbalance.strategies import Strategy, StrategySpec
from qbalance.workflow import workload


def test_empty_workload_does_not_warn_about_the_default_objective(tmp_path, caplog):
    dataset = save_dataset(tmp_path / "empty", [])
    with caplog.at_level("WARNING", logger=workload.__name__):
        balanced = (
            workload.Workload.from_dataset(dataset)
            .set_target("fake:generic:2")
            .adjust()
        )

    assert balanced.selections == {}
    assert "Objective term" not in caplog.text


@pytest.mark.parametrize("which_file", ["index", "circuit"])
def test_outputs_cannot_replace_an_input_dataset_file(tmp_path, which_file):
    dataset = save_dataset(tmp_path / "dataset", [QuantumCircuit(1, name="q")])
    filename = DATASET_INDEX if which_file == "index" else dataset.records[0].artifact
    destination = dataset.root / filename
    original = destination.read_bytes()

    with pytest.raises(ValueError, match="source dataset"):
        matrix.run_matrix(
            dataset.root, ["fake:generic:2"], [StrategySpec()], destination
        )
    balanced = workload.BalancedWorkload(dataset, "fake:generic:2", {})
    with pytest.raises(ValueError, match="source dataset"):
        balanced.to_download(destination, overwrite=True)
    assert destination.read_bytes() == original


def test_empty_exception_message_is_reported_as_a_failed_trial(tmp_path, monkeypatch):
    dataset = save_dataset(tmp_path / "dataset", [QuantumCircuit(1, name="q")])

    def fail(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(workload, "_compile_cached", fail)
    monkeypatch.setattr(matrix, "compile_ensemble", fail)
    with pytest.raises(QBalanceError, match="compile_failed.*TimeoutError"):
        workload.Workload.from_dataset(dataset).set_target("fake:generic:2").adjust(
            strategies=[StrategySpec()]
        )
    path = matrix.run_matrix(
        dataset.root, ["fake:generic:2"], [StrategySpec()], tmp_path / "matrix.json"
    )
    metrics = json.loads(path.read_text(encoding="utf-8"))["results"][0]["metrics"]
    assert metrics == {"compile_error": "TimeoutError"}


def test_archive_published_during_export_is_preserved(tmp_path, monkeypatch):
    destination = tmp_path / "work.zip"
    balanced = workload.BalancedWorkload(
        CircuitDataset(tmp_path, []), "fake:generic:2", {}
    )
    save = balanced.save

    def publish_while_staging(content):
        save(content)
        destination.write_bytes(b"other writer")

    monkeypatch.setattr(balanced, "save", publish_while_staging)
    with pytest.raises(FileExistsError):
        balanced.to_download(destination)
    assert destination.read_bytes() == b"other writer"


def test_rankings_mark_one_selection_for_partial_or_duplicate_legacy_history(tmp_path):
    strategy = Strategy(StrategySpec(), {"depth": 4})
    balanced = workload.BalancedWorkload(
        CircuitDataset(tmp_path, []),
        "fake:generic:2",
        {"a": strategy, "b": strategy},
        evaluation_history={"a": [strategy, strategy]},
    )

    ranks = balanced.candidate_rankings()
    assert [row["selected"] for row in ranks["a"]] == [True, False]
    assert [row["original_index"] for row in ranks["a"]] == [0, 1]
    assert len(ranks["b"]) == 1
    assert ranks["b"][0]["selected"] and ranks["b"][0]["original_index"] is None


def test_extreme_finite_objectives_keep_diagnostics_serializable(tmp_path):
    balanced = workload.BalancedWorkload(
        CircuitDataset(tmp_path, []),
        "fake:generic:2",
        {"q": Strategy(StrategySpec(), {"estimated_error": 1e308, "depth": 1e308})},
        baseline_metrics={"q": {"estimated_error": -1e308, "depth": 1e-308}},
        objective=Objective({"estimated_error": 1}),
    )

    diagnostics = balanced.selection_diagnostics()["q"]
    json.dumps(diagnostics, allow_nan=False)
    assert diagnostics["objective_delta"] is None
    assert diagnostics["objective_improved"] is False
    assert diagnostics["metric_deltas"]["estimated_error"]["delta"] is None
    assert diagnostics["metric_deltas"]["estimated_error"]["relative_delta"] == 2
    assert diagnostics["metric_deltas"]["depth"]["relative_delta"] is None
    assert workload._bandit_target(1e308, -1e308) == 2
    assert workload._bandit_target(1e308, 1e-308) is None


def test_objective_terms_cancel_before_scoring_or_comparing_candidates(tmp_path):
    objective = Objective({"first": 1.0, "second": 1.0, "third": -1.0})
    large = 1e308
    selected = {"first": large, "second": large, "third": large}
    baseline = {"first": large, "second": large, "third": large / 2}
    balanced = workload.BalancedWorkload(
        CircuitDataset(tmp_path, []),
        "fake:generic:2",
        {"q": Strategy(StrategySpec(), selected)},
        baseline_metrics={"q": baseline},
        objective=objective,
    )

    assert objective.score(selected) == large
    assert balanced.candidate_rankings()["q"][0]["selection_score"] == large
    diagnostics = balanced.selection_diagnostics()["q"]
    assert diagnostics["objective_delta"] == -large / 2
    assert diagnostics["objective_improved"] is True


def test_width_and_timing_constraints_invalidate_cached_transpilation():
    def fingerprint(width, pulse_alignment):
        target = Target(num_qubits=width, pulse_alignment=pulse_alignment)
        target.add_instruction(XGate())
        return workload._calibration_fingerprint(SimpleNamespace(target=target))

    assert len({fingerprint(3, 1), fingerprint(3, 16), fingerprint(5, 1)}) == 3


def test_circuit_without_a_fingerprint_skips_the_cache(tmp_path, monkeypatch):
    circuit = QuantumCircuit(1)
    circuit.x(0)

    def no_fingerprint(circuit):
        raise ValueError("unserializable circuit")

    def unexpected_entry(*args, **kwargs):
        pytest.fail("a circuit without a fingerprint cannot identify a cache entry")

    monkeypatch.setattr(workload, "fingerprint_circuit", no_fingerprint)
    monkeypatch.setattr(workload, "get_entry", unexpected_entry)
    monkeypatch.setattr(workload, "compile_ensemble", lambda qc, **kwargs: ([qc], {}))
    instances, _ = workload._compile_cached(
        circuit, object(), StrategySpec(), False, tmp_path
    )
    assert instances == [circuit]


def test_failed_calibration_lookup_bypasses_the_compile_cache(tmp_path, monkeypatch):
    dataset = save_dataset(tmp_path / "dataset", [QuantumCircuit(1, name="q")])

    def unavailable():
        raise TimeoutError()

    backend = SimpleNamespace(name="backend", properties=unavailable)
    monkeypatch.setattr(workload, "resolve_backend", lambda name: backend)
    calls = []

    def compile_circuit(circuit, **options):
        calls.append(circuit)
        return [circuit], {"depth": len(calls)}

    monkeypatch.setattr(workload, "compile_ensemble", compile_circuit)
    work = workload.Workload.from_dataset(dataset).set_target("backend")
    for expected in (1, 2):
        result = work.adjust(
            strategies=[StrategySpec(optimization_level=1, routing_method="sabre")],
            objective={"depth": 1},
            cache_root=tmp_path / "cache",
        )
        assert result.selections["q"].metrics["depth"] == expected
    assert not (tmp_path / "cache").exists()


def test_complex_metric_and_score_are_incomparable(tmp_path):
    metrics = {"depth": np.complex128(8 + 1j), "objective_score": np.complex128(3 + 2j)}
    balanced = workload.BalancedWorkload(
        CircuitDataset(tmp_path, []),
        "backend",
        {"q": Strategy(StrategySpec(), metrics)},
        objective=Objective({"depth": 1.0}),
    )
    assert (
        balanced.selection_diagnostics()["q"]["metric_deltas"]["depth"]["selected"]
        is None
    )
    assert balanced.candidate_rankings()["q"][0]["selection_score"] is None
    assert workload._bandit_target(metrics["objective_score"], 1.0) is None
