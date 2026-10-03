# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import shutil
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import typer

from qbalance import cli
from qbalance.benchmarking import matrix as matrix_mod
from qbalance.cutting import addon_cutting
from qbalance.errors import OptionalDependencyError
from qbalance.execution import ensemble
from qbalance.mitigation import zne
from qbalance.objectives import Objective, default_objective
from qbalance.reports import common as report_common
from qbalance.strategies import Strategy, StrategySpec
from qbalance.transpile import suppression
from qbalance.utils import measured_qubits_by_clbit
from qbalance.workflow import workload as wl
from tests.system_stubs import _Circ, as_ensemble


def test_cutting_and_workload_and_matrix_and_cli(monkeypatch, tmp_path):

    # Stub the real API surface: find_cuts, OptimizationParameters and
    # DeviceConstraints live at the qiskit_addon_cutting package root, and the
    # constraint is spelled qubits_per_subcircuit.
    cutting_mod = types.ModuleType("qiskit_addon_cutting")
    cutting_mod.DeviceConstraints = lambda qubits_per_subcircuit: types.SimpleNamespace(
        qubits_per_subcircuit=qubits_per_subcircuit
    )
    cutting_mod.OptimizationParameters = lambda max_backjumps, max_gamma: (
        types.SimpleNamespace(max_backjumps=max_backjumps, max_gamma=max_gamma)
    )
    cutting_mod.find_cuts = lambda circuit, optimization, constraints: (
        circuit,
        {"w": constraints.qubits_per_subcircuit},
    )
    monkeypatch.setitem(sys.modules, "qiskit_addon_cutting", cutting_mod)

    qc = _Circ()
    cut, meta = addon_cutting.find_cuts_best_effort(qc, max_subcircuit_qubits=1)
    assert cut is qc
    assert meta["w"] == 1

    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [record])

    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [
            StrategySpec(),
            StrategySpec(optimization_level=2),
        ],
    )
    monkeypatch.setattr(
        wl,
        "compile_ensemble",
        as_ensemble(
            lambda circuit, backend, spec, profile: (
                qc,
                {
                    "depth": spec.optimization_level + 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 0.1,
                    "measurement_flip_map": {},
                },
            )
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda backend, compiled, shots, seed_simulator: {"00": 5, "11": 5},
    )
    monkeypatch.setattr(
        ensemble, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(
        ensemble,
        "mitigate_twirled_counts",
        lambda backend, raw_counts, flip_maps, measured_qubits, seed: {"00": 1.0},
    )
    monkeypatch.setattr(
        ensemble, "fold_global_for_backend", lambda compiled, backend, f: compiled
    )
    monkeypatch.setattr(
        ensemble,
        "zne_extrapolate_counts",
        lambda factors, counts_pf, degree: {"00": 1.0},
    )
    monkeypatch.setattr(
        wl,
        "prepare_cutting_experiment",
        lambda circuit, max_subcircuit_qubits: None,
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    work = wl.Workload.from_dataset(ds).set_target("fake:generic:2")
    balanced = work.adjust(search="grid", execute=True, pareto=True, max_candidates=2)
    assert balanced.backend_spec == "fake:generic:2"
    summary = balanced.summary()
    assert "qbalance summary" in summary
    assert "candidate evaluations:" in summary
    assert "depth" in balanced.covars()
    diagnostics = balanced.selection_diagnostics()
    assert diagnostics["c0"]["evaluated_candidates"] == 2
    assert "objective deltas:" in summary
    assert len(balanced.evaluation_history["c0"]) == 2
    rankings = balanced.candidate_rankings()
    assert [row["rank"] for row in rankings["c0"]] == [1, 2]
    assert rankings["c0"][0]["objective_score"] <= rankings["c0"][1]["objective_score"]
    assert sum(1 for row in rankings["c0"] if row["selected"]) == 1

    out_dir = tmp_path / "out"
    balanced.save(out_dir)
    saved_payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert len(saved_payload["evaluation_history"]["c0"]) == 2
    assert len(saved_payload["candidate_rankings"]["c0"]) == 2
    assert saved_payload["selection_diagnostics"]["c0"]["evaluated_candidates"] == 2
    z = balanced.to_download(tmp_path / "bundle.zip", overwrite=True)
    assert z.exists()

    monkeypatch.setattr(
        wl, "load_compiled_ensemble", lambda entry: ([qc], {"depth": 1})
    )
    hit_c, hit_m = wl._compile_cached(qc, object(), StrategySpec(), False, tmp_path)
    assert hit_c == [qc]
    assert hit_m["depth"] == 1

    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    miss_c, miss_m = wl._compile_cached(qc, object(), StrategySpec(), False, tmp_path)
    assert miss_c == [qc]
    assert miss_m["depth"] >= 1

    with pytest.raises(RuntimeError):
        wl._choose([], pareto=False, objective=default_objective())

    chosen = wl._choose(
        [
            (StrategySpec(optimization_level=1), {"objective_score": 2}),
            (StrategySpec(optimization_level=2), {"objective_score": 1}),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen[0].optimization_level == 2

    monkeypatch.setattr(matrix_mod, "load_dataset", lambda d: ds)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(
            lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}})
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda backend, compiled, shots, seed_simulator: {"0": 1},
    )
    monkeypatch.setattr(
        ensemble, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(ensemble, "fold_global_for_backend", lambda c, backend, f: c)
    monkeypatch.setattr(
        ensemble,
        "zne_extrapolate_counts",
        lambda factors, counts_pf, degree: {"0": 1.0},
    )
    p = matrix_mod.run_matrix(
        dsroot, ["b"], [StrategySpec(zne=True)], tmp_path / "m2.json", execute=True
    )
    assert p.exists()

    monkeypatch.setattr(cli, "_make_tiny", lambda: [qc])

    class Saved:
        def __init__(self, root):

            self.root = root

        def __len__(self):

            return 1

    monkeypatch.setattr(
        cli, "save_dataset", lambda out, circuits, overwrite=False: Saved(out)
    )
    monkeypatch.setattr(cli, "run_matrix", lambda *a, **k: tmp_path / "mx.json")
    monkeypatch.setattr(
        cli, "render_markdown", lambda matrix_json, out: out / "report.md"
    )
    monkeypatch.setattr(
        cli, "render_html", lambda matrix_json, out: out / "report.html"
    )
    monkeypatch.setattr(cli, "list_plugins", lambda: {"g": ["x"]})
    monkeypatch.setattr(cli, "load_dataset", lambda d: ds)

    class DummyBW:
        def save(self, out, overwrite=False):

            _ = (out, overwrite)

        def summary(self):

            return "ok"

    monkeypatch.setattr(
        cli.Workload,
        "from_path",
        classmethod(
            lambda cls, p: types.SimpleNamespace(
                set_target=lambda b: types.SimpleNamespace(adjust=lambda **k: DummyBW())
            )
        ),
    )
    cli.dataset_cmd("examples", tmp_path / "a", overwrite=True)
    with pytest.raises(typer.BadParameter, match="Only 'examples' is supported"):
        cli.dataset_cmd("bad", tmp_path / "a", overwrite=True)


def test_additional_branch_coverage(monkeypatch, tmp_path):
    # resolver lazy-load branch

    from qbalance.backends import resolver as resolver_mod

    monkeypatch.setattr(resolver_mod, "_PLUGINS", None)
    monkeypatch.setattr(
        resolver_mod, "_load_backend_plugins", lambda: {"x": lambda s: s}
    )
    assert resolver_mod.resolve_backend("x:1") == "x:1"

    # matrix exec_error branch
    ds = types.SimpleNamespace(
        records=[types.SimpleNamespace(name="c0")],
        load_circuits=lambda: [_Circ()],
    )
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(
            lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}})
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    p = matrix_mod.run_matrix(
        tmp_path, ["b"], [StrategySpec()], tmp_path / "err.json", execute=True
    )
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert "exec_error" in payload["results"][0]["metrics"]

    # reports/common optional key branches + parse failures
    key = report_common.strategy_key(
        {
            "optimization_level": 2,
            "layout_method": "sabre",
            "routing_method": "sabre",
            "pauli_twirling": True,
            "num_twirls": 4,
            "dynamical_decoupling": True,
            "dd_sequence": "XX",
            "measurement_twirling": True,
            "mthree": True,
            "zne": True,
            "cutting": True,
            "max_subcircuit_qubits": 3,
        }
    )
    assert "layout=sabre" in key and "cut3" in key
    agg = report_common.aggregate(
        [{"metrics": {"depth": "oops", "two_qubit_ops": None}}]
    )
    assert np.isnan(agg["depth"])

    # zne branches
    c = _Circ()
    assert zne.fold_global(c, 1.0) is c
    probs = zne.zne_extrapolate_counts(
        [1.0, 3.0], [{"00": 9, "11": 1}, {"00": 8, "11": 2}], degree=1
    )
    assert pytest.approx(sum(probs.values())) == 1.0
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0], [{"0": 1}], degree=2)

    # suppression dependency/branch coverage
    monkeypatch.setitem(
        sys.modules, "qiskit.circuit", types.ModuleType("qiskit.circuit")
    )
    with pytest.raises(
        OptionalDependencyError, match="qiskit is required for pauli twirling"
    ):
        suppression.apply_pauli_twirling(_Circ())

    # candidates dedupe continue line via monkeypatched class equality
    import qbalance.search.candidates as cand_mod

    class D:
        def __init__(self, **kwargs):

            self.kwargs = kwargs

        def __hash__(self):

            return 0

        def __eq__(self, other):

            _ = other
            return True

    monkeypatch.setattr(cand_mod, "StrategySpec", D)
    out = cand_mod.default_candidate_strategies(max_candidates=3)
    assert len(out) == 1


def test_cli_full_commands(monkeypatch, tmp_path):

    rec = types.SimpleNamespace(name="c0", artifact="c0.qpy")
    ds = types.SimpleNamespace(records=[rec], load_circuits=lambda: [_Circ()])

    class BW:
        def save(self, out, overwrite=False):

            _ = (out, overwrite)

        def summary(self):

            return "sum"

    monkeypatch.setattr(
        cli.Workload,
        "from_path",
        classmethod(
            lambda cls, p: types.SimpleNamespace(
                set_target=lambda b: types.SimpleNamespace(adjust=lambda **k: BW())
            )
        ),
    )
    monkeypatch.setattr(cli, "run_matrix", lambda *a, **k: tmp_path / "m.json")
    monkeypatch.setattr(
        cli, "render_markdown", lambda matrix_json, out: out / "report.md"
    )
    monkeypatch.setattr(
        cli, "render_html", lambda matrix_json, out: out / "report.html"
    )
    monkeypatch.setattr(cli, "list_plugins", lambda: {"g": ["a"], "h": []})
    monkeypatch.setattr(cli, "load_dataset", lambda p: ds)

    backends = types.ModuleType("qbalance.backends")
    backends.resolve_backend = lambda b: object()
    monkeypatch.setitem(sys.modules, "qbalance.backends", backends)
    tp = types.ModuleType("qbalance.transpile.pipeline")
    tp.compile_ensemble = lambda qc, backend, spec, profile=False: ([qc], {"depth": 1})
    monkeypatch.setitem(sys.modules, "qbalance.transpile.pipeline", tp)

    qiskit = types.ModuleType("qiskit")
    qpy = types.SimpleNamespace(dump=lambda c, f: f.write(b"x"))
    qiskit.qpy = qpy
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)

    cli.adjust_cmd(tmp_path, "b", tmp_path / "o")
    cli.matrix_cmd(tmp_path, ["b"], tmp_path / "m.json")
    cli.report_cmd(tmp_path / "m.json", tmp_path, html=True)
    cli.plugins_cmd("list")
    with pytest.raises(typer.BadParameter, match="Only 'list' supported"):
        cli.plugins_cmd("bad")

    out = tmp_path / "compiled_out"
    cli.compile_cmd(
        tmp_path,
        "b",
        out,
        optimization_level=1,
        routing_method="sabre",
        layout_method=None,
        pauli_twirling=False,
        num_twirls=1,
        dynamical_decoupling=False,
        measurement_twirling=False,
        overwrite=True,
    )
    with pytest.raises(typer.BadParameter, match="use --overwrite"):
        cli.compile_cmd(
            tmp_path,
            "b",
            out,
            optimization_level=1,
            routing_method="sabre",
            layout_method=None,
            pauli_twirling=False,
            num_twirls=1,
            dynamical_decoupling=False,
            measurement_twirling=False,
            overwrite=False,
        )


def test_workload_additional_branches(monkeypatch, tmp_path):

    qc = _Circ()
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds2"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])

    work = wl.Workload.from_dataset(ds)
    with pytest.raises(ValueError):
        work.adjust()

    # save overwrite error path
    bw = wl.BalancedWorkload(
        ds,
        "b",
        {
            "c0": Strategy(
                spec=StrategySpec(),
                metrics={
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 0.1,
                    "compile_time_s": 0.1,
                },
            )
        },
        {
            "c0": {
                "depth": 2,
                "two_qubit_ops": 2,
                "estimated_error": 0.2,
                "compile_time_s": 0.2,
            }
        },
    )
    out = tmp_path / "save_out"
    bw.save(out)
    with pytest.raises(FileExistsError):
        bw.save(out, overwrite=False)
    zpath = tmp_path / "res.zip"
    bw.to_download(zpath, overwrite=True)
    with pytest.raises(FileExistsError):
        bw.to_download(zpath, overwrite=False)

    # adjust invalid search and execution error branches
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [StrategySpec(mthree=True, zne=True)],
    )
    monkeypatch.setattr(
        wl,
        "compile_ensemble",
        as_ensemble(
            lambda *a, **k: (
                _Circ(),
                {
                    "measurement_flip_map": {0: 1},
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 0.1,
                },
            )
        ),
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("exec fail")),
    )

    work2 = wl.Workload.from_dataset(ds).set_target("b")
    with pytest.raises(RuntimeError, match="No feasible candidate"):
        work2.adjust(search="bandit", execute=True, pareto=False, max_candidates=1)
    with pytest.raises(ValueError):
        work2.adjust(search="bad")


def test_run_matrix_validates_dataset_lengths_and_precomputes_strategies(
    monkeypatch, tmp_path
):

    ds_bad = types.SimpleNamespace(
        records=[types.SimpleNamespace(name="c0")],
        load_circuits=lambda: [],
    )
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds_bad)
    with pytest.raises(ValueError, match="length mismatch"):
        matrix_mod.run_matrix(tmp_path, ["b"], [StrategySpec()], tmp_path / "bad.json")

    # Serialization is pure, so each strategy is dumped once, not per trial.
    spec = StrategySpec(optimization_level=2)
    real_dump = StrategySpec.model_dump
    dumped = []

    def counting_dump(self, *args, **kwargs):
        dumped.append(self)
        return real_dump(self, *args, **kwargs)

    monkeypatch.setattr(StrategySpec, "model_dump", counting_dump)
    ds_ok = types.SimpleNamespace(
        records=[types.SimpleNamespace(name="c0"), types.SimpleNamespace(name="c1")],
        load_circuits=lambda: [_Circ(), _Circ()],
    )
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds_ok)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(
            lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}})
        ),
    )

    out = matrix_mod.run_matrix(tmp_path, ["b0", "b1"], [spec], tmp_path / "ok.json")
    assert out.exists()
    assert dumped == [spec]

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 4
    serialized = json.loads(json.dumps(real_dump(spec)))
    assert all(entry["strategy"] == serialized for entry in payload["results"])


def test_run_matrix_rejects_non_positive_shots(monkeypatch, tmp_path):

    ds = types.SimpleNamespace(records=[], load_circuits=lambda: [])
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds)

    with pytest.raises(ValueError, match="shots must be a positive integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", shots=0
        )

    with pytest.raises(ValueError, match="shots must be a positive integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", shots=-1
        )


def test_run_matrix_rejects_non_integer_shots(monkeypatch, tmp_path):

    ds = types.SimpleNamespace(records=[], load_circuits=lambda: [])
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds)

    with pytest.raises(ValueError, match="shots must be a positive integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", shots=1.5
        )

    with pytest.raises(ValueError, match="shots must be a positive integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", shots=True
        )


def test_run_matrix_rejects_non_integer_seed(monkeypatch, tmp_path):

    ds = types.SimpleNamespace(records=[], load_circuits=lambda: [])
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds)

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", seed=1.5
        )

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", seed=True
        )


def test_choose_pareto_treats_invalid_non_finite_metrics_as_infinite():

    chosen_spec, chosen_metrics = wl._choose(
        [
            (
                StrategySpec(optimization_level=0),
                {
                    "depth": "bad",
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": 5.0,
                },
            ),
            (
                StrategySpec(optimization_level=1),
                {
                    "depth": 2,
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": 3.0,
                },
            ),
            (
                StrategySpec(optimization_level=2),
                {
                    "depth": float("nan"),
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": 1.0,
                },
            ),
        ],
        pareto=True,
        objective=default_objective(),
    )

    assert chosen_spec.optimization_level == 1
    assert chosen_metrics["objective_score"] == 3.0


def test_choose_invalid_fallback_inputs_do_not_win_selection():

    chosen_spec, _ = wl._choose(
        [
            (
                StrategySpec(optimization_level=0),
                {"depth": "bad", "objective_score": "bad"},
            ),
            (
                StrategySpec(optimization_level=1),
                {"depth": 3, "two_qubit_ops": 1, "estimated_error": 0.1},
            ),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_pareto_invalid_fallback_inputs_do_not_win_selection():

    chosen_spec, _ = wl._choose(
        [
            (
                StrategySpec(optimization_level=0),
                {
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": "bad",
                    "compile_time_s": "bad",
                },
            ),
            (
                StrategySpec(optimization_level=1),
                {
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": "bad",
                    "compile_time_s": 2.0,
                },
            ),
        ],
        pareto=True,
        objective=Objective({"compile_time_s": 1.0}),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_handles_invalid_objective_scores():

    chosen_spec, _ = wl._choose(
        [
            (StrategySpec(optimization_level=0), {"objective_score": "bad"}),
            (StrategySpec(optimization_level=1), {"objective_score": 2.0}),
            (StrategySpec(optimization_level=2), {"objective_score": float("nan")}),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_rejects_finite_objective_score_when_no_finite_objective_terms():

    chosen_spec, _ = wl._choose(
        [
            (
                StrategySpec(optimization_level=0),
                {
                    "objective_score": 0.0,
                    "depth": "bad",
                    "two_qubit_ops": None,
                    "estimated_error": float("nan"),
                    "compile_time_s": "bad",
                },
            ),
            (
                StrategySpec(optimization_level=1),
                {"depth": 2, "two_qubit_ops": 1, "estimated_error": 0.1},
            ),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_pareto_tie_break_ignores_invalid_objective_scores():

    chosen_spec, chosen_metrics = wl._choose(
        [
            (
                StrategySpec(optimization_level=0),
                {
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": "bad",
                },
            ),
            (
                StrategySpec(optimization_level=1),
                {
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": 2.5,
                },
            ),
        ],
        pareto=True,
        objective=default_objective(),
    )

    assert chosen_spec.optimization_level == 1
    assert chosen_metrics["objective_score"] == 2.5


def test_choose_falls_back_to_objective_when_objective_score_missing():

    chosen_spec, _ = wl._choose(
        [
            (StrategySpec(optimization_level=0), {"depth": 5, "two_qubit_ops": 2}),
            (StrategySpec(optimization_level=1), {"depth": 2, "two_qubit_ops": 1}),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_pareto_tie_break_falls_back_to_objective_score_computation():

    chosen_spec, _ = wl._choose(
        [
            (
                StrategySpec(optimization_level=0),
                {
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 1.0,
                    "objective_score": "bad",
                },
            ),
            (
                StrategySpec(optimization_level=1),
                {
                    "depth": 1,
                    "two_qubit_ops": 1,
                    "estimated_error": 0.2,
                    "objective_score": float("nan"),
                },
            ),
        ],
        pareto=True,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_ignores_non_mapping_metrics_entries():

    chosen_spec, _ = wl._choose(
        [
            (StrategySpec(optimization_level=0), cast(Any, None)),
            (StrategySpec(optimization_level=1), {"objective_score": 1.0}),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_choose_pareto_handles_non_mapping_metrics_entries():

    chosen_spec, _ = wl._choose(
        [
            (StrategySpec(optimization_level=0), cast(Any, None)),
            (
                StrategySpec(optimization_level=1),
                {"depth": 1, "two_qubit_ops": 1, "estimated_error": 0.5},
            ),
        ],
        pareto=True,
        objective=default_objective(),
    )
    assert chosen_spec.optimization_level == 1


def test_compile_cache_separates_profile_mode(monkeypatch, tmp_path):
    calls = []

    def fake_compile(circuit, backend, spec, profile):
        calls.append(profile)
        metrics = {"depth": 1}
        if profile:
            metrics["pass_profile"] = {"passes": []}
        return circuit, metrics

    monkeypatch.setattr(wl, "fingerprint_circuit", lambda circuit: "fingerprint")
    monkeypatch.setattr(wl, "compile_ensemble", as_ensemble(fake_compile))

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    backend = types.SimpleNamespace(name=lambda: "backend")
    spec = StrategySpec()

    _, no_profile_metrics = wl._compile_cached(qc, backend, spec, False, tmp_path)
    _, profile_metrics = wl._compile_cached(qc, backend, spec, True, tmp_path)

    assert calls == [False, True]
    assert "pass_profile" not in no_profile_metrics
    assert "pass_profile" in profile_metrics


def test_load_balanced_workload_round_trip(tmp_path):
    dsroot = tmp_path / "ds_roundtrip"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"placeholder")
    (dsroot / "qbalance_dataset.json").write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {
                        "name": "c0",
                        "artifact": "c0.qpy",
                        "format": "qpy",
                        "metadata": {"family": "stub"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ds = wl.CircuitDataset(
        dsroot,
        [wl.CircuitRecord("c0", "c0.qpy", "qpy", {"family": "stub"})],
    )
    balanced = wl.BalancedWorkload(
        dataset=ds,
        backend_spec="fake:generic:2",
        selections={
            "c0": Strategy(
                spec=StrategySpec(optimization_level=2, routing_method="sabre"),
                metrics={"depth": 3, "two_qubit_ops": 1, "objective_score": 5.0},
            )
        },
        baseline_metrics={"c0": {"depth": 4, "two_qubit_ops": 2}},
        objective=Objective({"depth": 1.5, "two_qubit_ops": 2.0}),
        evaluation_history={
            "c0": [
                Strategy(
                    spec=StrategySpec(optimization_level=1),
                    metrics={"depth": 4, "objective_score": 6.0},
                ),
                Strategy(
                    spec=StrategySpec(optimization_level=2, routing_method="sabre"),
                    metrics={"depth": 3, "objective_score": 5.0},
                ),
            ]
        },
    )

    out = tmp_path / "balanced"
    balanced.save(out)
    loaded = wl.load_balanced_workload(out)

    assert loaded.backend_spec == balanced.backend_spec
    assert loaded.dataset.names() == ["c0"]
    assert loaded.objective.weights == {"depth": 1.5, "two_qubit_ops": 2.0}
    assert loaded.selections["c0"].spec.optimization_level == 2
    assert loaded.selections["c0"].metrics["objective_score"] == 5.0
    assert loaded.baseline_metrics["c0"]["depth"] == 4
    diagnostics = loaded.selection_diagnostics()["c0"]
    assert diagnostics["metric_deltas"]["depth"]["delta"] == -1.0
    assert diagnostics["objective_improved"] is True
    assert len(loaded.evaluation_history["c0"]) == 2
    assert loaded.evaluation_history["c0"][1].metrics["objective_score"] == 5.0


def test_selection_diagnostics_handles_missing_and_invalid_metrics(tmp_path):
    dsroot = tmp_path / "ds_diag_edges"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"placeholder")
    dataset = wl.CircuitDataset(
        dsroot,
        [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})],
    )
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={
            "c0": Strategy(
                spec=StrategySpec(),
                metrics={
                    "depth": "bad",
                    "two_qubit_ops": float("nan"),
                    "compile_time_s": float("inf"),
                },
            )
        },
        baseline_metrics={"c0": {"depth": None, "estimated_error": "bad"}},
        objective=Objective({"depth": 1.0, "estimated_error": 10.0}),
    )

    diagnostics = balanced.selection_diagnostics()["c0"]

    assert diagnostics["baseline_objective_score"] is None
    assert diagnostics["selected_objective_score"] is None
    assert diagnostics["objective_delta"] is None
    assert diagnostics["objective_improved"] is None
    assert diagnostics["objective_terms"] == {"baseline": {}, "selected": {}}
    assert diagnostics["metric_deltas"]["depth"] == {
        "baseline": None,
        "selected": None,
        "delta": None,
        "relative_delta": None,
    }
    json.dumps(balanced.selection_diagnostics(), allow_nan=False)
    assert "objective deltas:" not in balanced.summary()


def test_candidate_rankings_match_selection_score_and_are_json_safe(tmp_path):
    dsroot = tmp_path / "ds_ranking_edges"
    dsroot.mkdir()
    dataset = wl.CircuitDataset(
        dsroot,
        [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})],
    )
    selected = Strategy(
        spec=StrategySpec(optimization_level=2),
        metrics={"depth": 100.0, "objective_score": 1.0},
    )
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={"c0": selected},
        objective=Objective({"depth": 1.0}),
        evaluation_history={
            "c0": [
                Strategy(
                    spec=StrategySpec(optimization_level=0),
                    metrics={"depth": 1.0, "objective_score": 10.0},
                ),
                selected,
                Strategy(
                    spec=StrategySpec(optimization_level=3),
                    metrics={"depth": "bad", "objective_score": 0.0},
                ),
                Strategy(
                    spec=StrategySpec(optimization_level=1),
                    metrics={"not_an_objective_term": 5.0},
                ),
            ]
        },
    )

    rankings = balanced.candidate_rankings()["c0"]

    assert [row["original_index"] for row in rankings] == [1, 0, 2, 3]
    assert rankings[0]["selected"] is True
    assert rankings[0]["selection_score"] == 1.0
    assert rankings[0]["objective_score"] == 100.0
    assert rankings[2]["selection_score"] is None
    assert rankings[3]["selection_score"] is None
    json.dumps(rankings, allow_nan=False)


def test_saved_results_are_strict_json_and_reload_losslessly(tmp_path):
    """Regression: an infeasible candidate wrote ``Infinity`` into results.json.

    Its ``objective_score`` is +inf in memory, and ``json.dumps`` writes that
    (and NaN) as bare tokens that are not JSON, so strict parsers rejected the
    whole file.  They are written as null; ``strategy_failed`` still marks the
    candidate, so a reloaded workload ranks and diagnoses exactly as before.
    """
    dsroot = tmp_path / "ds_strict_json"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"artifact")
    dataset = wl.CircuitDataset(dsroot, [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})])
    selected = Strategy(
        spec=StrategySpec(optimization_level=2),
        metrics={"depth": 3.0, "objective_score": 3.0},
    )
    failed = Strategy(
        spec=StrategySpec(optimization_level=3),
        metrics={
            "compile_error": "boom",
            "strategy_failed": True,
            "strategy_failure_reason": "compile_failed",
            "objective_score": float("inf"),
        },
    )
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={"c0": selected},
        baseline_metrics={"c0": {"depth": 4.0, "compile_time_s": float("nan")}},
        objective=Objective({"depth": 1.0}),
        evaluation_history={"c0": [selected, failed]},
    )
    out = tmp_path / "balanced_strict_json"
    balanced.save(out)

    def reject_constant(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    text = (out / "results.json").read_text(encoding="utf-8")
    payload = json.loads(text, parse_constant=reject_constant)
    history = payload["evaluation_history"]["c0"]
    assert history[1]["metrics"]["objective_score"] is None
    assert payload["baseline_metrics"]["c0"]["compile_time_s"] is None

    reloaded = wl.load_balanced_workload(out)
    assert reloaded.candidate_rankings() == balanced.candidate_rankings()
    assert reloaded.selection_diagnostics() == balanced.selection_diagnostics()
    assert reloaded.summary() == balanced.summary()


def test_matrix_json_is_strict_json(tmp_path, monkeypatch):
    """A non-finite metric is written as null, not as a bare NaN token."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    monkeypatch.setattr(
        matrix_mod,
        "_evaluate_trial",
        lambda *args, **kwargs: {"depth": float("nan"), "ratio": float("-inf")},
    )
    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["fake:generic:2"],
        [StrategySpec()],
        tmp_path / "matrix.json",
    )

    def reject_constant(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    payload = json.loads(
        out.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    assert payload["results"][0]["metrics"] == {"depth": None, "ratio": None}


def test_load_balanced_workload_rejects_unknown_selection(tmp_path):
    out = tmp_path / "bad_balanced"
    dataset_dir = out / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "c0.qpy").write_bytes(b"placeholder")
    (dataset_dir / "qbalance_dataset.json").write_text(
        json.dumps(
            {
                "version": 1,
                "records": [{"name": "c0", "artifact": "c0.qpy", "format": "qpy"}],
            }
        ),
        encoding="utf-8",
    )
    (out / "results.json").write_text(
        json.dumps(
            {
                "backend_spec": "fake:generic:2",
                "objective": {"depth": 1.0},
                "selections": {"missing": {"spec": {}, "metrics": {}}},
                "baseline_metrics": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not present in dataset"):
        wl.load_balanced_workload(out)


def _write_saved_balanced_payload(tmp_path, payload, records=None):
    out = tmp_path / "saved_balanced"
    dataset_dir = out / "dataset"
    dataset_dir.mkdir(parents=True)
    if records is None:
        records = [{"name": "c0", "artifact": "c0.qpy", "format": "qpy"}]
    for record in records:
        (dataset_dir / record["artifact"]).write_bytes(b"placeholder")
    (dataset_dir / "qbalance_dataset.json").write_text(
        json.dumps({"version": 1, "records": records}), encoding="utf-8"
    )
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_load_balanced_workload_is_public_api():
    import qbalance
    import qbalance.workflow as workflow

    assert qbalance.load_balanced_workload is wl.load_balanced_workload
    assert workflow.load_balanced_workload is wl.load_balanced_workload


def test_load_balanced_workload_rejects_missing_selection(tmp_path):
    out = _write_saved_balanced_payload(
        tmp_path,
        {
            "backend_spec": "fake:generic:2",
            "objective": {"depth": 1.0},
            "selections": {},
            "baseline_metrics": {},
        },
    )

    with pytest.raises(ValueError, match="missing dataset circuits: c0"):
        wl.load_balanced_workload(out)


def test_load_balanced_workload_rejects_unknown_baseline(tmp_path):
    out = _write_saved_balanced_payload(
        tmp_path,
        {
            "backend_spec": "fake:generic:2",
            "objective": {"depth": 1.0},
            "selections": {"c0": {"spec": {}, "metrics": {}}},
            "baseline_metrics": {"ghost": {}},
        },
    )

    with pytest.raises(ValueError, match="baseline metrics reference circuits"):
        wl.load_balanced_workload(out)


def test_load_balanced_workload_wraps_invalid_strategy(tmp_path):
    out = _write_saved_balanced_payload(
        tmp_path,
        {
            "backend_spec": "fake:generic:2",
            "objective": {"depth": 1.0},
            "selections": {"c0": {"spec": {"optimization_level": 9}, "metrics": {}}},
            "baseline_metrics": {},
        },
    )

    with pytest.raises(ValueError, match="Selection for 'c0' has an invalid spec"):
        wl.load_balanced_workload(out)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "results must be a JSON object"),
        ({"backend_spec": "", "selections": {}}, "non-empty backend_spec"),
        (
            {"backend_spec": "b", "objective": [], "selections": {}},
            "objective must be a JSON object",
        ),
        (
            {"backend_spec": "b", "objective": {}, "selections": []},
            "selections must be a JSON object",
        ),
        (
            {"backend_spec": "b", "objective": {}, "selections": {"c0": []}},
            "selection for 'c0' must be a JSON object",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": []}},
            },
            "selection for 'c0' metrics must be a JSON object",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "baseline_metrics": [],
            },
            "baseline_metrics must be a JSON object",
        ),
    ],
)
def test_load_balanced_workload_rejects_malformed_payloads(tmp_path, payload, message):
    out = _write_saved_balanced_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        wl.load_balanced_workload(out)


@pytest.mark.parametrize("history_value", [None, pytest.param("missing", id="missing")])
def test_load_balanced_workload_accepts_legacy_payload_without_history(
    tmp_path, history_value
):
    payload = {
        "backend_spec": "fake:generic:2",
        "objective": {"depth": 1.0},
        "selections": {"c0": {"spec": {}, "metrics": {}}},
        "baseline_metrics": {},
    }
    if history_value != "missing":
        payload["evaluation_history"] = history_value
    out = _write_saved_balanced_payload(tmp_path, payload)

    loaded = wl.load_balanced_workload(out)

    assert loaded.evaluation_history == {}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": [],
            },
            "evaluation_history must be a JSON object",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {"c0": {}},
            },
            "evaluation history for 'c0' must be a list",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {"": []},
            },
            "evaluation history names must be non-empty strings",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {"c0": [[]]},
            },
            "evaluation history entry 0 for 'c0' must be a JSON object",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {"c0": [{"metrics": {}}]},
            },
            "Evaluation history entry 0 for 'c0' must include a spec object",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {"c0": [{"spec": {}, "metrics": []}]},
            },
            "evaluation history entry 0 for 'c0' metrics must be a JSON object",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {
                    "c0": [{"spec": {"optimization_level": 7}, "metrics": {}}]
                },
            },
            "Evaluation history entry 0 for 'c0' has an invalid spec",
        ),
        (
            {
                "backend_spec": "b",
                "objective": {},
                "selections": {"c0": {"spec": {}, "metrics": {}}},
                "evaluation_history": {"ghost": []},
            },
            "evaluation history references circuits not present",
        ),
    ],
)
def test_load_balanced_workload_rejects_malformed_evaluation_history(
    tmp_path, payload, message
):
    out = _write_saved_balanced_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        wl.load_balanced_workload(out)


def test_cli_seed_and_cache_options_forwarded(monkeypatch, tmp_path):
    adjust_kwargs = {}

    class BW:
        def save(self, out, overwrite=False):
            _ = (out, overwrite)

        def summary(self):
            return "sum"

    monkeypatch.setattr(
        cli.Workload,
        "from_path",
        classmethod(
            lambda cls, p: types.SimpleNamespace(
                set_target=lambda b: types.SimpleNamespace(
                    adjust=lambda **k: adjust_kwargs.update(k) or BW()
                )
            )
        ),
    )

    cache_root = tmp_path / "cache"
    cli.adjust_cmd(
        tmp_path,
        "b",
        tmp_path / "o",
        cache_root=cache_root,
        seed=123,
        shots=77,
    )
    assert adjust_kwargs["cache_root"] == cache_root
    assert adjust_kwargs["seed"] == 123
    assert adjust_kwargs["shots"] == 77

    matrix_kwargs = {}
    monkeypatch.setattr(
        cli,
        "run_matrix",
        lambda *a, **k: matrix_kwargs.update(k) or tmp_path / "m.json",
    )
    cli.matrix_cmd(tmp_path, ["b"], tmp_path / "m.json", seed=456, shots=88)
    assert matrix_kwargs["seed"] == 456
    assert matrix_kwargs["shots"] == 88
    # A single flip pattern relabels readout error instead of averaging it.
    twirled = [
        spec
        for spec in matrix_kwargs["strategies"]
        if spec.measurement_twirling or spec.pauli_twirling
    ]
    assert twirled and all(spec.num_twirls > 1 for spec in twirled)


def test_workload_adjust_validates_numeric_options(tmp_path):
    work = wl.Workload(dataset=types.SimpleNamespace()).set_target("b")
    with pytest.raises(ValueError, match="shots"):
        work.adjust(shots=0)
    with pytest.raises(ValueError, match="seed"):
        work.adjust(seed=True)
    with pytest.raises(ValueError, match="max_candidates"):
        work.adjust(max_candidates=True)
    with pytest.raises(ValueError, match="warmup"):
        work.adjust(warmup=-1)


def test_run_matrix_rejects_empty_or_string_sequences(tmp_path):
    with pytest.raises(ValueError, match="backend_specs"):
        matrix_mod.run_matrix(tmp_path, [], [StrategySpec()], tmp_path / "x.json")
    with pytest.raises(ValueError, match="backend_specs"):
        matrix_mod.run_matrix(
            tmp_path, "backend", [StrategySpec()], tmp_path / "x.json"
        )
    with pytest.raises(ValueError, match="strategies"):
        matrix_mod.run_matrix(tmp_path, ["b"], [], tmp_path / "x.json")
    with pytest.raises(ValueError, match="strategies"):
        matrix_mod.run_matrix(tmp_path, ["b"], "strategy", tmp_path / "x.json")


def test_run_matrix_writes_reproducibility_metadata(monkeypatch, tmp_path):
    ds = types.SimpleNamespace(
        records=[types.SimpleNamespace(name="c0")],
        load_circuits=lambda: [_Circ()],
    )
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(
            lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}})
        ),
    )

    out = matrix_mod.run_matrix(
        tmp_path,
        ["b0"],
        [StrategySpec()],
        tmp_path / "metadata.json",
        execute=False,
        shots=12,
        seed=34,
        profile=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"] == {
        "dataset_dir": str(tmp_path),
        "backends": ["b0"],
        "execute": False,
        "shots": 12,
        "seed": 34,
        "profile": True,
    }


def test_workload_adjust_accepts_integral_types_and_cache_root_string(
    monkeypatch, tmp_path
):
    qc = _Circ()
    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds_integral"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [record])
    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])
    monkeypatch.setattr(wl, "resolve_backend", lambda b: object())
    seen_cache_roots = []

    def fake_compile_cached(
        circuit, backend, spec, profile, cache_root, backend_key=None
    ):
        seen_cache_roots.append(cache_root)
        return circuit, {
            "depth": 1,
            "two_qubit_ops": 0,
            "estimated_error": 0.0,
            "compile_time_s": 0.0,
        }

    monkeypatch.setattr(wl, "_compile_cached", as_ensemble(fake_compile_cached))
    result = (
        wl.Workload.from_dataset(ds)
        .set_target("b")
        .adjust(
            search="bandit",
            max_candidates=np.int64(1),
            warmup=np.int64(0),
            seed=np.int64(3),
            shots=np.int64(4),
            cache_root=str(tmp_path / "cache"),
        )
    )
    assert result.selections["c0"].metrics["objective_score"] == 1.0
    assert all(root == tmp_path / "cache" for root in seen_cache_roots)


def test_bandit_skips_non_finite_observations(monkeypatch, tmp_path):
    qc = _Circ()
    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds_non_finite"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [record])
    monkeypatch.setattr(ds, "load_circuits", lambda: [qc])
    monkeypatch.setattr(wl, "resolve_backend", lambda b: object())
    specs = [StrategySpec(optimization_level=0), StrategySpec(optimization_level=1)]
    monkeypatch.setattr(wl, "default_candidate_strategies", lambda **k: specs)

    def fake_compile_cached(
        circuit, backend, spec, profile, cache_root, backend_key=None
    ):
        if spec.optimization_level == 0:
            return circuit, {"depth": float("inf")}
        return circuit, {"depth": 1, "two_qubit_ops": 0, "estimated_error": 0.0}

    monkeypatch.setattr(wl, "_compile_cached", as_ensemble(fake_compile_cached))
    result = (
        wl.Workload.from_dataset(ds)
        .set_target("b")
        .adjust(search="bandit", max_candidates=2, warmup=2)
    )
    assert result.selections["c0"].spec.optimization_level == 1


def test_failed_strategy_is_infeasible_even_with_good_compile_metrics():
    failed = StrategySpec(optimization_level=0, mthree=True)
    healthy = StrategySpec(optimization_level=1)
    chosen_spec, chosen_metrics = wl._choose(
        [
            (
                failed,
                {
                    "depth": 1,
                    "two_qubit_ops": 0,
                    "estimated_error": 0.0,
                    "mthree_error": "calibration failed",
                    "strategy_failed": True,
                    "strategy_failure_reason": "mthree_failed",
                    "objective_score": float("inf"),
                },
            ),
            (
                healthy,
                {
                    "depth": 10,
                    "two_qubit_ops": 0,
                    "estimated_error": 0.0,
                    "objective_score": 10.0,
                },
            ),
        ],
        pareto=False,
        objective=default_objective(),
    )
    assert chosen_spec == healthy
    assert chosen_metrics["objective_score"] == 10.0


def test_failed_strategy_cannot_dominate_pareto_front():
    failed = StrategySpec(optimization_level=0, mthree=True)
    healthy = StrategySpec(optimization_level=1)
    chosen_spec, chosen_metrics = wl._choose(
        [
            (
                failed,
                {
                    "depth": 1,
                    "two_qubit_ops": 0,
                    "estimated_error": 0.0,
                    "strategy_failed": True,
                    "strategy_failure_reason": "mthree_failed",
                    "objective_score": float("inf"),
                },
            ),
            (
                healthy,
                {
                    "depth": 10,
                    "two_qubit_ops": 1,
                    "estimated_error": 0.1,
                    "objective_score": 13.0,
                },
            ),
        ],
        pareto=True,
        objective=default_objective(),
    )
    assert chosen_spec == healthy
    assert chosen_metrics["objective_score"] == 13.0


def test_choose_raises_when_every_candidate_is_infeasible():
    with pytest.raises(RuntimeError, match="No feasible candidate"):
        wl._choose(
            [
                (
                    StrategySpec(optimization_level=0),
                    {
                        "depth": 1,
                        "strategy_failed": True,
                        "strategy_failure_reason": "execution_failed",
                        "objective_score": float("inf"),
                    },
                ),
                (StrategySpec(optimization_level=1), {"depth": "bad"}),
            ],
            pareto=False,
            objective=default_objective(),
        )


def test_strategy_failure_reason_marks_requested_runtime_failures():
    assert (
        wl._strategy_failure_reason(
            {"exec_error": "backend unavailable"}, StrategySpec(), execute=True
        )
        == "execution_failed"
    )
    assert (
        wl._strategy_failure_reason(
            {"mthree_error": "bad calibration"},
            StrategySpec(mthree=True),
            execute=False,
        )
        == "mthree_failed"
    )
    assert (
        wl._strategy_failure_reason(
            {"zne_error": "fold failed"}, StrategySpec(zne=True), execute=False
        )
        == "zne_failed"
    )
    assert (
        wl._strategy_failure_reason(
            {"exec_error": "ignored"}, StrategySpec(), execute=False
        )
        is None
    )


def test_compile_cache_key_separates_backends_sharing_a_display_name(tmp_path):
    """Regression: backend display names are not unique cache identities.

    ``fake:generic:5:1`` and ``fake:generic:5:7`` both report the name
    ``generic_backend_5q`` while carrying different calibration data, so keying
    the compile cache on the name alone served one backend's compiled circuit
    and calibration-derived metrics for the other.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from qbalance.backends import resolve_backend

    first = resolve_backend("fake:generic:5:1")
    second = resolve_backend("fake:generic:5:7")
    assert first.name == second.name

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    spec = StrategySpec(optimization_level=1, routing_method="sabre")

    cache_root = tmp_path / "cache"
    _, first_metrics = wl._compile_cached(
        qc,
        first,
        spec,
        profile=False,
        cache_root=cache_root,
        backend_key="fake:generic:5:1",
    )
    _, second_metrics = wl._compile_cached(
        qc,
        second,
        spec,
        profile=False,
        cache_root=cache_root,
        backend_key="fake:generic:5:7",
    )

    assert first_metrics["estimated_error"] != second_metrics["estimated_error"]

    # The same backend must still hit the cache.
    _, repeat_metrics = wl._compile_cached(
        qc,
        first,
        spec,
        profile=False,
        cache_root=cache_root,
        backend_key="fake:generic:5:1",
    )
    assert repeat_metrics["estimated_error"] == first_metrics["estimated_error"]


def test_adjust_threads_the_backend_spec_into_the_compile_cache_key(
    monkeypatch, tmp_path
):
    seen: list = []
    real_compile_cached = wl._compile_cached

    def recording_compile_cached(*args, **kwargs):
        seen.append(kwargs.get("backend_key"))
        return real_compile_cached(*args, **kwargs)

    monkeypatch.setattr(wl, "_compile_cached", recording_compile_cached)

    dataset_dir = tmp_path / "ds"
    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    save_dataset(dataset_dir, _make_tiny()[:1], overwrite=True)

    wl.Workload.from_path(dataset_dir).set_target("fake:generic:5").adjust(
        strategies=[StrategySpec(optimization_level=0)],
        cache_root=tmp_path / "cache",
    )

    from qbalance.backends import resolve_backend

    fingerprint = wl._calibration_fingerprint(resolve_backend("fake:generic:5"))
    assert seen
    assert set(seen) == {f"fake:generic:5|cal={fingerprint}"}


def test_a_recalibrated_backend_does_not_reuse_stale_compiles(monkeypatch, tmp_path):
    """Regression: the compile cache ignored calibration data.

    A device keeps its name and spec across recalibrations, but the
    noise-aware layout and ``estimated_error`` derive from its error rates, so
    a cache keyed on the spec kept serving layouts and error estimates computed
    from calibrations that no longer applied.
    """
    pytest.importorskip("qiskit")
    from qbalance.backends import resolve_backend
    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    before = resolve_backend("fake:generic:5")
    after = resolve_backend("fake:generic:5")
    assert wl._calibration_fingerprint(before) == wl._calibration_fingerprint(after)
    # Recalibrate: every two-qubit gate on the device gets worse.
    for name in after.target.operation_names:
        for qargs, props in after.target[name].items():
            if qargs is not None and len(qargs) == 2 and props is not None:
                props.error = min(1.0, 5 * (props.error or 0.01))
    assert wl._calibration_fingerprint(before) != wl._calibration_fingerprint(after)
    assert wl._calibration_fingerprint(object()) == "none"
    # A target whose operations cannot be listed cannot be described.
    broken = types.SimpleNamespace(target=types.SimpleNamespace(operation_names=None))
    assert wl._calibration_fingerprint(broken) == "unknown"

    save_dataset(tmp_path / "ds", _make_tiny()[:1], overwrite=True)
    spec = StrategySpec(optimization_level=1, routing_method="sabre")

    def _adjust(backend):
        monkeypatch.setattr(wl, "resolve_backend", lambda _spec: backend)
        balanced = (
            wl.Workload.from_path(tmp_path / "ds")
            .set_target("fake:generic:5")
            .adjust(strategies=[spec], cache_root=tmp_path / "cache")
        )
        (selection,) = balanced.selections.values()
        return selection.metrics["estimated_error"]

    stale = _adjust(before)
    fresh = _adjust(after)
    assert fresh > stale
    assert _adjust(before) == stale


def test_compile_cache_survives_a_corrupt_entry(tmp_path, monkeypatch):
    """Regression: a half-written cache entry used to abort the whole run.

    The compile cache lives in the platform cache directory and persists across
    runs, so any interrupted run left a truncated ``meta.json`` or
    ``compiled.qpy`` that made every later run fail with a raw JSON or QPY
    error.  A cache only saves work; it must never be a failure mode.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from qbalance.backends import resolve_backend

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    backend = resolve_backend("fake:generic:5")
    spec = StrategySpec(optimization_level=1, routing_method="sabre")
    cache_root = tmp_path / "cache"

    def compile_once():
        return wl._compile_cached(
            qc, backend, spec, profile=False, cache_root=cache_root, backend_key="b"
        )[1]["estimated_error"]

    expected = compile_once()
    assert compile_once() == expected  # served from cache

    for corrupt in (b"{ truncated", b"", b"\x00\x01"):
        for name in ("meta.json", "compiled.qpy"):
            for path in cache_root.rglob(name):
                path.write_bytes(corrupt)
            assert compile_once() == expected
            # The bad entry is healed by the recompile that replaced it.
            assert compile_once() == expected

    # A cache that cannot be written must not fail the compile either.
    monkeypatch.setattr(
        wl,
        "save_compiled",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")),
    )
    shutil.rmtree(cache_root)
    assert compile_once() == expected


def test_save_compiled_writes_atomically_and_leaves_no_partials(tmp_path):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from qbalance.cache import get_entry, load_compiled, save_compiled

    qc = QuantumCircuit(1)
    qc.h(0)
    entry = get_entry("a" * 64, root=tmp_path)

    save_compiled(entry, qc, {"depth": 1})

    assert sorted(p.name for p in entry.dir.iterdir()) == ["compiled.qpy", "meta.json"]
    loaded, meta = load_compiled(entry)
    assert meta["depth"] == 1
    assert loaded.num_qubits == 1

    # A circuit QPY cannot serialize must not leave a partial entry behind.
    entry2 = get_entry("b" * 64, root=tmp_path)
    with pytest.raises(TypeError, match="not a supported data type"):
        save_compiled(entry2, object(), {"depth": 1})
    assert not entry2.dir.exists() or list(entry2.dir.iterdir()) == []


def test_find_cuts_best_effort_uses_the_real_addon_api():
    """Regression: the wrapper called qiskit-addon-cutting two ways it never had.

    It imported from a ``qiskit_addon_cutting.cutting`` submodule that does not
    exist and passed ``DeviceConstraints(max_subcircuit_width=...)`` instead of
    ``qubits_per_subcircuit``.  Worse, the import error was reported as a missing
    optional dependency, sending users to reinstall a package they already had.
    """
    pytest.importorskip("qiskit_addon_cutting")
    from qiskit import QuantumCircuit

    from qbalance.cutting.addon_cutting import find_cuts_best_effort

    circuit = QuantumCircuit(6)
    circuit.h(0)
    for control, target in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 5),
        (1, 4),
        (2, 5),
    ]:
        circuit.cx(control, target)

    cut, meta = find_cuts_best_effort(circuit, max_subcircuit_qubits=3)

    assert cut.num_qubits == circuit.num_qubits
    assert isinstance(meta, dict)
    assert "sampling_overhead" in meta
    # Cutting must have actually replaced gates with QPD placeholders.
    assert any(name.startswith("qpd") for name in cut.count_ops())


def test_missing_cutting_dependency_is_reported_as_such(monkeypatch):
    import builtins

    from qbalance.cutting import addon_cutting
    from qbalance.errors import OptionalDependencyError

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "qiskit_addon_cutting":
            raise ImportError("no cutting addon")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "qiskit_addon_cutting", raising=False)

    with pytest.raises(
        OptionalDependencyError, match="qiskit-addon-cutting is required"
    ):
        addon_cutting.find_cuts_best_effort(object(), 3)


def test_a_skipped_cutting_candidate_says_why(monkeypatch, caplog):
    """Regression: a skipped candidate left no trace at all.

    It is absent from evaluation_history and from the rankings, so a broken
    cutting integration looked exactly like "cutting was simply not selected".
    """
    monkeypatch.setattr(
        wl,
        "prepare_cutting_experiment",
        lambda circuit, width: (_ for _ in ()).throw(RuntimeError("cannot cut")),
    )

    with caplog.at_level("WARNING", logger="qbalance.workflow.workload"):
        metrics = wl._evaluate_candidate(
            object(),
            object(),
            StrategySpec(optimization_level=1, cutting=True, max_subcircuit_qubits=3),
            objective=default_objective(),
            execute=False,
            shots=10,
            seed=0,
            profile=False,
            cache_root=None,
        )

    # Recorded as infeasible, with the reason, rather than silently dropped.
    assert metrics["cutting_error"] == "cannot cut"
    assert metrics["strategy_failure_reason"] == "cutting_failed"
    assert "cannot cut" in caplog.text
    assert "Candidate is infeasible" in caplog.text


def test_executed_runs_are_reproducible_cold_or_warm(tmp_path):
    """A fixed seed reproduces execution and mitigation results too.

    Twirl instances, M3 calibration and ZNE folds all run on seeded
    simulators, so two cold-cache runs, and a warm-cache run whose compiled
    circuits come back from the QPY cache, must agree on every execution
    metric -- not just on the compile metrics.
    """
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("mthree")
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset
    from qbalance.objectives import Objective

    bell = QuantumCircuit(2, 2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])
    save_dataset(tmp_path / "ds", [bell])
    specs = [
        StrategySpec(optimization_level=1, routing_method="sabre"),
        StrategySpec(measurement_twirling=True, num_twirls=3, mthree=True),
        StrategySpec(measurement_twirling=True, num_twirls=3, zne=True),
    ]
    objective = Objective({"depth": 1.0, "raw_parity_expval": -1.0})

    def run(cache_name):
        balanced = (
            wl.Workload.from_path(tmp_path / "ds")
            .set_target("fake:generic:5")
            .adjust(
                strategies=specs,
                objective=objective,
                execute=True,
                shots=500,
                seed=3,
                cache_root=tmp_path / cache_name,
            )
        )
        return [
            {k: v for k, v in h.metrics.items() if k != "compile_time_s"}
            for h in balanced.evaluation_history["bell"]
        ]

    cold = run("cache-a")
    assert cold == run("cache-b")
    assert cold == run("cache-a")  # warm: circuits reloaded from the cache
    assert {"mitigated_parity_expval", "zne_parity_expval"} <= set().union(*cold)


def test_adjust_is_reproducible_across_cold_caches(tmp_path):
    """A fixed seed must reproduce selections and compile metrics exactly.

    Two runs share a seed but not a cache, so every circuit is genuinely
    recompiled.  Only ``compile_time_s`` (wall clock) and the default
    objective's ``0.1 * compile_time_s`` term may differ, which is why this
    checks a time-free objective for bit equality.
    """
    pytest.importorskip("qiskit")

    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset
    from qbalance.objectives import Objective

    dataset_dir = tmp_path / "ds"
    save_dataset(dataset_dir, _make_tiny(), overwrite=True)
    objective = Objective(
        weights={"depth": 1.0, "two_qubit_ops": 2.0, "estimated_error": 10.0}
    )

    def run(cache_name):
        workload = wl.Workload.from_path(dataset_dir).set_target("fake:generic:5")
        balanced = workload.adjust(
            objective=objective,
            search="bandit",
            pareto=True,
            max_candidates=8,
            seed=42,
            cache_root=tmp_path / cache_name,
        )
        out_dir = tmp_path / f"out-{cache_name}"
        balanced.save(out_dir, overwrite=True)
        payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        return _without_compile_time(payload), balanced

    first, first_workload = run("cache-a")
    second, _ = run("cache-b")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    # A different seed must actually explore differently.
    other = (
        wl.Workload.from_path(dataset_dir)
        .set_target("fake:generic:5")
        .adjust(
            objective=objective,
            search="bandit",
            pareto=True,
            max_candidates=8,
            seed=7,
            cache_root=tmp_path / "cache-c",
        )
    )
    assert [s.spec for s in other.evaluation_history["qft4"]] != [
        s.spec for s in first_workload.evaluation_history["qft4"]
    ]


def _without_compile_time(value):
    if isinstance(value, dict):
        return {
            key: _without_compile_time(item)
            for key, item in value.items()
            if key != "compile_time_s"
        }
    if isinstance(value, list):
        return [_without_compile_time(item) for item in value]
    return value


def test_reporting_survives_a_workload_with_no_selections(tmp_path):
    """Regression: summary() and covars() crashed on an empty workload.

    ``adjust`` legitimately returns no selections for an empty dataset or an
    empty ``split`` half, but the distance helpers reject empty samples, so
    reporting raised "Input samples must be non-empty" from deep inside the
    diagnostics -- and ``save`` failed after already writing results.json,
    leaving a partial output directory.
    """
    pytest.importorskip("qiskit")

    from qbalance.dataset import load_dataset, save_dataset

    dataset_dir = tmp_path / "empty"
    save_dataset(dataset_dir, [], overwrite=True)
    assert len(load_dataset(dataset_dir)) == 0

    balanced = (
        wl.Workload.from_path(dataset_dir)
        .set_target("fake:generic:5")
        .adjust(cache_root=tmp_path / "cache")
    )
    assert balanced.selections == {}

    summary = balanced.summary()
    assert "circuits: 0" in summary
    assert "dist[depth]: n/a" in summary
    assert "dist[two_qubit_ops]: n/a" in summary

    covars = balanced.covars()
    assert set(covars) == {"depth", "two_qubit_ops", "estimated_error"}
    assert all(math.isnan(value) for row in covars.values() for value in row.values())

    out_dir = tmp_path / "out"
    balanced.save(out_dir, overwrite=True)
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "dataset",
        "results.json",
        "summary.txt",
    ]
    assert wl.load_balanced_workload(out_dir).selections == {}


def test_reporting_survives_an_empty_dataset_split(tmp_path):
    pytest.importorskip("qiskit")

    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import load_dataset, save_dataset

    save_dataset(tmp_path / "full", _make_tiny(), overwrite=True)
    train, test = load_dataset(tmp_path / "full").split(seed=0, frac_train=0.0)
    assert len(train) == 0 and len(test) == 3

    balanced = (
        wl.Workload.from_dataset(train)
        .set_target("fake:generic:5")
        .adjust(cache_root=tmp_path / "cache")
    )
    balanced.save(tmp_path / "out", overwrite=True)
    assert (tmp_path / "out" / "summary.txt").exists()


def test_non_empty_workloads_still_report_numeric_distances(tmp_path):
    pytest.importorskip("qiskit")

    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    save_dataset(tmp_path / "ds", _make_tiny(), overwrite=True)
    balanced = (
        wl.Workload.from_path(tmp_path / "ds")
        .set_target("fake:generic:5")
        .adjust(max_candidates=4, seed=0, cache_root=tmp_path / "cache")
    )

    distance_lines = [
        line for line in balanced.summary().splitlines() if "dist[" in line
    ]
    assert distance_lines
    assert all("n/a" not in line for line in distance_lines)
    assert all(
        not math.isnan(value)
        for row in balanced.covars().values()
        for value in row.values()
    )


def test_summary_reports_how_many_circuits_improved(tmp_path):
    """Mutation testing found the improved count unasserted.

    The summary line is the headline result of a run, so inverting the
    comparison that produces it must not go unnoticed.
    """
    from qbalance.dataset import CircuitDataset
    from qbalance.objectives import Objective

    dataset = CircuitDataset(tmp_path, [])
    objective = Objective(weights={"depth": 1.0})
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:5",
        selections={
            "better": Strategy(spec=StrategySpec(), metrics={"depth": 1.0}),
            "worse": Strategy(spec=StrategySpec(), metrics={"depth": 9.0}),
            "same": Strategy(spec=StrategySpec(), metrics={"depth": 5.0}),
        },
        baseline_metrics={
            "better": {"depth": 5.0},
            "worse": {"depth": 5.0},
            "same": {"depth": 5.0},
        },
        objective=objective,
    )

    diagnostics = balanced.selection_diagnostics()
    assert diagnostics["better"]["objective_improved"] is True
    assert diagnostics["worse"]["objective_improved"] is False
    assert (
        diagnostics["same"]["objective_improved"] is True
    )  # equal counts as no regression

    line = next(
        line for line in balanced.summary().splitlines() if "objective deltas" in line
    )
    # The headline counts strict gains; a tie is reported as unchanged.
    assert "improved=1/3 unchanged=1/3" in line


def test_to_download_writes_a_fresh_zip_without_overwrite(tmp_path):
    """Mutation testing found this path unexercised.

    ``overwrite=False`` must only refuse when the zip already exists.
    """
    from qbalance.dataset import CircuitDataset

    balanced = wl.BalancedWorkload(
        dataset=CircuitDataset(tmp_path / "ds", []),
        backend_spec="fake:generic:5",
        selections={},
    )
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "qbalance_dataset.json").write_text(
        json.dumps({"version": 1, "records": []}), encoding="utf-8"
    )

    target = tmp_path / "bundle.zip"
    assert not target.exists()
    assert balanced.to_download(target, overwrite=False) == target
    assert target.is_file()

    with pytest.raises(FileExistsError):
        balanced.to_download(target, overwrite=False)


def test_save_creates_missing_parent_directories(tmp_path):
    """Mutation testing found nested output paths unexercised."""
    from qbalance.dataset import CircuitDataset

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    (dataset_dir / "qbalance_dataset.json").write_text(
        json.dumps({"version": 1, "records": []}), encoding="utf-8"
    )
    balanced = wl.BalancedWorkload(
        dataset=CircuitDataset(dataset_dir, []),
        backend_spec="fake:generic:5",
        selections={},
    )

    nested = tmp_path / "a" / "b" / "c"
    assert not nested.parent.exists()
    balanced.save(nested, overwrite=False)
    assert (nested / "results.json").is_file()


def test_selection_diagnostics_reports_relative_deltas(tmp_path):
    """Relative deltas must be real ratios, and must skip a zero baseline.

    ``relative_delta`` divides by the baseline magnitude, so the zero-baseline
    guard is what keeps the diagnostics JSON-serializable instead of raising.
    Exercise both sides of that guard with numbers, not just the all-``None``
    degenerate case.
    """
    dsroot = tmp_path / "ds_rel_delta"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"placeholder")
    dataset = wl.CircuitDataset(dsroot, [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})])
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={
            "c0": Strategy(
                spec=StrategySpec(),
                metrics={"depth": 8.0, "two_qubit_ops": 5.0},
            )
        },
        baseline_metrics={"c0": {"depth": 10.0, "two_qubit_ops": 0.0}},
        objective=Objective({"depth": 1.0}),
    )

    deltas = balanced.selection_diagnostics()["c0"]["metric_deltas"]

    assert deltas["depth"] == {
        "baseline": 10.0,
        "selected": 8.0,
        "delta": -2.0,
        "relative_delta": -0.2,
    }
    # A zero baseline has no meaningful ratio; the delta still stands.
    assert deltas["two_qubit_ops"] == {
        "baseline": 0.0,
        "selected": 5.0,
        "delta": 5.0,
        "relative_delta": None,
    }


def test_measured_qubits_by_clbit_skips_malformed_measurements():
    """Only one-qubit/one-clbit measurements define the clbit -> qubit map.

    A measurement carrying more than one qubit or no clbit at all cannot say
    which qubit feeds which classical bit.  Mapping one anyway would hand
    mthree the wrong physical qubits and silently degrade the correction.
    """
    from tests.system_stubs import _I, _Q

    class _MalformedCirc:
        num_qubits = 3
        data = [
            (_I("measure"), [_Q(2)], [_Q(0)]),
            (_I("measure"), [_Q(0), _Q(1)], [_Q(1)]),
            (_I("measure"), [_Q(1)], []),
            (_I("barrier"), [_Q(0)], []),
        ]

    assert measured_qubits_by_clbit(_MalformedCirc()) == [2]


def test_selection_diagnostics_handles_a_metric_present_on_only_one_side(tmp_path):
    """A metric missing from one side has no delta, and must not raise.

    Baselines and selections are independent metric dicts, so a key can easily
    exist on one and not the other -- subtracting them would raise TypeError
    mid-report.
    """
    dsroot = tmp_path / "ds_one_sided"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"placeholder")
    dataset = wl.CircuitDataset(dsroot, [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})])
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={"c0": Strategy(spec=StrategySpec(), metrics={"depth": 4.0})},
        baseline_metrics={"c0": {"two_qubit_ops": 7.0}},
        objective=Objective({"depth": 1.0}),
    )

    deltas = balanced.selection_diagnostics()["c0"]["metric_deltas"]

    assert deltas["depth"] == {
        "baseline": None,
        "selected": 4.0,
        "delta": None,
        "relative_delta": None,
    }
    assert deltas["two_qubit_ops"] == {
        "baseline": 7.0,
        "selected": None,
        "delta": None,
        "relative_delta": None,
    }


def _one_circuit_workload(tmp_path, name="ds_dl"):
    """Build a minimal saveable BalancedWorkload."""
    dsroot = tmp_path / name
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"artifact")
    dataset = wl.CircuitDataset(dsroot, [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})])
    return wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={"c0": Strategy(spec=StrategySpec(), metrics={"depth": 1.0})},
        baseline_metrics={"c0": {"depth": 2.0}},
        objective=Objective({"depth": 1.0}),
    )


def _file_contents(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_a_failed_overwrite_keeps_the_previous_save(tmp_path, monkeypatch):
    """Regression: save(overwrite=True) deleted out_dir before writing.

    A failure while writing -- a full disk, an unreadable artifact -- then
    left a half-written directory where a complete earlier save had been.
    """
    balanced = _one_circuit_workload(tmp_path, name="ds_keep")
    out = tmp_path / "saved"
    balanced.save(out)
    before = _file_contents(out)

    def _disk_full(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(wl.shutil, "copy2", _disk_full)
    with pytest.raises(OSError, match="disk full"):
        balanced.save(out, overwrite=True)

    assert _file_contents(out) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds_keep", "saved"]


def test_a_failed_export_keeps_the_previous_archive(tmp_path, monkeypatch):
    """Regression: to_download truncated an existing archive before writing."""
    balanced = _one_circuit_workload(tmp_path, name="ds_zip_keep")
    zip_path = tmp_path / "workload.zip"
    zip_path.write_bytes(b"precious")

    def _disk_full(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(wl.zipfile.ZipFile, "write", _disk_full)
    with pytest.raises(OSError, match="disk full"):
        balanced.to_download(zip_path, overwrite=True)

    assert zip_path.read_bytes() == b"precious"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds_zip_keep", "workload.zip"]


def test_save_treats_a_dangling_link_as_an_existing_path(tmp_path):
    """A dangling link reports exists() == False, yet a save would replace it."""
    balanced = _one_circuit_workload(tmp_path, name="ds_link")
    link = tmp_path / "saved"
    try:
        link.symlink_to(tmp_path / "missing", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")

    with pytest.raises(FileExistsError):
        balanced.save(link)
    with pytest.raises(NotADirectoryError):
        balanced.save(link, overwrite=True)
    assert link.is_symlink()


def test_to_download_refuses_to_overwrite_by_default(tmp_path):
    """Exporting must not silently destroy an existing archive."""
    balanced = _one_circuit_workload(tmp_path)
    zip_path = tmp_path / "workload.zip"
    zip_path.write_bytes(b"precious")

    with pytest.raises(FileExistsError):
        balanced.to_download(zip_path)

    assert zip_path.read_bytes() == b"precious"


def test_to_download_overwrites_when_asked(tmp_path):
    balanced = _one_circuit_workload(tmp_path, name="ds_dl2")
    zip_path = tmp_path / "workload.zip"
    zip_path.write_bytes(b"precious")

    out = balanced.to_download(zip_path, overwrite=True)

    assert out == zip_path
    assert zip_path.read_bytes() != b"precious"


def test_grid_search_never_consults_the_bandit(tmp_path, monkeypatch):
    """Grid search evaluates every candidate; the bandit is for the other mode."""
    from tests.system_stubs import _Circ

    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds_grid"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])

    class _BanditThatMustNotBeUsed:
        def __init__(self, **kwargs):

            _ = kwargs

        def observe(self, *a, **k):

            raise AssertionError("grid search must not consult the bandit")

        def propose(self, *a, **k):

            raise AssertionError("grid search must not consult the bandit")

    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(wl, "BanditSearcher", _BanditThatMustNotBeUsed)
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [
            StrategySpec(seed_transpiler=i) for i in range(3)
        ],
    )
    monkeypatch.setattr(
        wl,
        "compile_ensemble",
        as_ensemble(lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})),
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    balanced = (
        wl.Workload.from_dataset(ds)
        .set_target("fake:generic:2")
        .adjust(search="grid", max_candidates=3)
    )

    assert balanced.selection_diagnostics()["c0"]["evaluated_candidates"] == 3


def test_candidate_is_not_cut_without_the_cutting_flag(tmp_path, monkeypatch):
    """max_subcircuit_qubits alone must not trigger circuit cutting.

    The width limit is meaningful only when cutting is requested; acting on it
    by itself would cut circuits the caller never asked to cut, and a cutting
    failure silently drops the candidate.
    """
    from tests.system_stubs import _Circ

    def _must_not_cut(*a, **k):

        raise AssertionError("cutting must not run when spec.cutting is False")

    monkeypatch.setattr(wl, "prepare_cutting_experiment", _must_not_cut)
    monkeypatch.setattr(
        wl, "_compile_cached", as_ensemble(lambda *a, **k: (_Circ(), {"depth": 2.0}))
    )

    metrics = wl._evaluate_candidate(
        _Circ(),
        types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
        StrategySpec(cutting=False, max_subcircuit_qubits=2),
        objective=Objective({"depth": 1.0}),
        execute=False,
        shots=16,
        seed=0,
        profile=False,
        cache_root=tmp_path,
    )

    assert metrics is not None


def test_mitigation_receives_raw_counts_and_each_instance_flip_map(
    tmp_path, monkeypatch
):
    """M3 corrects the counts as measured; twirl flips are undone afterwards.

    M3 calibrates the physical readout channel.  Where a twirl flip was
    applied, the untwirled bit's 0->1 and 1->0 error rates are swapped
    relative to that calibration, so correcting untwirled counts silently
    biases every asymmetric-readout result (a true 50/50 state comes back as
    roughly 26/74 at a 20% 1->0 error rate).  The workload therefore hands
    mitigation the raw counts of every instance together with that
    instance's own flip map.
    """
    from tests.system_stubs import _Circ

    seen: list = []

    def _capture(backend, raw_counts, flip_maps, measured_qubits, seed, clbits):

        assert seed == 3
        # The stub circuit's measurement writes no classical bit it can name.
        assert clbits is None
        seen.append((list(raw_counts), list(flip_maps)))
        return {"0": 0.25, "1": 0.75}

    ensemble_metrics = {
        "measurement_flip_map": {0: 1},
        "measurement_flip_maps": [{0: 1}, {}],
        "twirl_instances": 2,
    }
    monkeypatch.setattr(
        wl,
        "_compile_cached",
        lambda *a, **k: ([_Circ(), _Circ()], dict(ensemble_metrics)),
    )
    shots_seen: list = []

    def _run(backend, circuit, shots, seed_simulator):

        shots_seen.append((shots, seed_simulator))
        return {"0": 8, "1": 2} if len(shots_seen) == 1 else {"0": 1, "1": 9}

    monkeypatch.setattr(ensemble, "run_counts", _run)
    monkeypatch.setattr(ensemble, "mitigate_twirled_counts", _capture)

    metrics = wl._evaluate_candidate(
        _Circ(),
        types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
        StrategySpec(mthree=True, measurement_twirling=True, num_twirls=2),
        objective=Objective({"depth": 1.0}),
        execute=True,
        shots=20,
        seed=3,
        profile=False,
        cache_root=tmp_path,
    )

    # Every instance ran, on its own share of the shots and its own seed ...
    assert shots_seen == [(10, 3), (10, 4)]
    # ... mitigation saw each instance's counts exactly as measured, paired
    # with the flip map that instance was compiled with ...
    assert seen == [([{"0": 8, "1": 2}, {"0": 1, "1": 9}], [{0: 1}, {}])]
    # ... and the raw metrics pool the untwirled instances: instance 0 flips
    # back to {"1": 8, "0": 2}, instance 1 is unflipped.
    assert metrics is not None
    assert metrics["raw_top_prob"] == pytest.approx(17 / 20)
    assert metrics["mitigated_top_prob"] == 0.75


def test_compile_cache_key_separates_backends_sharing_a_class(tmp_path, monkeypatch):
    """Backend identity comes from the backend's name, not its Python class.

    Every fake backend shares one class, so keying on the class name would let
    calibration-derived metrics leak between different devices.
    """
    from tests.system_stubs import _Circ

    keys: list[str] = []

    def _record(key_hash, root=None):

        keys.append(key_hash)
        return types.SimpleNamespace(dir=tmp_path)

    monkeypatch.setattr(wl, "get_entry", _record)
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(
        wl, "compile_ensemble", as_ensemble(lambda *a, **k: (_Circ(), {}))
    )
    monkeypatch.setattr(wl, "fingerprint_circuit", lambda c: "fingerprint")

    class _Backend:
        def __init__(self, label):

            self._label = label

        def name(self):

            return self._label

    for label in ("alpha", "beta"):
        wl._compile_cached(_Circ(), _Backend(label), StrategySpec(), False, tmp_path)

    assert len(keys) == 2
    assert keys[0] != keys[1]


def test_loading_rejects_an_empty_selection_name(tmp_path):
    """An empty selection name is not a circuit name.

    It would otherwise be reported as "references circuits not present in the
    dataset", which points the reader at the dataset instead of the malformed
    key that is actually at fault.
    """
    out_dir = tmp_path / "saved"
    (out_dir / "dataset").mkdir(parents=True)
    (out_dir / "dataset" / "qbalance_dataset.json").write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {
                        "name": "c0",
                        "artifact": "c0.qpy",
                        "format": "qpy",
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "dataset" / "c0.qpy").write_bytes(b"artifact")
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "backend_spec": "fake:generic:2",
                "objective": {"depth": 1.0},
                "selections": {
                    "": {"spec": StrategySpec().model_dump(), "metrics": {}}
                },
                "baseline_metrics": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty strings"):
        wl.load_balanced_workload(out_dir)


def test_regression_guard_is_off_by_default(tmp_path, monkeypatch):
    """adjust() explores freely unless the caller opts into the safety rail.

    The guard replaces a worse-than-baseline selection with the baseline, so
    turning it on by default would silently change what every caller ships.
    """
    from tests.system_stubs import _Circ

    guarded: list[int] = []

    def _record_guard(
        baseline_spec, baseline_metrics, chosen_spec, chosen_metrics, obj
    ):

        guarded.append(1)
        return chosen_spec, chosen_metrics

    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds_guard"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])

    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(wl, "_guard_against_regression", _record_guard)
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [StrategySpec()],
    )
    monkeypatch.setattr(
        wl,
        "compile_ensemble",
        as_ensemble(lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})),
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    wl.Workload.from_dataset(ds).set_target("fake:generic:2").adjust(
        search="grid", max_candidates=1
    )

    assert guarded == []


def test_adjust_does_not_use_pareto_selection_by_default(tmp_path, monkeypatch):
    """Pareto selection is opt-in.

    It can pick a different strategy than the plain minimum-objective rule,
    so turning it on by default would quietly change every caller's result.
    """
    from tests.system_stubs import _Circ

    def _must_not_run(*a, **k):

        raise AssertionError("pareto selection must be opt-in")

    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds_pareto_default"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])

    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(wl, "pareto_front", _must_not_run)
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [
            StrategySpec(seed_transpiler=i) for i in range(2)
        ],
    )
    monkeypatch.setattr(
        wl,
        "compile_ensemble",
        as_ensemble(lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})),
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)

    balanced = (
        wl.Workload.from_dataset(ds)
        .set_target("fake:generic:2")
        .adjust(search="grid", max_candidates=2)
    )

    assert "c0" in balanced.selections


def test_adjust_requires_a_positive_candidate_budget(tmp_path):
    """A zero or negative budget evaluates nothing and cannot select."""
    dsroot = tmp_path / "ds_budget"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})])
    workload = wl.Workload.from_dataset(ds).set_target("fake:generic:2")

    for budget in (0, -1):
        with pytest.raises(ValueError, match="max_candidates must be a positive"):
            workload.adjust(max_candidates=budget)


def test_measured_qubits_by_clbit_falls_back_to_the_full_width(tmp_path):
    """With no recoverable mapping, every qubit is assumed measured in order.

    Returning an empty list instead would hand mthree no qubits to correct,
    which mis-mitigates silently rather than failing.  The width guard also
    has to survive a backend stub that reports no width at all.
    """
    from tests.system_stubs import _I, _Q

    class _NoMeasurements:
        num_qubits = 3
        data = [(_I("h"), [_Q(0)], []), (_I("cx"), [_Q(0), _Q(1)], [])]

    class _NoWidth:
        data = []

    class _NullWidth:
        num_qubits = None
        data = []

    assert measured_qubits_by_clbit(_NoMeasurements()) == [0, 1, 2]
    assert measured_qubits_by_clbit(_NoWidth()) == []
    assert measured_qubits_by_clbit(_NullWidth()) == []


def test_to_download_creates_missing_parent_directories(tmp_path):
    """The archive path may name a directory that does not exist yet.

    The staging directory is created beside the archive, so the whole parent
    chain has to exist first.
    """
    balanced = _one_circuit_workload(tmp_path, name="ds_dl_nested")
    zip_path = tmp_path / "deep" / "nested" / "workload.zip"

    out = balanced.to_download(zip_path)

    assert out == zip_path
    assert zip_path.is_file()


def test_save_refuses_to_overwrite_by_default(tmp_path):
    """An existing output directory is not replaced unless asked."""
    balanced = _one_circuit_workload(tmp_path, name="ds_save_default")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("precious", encoding="utf-8")

    with pytest.raises(FileExistsError):
        balanced.save(out_dir)

    assert (out_dir / "keep.txt").read_text(encoding="utf-8") == "precious"


def test_save_refuses_to_overwrite_a_directory_holding_the_source_dataset(tmp_path):
    """overwrite=True deletes the target, so it must not contain the source.

    The guard covers two shapes: the output is the dataset directory itself,
    or it is an ancestor of it.  Requiring both would make the guard
    unreachable, and save() would then rmtree the dataset it is reading.
    """
    balanced = _one_circuit_workload(tmp_path, name="ds_self")
    dataset_root = Path(balanced.dataset.root)

    # The output *is* the dataset directory.
    with pytest.raises(ValueError, match="source dataset"):
        balanced.save(dataset_root, overwrite=True)

    # The output is an ancestor of the dataset directory.
    with pytest.raises(ValueError, match="source dataset"):
        balanced.save(dataset_root.parent, overwrite=True)

    assert (dataset_root / "c0.qpy").is_file()


def test_to_download_cleanup_does_not_mask_the_original_failure(tmp_path, monkeypatch):
    """Discarding the staging tree must not replace the real error.

    The cleanup runs in a finally while the failure that caused it is already
    propagating; without ignore_errors a failing rmtree is what the caller
    sees instead.  Same contract as save_dataset and atomic_write_bytes.
    """
    balanced = _one_circuit_workload(tmp_path, name="ds_dl_mask")

    real_rmtree = wl.shutil.rmtree
    armed = {"on": False}

    def _explode(*a, **k):

        # save() has finished staging by now, so only the cleanup that runs
        # while this error propagates should see the failing rmtree.
        armed["on"] = True
        raise RuntimeError("archive failed")

    def _rmtree(path, ignore_errors=False):

        if not armed["on"]:
            # Anything save() removes before the archive step must behave
            # normally; only the cleanup under the propagating error fails.
            return real_rmtree(path, ignore_errors=ignore_errors)
        if ignore_errors:
            return
        raise OSError("rmtree refused")

    monkeypatch.setattr(wl.zipfile, "ZipFile", _explode)
    monkeypatch.setattr(wl.shutil, "rmtree", _rmtree)

    with pytest.raises(RuntimeError, match="archive failed"):
        balanced.to_download(tmp_path / "out.zip")


def test_matrix_output_is_written_atomically(tmp_path, monkeypatch):
    """A failed write must not destroy the previous results file.

    The matrix runs compiles (and optionally executions) across every backend
    x circuit x strategy before writing anything, and the file it produces is
    read back by ``qbalance report``.  A plain write truncates the destination
    the moment it opens, so an interrupted write loses the old results and
    leaves a partial file the next step rejects.
    """
    dsroot = tmp_path / "ds_atomic"
    dsroot.mkdir()
    (dsroot / "c0.qpy").write_bytes(b"placeholder")
    dataset = wl.CircuitDataset(dsroot, [wl.CircuitRecord("c0", "c0.qpy", "qpy", {})])
    monkeypatch.setattr(dataset, "load_circuits", lambda: [object()])

    monkeypatch.setattr(matrix_mod, "load_dataset", lambda d: dataset)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(lambda qc, backend, spec, profile: (qc, {"depth": 1.0})),
    )

    out_json = tmp_path / "matrix.json"
    out_json.write_text("previous results", encoding="utf-8")

    import qbalance.utils as utils_mod

    def _failing_replace(src, dst):

        raise OSError("replace failed")

    monkeypatch.setattr(utils_mod.os, "replace", _failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        matrix_mod.run_matrix(dsroot, ["b"], [StrategySpec()], out_json)

    # The previous file is untouched and no partial artifact is left beside it.
    assert out_json.read_text(encoding="utf-8") == "previous results"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds_atomic", "matrix.json"]


def test_documented_adjust_parameters_match_the_signature():
    """The adjust() table in the API reference is a user-facing contract.

    A parameter added to the signature but not the table is undiscoverable,
    and a default that drifts sends readers to the wrong conclusion about
    what a bare call does.  Neither shows up as a test failure anywhere else.
    """
    import inspect
    import re

    doc_path = Path(__file__).resolve().parents[1] / "docs" / "api-references.md"
    if not doc_path.is_file():
        pytest.skip("docs are not present in this checkout")

    # The table runs from its header to the first blank line.
    block = (
        doc_path.read_text(encoding="utf-8")
        .split("| Parameter | Default | Description |")[1]
        .split("\n\n")[0]
    )
    documented = dict(
        re.findall(r"^\|\s*`([a-z_]+)`\s*\|\s*`([^`]*)`\s*\|", block, re.M)
    )
    actual = {
        name: param.default
        for name, param in inspect.signature(wl.Workload.adjust).parameters.items()
        if name != "self"
    }

    assert set(documented) == set(actual)

    def rendered(value):

        # Identity checks, not equality: 0 == False in Python, so a dict
        # lookup keyed on False would claim seed=0 is documented wrong.
        if value is None:
            return "None"
        if value is True:
            return "True"
        if value is False:
            return "False"
        return repr(value).replace("'", '"')

    assert {name: rendered(value) for name, value in actual.items()} == documented


def test_zne_extrapolates_against_realized_fold_factors(tmp_path, monkeypatch):
    """Regression: the fit used requested factors, not the ones folding ran.

    Folding only realizes odd factors, so the default ``(1, 2, 3)`` runs at
    ``(1, 3, 3)``.  Fitting against ``(1, 2, 3)`` placed the second sample at
    the wrong noise level and biased the zero-noise estimate.
    """
    from tests.system_stubs import _Circ

    captured: list[list[float]] = []

    def _extrapolate(factors, counts_pf, degree):

        captured.append(list(factors))
        return {"0": 1.0}

    monkeypatch.setattr(
        wl,
        "_compile_cached",
        as_ensemble(lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})),
    )
    monkeypatch.setattr(ensemble, "run_counts", lambda *a, **k: {"0": 8, "1": 2})
    monkeypatch.setattr(ensemble, "fold_global_for_backend", lambda c, backend, f: c)
    monkeypatch.setattr(ensemble, "zne_extrapolate_counts", _extrapolate)

    metrics = wl._evaluate_candidate(
        _Circ(),
        types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
        StrategySpec(zne=True, zne_factors=(1.0, 2.0, 3.0, 4.0)),
        objective=Objective({"depth": 1.0}),
        execute=False,
        shots=10,
        seed=0,
        profile=False,
        cache_root=tmp_path,
    )

    assert captured == [[1.0, 3.0, 3.0, 5.0]]
    assert metrics is not None
    assert metrics["zne_realized_factors"] == [1.0, 3.0, 3.0, 5.0]
    assert "zne_error" not in metrics

    # The matrix runner extrapolates the same way.
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.h(0)
    qc.measure(0, 0)
    dsroot = tmp_path / "zne_matrix_ds"
    save_dataset(dsroot, [qc])
    captured.clear()
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(
            lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}})
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda backend, compiled, shots, seed_simulator: {"0": 1},
    )
    monkeypatch.setattr(ensemble, "fold_global_for_backend", lambda c, backend, f: c)
    monkeypatch.setattr(ensemble, "zne_extrapolate_counts", _extrapolate)

    out = matrix_mod.run_matrix(
        dsroot,
        ["b"],
        [StrategySpec(zne=True, zne_factors=(1.0, 2.0, 3.0))],
        tmp_path / "zne.json",
        execute=True,
    )

    assert captured == [[1.0, 3.0, 3.0]]
    row = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert row["metrics"]["zne_realized_factors"] == [1.0, 3.0, 3.0]


def test_a_cut_candidate_whose_subexperiments_cannot_compile_is_skipped(
    tmp_path, monkeypatch, caplog
):
    """A subexperiment the backend cannot run skips that one candidate.

    For example a subcircuit wider than the device: the failure belongs to
    this candidate and must not abort the whole run.
    """
    from qbalance.cutting.addon_cutting import CuttingExperiment
    from tests.system_stubs import _Circ

    experiment = CuttingExperiment(
        subexperiments={0: [_Circ()]},
        coefficients=[(1.0, None)],
        subobservables={0: None},
        measured_qubits=[0],
        metadata={"cut_count": 1, "sampling_overhead": 9.0},
    )
    monkeypatch.setattr(
        wl, "prepare_cutting_experiment", lambda circuit, width: experiment
    )

    def _fail(*a, **k):

        raise RuntimeError("subcircuit is wider than the device")

    monkeypatch.setattr(wl, "_compile_cached", _fail)

    with caplog.at_level("WARNING", logger="qbalance.workflow.workload"):
        metrics = wl._evaluate_candidate(
            _Circ(),
            object(),
            StrategySpec(optimization_level=1, cutting=True, max_subcircuit_qubits=3),
            objective=default_objective(),
            execute=False,
            shots=10,
            seed=0,
            profile=False,
            cache_root=tmp_path,
        )

    assert metrics["compile_error"] == "subcircuit is wider than the device"
    assert metrics["strategy_failure_reason"] == "compile_failed"
    assert metrics["sampling_overhead"] == 9.0  # the cut it could not compile
    assert "could not be compiled" in caplog.text
    assert "wider than the device" in caplog.text

    # An uncut candidate that cannot compile is recorded as infeasible: it
    # stays in the history with its reason instead of aborting the run --
    # which would also stop a cut candidate from ever handling a circuit
    # wider than the device.
    metrics = wl._evaluate_candidate(
        _Circ(),
        object(),
        StrategySpec(optimization_level=1),
        objective=default_objective(),
        execute=False,
        shots=10,
        seed=0,
        profile=False,
        cache_root=tmp_path,
    )
    assert metrics["compile_error"] == "subcircuit is wider than the device"
    assert metrics["strategy_failed"] is True
    assert metrics["strategy_failure_reason"] == "compile_failed"
    assert metrics["objective_score"] == float("inf")


def test_a_circuit_wider_than_the_backend_is_handled_by_cutting(tmp_path):
    """Regression: one circuit wider than the device aborted the whole run.

    Its uncut baseline and candidates raise TranspilerError at compile time.
    That used to propagate out of ``adjust`` and ``run_matrix``, losing every
    other circuit's results -- and preventing the cutting candidate, whose
    subcircuits do fit, from ever being evaluated.
    """
    pytest.importorskip("qiskit_addon_cutting")
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset
    from qbalance.reports.common import failed_trials

    wide = QuantumCircuit(7, 7, name="wide")
    wide.h(0)
    for qubit in range(6):
        wide.cx(qubit, qubit + 1)
    wide.measure(range(7), range(7))
    bell = QuantumCircuit(2, 2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])
    save_dataset(tmp_path / "ds", [wide, bell])
    cut = StrategySpec(optimization_level=1, cutting=True, max_subcircuit_qubits=4)
    specs = [StrategySpec(optimization_level=2), cut]

    balanced = (
        wl.Workload.from_path(tmp_path / "ds")
        .set_target("fake:generic:5")
        .adjust(strategies=specs, cache_root=tmp_path / "cache")
    )

    assert balanced.selections["wide"].spec == cut
    assert balanced.selections["wide"].metrics["cut_count"] >= 1
    assert balanced.baseline_metrics["wide"]["strategy_failure_reason"] == (
        "compile_failed"
    )
    uncut = balanced.evaluation_history["wide"][0].metrics
    assert uncut["compile_error"]
    assert uncut["strategy_failure_reason"] == "compile_failed"
    assert balanced.selections["bell"].spec in specs
    # Reporting copes with a baseline that has no metrics to compare.
    assert "circuits: 2" in balanced.summary()
    assert balanced.selection_diagnostics()["wide"]["objective_improved"] is None

    out = matrix_mod.run_matrix(
        tmp_path / "ds", ["fake:generic:5"], specs, tmp_path / "m.json"
    )
    rows = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert len(rows) == 4
    assert failed_trials(rows) == 1
    assert "compile_error" in rows[0]["metrics"]


def test_executing_a_circuit_without_measurements_says_so():
    from qiskit import QuantumCircuit

    from qbalance.execution import run_counts

    qc = QuantumCircuit(2, name="unmeasured")
    qc.h(0)
    with pytest.raises(ValueError, match="'unmeasured' has no measurements"):
        run_counts(object(), qc, shots=10)


def test_compile_cache_key_includes_the_toolchain(tmp_path, monkeypatch):
    """A qiskit upgrade or a compile-pipeline change must miss the cache.

    Otherwise results compiled by the previous toolchain keep being served as
    if the running code had produced them.
    """
    from tests.system_stubs import _Circ

    keys: list[str] = []

    def _record(key_hash, root=None):

        keys.append(key_hash)
        return types.SimpleNamespace(dir=tmp_path)

    monkeypatch.setattr(wl, "get_entry", _record)
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(
        wl, "compile_ensemble", as_ensemble(lambda *a, **k: (_Circ(), {}))
    )
    monkeypatch.setattr(wl, "fingerprint_circuit", lambda c: "fingerprint")

    def _key() -> str:

        wl._compile_cached(_Circ(), object(), StrategySpec(), False, tmp_path)
        return keys[-1]

    real_qiskit_version = wl._qiskit_version
    baseline = _key()
    assert _key() == baseline

    monkeypatch.setattr(wl, "_qiskit_version", lambda: "0.0.0-other")
    other_qiskit = _key()
    monkeypatch.setattr(wl, "_qiskit_version", real_qiskit_version)
    monkeypatch.setattr(wl, "_COMPILE_CACHE_VERSION", wl._COMPILE_CACHE_VERSION + 1)
    other_pipeline = _key()

    assert len({baseline, other_qiskit, other_pipeline}) == 3


def test_saved_workload_built_on_a_split_reloads(tmp_path):
    """Regression: saving copied the source index, which lists both halves.

    The artifacts of the records outside this workload were not copied, so
    load_balanced_workload rejected the workload's own saved output.
    """
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    circuits = []
    for name in ("a", "b", "c", "d"):
        qc = QuantumCircuit(2, 2, name=name)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)
    dataset = save_dataset(tmp_path / "ds", circuits)
    train, test = dataset.split(seed=0, frac_train=0.5)
    assert len(train) == 2 and len(test) == 2

    balanced = (
        wl.Workload.from_dataset(train)
        .set_target("fake:generic:3")
        .adjust(
            strategies=[StrategySpec(optimization_level=1)],
            cache_root=tmp_path / "cache",
        )
    )
    balanced.save(tmp_path / "out")

    loaded = wl.load_balanced_workload(tmp_path / "out")
    assert sorted(loaded.dataset.names()) == sorted(train.names())
    assert set(loaded.selections) == set(train.names())
    assert len(loaded.dataset.load_circuits()) == 2


def test_adjust_rejects_a_negative_seed_up_front(tmp_path):
    """The seed seeds numpy's generator, which rejects negative values."""
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    ds = wl.CircuitDataset(tmp_path, [rec])

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        wl.Workload.from_dataset(ds).set_target("fake:generic:2").adjust(seed=-1)


@pytest.mark.parametrize("flag", ["pareto", "execute", "profile", "allow_regression"])
def test_adjust_rejects_a_flag_that_is_not_a_boolean(tmp_path, flag):
    """Regression: flags were only truth-tested, so execute="no" executed."""
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    ds = wl.CircuitDataset(tmp_path, [rec])

    with pytest.raises(ValueError, match=f"{flag} must be a boolean"):
        wl.Workload.from_dataset(ds).set_target("fake:generic:2").adjust(**{flag: "no"})


def test_adjust_accepts_numpy_booleans_and_a_mapping_objective(tmp_path):
    """NumPy booleans are booleans, and a weight mapping is an objective.

    ``objective={"depth": 1.0}`` used to fail deep inside with
    "'dict' object has no attribute 'score'".
    """
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    ds = save_dataset(tmp_path / "ds", [qc])
    workload = wl.Workload.from_dataset(ds).set_target("fake:generic:3")
    strategies = [StrategySpec(optimization_level=0), StrategySpec()]

    balanced = workload.adjust(
        objective={"depth": 1.0},
        strategies=strategies,
        pareto=np.bool_(False),
        cache_root=tmp_path / "cache",
    )
    assert balanced.objective.weights == {"depth": 1.0}
    reference = workload.adjust(
        objective=Objective({"depth": 1.0}),
        strategies=strategies,
        cache_root=tmp_path / "cache",
    )
    assert balanced.selections["bell"].spec == reference.selections["bell"].spec

    with pytest.raises(ValueError, match="objective must be an Objective"):
        workload.adjust(objective="depth", strategies=strategies)


def test_run_matrix_takes_strategy_mappings_and_any_iterable_of_specs(
    tmp_path, monkeypatch
):
    """Strategies arrive as adjust takes them; repeats keep their rows."""
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        as_ensemble(lambda qc, backend, spec, profile: (qc, {"depth": 1.0})),
    )

    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        (spec for spec in ["b"]),
        [{"optimization_level": 2}, StrategySpec(optimization_level=2)],
        tmp_path / "m.json",
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["backends"] == ["b"]
    assert [row["strategy"]["optimization_level"] for row in payload["results"]] == [
        2,
        2,
    ]

    for bad, message in [
        ({"strategies": [{"optimisation_level": 2}]}, "Invalid strategy entry 0"),
        ({"strategies": StrategySpec()}, "not a single strategy"),
        ({"backend_specs": [object()]}, "backend spec strings"),
        ({"execute": "no"}, "execute must be a boolean"),
    ]:
        arguments = {
            "dataset_dir": tmp_path / "ds",
            "backend_specs": ["b"],
            "strategies": [StrategySpec()],
            "out_json": tmp_path / "bad.json",
            **bad,
        }
        with pytest.raises(ValueError, match=message):
            matrix_mod.run_matrix(**arguments)


def test_no_feasible_candidate_error_says_why():
    """Every candidate failing the same way must say how it failed."""
    evals = [
        (
            StrategySpec(optimization_level=1),
            {
                "strategy_failed": True,
                "strategy_failure_reason": "execution_failed",
                "exec_error": "No counts for experiment",
            },
        ),
        (
            StrategySpec(optimization_level=2),
            {
                "strategy_failed": True,
                "strategy_failure_reason": "execution_failed",
                "exec_error": "No counts for experiment",
            },
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match=r"No feasible candidate.*execution_failed \(No counts for experiment\)",
    ):
        wl._choose(evals, pareto=False, objective=default_objective())


def test_zne_folds_and_runs_every_twirl_instance(tmp_path, monkeypatch):
    """Each twirl instance is folded, run on its shot share, and untwirled."""
    from tests.system_stubs import _Circ

    first, second = _Circ(), _Circ()
    monkeypatch.setattr(
        wl,
        "_compile_cached",
        lambda *a, **k: (
            [first, second],
            {
                "measurement_flip_map": {0: 1},
                "measurement_flip_maps": [{0: 1}, {}],
                "twirl_instances": 2,
            },
        ),
    )
    folds: list = []

    def _fold(circuit, backend, factor):

        folds.append((circuit, factor))
        return circuit

    runs: list = []

    def _run(backend, circuit, shots, seed_simulator):

        runs.append((circuit, shots, seed_simulator))
        return {"0": shots}

    captured: list = []

    def _extrapolate(factors, counts_pf, degree):

        captured.append((list(factors), [dict(c) for c in counts_pf]))
        return {"0": 1.0}

    monkeypatch.setattr(ensemble, "fold_global_for_backend", _fold)
    monkeypatch.setattr(ensemble, "run_counts", _run)
    monkeypatch.setattr(ensemble, "zne_extrapolate_counts", _extrapolate)

    metrics = wl._evaluate_candidate(
        _Circ(),
        types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
        StrategySpec(zne=True, zne_factors=(1.0, 3.0)),
        objective=Objective({"depth": 1.0}),
        execute=False,
        shots=11,
        seed=5,
        profile=False,
        cache_root=tmp_path,
    )

    assert metrics is not None and "zne_error" not in metrics
    # Factor 1.0 is the unfolded circuit, whose runs are the primary ones:
    # it is neither folded nor run again.
    assert folds == [(first, 3.0), (second, 3.0)]
    # The unfolded run of both instances, then both instances folded, each on
    # its own share (6 + 5 = 11) and seed.
    assert [(shots, seed) for _, shots, seed in runs] == [(6, 5), (5, 6)] * 2
    # Instance 0 was flipped: its "0" counts come back as "1".
    assert captured == [([1.0, 3.0], [{"1": 6, "0": 5}, {"1": 6, "0": 5}])]
    assert metrics["zne_parity_expval"] == 1.0
    assert metrics["raw_parity_expval"] == pytest.approx((5 - 6) / 11)


def test_baseline_is_executed_when_execution_is_requested(tmp_path, monkeypatch):
    """With execute=True the baseline carries the same execution metrics.

    A compile-only baseline cannot be compared with executed candidates on
    execution-derived objective terms, which is what the regression guard
    and the diagnostics need.
    """
    from tests.system_stubs import _Circ

    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    ds = wl.CircuitDataset(tmp_path, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(
        wl,
        "_compile_cached",
        lambda *a, **k: ([_Circ()], {"depth": 1, "measurement_flip_map": {}}),
    )
    monkeypatch.setattr(ensemble, "run_counts", lambda *a, **k: {"00": 3, "11": 1})
    objective = Objective({"depth": 1.0, "raw_top_prob": -1.0})

    for execute in (True, False):
        balanced = (
            wl.Workload.from_dataset(ds)
            .set_target("b")
            .adjust(
                strategies=[StrategySpec(optimization_level=2)],
                objective=objective,
                execute=execute,
            )
        )
        baseline = balanced.baseline_metrics["c0"]
        if execute:
            assert baseline["raw_top_prob"] == 0.75
            assert baseline["objective_score"] == pytest.approx(1.0 - 0.75)
            diagnostics = balanced.selection_diagnostics()["c0"]
            assert "raw_top_prob" in diagnostics["comparable_objective_terms"]
        else:
            assert "raw_top_prob" not in baseline


def test_a_candidate_equal_to_the_baseline_is_not_executed_twice(tmp_path, monkeypatch):
    """Regression: the default pool's opt1+sabre candidate re-ran the baseline.

    Same spec, inputs and seeds give the same result, so the second set of
    executions (jobs, on hardware) bought nothing.
    """
    from tests.system_stubs import _Circ

    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    ds = wl.CircuitDataset(tmp_path, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(
        wl,
        "_compile_cached",
        lambda *a, **k: ([_Circ()], {"depth": 1, "measurement_flip_map": {}}),
    )
    runs: list = []

    def _run(*args, **kwargs):
        runs.append(args)
        return {"00": 3, "11": 1}

    monkeypatch.setattr(ensemble, "run_counts", _run)
    baseline_spec = StrategySpec(optimization_level=1, routing_method="sabre")

    balanced = (
        wl.Workload.from_dataset(ds)
        .set_target("b")
        .adjust(
            strategies=[baseline_spec, StrategySpec(optimization_level=2)],
            objective=Objective({"depth": 1.0}),
            execute=True,
        )
    )

    # The baseline once, the other candidate once.
    assert len(runs) == 2
    history = balanced.evaluation_history["c0"]
    assert history[0].spec == baseline_spec
    assert history[0].metrics == balanced.baseline_metrics["c0"]
    # An independent copy: nothing done to one can leak into the other.
    assert history[0].metrics is not balanced.baseline_metrics["c0"]


def test_selection_diagnostics_compare_only_shared_objective_terms(tmp_path):
    """A term only one side reports is neither an improvement nor a regression."""
    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    balanced = wl.BalancedWorkload(
        dataset=wl.CircuitDataset(tmp_path, [record]),
        backend_spec="b",
        selections={
            "c0": Strategy(
                spec=StrategySpec(mthree=True),
                metrics={"depth": 6.0, "mitigated_top_prob": 0.9},
            )
        },
        baseline_metrics={"c0": {"depth": 5.0}},
        objective=Objective({"depth": 1.0, "mitigated_top_prob": -10.0}),
    )

    diagnostics = balanced.selection_diagnostics()["c0"]

    # The full scores differ by -8, but only depth is comparable: +1.
    assert diagnostics["baseline_objective_score"] == 5.0
    assert diagnostics["selected_objective_score"] == pytest.approx(-3.0)
    assert diagnostics["comparable_objective_terms"] == ["depth"]
    assert diagnostics["objective_delta"] == 1.0
    assert diagnostics["objective_improved"] is False


def test_cutting_candidate_runs_its_subexperiments_end_to_end(tmp_path):
    """Regression: cutting candidates never evaluated.

    A measured circuit was rejected by the cut finder, and the cut circuit it
    returned for unmeasured ones -- QPD placeholders -- cannot be compiled.
    The candidate now compiles every subexperiment and, when executed,
    reconstructs the Z-parity of the measured qubits.
    """
    pytest.importorskip("qiskit_addon_cutting")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(6, 6, name="chain6")
    qc.ry(0.7, 0)
    for qubit in range(5):
        qc.cx(qubit, qubit + 1)
    qc.measure(range(6), range(6))
    exact = (
        Statevector(qc.remove_final_measurements(inplace=False))
        .expectation_value(SparsePauliOp("Z" * 6))
        .real
    )
    dataset = save_dataset(tmp_path / "ds", [qc])
    cut_spec = StrategySpec(optimization_level=1, cutting=True, max_subcircuit_qubits=4)

    balanced = (
        wl.Workload.from_dataset(dataset)
        .set_target("aer:simulator")
        .adjust(
            strategies=[cut_spec],
            execute=True,
            shots=20_000,
            cache_root=tmp_path / "cache",
        )
    )

    metrics = balanced.evaluation_history["chain6"][0].metrics
    assert "strategy_failed" not in metrics
    assert metrics["cut_count"] >= 1
    assert metrics["sampling_overhead"] == 9.0 ** metrics["cut_count"]
    assert max(metrics["subcircuit_widths"]) <= 4
    assert metrics["num_subexperiments"] > 0
    assert metrics["cut_parity_expval"] == pytest.approx(exact, abs=0.05)

    # The matrix runner evaluates cutting the same way.
    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["aer:simulator"],
        [cut_spec],
        tmp_path / "matrix.json",
        execute=True,
        shots=20_000,
    )
    row = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert row["metrics"]["cut_count"] == metrics["cut_count"]
    assert row["metrics"]["cut_parity_expval"] == pytest.approx(exact, abs=0.05)


@pytest.mark.parametrize(
    "case",
    ["idle_measured_qubit", "disconnected_parts", "idle_qubit_only_widens"],
)
def test_cutting_respects_the_width_limit_with_idle_or_separate_qubits(case):
    """Regression: two circuit shapes broke the width limit or crashed.

    qiskit-addon-cutting labels a qubit no gate touches ``None`` and then
    fails looking that partition up, so any cut circuit with an idle measured
    qubit raised ``KeyError: None``.  And a circuit wider than the limit whose
    connected parts each fit needed no cut, so it was reported as fitting and
    run whole, wider than ``max_subcircuit_qubits``.
    """
    pytest.importorskip("qiskit_addon_cutting")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator

    from qbalance.cutting.addon_cutting import (
        prepare_cutting_experiment,
        run_cutting_experiment,
    )
    from qbalance.transpile.pipeline import compile_ensemble

    if case == "idle_measured_qubit":
        qc = QuantumCircuit(5, 5)
        qc.ry(0.9, 0)
        for qubit in range(3):
            qc.cx(qubit, qubit + 1)
        expected_cuts = 1
    elif case == "disconnected_parts":
        qc = QuantumCircuit(4, 4)
        qc.ry(0.9, 0)
        qc.cx(0, 1)
        qc.ry(0.3, 2)
        qc.cx(2, 3)
        expected_cuts = 0
    else:
        qc = QuantumCircuit(3, 3)
        qc.ry(0.9, 0)
        qc.cx(0, 1)
        expected_cuts = 0
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    exact = (
        Statevector(qc.remove_final_measurements(inplace=False))
        .expectation_value(SparsePauliOp("Z" * qc.num_qubits))
        .real
    )

    experiment = prepare_cutting_experiment(qc, 2)

    assert experiment is not None
    assert experiment.metadata["cut_count"] == expected_cuts
    assert max(experiment.metadata["subcircuit_widths"]) <= 2
    assert experiment.measured_qubits == list(range(qc.num_qubits))
    if expected_cuts == 0:
        assert experiment.metadata["sampling_overhead"] == 1.0
    simulator = AerSimulator()
    compiled = {
        label: [compile_ensemble(sub, simulator, StrategySpec())[0][0] for sub in subs]
        for label, subs in experiment.subexperiments.items()
    }
    value = run_cutting_experiment(
        experiment, compiled, simulator, shots=40_000, seed=3
    )
    assert value == pytest.approx(exact, abs=0.03)


def test_cutting_leaves_narrow_or_gateless_circuits_uncut():
    pytest.importorskip("qiskit_addon_cutting")
    from qiskit import QuantumCircuit

    from qbalance.cutting.addon_cutting import prepare_cutting_experiment

    narrow = QuantumCircuit(2, 2)
    narrow.h(0)
    narrow.cx(0, 1)
    narrow.measure([0, 1], [0, 1])
    gateless = QuantumCircuit(4, 4)
    gateless.measure(range(4), range(4))

    assert prepare_cutting_experiment(narrow, 2) is None
    assert prepare_cutting_experiment(gateless, 2) is None


def test_matrix_rejects_a_directory_as_output_before_running(tmp_path, monkeypatch):
    """Regression: an --out directory failed only after every trial ran."""
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])

    def _unexpected(*a, **k):
        raise AssertionError("no trial may run before the output is checked")

    monkeypatch.setattr(matrix_mod, "resolve_backend", _unexpected)
    with pytest.raises(ValueError, match="is a directory"):
        matrix_mod.run_matrix(
            tmp_path / "ds", ["b"], [StrategySpec()], out_json=tmp_path
        )


def test_matrix_records_a_cutting_error_instead_of_crashing(tmp_path, monkeypatch):
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "prepare_cutting_experiment",
        lambda circuit, width: (_ for _ in ()).throw(ValueError("mid-circuit")),
    )

    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["b"],
        [StrategySpec(cutting=True, max_subcircuit_qubits=2)],
        tmp_path / "m.json",
    )

    row = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert row["metrics"] == {"cutting_error": "mid-circuit"}


def test_matrix_records_which_cut_failed_to_compile(tmp_path, monkeypatch):
    """A cut whose subexperiments cannot compile keeps the cut's description.

    ``adjust`` records the same keys for the same failure, so a matrix row and
    an audit-history entry describe it identically.
    """
    from qiskit import QuantumCircuit

    from qbalance.cutting.addon_cutting import CuttingExperiment
    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    experiment = CuttingExperiment(
        subexperiments={0: [_Circ()]},
        coefficients=[(1.0, None)],
        subobservables={0: None},
        measured_qubits=[0],
        metadata={"cut_count": 1, "sampling_overhead": 9.0},
    )
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod, "prepare_cutting_experiment", lambda circuit, width: experiment
    )

    def _fail(*a, **k):
        raise RuntimeError("subcircuit is wider than the device")

    monkeypatch.setattr(matrix_mod, "compile_ensemble", _fail)

    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["b"],
        [StrategySpec(cutting=True, max_subcircuit_qubits=2)],
        tmp_path / "m.json",
        execute=True,
    )

    row = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert row["metrics"] == {
        "compile_error": "subcircuit is wider than the device",
        "cut_count": 1,
        "sampling_overhead": 9.0,
    }


def test_matrix_runs_every_twirl_instance(tmp_path, monkeypatch):
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    first, second = object(), object()
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        lambda qc, backend, spec, profile: (
            [first, second],
            {"measurement_flip_maps": [{0: 1}, {}], "twirl_instances": 2},
        ),
    )
    runs: list = []

    def _run(backend, circuit, shots, seed_simulator):

        runs.append((circuit, shots, seed_simulator))
        return {"0": shots}

    monkeypatch.setattr(ensemble, "run_counts", _run)

    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["b"],
        [StrategySpec(measurement_twirling=True, num_twirls=2)],
        tmp_path / "m.json",
        execute=True,
        shots=9,
        seed=2,
    )

    assert runs == [(first, 5, 2), (second, 4, 3)]
    row = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert row["metrics"]["counts"] == {"1": 5, "0": 4}
    assert row["metrics"]["shots"] == 9


def test_matrix_applies_mthree_to_raw_instance_counts(tmp_path, monkeypatch):
    """A matrix trial must honour ``mthree=True`` the way ``adjust`` does.

    M3 corrects the physical readout, so it needs each instance's counts as
    measured together with that instance's flip map, not the pooled,
    already-untwirled counts.
    """
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    compiled = QuantumCircuit(3, 1)
    compiled.measure(2, 0)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: "bk")
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        lambda qc, backend, spec, profile: (
            [compiled, compiled],
            {"measurement_flip_maps": [{0: 1}, {}], "twirl_instances": 2},
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda backend, circuit, shots, seed_simulator: {"0": shots},
    )
    seen: dict = {}

    def _mitigate(backend, raw_counts, flip_maps, measured_qubits, seed, clbits):

        seen.update(
            backend=backend,
            raw_counts=raw_counts,
            flip_maps=flip_maps,
            measured_qubits=measured_qubits,
            seed=seed,
            clbits=clbits,
        )
        return {"0": 0.5, "1": 0.5}

    monkeypatch.setattr(ensemble, "mitigate_twirled_counts", _mitigate)
    strategy = StrategySpec(measurement_twirling=True, num_twirls=2, mthree=True)

    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["b"],
        [strategy],
        tmp_path / "m.json",
        execute=True,
        shots=9,
        seed=4,
    )

    metrics = json.loads(out.read_text(encoding="utf-8"))["results"][0]["metrics"]
    assert seen == {
        "backend": "bk",
        "raw_counts": [{"0": 5}, {"0": 4}],
        "flip_maps": [{0: 1}, {}],
        "measured_qubits": [2],
        "seed": 4,
        "clbits": [0],
    }
    assert metrics["mthree_probs"] == {"0": 0.5, "1": 0.5}
    assert metrics["counts"] == {"1": 5, "0": 4}

    def _unavailable(*args, **kwargs):

        raise RuntimeError("mthree is not installed")

    monkeypatch.setattr(ensemble, "mitigate_twirled_counts", _unavailable)
    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["b"],
        [strategy],
        tmp_path / "m.json",
        execute=True,
        shots=9,
    )

    metrics = json.loads(out.read_text(encoding="utf-8"))["results"][0]["metrics"]
    assert metrics["mthree_error"] == "mthree is not installed"
    assert "mthree_probs" not in metrics
    # A mitigation failure leaves the unmitigated result intact.
    assert metrics["counts"] == {"1": 5, "0": 4}
    assert "exec_error" not in metrics


def test_matrix_reports_a_zne_failure_as_zne_error(tmp_path, monkeypatch):
    """A failed extrapolation is not a failed execution, matching ``adjust``."""
    from qiskit import QuantumCircuit

    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.measure(0, 0)
    save_dataset(tmp_path / "ds", [qc])
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_ensemble",
        lambda qc, backend, spec, profile: ([qc], {"measurement_flip_map": {}}),
    )
    monkeypatch.setattr(
        ensemble,
        "run_counts",
        lambda backend, circuit, shots, seed_simulator: {"0": shots},
    )

    def _cannot_fold(*args, **kwargs):

        raise ValueError("cannot fold a mid-circuit measurement")

    monkeypatch.setattr(ensemble, "fold_global_for_backend", _cannot_fold)

    out = matrix_mod.run_matrix(
        tmp_path / "ds",
        ["b"],
        [StrategySpec(zne=True)],
        tmp_path / "m.json",
        execute=True,
        shots=8,
    )

    metrics = json.loads(out.read_text(encoding="utf-8"))["results"][0]["metrics"]
    assert metrics["zne_error"] == "cannot fold a mid-circuit measurement"
    assert "zne_probs" not in metrics
    assert "exec_error" not in metrics
    assert metrics["counts"] == {"0": 8}


def test_bandit_observes_scores_relative_to_the_circuit_baseline():
    """One surrogate serves every circuit, so it must not see raw scales.

    Raw scores scale with circuit size; pooled, a large circuit's scores
    would be credited to whichever strategies happened to run on it.
    """
    # The same 20% improvement on circuits of very different size.
    assert wl._bandit_target(8.0, 10.0) == pytest.approx(-0.2)
    assert wl._bandit_target(80.0, 100.0) == pytest.approx(-0.2)
    # Negative baselines (custom objectives) keep the direction of change.
    assert wl._bandit_target(-12.0, -10.0) == pytest.approx(-0.2)
    assert wl._bandit_target(3.0, 0.0) == 3.0
    for score, baseline in ((float("inf"), 1.0), (1.0, float("inf")), ("x", 1.0)):
        assert wl._bandit_target(score, baseline) is None


def _budget_workload(tmp_path, monkeypatch, sizes):
    """Three circuits whose depth falls with the optimization level."""
    from tests.system_stubs import _Circ

    records = [
        wl.CircuitRecord(name=f"c{i}", artifact=f"c{i}.qpy", format="qpy")
        for i in range(len(sizes))
    ]
    circuits = [_Circ() for _ in records]
    size_of = {id(circuit): size for circuit, size in zip(circuits, sizes)}
    dataset = wl.CircuitDataset(tmp_path, records)
    monkeypatch.setattr(dataset, "load_circuits", lambda: circuits)
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    calls: list = []

    def _compile(circuit, backend, spec, profile, cache_root, backend_key=None):

        calls.append(spec)
        size = size_of[id(circuit)]
        return [circuit], {"depth": size * (4 - spec.optimization_level) / 4 + 1}

    monkeypatch.setattr(wl, "_compile_cached", _compile)
    candidates = [
        StrategySpec(optimization_level=level, seed_transpiler=seed)
        for level in (0, 1, 2, 3)
        for seed in (0, 1, 2)
    ]
    return wl.Workload.from_dataset(dataset).set_target("b"), candidates, calls


def test_max_evaluations_caps_candidate_evaluations_per_circuit(tmp_path, monkeypatch):
    work, candidates, _ = _budget_workload(tmp_path, monkeypatch, [10, 20, 30])

    for search in ("grid", "bandit"):
        balanced = work.adjust(
            strategies=candidates,
            objective=Objective({"depth": 1.0}),
            search=search,
            warmup=2,
            max_evaluations=4,
        )
        assert all(len(h) == 4 for h in balanced.evaluation_history.values())
        if search == "grid":
            assert [s.spec for s in balanced.evaluation_history["c0"]] == candidates[:4]

    unbudgeted = work.adjust(strategies=candidates, search="bandit", warmup=2)
    assert all(len(h) == 12 for h in unbudgeted.evaluation_history.values())

    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_evaluations"):
            work.adjust(strategies=candidates, max_evaluations=bad)


def test_bandit_spends_a_budget_on_the_best_candidates(tmp_path, monkeypatch):
    """Regression: bandit search evaluated every candidate, so it never mattered.

    With a budget of 5 of 12 candidates, grid search only reaches the first
    five (optimization levels 0 and 1); the surrogate must steer the rest of
    the budget to optimization level 3 on every circuit, whatever the seed.
    """
    work, candidates, _ = _budget_workload(tmp_path, monkeypatch, [10, 20, 30])

    for seed in range(5):
        for search, expected in (("grid", 1), ("bandit", 3)):
            balanced = work.adjust(
                strategies=candidates,
                objective=Objective({"depth": 1.0}),
                search=search,
                warmup=2,
                max_evaluations=5,
                seed=seed,
            )
            levels = {s.spec.optimization_level for s in balanced.selections.values()}
            assert levels == {expected}, (search, seed, levels)


def test_pareto_selection_weighs_the_cut_sampling_overhead():
    """Regression: the Pareto axes ignored sampling_overhead.

    A cut candidate with shallower subexperiments then dominated the uncut
    candidate, removing it from the front, so pareto=True chose the cut even
    though its 9x shot cost makes it the worse strategy under the objective.
    """
    objective = default_objective()
    uncut = {
        "depth": 10,
        "two_qubit_ops": 4,
        "estimated_error": 0.05,
        "compile_time_s": 0.0,
        "sampling_overhead": 1.0,
    }
    cut = {
        "depth": 8,
        "two_qubit_ops": 3,
        "estimated_error": 0.04,
        "compile_time_s": 0.0,
        "sampling_overhead": 9.0,
    }
    for metrics in (uncut, cut):
        metrics["objective_score"] = objective.score(metrics)
    evals = [
        (StrategySpec(optimization_level=1), uncut),
        (StrategySpec(cutting=True, max_subcircuit_qubits=4), cut),
    ]
    assert cut["objective_score"] > uncut["objective_score"]

    for pareto in (False, True):
        chosen, _ = wl._choose(evals, pareto=pareto, objective=objective)
        assert chosen.cutting is False, pareto


def test_distribution_diagnostics_skip_missing_metrics_instead_of_zeroing(tmp_path):
    """A circuit missing a metric must not enter the distances as ``0.0``.

    Every selected circuit has the same depth as its baseline, so the two
    samples are identical; an invented zero for the circuit whose baseline
    lacks ``depth`` would make them differ.
    """
    dataset = wl.CircuitDataset(
        tmp_path,
        [wl.CircuitRecord(f"c{i}", f"c{i}.qpy", "qpy") for i in range(3)],
    )
    balanced = wl.BalancedWorkload(
        dataset=dataset,
        backend_spec="fake:generic:2",
        selections={
            "c0": Strategy(spec=StrategySpec(), metrics={"depth": 5}),
            "c1": Strategy(spec=StrategySpec(), metrics={"depth": 5}),
            "c2": Strategy(spec=StrategySpec(), metrics={"depth": 5}),
        },
        baseline_metrics={"c0": {"depth": 5}, "c1": {"depth": 5}, "c2": {}},
    )

    assert wl._finite_samples(
        [{"depth": 5}, {}, {"depth": float("nan")}, {"depth": "7"}], "depth"
    ) == [5.0, 7.0]
    assert balanced.covars()["depth"] == {"emd": 0.0, "cvm": 0.0, "ks": 0.0}
    assert "dist[depth]: EMD=0  CVM=0  KS=0" in balanced.summary()


def _bell_workload(tmp_path):
    from qiskit import QuantumCircuit

    from qbalance import save_dataset

    bell = QuantumCircuit(2, 2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])
    dataset = save_dataset(tmp_path / "ds_objective_terms", [bell])
    return wl.Workload.from_dataset(dataset).set_target("fake:generic:3")


def test_adjust_warns_about_objective_terms_no_metric_reports(tmp_path, caplog):
    """Regression: a misspelled objective term was dropped without a trace.

    ``Objective.score`` skips a term whose metric is absent, so
    ``two_qubit_op`` (for ``two_qubit_ops``) silently took no part in the
    selection -- and an objective made only of such terms scored every
    candidate 0 and selected the first one.
    """
    workload = _bell_workload(tmp_path)

    with caplog.at_level("WARNING", logger="qbalance.workflow.workload"):
        balanced = workload.adjust(
            objective=Objective({"depth": 1.0, "two_qubit_op": 2.0}),
            strategies=[StrategySpec(optimization_level=1)],
            cache_root=tmp_path / "cache",
        )

    assert "'two_qubit_op'" in caplog.text
    assert "'depth'" not in caplog.text
    assert balanced.selections["bell"].spec == StrategySpec(optimization_level=1)


def test_adjust_does_not_warn_when_every_objective_term_is_reported(tmp_path, caplog):
    workload = _bell_workload(tmp_path)

    with caplog.at_level("WARNING", logger="qbalance.workflow.workload"):
        workload.adjust(
            strategies=[StrategySpec(optimization_level=1)],
            cache_root=tmp_path / "cache",
        )

    assert "Objective term" not in caplog.text
