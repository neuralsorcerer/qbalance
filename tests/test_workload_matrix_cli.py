# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sys
import types
from typing import Any, cast

import numpy as np
import pytest

from qbalance import cli
from qbalance.benchmarking import matrix as matrix_mod
from qbalance.cutting import addon_cutting
from qbalance.mitigation import zne
from qbalance.objectives import Objective, default_objective
from qbalance.reports import common as report_common
from qbalance.strategies import Strategy, StrategySpec
from qbalance.transpile import suppression
from qbalance.workflow import workload as wl
from tests.system_stubs import _Circ


def test_cutting_and_workload_and_matrix_and_cli(monkeypatch, tmp_path):

    cutting_mod = types.ModuleType("qiskit_addon_cutting.cutting")
    cutting_mod.DeviceConstraints = lambda max_subcircuit_width: types.SimpleNamespace(
        max_subcircuit_width=max_subcircuit_width
    )
    cutting_mod.OptimizationParameters = lambda max_backjumps, max_gamma: (
        types.SimpleNamespace(max_backjumps=max_backjumps, max_gamma=max_gamma)
    )
    cutting_mod.find_cuts = lambda circuit, optimization, constraints: (
        circuit,
        {"w": constraints.max_subcircuit_width},
    )
    monkeypatch.setitem(sys.modules, "qiskit_addon_cutting.cutting", cutting_mod)

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
        "compile_one",
        lambda circuit, backend, spec, profile: (
            qc,
            {
                "depth": spec.optimization_level + 1,
                "two_qubit_ops": 1,
                "estimated_error": 0.1,
                "measurement_flip_map": {},
            },
        ),
    )
    monkeypatch.setattr(
        wl,
        "run_counts",
        lambda backend, compiled, shots, seed_simulator: {"00": 5, "11": 5},
    )
    monkeypatch.setattr(
        wl, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(
        wl,
        "apply_mthree_mitigation",
        lambda backend, counts, measured_qubits, shots: {"00": 1.0},
    )
    monkeypatch.setattr(wl, "fold_global", lambda compiled, f: compiled)
    monkeypatch.setattr(
        wl, "zne_extrapolate_counts", lambda factors, counts_pf, degree: {"00": 1.0}
    )
    monkeypatch.setattr(
        wl,
        "find_cuts_best_effort",
        lambda working, max_subcircuit_qubits: (working, {"x": 1}),
    )
    monkeypatch.setattr(wl, "load_compiled", lambda entry: None)
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

    monkeypatch.setattr(wl, "load_compiled", lambda entry: (qc, {"depth": 1}))
    hit_c, hit_m = wl._compile_cached(qc, object(), StrategySpec(), False, tmp_path)
    assert hit_c is qc
    assert hit_m["depth"] == 1

    monkeypatch.setattr(wl, "load_compiled", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    miss_c, miss_m = wl._compile_cached(qc, object(), StrategySpec(), False, tmp_path)
    assert miss_c is qc
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
        "compile_one",
        lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}}),
    )
    monkeypatch.setattr(
        matrix_mod,
        "run_counts",
        lambda backend, compiled, shots, seed_simulator: {"0": 1},
    )
    monkeypatch.setattr(
        matrix_mod, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(matrix_mod, "fold_global", lambda c, f: c)
    monkeypatch.setattr(
        matrix_mod,
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
    with pytest.raises(Exception):
        cli.dataset_cmd("bad", tmp_path / "a", overwrite=True)


def test_additional_branch_coverage(monkeypatch, tmp_path):
    # resolver lazy-load branch

    from qbalance.backends import resolver as resolver_mod

    resolver_mod._PLUGINS = None
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
        "compile_one",
        lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}}),
    )
    monkeypatch.setattr(
        matrix_mod,
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
    with pytest.raises(Exception):
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
    tp.compile_one = lambda qc, backend, spec, profile=False: (qc, {"depth": 1})
    monkeypatch.setitem(sys.modules, "qbalance.transpile.pipeline", tp)

    qiskit = types.ModuleType("qiskit")
    qpy = types.SimpleNamespace(dump=lambda c, f: f.write(b"x"))
    qiskit.qpy = qpy
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)

    cli.adjust_cmd(tmp_path, "b", tmp_path / "o")
    cli.matrix_cmd(tmp_path, ["b"], tmp_path / "m.json")
    cli.report_cmd(tmp_path / "m.json", tmp_path, html=True)
    cli.plugins_cmd("list")
    with pytest.raises(Exception):
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
    with pytest.raises(Exception):
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
        lambda max_candidates, seed: [
            StrategySpec(mthree=True, zne=True, cutting=True, max_subcircuit_qubits=1)
        ],
    )
    monkeypatch.setattr(
        wl,
        "find_cuts_best_effort",
        lambda working, max_subcircuit_qubits: (working, {"cuts": 1}),
    )
    monkeypatch.setattr(
        wl,
        "compile_one",
        lambda *a, **k: (
            _Circ(),
            {
                "measurement_flip_map": {0: 1},
                "depth": 1,
                "two_qubit_ops": 1,
                "estimated_error": 0.1,
            },
        ),
    )
    monkeypatch.setattr(wl, "load_compiled", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(
        wl,
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

    class CountingSpec:
        def __init__(self):

            self.calls = 0

        def model_dump(self):

            self.calls += 1
            return {"id": 1}

    spec = CountingSpec()
    ds_ok = types.SimpleNamespace(
        records=[types.SimpleNamespace(name="c0"), types.SimpleNamespace(name="c1")],
        load_circuits=lambda: [_Circ(), _Circ()],
    )
    monkeypatch.setattr(matrix_mod, "load_dataset", lambda p: ds_ok)
    monkeypatch.setattr(matrix_mod, "resolve_backend", lambda b: object())
    monkeypatch.setattr(
        matrix_mod,
        "compile_one",
        lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}}),
    )

    out = matrix_mod.run_matrix(tmp_path, ["b0", "b1"], [spec], tmp_path / "ok.json")
    assert out.exists()
    assert spec.calls == 1

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 4
    assert all(entry["strategy"] == {"id": 1} for entry in payload["results"])


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

    with pytest.raises(ValueError, match="seed must be an integer"):
        matrix_mod.run_matrix(
            tmp_path, ["b"], [StrategySpec()], tmp_path / "x.json", seed=1.5
        )

    with pytest.raises(ValueError, match="seed must be an integer"):
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
    monkeypatch.setattr(wl, "compile_one", fake_compile)

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
        "compile_one",
        lambda qc, backend, spec, profile: (qc, {"measurement_flip_map": {}}),
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

    def fake_compile_cached(circuit, backend, spec, profile, cache_root):
        seen_cache_roots.append(cache_root)
        return circuit, {
            "depth": 1,
            "two_qubit_ops": 0,
            "estimated_error": 0.0,
            "compile_time_s": 0.0,
        }

    monkeypatch.setattr(wl, "_compile_cached", fake_compile_cached)
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

    def fake_compile_cached(circuit, backend, spec, profile, cache_root):
        if spec.optimization_level == 0:
            return circuit, {"depth": float("inf")}
        return circuit, {"depth": 1, "two_qubit_ops": 0, "estimated_error": 0.0}

    monkeypatch.setattr(wl, "_compile_cached", fake_compile_cached)
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
