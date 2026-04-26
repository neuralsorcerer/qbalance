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
    cutting_mod.OptimizationParameters = (
        lambda max_backjumps, max_gamma: types.SimpleNamespace(
            max_backjumps=max_backjumps, max_gamma=max_gamma
        )
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
    assert "qbalance summary" in balanced.summary()
    assert "depth" in balanced.covars()

    out_dir = tmp_path / "out"
    balanced.save(out_dir)
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
    res = work2.adjust(search="bandit", execute=True, pareto=False, max_candidates=1)
    assert "exec_error" in next(iter(res.selections.values())).metrics
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
