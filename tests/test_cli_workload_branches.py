# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import builtins
import runpy
import sys
import types

import numpy as np
import pytest

from qbalance import cli
from qbalance.mitigation import zne
from qbalance.objectives import default_objective
from qbalance.strategies import Strategy, StrategySpec
from qbalance.transpile import noise_aware_layout as nal
from qbalance.transpile import pipeline, suppression
from qbalance.workflow import workload as wl
from tests.system_stubs import _I, _PM, _Q, _Circ


def test_remaining_branch_coverage(monkeypatch, tmp_path):

    # zne parity-adjust branch (covers scaling + renormalization)
    probs = zne.zne_extrapolate_counts(
        [1.0, 3.0],
        [{"00": 4, "11": 2, "01": 2}, {"00": 5, "11": 1, "01": 2}],
        degree=1,
    )
    assert pytest.approx(sum(probs.values())) == 1.0

    # noise-aware fallback branches
    class BadBackend:
        def properties(self):

            raise RuntimeError("no props")

    bad = BadBackend()
    assert nal._safe_get_qubit_readout_error(bad, 0) is None
    assert nal._safe_get_t1(bad, 0) is None
    assert nal._safe_get_t2(bad, 0) is None
    assert nal._safe_get_2q_error(bad, "cx", 0, 1) is None
    assert nal.estimate_circuit_error(bad, object()) == 1.0

    assert (
        nal.noise_aware_initial_layout(types.SimpleNamespace(), types.SimpleNamespace())
        is None
    )

    small_backend = types.SimpleNamespace(
        properties=lambda: types.SimpleNamespace(qubits=[[]]),
        num_qubits=1,
        qubits=[0],
    )
    assert nal.noise_aware_initial_layout(small_backend, _Circ()) is None

    # suppression branches
    lib = types.ModuleType("qiskit.circuit.library")
    lib.XGate = type("XGate", (), {})
    lib.YGate = type("YGate", (), {})
    monkeypatch.setitem(sys.modules, "qiskit.circuit.library", lib)
    assert len(suppression._dd_sequence("YY")) == 2
    assert len(suppression._dd_sequence("other")) == 4

    class BNoTarget:
        target = None

        @staticmethod
        def configuration():

            return types.SimpleNamespace(basis_gates=["x", "cx"])

    transpiler = types.ModuleType("qiskit.transpiler")

    class PM3:
        def __init__(self):

            self.steps = []

        def append(self, x):

            self.steps.append(x)

        def run(self, out, callback=None):

            _ = callback
            return out

    transpiler.PassManager = PM3
    monkeypatch.setitem(sys.modules, "qiskit.transpiler", transpiler)
    eqlib = types.ModuleType("qiskit.circuit.equivalence_library")
    eqlib.SessionEquivalenceLibrary = object()
    monkeypatch.setitem(sys.modules, "qiskit.circuit.equivalence_library", eqlib)
    passes = types.ModuleType("qiskit.transpiler.passes")
    for name in [
        "ALAPScheduleAnalysis",
        "ApplyLayout",
        "BasisTranslator",
        "EnlargeWithAncilla",
        "FullAncillaAllocation",
        "PadDynamicalDecoupling",
        "SetLayout",
        "Unroll3qOrMore",
    ]:
        setattr(passes, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    monkeypatch.setitem(sys.modules, "qiskit.transpiler.passes", passes)
    assert suppression.build_dd_pass_manager(BNoTarget())

    qiskit = types.ModuleType("qiskit")
    qiskit.QuantumCircuit = object
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)
    orig_default_rng = np.random.default_rng
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda seed=None: types.SimpleNamespace(integers=lambda a, b: 1),
    )

    class _CircWithCargs(_Circ):
        def __init__(self):

            super().__init__()
            self.data = [(_I("measure"), [_Q(0)], [types.SimpleNamespace(index=0)])]

        def x(self, qb):

            _ = qb

        def copy(self):

            return _CircWithCargs()

    tw, fmap = suppression.apply_measurement_twirling(_CircWithCargs(), seed=1)
    assert isinstance(tw, _CircWithCargs)
    assert fmap
    counts = suppression.apply_measurement_untwirl_counts({"0": 3}, {})
    assert counts == {"0": 3}
    monkeypatch.setattr(np.random, "default_rng", orig_default_rng)

    # pipeline branches: noise-aware pass-manager override, DD and measurement failures, error estimate failure
    ppm = types.ModuleType("qiskit.transpiler.preset_passmanagers")
    ppm.generate_translation_passmanager = lambda **kwargs: _PM(_Circ())
    ppm.generate_unroll_3q = lambda **kwargs: _PM(_Circ())
    monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", ppm)
    conv = types.ModuleType("qiskit.converters")
    conv.circuit_to_dag = lambda c: c
    monkeypatch.setitem(sys.modules, "qiskit.converters", conv)
    monkeypatch.setattr(
        pipeline, "noise_aware_initial_layout", lambda backend, tw: {0: 0}
    )
    monkeypatch.setattr(
        pipeline,
        "build_dd_pass_manager",
        lambda backend, seq: (_ for _ in ()).throw(RuntimeError("dd")),
    )
    monkeypatch.setattr(
        pipeline,
        "apply_measurement_twirling",
        lambda out, seed: (_ for _ in ()).throw(RuntimeError("mt")),
    )
    monkeypatch.setattr(
        pipeline,
        "estimate_circuit_error",
        lambda backend, out: (_ for _ in ()).throw(RuntimeError("ee")),
    )
    _, met = pipeline.compile_one(
        _Circ(),
        types.SimpleNamespace(
            target=types.SimpleNamespace(operation_names=["x", "cx"])
        ),
        StrategySpec(
            layout_method="qbalance_noise_aware",
            dynamical_decoupling=True,
            measurement_twirling=True,
        ),
        profile=False,
    )
    assert met["estimated_error"] is None

    # cli branches: overwrite existing output (shutil.rmtree), qiskit import failure branch, and __main__ invocation
    rec = types.SimpleNamespace(name="c0", artifact="c0.qpy")
    ds = types.SimpleNamespace(records=[rec], load_circuits=lambda: [_Circ()])
    monkeypatch.setattr(cli, "load_dataset", lambda p: ds)
    original_from_path = cli.Workload.from_path
    monkeypatch.setattr(
        cli.Workload,
        "from_path",
        classmethod(lambda cls, p: types.SimpleNamespace(set_target=lambda b: None)),
    )
    backends = types.ModuleType("qbalance.backends")
    backends.resolve_backend = lambda b: object()
    monkeypatch.setitem(sys.modules, "qbalance.backends", backends)
    tp = types.ModuleType("qbalance.transpile.pipeline")
    tp.compile_one = lambda qc, backend, spec, profile=False: (qc, {"depth": 1})
    monkeypatch.setitem(sys.modules, "qbalance.transpile.pipeline", tp)

    out = tmp_path / "compiled_existing"
    out.mkdir()

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):

        if name == "qiskit":
            raise ImportError("forced")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
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

    monkeypatch.setattr(builtins, "__import__", orig_import)
    monkeypatch.setattr(cli.Workload, "from_path", original_from_path)
    import typer

    monkeypatch.setattr(
        typer.main.Typer,
        "__call__",
        lambda self, *a, **k: (_ for _ in ()).throw(SystemExit(0)),
    )
    sys.modules.pop("qbalance.cli", None)
    with pytest.raises(SystemExit):
        runpy.run_module("qbalance.cli", run_name="__main__")

    # workload branches: from_path, save overwrite removal, tmp cleanup in to_download,
    # bandit-propose completion, cutting failure continue, mthree/zne error capture
    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "wlds"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds_real = wl.CircuitDataset(dsroot, [record])
    monkeypatch.setattr(wl, "load_dataset", lambda p: ds_real)
    assert wl.Workload.from_path(dsroot).dataset.root == dsroot

    bw = wl.BalancedWorkload(
        dataset=ds_real,
        backend_spec="b",
        selections={
            "c0": Strategy(
                spec=StrategySpec(), metrics={"depth": 1, "two_qubit_ops": 1}
            )
        },
        baseline_metrics={"c0": {"depth": 1, "two_qubit_ops": 1}},
        objective=default_objective(),
    )
    out_dir = tmp_path / "save_dir"
    out_dir.mkdir()
    (out_dir / "old.txt").write_text("x", encoding="utf-8")
    bw.save(out_dir, overwrite=True)

    tmp_bundle_dir = tmp_path / "bundle_dir"
    tmp_bundle_dir.mkdir()
    zpath = bw.to_download(tmp_path / "bundle.zip", overwrite=True)
    assert zpath.exists()

    monkeypatch.setattr(ds_real, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [
            StrategySpec(cutting=True, max_subcircuit_qubits=1),
            StrategySpec(mthree=True, zne=True),
        ],
    )

    class Bnd:
        def observe(self, *a, **k):

            return None

        def propose(self, candidates, rng):

            _ = rng
            return candidates[0]

    monkeypatch.setattr(wl, "BanditSearcher", Bnd)
    monkeypatch.setattr(
        wl, "compile_one", lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})
    )
    monkeypatch.setattr(wl, "load_compiled", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(
        wl,
        "find_cuts_best_effort",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cut")),
    )
    monkeypatch.setattr(wl, "run_counts", lambda *a, **k: {"00": 4, "11": 4})
    monkeypatch.setattr(
        wl, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(
        wl,
        "apply_mthree_mitigation",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("m3")),
    )
    monkeypatch.setattr(
        wl, "fold_global_for_backend", lambda compiled, backend, f: compiled
    )
    monkeypatch.setattr(
        wl,
        "zne_extrapolate_counts",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("zne")),
    )

    with pytest.raises(RuntimeError, match="No feasible candidate"):
        (
            wl.Workload.from_dataset(ds_real)
            .set_target("fake:generic:2")
            .adjust(search="bandit", execute=True, max_candidates=2)
        )


def test_workload_adjust_raises_on_dataset_circuit_record_mismatch(
    monkeypatch, tmp_path
):
    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "wlds_mismatch"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [record])

    monkeypatch.setattr(ds, "load_circuits", lambda: [])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        wl.Workload.from_dataset(ds).set_target("fake:generic:2").adjust()


def test_overwrite_refuses_to_delete_a_non_directory(tmp_path):
    """Regression: --overwrite onto a file raised a raw errno traceback."""
    import pytest

    from qbalance.dataset import CircuitDataset
    from qbalance.workflow.workload import BalancedWorkload

    target = tmp_path / "notes.txt"
    target.write_text("user data", encoding="utf-8")

    workload = BalancedWorkload(
        dataset=CircuitDataset(tmp_path, []), backend_spec="b", selections={}
    )
    with pytest.raises(NotADirectoryError) as excinfo:
        workload.save(target, overwrite=True)
    # The guard must fire before rmtree, with an explanation rather than an errno.
    assert "is not a directory" in str(excinfo.value)
    assert "Cannot overwrite" in str(excinfo.value)
    assert target.read_text(encoding="utf-8") == "user data"

    with pytest.raises(FileExistsError):
        workload.save(target, overwrite=False)
    assert target.read_text(encoding="utf-8") == "user data"


def test_compile_cmd_overwrite_refuses_to_delete_a_non_directory(tmp_path):
    import pytest
    import typer

    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    dataset_dir = tmp_path / "ds"
    save_dataset(dataset_dir, _make_tiny()[:1], overwrite=True)
    target = tmp_path / "notes.txt"
    target.write_text("user data", encoding="utf-8")

    def run(out, overwrite):
        cli.compile_cmd(
            dataset_dir,
            backend="fake:generic:5",
            out=out,
            optimization_level=1,
            routing_method="sabre",
            layout_method=None,
            pauli_twirling=False,
            num_twirls=1,
            dynamical_decoupling=False,
            measurement_twirling=False,
            overwrite=overwrite,
        )

    with pytest.raises(typer.BadParameter) as excinfo:
        run(target, overwrite=True)
    assert "not a directory" in str(excinfo.value)
    assert target.read_text(encoding="utf-8") == "user data"

    # Directories still overwrite normally.
    out_dir = tmp_path / "compiled"
    run(out_dir, overwrite=False)
    run(out_dir, overwrite=True)
    assert (out_dir / "meta.json").exists()


def test_cli_help_keeps_the_extra_name_in_install_hints():
    """Rich markup must not eat the extra out of an install hint.

    Typer renders option help as Rich markup, where ``[aer]`` and ``[report]``
    parse as style tags and are deleted.  That silently turns "install
    qbalance[aer]" into "install qbalance" -- removing the one piece of
    information the hint exists to carry, and leaving an unbalanced paren.
    """
    from typer.testing import CliRunner

    from qbalance.cli import app

    runner = CliRunner()

    def rendered(*args):

        result = runner.invoke(app, list(args), env={"COLUMNS": "200"})
        assert result.exit_code == 0
        # Strip the box drawing and collapse the wrapping Rich applies.
        return " ".join(result.output.replace("│", " ").split())

    assert "qbalance[aer]" in rendered("adjust", "--help")
    assert "qbalance[report]" in rendered("report", "--help")
