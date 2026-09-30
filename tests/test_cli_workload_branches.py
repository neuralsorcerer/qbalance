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
from qbalance.execution import ensemble
from qbalance.mitigation import zne
from qbalance.objectives import default_objective
from qbalance.strategies import Strategy, StrategySpec
from qbalance.transpile import noise_aware_layout as nal
from qbalance.transpile import pipeline, suppression
from qbalance.workflow import workload as wl
from tests.system_stubs import _I, _PM, _Q, _Circ, as_ensemble


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
    # Without copy_empty_like an X can only be appended after the measurement,
    # where it cannot flip the recorded bit, so no flip may be recorded.
    assert fmap == {}
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
    tp.compile_ensemble = lambda qc, backend, spec, profile=False: ([qc], {"depth": 1})
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
        def __init__(self, **kwargs):

            _ = kwargs

        def observe(self, *a, **k):

            return None

        def propose(self, candidates, rng):

            _ = rng
            return candidates[0]

    monkeypatch.setattr(wl, "BanditSearcher", Bnd)
    monkeypatch.setattr(
        wl,
        "compile_ensemble",
        as_ensemble(lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})),
    )
    monkeypatch.setattr(wl, "load_compiled_ensemble", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(
        wl,
        "prepare_cutting_experiment",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cut")),
    )
    monkeypatch.setattr(ensemble, "run_counts", lambda *a, **k: {"00": 4, "11": 4})
    monkeypatch.setattr(
        ensemble, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(
        ensemble,
        "mitigate_twirled_counts",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("m3")),
    )
    monkeypatch.setattr(
        ensemble, "fold_global_for_backend", lambda compiled, backend, f: compiled
    )
    monkeypatch.setattr(
        ensemble,
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


def test_cli_echoes_a_path_containing_brackets_verbatim(tmp_path):
    """A path the CLI reports back must be the path it actually wrote.

    Confirmation lines interpolate user-supplied paths into a Rich markup
    string, where a bracketed segment starting with a letter parses as a style
    tag and disappears -- so ``run[cache]/ds`` was reported as ``run/ds``,
    handing the user a path that does not exist.
    """
    from typer.testing import CliRunner

    from qbalance.cli import app

    out = tmp_path / "run[cache]" / "ds"
    result = CliRunner().invoke(
        app, ["dataset", "examples", "-o", str(out)], env={"COLUMNS": "220"}
    )

    assert result.exit_code == 0, result.output
    assert out.is_dir()
    assert "run[cache]" in " ".join(result.output.split())


def test_compile_overwrite_refuses_to_delete_the_source_dataset(tmp_path):
    """Regression: ``compile DATA --out DATA --overwrite`` deleted the dataset.

    The circuits were already loaded, so the compile then succeeded -- after
    removing the directory it had been compiled from.
    """
    from qiskit import QuantumCircuit
    from typer.testing import CliRunner

    from qbalance.cli import app
    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(1, 1, name="c0")
    qc.h(0)
    qc.measure(0, 0)
    dataset_dir = tmp_path / "project" / "ds"
    save_dataset(dataset_dir, [qc])

    for out in (dataset_dir, dataset_dir.parent):
        result = CliRunner().invoke(
            app,
            [
                "compile",
                str(dataset_dir),
                "-b",
                "fake:generic:2",
                "-o",
                str(out),
                "--overwrite",
            ],
            env={"COLUMNS": "220"},
        )
        assert result.exit_code != 0
        assert (dataset_dir / "qbalance_dataset.json").is_file()


def test_compile_keeps_one_output_per_record_when_stems_collide(tmp_path):
    """Records "bell.qpy" and "bell.qasm" share a stem but are two circuits."""
    import json

    from qiskit import QuantumCircuit, qpy
    from typer.testing import CliRunner

    from qbalance.cli import app

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    first = QuantumCircuit(1, 1)
    first.x(0)
    first.measure(0, 0)
    with (dataset_dir / "bell.qpy").open("wb") as handle:
        qpy.dump(first, handle)
    (dataset_dir / "bell.qasm").write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
        "measure q[0] -> c[0];\n",
        encoding="utf-8",
    )
    (dataset_dir / "qbalance_dataset.json").write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {"name": "a", "artifact": "bell.qpy", "format": "qpy"},
                    {"name": "b", "artifact": "bell.qasm", "format": "qasm"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "compiled_out"
    result = CliRunner().invoke(
        app,
        ["compile", str(dataset_dir), "-b", "fake:generic:2", "-o", str(out)],
        env={"COLUMNS": "220"},
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (out / "compiled").iterdir()) == [
        "bell.qpy",
        "bell_1.qpy",
    ]
    # meta.json says which file holds which record: the names alone cannot.
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["artifacts"] == {"a": "compiled/bell.qpy", "b": "compiled/bell_1.qpy"}
    with (out / meta["artifacts"]["b"]).open("rb") as handle:
        assert qpy.load(handle)[0].count_ops().get("x", 0) == 0


def test_a_failed_compile_keeps_the_previous_output(tmp_path, monkeypatch):
    """Regression: compile --overwrite deleted --out before compiling.

    A circuit that then failed to compile left a half-written directory where
    the previous, complete output had been.
    """
    from qiskit import QuantumCircuit
    from typer.testing import CliRunner

    from qbalance.cli import app
    from qbalance.dataset import save_dataset

    circuit = QuantumCircuit(1, 1, name="c0")
    circuit.x(0)
    circuit.measure(0, 0)
    dataset_dir = tmp_path / "ds"
    save_dataset(dataset_dir, [circuit])
    out = tmp_path / "compiled_out"
    args = ["compile", str(dataset_dir), "-b", "fake:generic:2", "-o", str(out)]

    first = CliRunner().invoke(app, args, env={"COLUMNS": "220"})
    assert first.exit_code == 0, first.output
    before = {
        str(path.relative_to(out)): path.read_bytes()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }

    def _cannot_compile(*a, **k):
        raise RuntimeError("cannot compile")

    monkeypatch.setattr(pipeline, "compile_ensemble", _cannot_compile)
    second = CliRunner().invoke(app, [*args, "--overwrite"], env={"COLUMNS": "220"})

    assert second.exit_code != 0
    assert {
        str(path.relative_to(out)): path.read_bytes()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    } == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["compiled_out", "ds"]


def test_compile_checks_the_output_before_loading_the_dataset(tmp_path):
    """--out is validated before any work, like adjust and dataset."""
    from typer.testing import CliRunner

    from qbalance.cli import app

    taken = tmp_path / "taken"
    taken.write_text("not a directory", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "compile",
            str(tmp_path / "missing"),
            "-b",
            "fake:generic:2",
            "-o",
            str(taken),
        ],
        env={"COLUMNS": "220"},
    )

    # The output problem is reported, not the missing dataset.
    assert result.exit_code == 2
    assert "exists" in result.output
    assert taken.read_text(encoding="utf-8") == "not a directory"


def test_compile_writes_every_twirl_instance(tmp_path):
    """The compiled QPY holds the whole ensemble, in flip-map order."""
    import json

    from qiskit import QuantumCircuit, qpy
    from typer.testing import CliRunner

    from qbalance.cli import app
    from qbalance.dataset import save_dataset

    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    save_dataset(tmp_path / "ds", [qc])

    out = tmp_path / "compiled"
    result = CliRunner().invoke(
        app,
        [
            "compile",
            str(tmp_path / "ds"),
            "-b",
            "fake:generic:3",
            "-o",
            str(out),
            "--pauli-twirling",
            "--num-twirls",
            "3",
            "--meas-twirl",
        ],
        env={"COLUMNS": "220"},
    )

    assert result.exit_code == 0, result.output
    with (out / "compiled" / "bell.qpy").open("rb") as handle:
        instances = qpy.load(handle)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert len(instances) == 3
    assert meta["circuits"]["bell"]["twirl_instances"] == 3
    assert len(meta["circuits"]["bell"]["measurement_flip_maps"]) == 3


def test_cli_reports_user_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    """Regression: every bad input printed a full traceback.

    A missing dataset, an existing output directory, an unknown backend or an
    invalid option are expected input errors; burying their one-line message
    in a traceback made them read like crashes.  Genuine bugs keep theirs.
    """
    import sys

    from qbalance import cli

    def invoke(*argv):
        monkeypatch.setattr(sys, "argv", ["qbalance", *argv])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        captured = capsys.readouterr()
        return exit_info.value.code, types.SimpleNamespace(
            err=_plain(captured.err), out=_plain(captured.out)
        )

    code, captured = invoke(
        "adjust",
        str(tmp_path / "nowhere"),
        "-b",
        "fake:generic:5",
        "-o",
        str(tmp_path / "o"),
    )
    assert code == 1
    assert "Error:" in captured.err and "nowhere" in captured.err
    assert "Traceback" not in captured.err + captured.out

    # The file name arrives whole: the error line is never hard-wrapped.
    # (Windows quotes the path with doubled backslashes, so match the name.)
    assert "qbalance_dataset.json" in captured.err

    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    save_dataset(tmp_path / "ds", _make_tiny()[:1])
    code, captured = invoke(
        "adjust",
        str(tmp_path / "ds"),
        "-b",
        "nosuch:backend",
        "-o",
        str(tmp_path / "o"),
    )
    assert code == 1 and "Unknown backend kind 'nosuch'" in captured.err

    code, captured = invoke("plugins", "list")
    assert code == 0

    def boom(*args, **kwargs):
        raise KeyError("internal bug")

    monkeypatch.setattr(cli.Workload, "from_path", boom)
    monkeypatch.setattr(
        sys, "argv", ["qbalance", "adjust", str(tmp_path), "-b", "b", "-o", "o"]
    )
    with pytest.raises(KeyError, match="internal bug"):
        cli.main()


def _plain(output: str) -> str:
    """Rich output as plain words: no colors, panel borders or line wrapping.

    CI forces colored output, and the usage-error panel wraps its message,
    so a phrase can span ANSI codes, borders and a line break.
    """
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    text = re.sub(r"[\u2500-\u257f]", " ", text)
    return " ".join(text.split())


def test_compile_reports_an_uncompilable_circuit_without_a_traceback(
    tmp_path, monkeypatch, capsys
):
    """Regression: Qiskit's TranspilerError escaped compile as a traceback.

    A circuit wider than the device, or an unknown layout or routing method,
    is bad input -- adjust and matrix record it per candidate -- yet compile
    crashed with a full traceback.  It is now one line naming the circuit.
    """
    import sys

    from qbalance import cli
    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    dataset_dir = tmp_path / "ds"
    save_dataset(dataset_dir, _make_tiny())  # qft4 needs 4 qubits

    def invoke(*extra):
        argv = ["qbalance", "compile", str(dataset_dir), "-o", str(tmp_path / "o")]
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        captured = capsys.readouterr()
        return exit_info.value.code, _plain(captured.err + captured.out)

    code, output = invoke("-b", "fake:generic:2")
    assert code == 1
    assert "Traceback" not in output
    assert (
        "Error: Circuit 'ghz3' could not be compiled for fake:generic:2: "
        "Number of qubits greater than device." in output
    )

    code, output = invoke("-b", "fake:generic:5", "--routing-method", "bogus")
    assert code == 1
    assert "Traceback" not in output
    assert "Invalid plugin name bogus for stage routing" in output
    assert not (tmp_path / "o").exists()


def test_cli_checks_the_output_directory_before_running(tmp_path, monkeypatch):
    """Regression: adjust found an unusable --out only when saving.

    The whole search ran first, then save() refused the existing directory
    (or the one holding the dataset) and the results were lost.
    """
    from typer.testing import CliRunner

    from qbalance import cli
    from qbalance.builtin_data import _make_tiny
    from qbalance.dataset import save_dataset

    save_dataset(tmp_path / "ds", _make_tiny()[:1])
    (tmp_path / "taken").mkdir()
    ran: list = []
    monkeypatch.setattr(
        cli.Workload, "from_path", classmethod(lambda cls, d: ran.append(d))
    )

    for out, extra, message in (
        (tmp_path / "taken", [], "use --overwrite"),
        (tmp_path / "ds", ["--overwrite"], "contains the source dataset"),
        (tmp_path, ["--overwrite"], "contains the source dataset"),
    ):
        result = CliRunner().invoke(
            cli.app,
            [
                "adjust",
                str(tmp_path / "ds"),
                "-b",
                "fake:generic:5",
                "-o",
                str(out),
                *extra,
            ],
            env={"COLUMNS": "220"},
        )
        assert result.exit_code == 2, result.output
        assert message in _plain(result.output)

    result = CliRunner().invoke(
        cli.app,
        ["dataset", "examples", "-o", str(tmp_path / "taken")],
        env={"COLUMNS": "220"},
    )
    assert result.exit_code == 2 and "use --overwrite" in _plain(result.output)
    assert ran == []  # nothing was computed


def test_cli_help_describes_every_command_and_option():
    """Regression: --help printed boilerplate docstrings.

    The command help was the generated docstring -- "Adjust cmd used by the
    qbalance workflow", an "Args:" list of ``typer.Option(...)`` reprs and
    "Returns: Computed value produced by this routine" -- and several options
    had no help at all.
    """
    import typer.main
    from typer.testing import CliRunner

    from qbalance.cli import app

    runner = CliRunner()
    # Recent Typer releases no longer build on click's classes, so the
    # command tree is inspected by attribute rather than by type.
    commands = typer.main.get_command(app).commands
    assert set(commands) == {
        "dataset",
        "adjust",
        "matrix",
        "report",
        "plugins",
        "compile",
    }
    for name, command in commands.items():
        output = runner.invoke(app, [name, "--help"], env={"COLUMNS": "200"}).output
        for boilerplate in ("used by the qbalance workflow", "typer.Option", "Args:"):
            assert boilerplate not in output, (name, boilerplate)
        for param in command.params:
            if getattr(param, "param_type_name", "") == "option":
                assert param.help, (name, param.name)
