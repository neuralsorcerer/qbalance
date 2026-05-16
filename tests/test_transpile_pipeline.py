# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types

import pytest

from qbalance.strategies import StrategySpec
from qbalance.transpile import noise_aware_layout as nal
from qbalance.transpile import pipeline, suppression
from tests.system_stubs import _PM, _Circ


def test_noise_layout_suppression_and_pipeline(monkeypatch):

    class Props:
        qubits = [
            [
                types.SimpleNamespace(name="readout_error", value=0.1),
                types.SimpleNamespace(name="T1", value=100),
                types.SimpleNamespace(name="T2", value=200),
            ]
            for _ in range(2)
        ]

        @staticmethod
        def gate_error(name, pair):

            _ = (name, pair)
            return 0.05

    backend = types.SimpleNamespace(
        properties=lambda: Props(),
        num_qubits=2,
        qubits=[0, 1],
        target=types.SimpleNamespace(operation_names=["x", "cx"]),
    )

    assert nal._safe_get_qubit_readout_error(backend, 0) == 0.1
    assert nal._safe_get_t1(backend, 0) == 100.0
    assert nal._safe_get_t2(backend, 0) == 200.0
    assert nal._safe_get_2q_error(backend, "cx", 0, 1) == 0.05
    assert nal.estimate_circuit_error(backend, _Circ()) > 0

    transpiler = types.ModuleType("qiskit.transpiler")

    class Layout(dict):
        pass

    transpiler.Layout = Layout
    monkeypatch.setitem(sys.modules, "qiskit.transpiler", transpiler)
    layout = nal.noise_aware_initial_layout(backend, _Circ())
    assert layout is not None

    circuit_mod = types.ModuleType("qiskit.circuit")
    circuit_mod.pauli_twirl_2q_gates = lambda circuit, seed, num_twirls, target: (
        [circuit] * num_twirls
    )
    monkeypatch.setitem(sys.modules, "qiskit.circuit", circuit_mod)
    assert len(suppression.apply_pauli_twirling(_Circ(), num_twirls=2)) == 2

    lib = types.ModuleType("qiskit.circuit.library")
    lib.XGate = type("XGate", (), {})
    lib.YGate = type("YGate", (), {})
    monkeypatch.setitem(sys.modules, "qiskit.circuit.library", lib)
    assert len(suppression._dd_sequence("XY4")) == 4

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

    class PM2:
        def __init__(self):

            self.steps = []

        def append(self, x):

            self.steps.append(x)

        def run(self, out):

            return out

    transpiler.PassManager = PM2
    assert suppression.build_dd_pass_manager(backend)

    qiskit = types.ModuleType("qiskit")
    qiskit.QuantumCircuit = object
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)
    tw, flip_map = suppression.apply_measurement_twirling(_Circ(), seed=0)
    assert isinstance(tw, _Circ)
    assert isinstance(flip_map, dict)
    assert suppression.apply_measurement_untwirl_counts({"01": 1}, {0: 1})

    ppm = types.ModuleType("qiskit.transpiler.preset_passmanagers")
    ppm.generate_preset_pass_manager = lambda **kwargs: _PM(_Circ())
    monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", ppm)
    conv = types.ModuleType("qiskit.converters")
    conv.circuit_to_dag = lambda c: c
    monkeypatch.setitem(sys.modules, "qiskit.converters", conv)
    monkeypatch.setattr(
        pipeline,
        "apply_pauli_twirling",
        lambda circuit, num_twirls, seed, target: [circuit],
    )
    monkeypatch.setattr(pipeline, "estimate_circuit_error", lambda backend, out: 0.123)
    monkeypatch.setattr(
        pipeline, "build_dd_pass_manager", lambda backend, seq: _PM(_Circ())
    )
    monkeypatch.setattr(
        pipeline, "apply_measurement_twirling", lambda out, seed: (out, {0: 1})
    )
    out, met = pipeline.compile_one(
        _Circ(),
        backend,
        StrategySpec(
            pauli_twirling=True, dynamical_decoupling=True, measurement_twirling=True
        ),
        profile=True,
    )
    assert out is not None
    assert met["estimated_error"] == 0.123


def test_noise_aware_helpers_support_qiskit_bits_without_public_index():
    from qiskit import QuantumCircuit

    from qbalance.transpile import noise_aware_layout as nal

    class Backend:
        num_qubits = 2

        @staticmethod
        def properties():
            return None

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    error = nal.estimate_circuit_error(Backend(), qc)
    assert 0.0 < error < 1.0
    assert nal.noise_aware_initial_layout(Backend(), qc) is not None


def test_noise_aware_helpers_ignore_nonfinite_calibration_values():
    class Props:
        qubits = [
            [
                types.SimpleNamespace(name="readout_error", value=float("nan")),
                types.SimpleNamespace(name="T1", value=float("inf")),
                types.SimpleNamespace(name="T2", value="bad"),
            ]
        ]

        @staticmethod
        def gate_error(name, pair):
            _ = (name, pair)
            return 5.0

    backend = types.SimpleNamespace(properties=lambda: Props())

    assert nal._safe_get_qubit_readout_error(backend, 0) is None
    assert nal._safe_get_t1(backend, 0) is None
    assert nal._safe_get_t2(backend, 0) is None
    assert nal._safe_get_2q_error(backend, "cx", 0, 1) == 1.0


def test_measurement_twirling_inserts_flip_before_measurement(monkeypatch):
    from qiskit import QuantumCircuit

    monkeypatch.setattr(
        suppression.np.random,
        "default_rng",
        lambda seed=None: types.SimpleNamespace(integers=lambda a, b: 1),
    )

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    twirled, flip_map = suppression.apply_measurement_twirling(qc, seed=123)

    assert flip_map == {0: 1}
    assert [inst.operation.name for inst in twirled.data] == ["h", "x", "measure"]


def test_measurement_twirling_skips_nonterminal_measurements(monkeypatch):
    from qiskit import QuantumCircuit

    monkeypatch.setattr(
        suppression.np.random,
        "default_rng",
        lambda seed=None: types.SimpleNamespace(integers=lambda a, b: 1),
    )

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)

    twirled, flip_map = suppression.apply_measurement_twirling(qc, seed=123)

    assert flip_map == {}
    assert [inst.operation.name for inst in twirled.data] == ["h", "measure", "x"]


def test_compile_one_dd_with_backendv2_target(caplog):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=2)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    _, metrics = pipeline.compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(dynamical_decoupling=True, dd_sequence="XY4"),
        profile=False,
    )

    assert metrics["dd_applied"] is True
    assert "DD insertion failed" not in caplog.text


def test_dd_sequence_compatibility_helpers():
    class XGate:
        pass

    class YGate:
        pass

    requested = [XGate(), YGate(), XGate(), YGate()]

    assert suppression._operation_names(
        types.SimpleNamespace(operation_names=["X", " cx "])
    ) == {
        "x",
        "cx",
    }
    assert suppression._operation_names(
        types.SimpleNamespace(operation_names=lambda: ["measure", "Delay"])
    ) == {"measure", "delay"}

    backend = types.SimpleNamespace(
        configuration=lambda: types.SimpleNamespace(basis_gates=["X", "SX"])
    )
    assert suppression._backend_basis_gates(backend) == {"x", "sx"}
    assert suppression._backend_basis_gates(object()) == set()

    compatible = suppression._compatible_dd_sequence(requested, {"x", "sx"})
    assert [suppression._gate_name(gate) for gate in compatible] == ["x", "x"]

    unchanged = suppression._compatible_dd_sequence(requested, {"x", "y"})
    assert unchanged is requested


def test_build_dd_pass_manager_without_basis_skips_translator(monkeypatch):
    class PassRecorder:
        def __init__(self):
            self.steps = []

        def append(self, item):
            self.steps.append(item)

    class NamedPass:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(
        suppression, "_dd_sequence", lambda name: [types.SimpleNamespace(name="x")]
    )

    eqlib = types.ModuleType("qiskit.circuit.equivalence_library")
    eqlib.SessionEquivalenceLibrary = object()
    monkeypatch.setitem(sys.modules, "qiskit.circuit.equivalence_library", eqlib)

    transpiler = types.ModuleType("qiskit.transpiler")
    transpiler.PassManager = PassRecorder
    monkeypatch.setitem(sys.modules, "qiskit.transpiler", transpiler)

    passes = types.ModuleType("qiskit.transpiler.passes")
    passes.Unroll3qOrMore = NamedPass
    passes.BasisTranslator = NamedPass
    passes.ALAPScheduleAnalysis = NamedPass
    passes.PadDynamicalDecoupling = NamedPass
    monkeypatch.setitem(sys.modules, "qiskit.transpiler.passes", passes)

    pm = suppression.build_dd_pass_manager(object())

    assert len(pm.steps) == 3
    assert all(not step.args for step in pm.steps[1:])
    assert pm.steps[1].kwargs == {"durations": None}
    assert pm.steps[2].kwargs == {
        "durations": None,
        "dd_sequence": [types.SimpleNamespace(name="x")],
    }
