# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types

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
    circuit_mod.pauli_twirl_2q_gates = (
        lambda circuit, seed, num_twirls, target: [circuit] * num_twirls
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
