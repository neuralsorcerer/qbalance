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

        def run(self, out, callback=None):

            if callback is not None:
                callback(pass_=object(), time=0.1, count=1)
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
    ppm.generate_translation_passmanager = lambda **kwargs: _PM(_Circ())
    ppm.generate_unroll_3q = lambda **kwargs: _PM(_Circ())
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
    # profile=True has to install the callback on the pass-manager run; the
    # key is present either way, so an empty report is what exposes a run
    # that silently dropped it.
    assert met["pass_profile"]["passes"]
    assert met["pass_profile"]["total_time_s"] > 0.0


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

    # The same compile without the flag must report the metric as False;
    # otherwise "dd_applied" says nothing about whether DD actually ran.
    _, plain_metrics = pipeline.compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(dynamical_decoupling=False),
        profile=False,
    )
    assert plain_metrics["dd_applied"] is False


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


def test_measurement_untwirl_counts_preserves_register_separators_and_accepts_json_keys():
    counts = {"01 0": 2, "00 1": 3}

    out = suppression.apply_measurement_untwirl_counts(counts, {"1": 1})

    assert out == {"00 0": 2, "01 1": 3}


def test_measurement_flip_map_normalization_ignores_invalid_and_even_flips():
    assert suppression.normalize_measurement_flip_map(
        {"0": "1", 1: 2, "bad": 1, -1: 1, 2: True, 3: False}
    ) == {0: 1, 2: 1}
    assert suppression.normalize_measurement_flip_map(None) == {}
    assert suppression.apply_measurement_untwirl_counts({"000": 1}, {0: 2}) == {
        "000": 1
    }
    assert suppression.apply_measurement_untwirl_counts(
        {"00": 2, "01": 3}, {0: 1, 2: 1}
    ) == {"01": 2, "00": 3}


def test_compile_one_honors_optimization_level_and_respects_coupling_map():
    """Regression: compile knobs must reach Qiskit and the result must be routed.

    A translation-only pass manager silently ignored ``optimization_level``,
    ``routing_method`` and ``seed_transpiler`` and emitted circuits with
    two-qubit gates on non-adjacent physical qubits.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    backend = GenericBackendV2(
        num_qubits=5, coupling_map=CouplingMap.from_line(5), seed=11
    )
    edges = {tuple(edge) for edge in backend.coupling_map}

    qc = QuantumCircuit(5, 5, name="star")
    qc.h(0)
    for target in (1, 2, 3, 4):
        qc.cx(0, target)
    qc.measure(range(5), range(5))

    depths = {}
    for level in (0, 1, 2, 3):
        compiled, metrics = pipeline.compile_one(
            qc,
            backend=backend,
            spec=StrategySpec(
                optimization_level=level, routing_method="sabre", layout_method="sabre"
            ),
            profile=False,
        )
        violations = [
            instruction.operation.name
            for instruction in compiled.data
            if len(instruction.qubits) == 2
            and instruction.operation.name not in ("barrier", "delay")
            and tuple(compiled.find_bit(bit).index for bit in instruction.qubits)
            not in edges
        ]
        assert violations == []
        assert compiled.num_qubits == backend.num_qubits
        depths[level] = metrics["depth"]

    # The knobs must actually change the compilation result.
    assert len(set(depths.values())) > 1


def test_compile_one_noise_aware_layout_is_applied_and_routed():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    backend = GenericBackendV2(
        num_qubits=5, coupling_map=CouplingMap.from_line(5), seed=11
    )
    edges = {tuple(edge) for edge in backend.coupling_map}

    qc = QuantumCircuit(4, 4, name="chain")
    qc.h(0)
    for a, b in ((0, 1), (1, 2), (2, 3), (0, 3)):
        qc.cx(a, b)
    qc.measure(range(4), range(4))

    compiled, _ = pipeline.compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(
            optimization_level=2,
            layout_method=pipeline.NOISE_AWARE_LAYOUT,
            routing_method="sabre",
        ),
        profile=False,
    )

    assert compiled.num_qubits == backend.num_qubits
    for instruction in compiled.data:
        if len(instruction.qubits) == 2 and instruction.operation.name not in (
            "barrier",
            "delay",
        ):
            pair = tuple(compiled.find_bit(bit).index for bit in instruction.qubits)
            assert pair in edges


def test_compile_one_rejects_unknown_transpiler_methods():
    """An unusable strategy must fail loudly rather than silently degrade."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.exceptions import TranspilerError

    backend = GenericBackendV2(num_qubits=3, seed=3)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    with pytest.raises(TranspilerError):
        pipeline.compile_one(
            qc,
            backend=backend,
            spec=StrategySpec(routing_method="definitely_not_a_router"),
            profile=False,
        )


def test_measurement_twirling_skips_measurement_reused_by_a_later_instruction(
    monkeypatch,
):
    """Regression: a flip is only correctable when nothing later observes it.

    Measuring the same qubit twice used to record one flip per classical bit
    while the inserted ``X`` gates compounded, so untwirling corrupted the
    later bit.  The same applies when a later measurement overwrites the
    classical bit that carries the correction.
    """
    from qiskit import QuantumCircuit

    monkeypatch.setattr(
        suppression.np.random,
        "default_rng",
        lambda seed=None: types.SimpleNamespace(integers=lambda a, b: 1),
    )

    repeated_qubit = QuantumCircuit(1, 2)
    repeated_qubit.h(0)
    repeated_qubit.measure(0, 0)
    repeated_qubit.measure(0, 1)
    twirled, flip_map = suppression.apply_measurement_twirling(repeated_qubit, seed=0)
    assert [inst.operation.name for inst in twirled.data] == [
        "h",
        "measure",
        "x",
        "measure",
    ]
    assert flip_map == {1: 1}

    overwritten_clbit = QuantumCircuit(2, 1)
    overwritten_clbit.h(0)
    overwritten_clbit.measure(0, 0)
    overwritten_clbit.measure(1, 0)
    twirled, flip_map = suppression.apply_measurement_twirling(
        overwritten_clbit, seed=0
    )
    assert [inst.operation.name for inst in twirled.data] == [
        "h",
        "measure",
        "x",
        "measure",
    ]
    assert flip_map == {0: 1}


def test_measurement_twirling_still_twirls_independent_and_delayed_measurements(
    monkeypatch,
):
    from qiskit import QuantumCircuit

    monkeypatch.setattr(
        suppression.np.random,
        "default_rng",
        lambda seed=None: types.SimpleNamespace(integers=lambda a, b: 1),
    )

    independent = QuantumCircuit(2, 2)
    independent.h(0)
    independent.cx(0, 1)
    independent.barrier()
    independent.measure(0, 0)
    independent.measure(1, 1)
    _, flip_map = suppression.apply_measurement_twirling(independent, seed=0)
    assert flip_map == {0: 1, 1: 1}

    delayed = QuantumCircuit(1, 1)
    delayed.h(0)
    delayed.measure(0, 0)
    delayed.delay(16, 0)
    _, flip_map = suppression.apply_measurement_twirling(delayed, seed=0)
    assert flip_map == {0: 1}


@pytest.mark.parametrize(
    "builder,label",
    [
        (lambda: _single_register_circuit(), "single register"),
        (lambda: _multi_register_circuit(), "multiple classical registers"),
        (lambda: _permuted_mapping_circuit(), "permuted qubit to clbit mapping"),
        (lambda: _repeated_measurement_circuit(), "same qubit measured twice"),
    ],
)
def test_measurement_twirl_untwirl_round_trip_preserves_the_distribution(
    builder, label
):
    """Twirling plus untwirling must reproduce the untwirled distribution.

    This is the property the flip map exists to guarantee, and it exercises the
    count-key bit order (little-endian, register separators preserved) that
    untwirling depends on.
    """
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator

    simulator = AerSimulator()
    shots = 20000
    circuit = builder()

    def distribution(counts):
        total = sum(counts.values())
        return {key: value / total for key, value in counts.items()}

    reference = distribution(
        simulator.run(circuit, shots=shots, seed_simulator=1).result().get_counts()
    )

    for seed in range(4):
        twirled, flip_map = suppression.apply_measurement_twirling(circuit, seed=seed)
        raw = (
            simulator.run(twirled, shots=shots, seed_simulator=1).result().get_counts()
        )
        corrected = distribution(
            suppression.apply_measurement_untwirl_counts(raw, flip_map)
        )
        total_variation = 0.5 * sum(
            abs(reference.get(key, 0.0) - corrected.get(key, 0.0))
            for key in set(reference) | set(corrected)
        )
        assert total_variation < 0.03, (label, seed, total_variation)


def _single_register_circuit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.ry(1.1, 0)
    qc.cx(0, 1)
    qc.ry(0.4, 2)
    qc.measure(range(3), range(3))
    return qc


def _multi_register_circuit():
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(3, "q")
    first = ClassicalRegister(2, "ca")
    second = ClassicalRegister(1, "cb")
    qc = QuantumCircuit(qubits, first, second)
    qc.ry(0.9, 0)
    qc.h(1)
    qc.ry(1.4, 2)
    qc.cx(0, 1)
    qc.measure(qubits[0], first[0])
    qc.measure(qubits[1], first[1])
    qc.measure(qubits[2], second[0])
    return qc


def _permuted_mapping_circuit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.ry(0.9, 0)
    qc.h(1)
    qc.ry(1.4, 2)
    qc.measure(0, 2)
    qc.measure(1, 0)
    qc.measure(2, 1)
    return qc


def _repeated_measurement_circuit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 2)
    qc.ry(1.2, 0)
    qc.measure(0, 0)
    qc.measure(0, 1)
    return qc


def test_twirled_ensembles_reuse_one_pass_manager(monkeypatch):
    """Building a preset pass manager for a large backend is not free.

    Only the noise-aware layout varies per twirled circuit, so an eight-twirl
    strategy must build one pass manager, not eight identical ones.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=5, seed=2)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    builds = []
    real_generate = pipeline._generate_pm

    def counting(backend_arg, spec_arg, initial_layout=None):
        builds.append(initial_layout)
        return real_generate(backend_arg, spec_arg, initial_layout=initial_layout)

    monkeypatch.setattr(pipeline, "_generate_pm", counting)

    _, metrics = pipeline.compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(
            optimization_level=1,
            routing_method="sabre",
            pauli_twirling=True,
            num_twirls=8,
        ),
        profile=False,
    )
    assert len(builds) == 1
    assert metrics["depth"] > 0

    # The noise-aware layout is per circuit, so it must still rebuild each time.
    builds.clear()
    ensemble = []

    def capture(circuit, num_twirls, seed, target):
        ensemble.extend([circuit] * num_twirls)
        return list(ensemble)

    monkeypatch.setattr(pipeline, "apply_pauli_twirling", capture)
    pipeline.compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(
            optimization_level=1,
            routing_method="sabre",
            layout_method=pipeline.NOISE_AWARE_LAYOUT,
            pauli_twirling=True,
            num_twirls=4,
        ),
        profile=False,
    )
    assert len(builds) == 4


def test_count_two_qubit_ops_counts_exactly_the_two_qubit_gates():
    """Mutation testing found this metric unverified.

    ``_count_two_qubit_ops`` produces the ``two_qubit_ops`` metric the default
    objective weights at 2.0, so a wrong count silently mis-ranks every
    candidate. Inverting its condition previously broke no test.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    empty = QuantumCircuit(2)
    assert pipeline._count_two_qubit_ops(empty) == 0

    single_qubit_only = QuantumCircuit(3)
    single_qubit_only.h(0)
    single_qubit_only.x(1)
    single_qubit_only.rz(0.3, 2)
    assert pipeline._count_two_qubit_ops(single_qubit_only) == 0

    mixed = QuantumCircuit(3, 3)
    mixed.h(0)
    mixed.cx(0, 1)
    mixed.cx(1, 2)
    mixed.measure(range(3), range(3))
    assert pipeline._count_two_qubit_ops(mixed) == 2

    # Barriers and delays span qubits but are directives, not gates: a
    # two-qubit barrier must not inflate the count.
    with_directives = QuantumCircuit(3, 3)
    with_directives.h(0)
    with_directives.cx(0, 1)
    with_directives.barrier(0, 1)
    with_directives.delay(16, 0)
    with_directives.measure(range(3), range(3))
    assert pipeline._count_two_qubit_ops(with_directives) == 1

    three_qubit = QuantumCircuit(3)
    three_qubit.ccx(0, 1, 2)
    assert pipeline._count_two_qubit_ops(three_qubit) == 0


def test_compile_one_reports_the_two_qubit_count_of_the_compiled_circuit():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=5, seed=4)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    compiled, metrics = pipeline.compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(optimization_level=1, routing_method="sabre"),
        profile=False,
    )
    assert metrics["two_qubit_ops"] == pipeline._count_two_qubit_ops(compiled)
    assert metrics["two_qubit_ops"] == sum(
        1
        for instruction in compiled.data
        if len(instruction.qubits) == 2
        and instruction.operation.name not in ("barrier", "delay")
    )
