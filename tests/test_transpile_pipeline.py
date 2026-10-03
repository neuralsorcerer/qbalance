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
    """A twirled strategy transpiles once, whatever its twirl count.

    Twirling is applied to the compiled circuit, so an eight-twirl strategy
    builds one pass manager -- including with the noise-aware layout, which
    depends only on the circuit being compiled.
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

    builds.clear()
    twirled_inputs = []

    def capture(circuit, num_twirls, seed, target):
        twirled_inputs.append(circuit)
        return [circuit.copy() for _ in range(num_twirls)]

    monkeypatch.setattr(pipeline, "apply_pauli_twirling", capture)
    instances, metrics = pipeline.compile_ensemble(
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
    assert len(builds) == 1
    assert builds[0] is not None  # the noise-aware layout reached the compile
    # The twirl acts on the compiled, backend-native circuit.
    assert len(twirled_inputs) == 1
    assert twirled_inputs[0].num_qubits == backend.num_qubits
    assert len(instances) == 4
    assert metrics["twirl_instances"] == 4


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


def test_target_fallback_warning_names_the_backend(caplog):
    """The warning must identify which backend lost optimization_level.

    It fires only for backends without a transpiler Target -- BackendV1-style
    objects, whose ``name`` is a method.  Printing the attribute directly puts
    a bound-method repr in the log instead of the name.
    """
    import logging
    import types as _types

    class V1Style:
        target = None

        def name(self):

            return "ibmq_fake_device"

        def configuration(self):

            return _types.SimpleNamespace(basis_gates=["cx", "x", "rz", "sx"])

    with caplog.at_level(logging.WARNING, logger="qbalance.transpile.pipeline"):
        pipeline._generate_pm(V1Style(), StrategySpec())

    assert "Backend ibmq_fake_device exposes no transpiler Target" in caplog.text
    assert "bound method" not in caplog.text


def test_estimate_circuit_error_matches_the_independent_error_product():
    """The estimate is 1 - product(1 - e_i) over the circuit's operations.

    This metric carries the heaviest weight in the default objective, so an
    error here skews every selection.  The existing assertions only bound it
    to (0, 1), which a quite different formula would also satisfy -- pin the
    arithmetic against the documented per-operation defaults.
    """
    from qiskit import QuantumCircuit

    class NoCalibration:
        """No calibration data, so every operation takes its default."""

        num_qubits = 2
        target = None

        @staticmethod
        def properties():

            return None

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    circuit.barrier()
    circuit.measure([0, 1], [0, 1])

    # Defaults: two-qubit 0.01, single-qubit 0.001, readout 0.02.
    expected = 1.0 - (0.99**2) * 0.999 * (0.98**2)
    error = nal.estimate_circuit_error(NoCalibration(), circuit)
    assert error == pytest.approx(expected, rel=1e-12, abs=0.0)

    # A barrier spans two qubits but is a scheduling directive, not a gate;
    # billing it as a two-qubit error would inflate the estimate.
    without_barrier = QuantumCircuit(2, 2)
    without_barrier.h(0)
    without_barrier.cx(0, 1)
    without_barrier.cx(0, 1)
    without_barrier.measure([0, 1], [0, 1])
    assert nal.estimate_circuit_error(NoCalibration(), without_barrier) == error


def test_compile_one_does_not_profile_by_default():
    """Profiling is opt-in; it installs a callback and adds a report.

    Turning it on by default would put a pass_profile section into every
    compiled artifact and pay for the instrumentation on every call.
    """
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=2)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    _, metrics = pipeline.compile_one(qc, backend=backend, spec=StrategySpec())
    assert "pass_profile" not in metrics

    _, profiled = pipeline.compile_one(
        qc, backend=backend, spec=StrategySpec(), profile=True
    )
    assert profiled["pass_profile"]["passes"]


def test_two_qubit_counting_agrees_across_the_two_implementations():
    """Both counters must exclude scheduling directives identically.

    ``extract_circuit_metrics`` and ``pipeline._count_two_qubit_ops`` both
    report two_qubit_ops, in separate implementations (sharing
    ``utils.SCHEDULING_DIRECTIVES``).  If they ever diverge, the metric
    silently depends on which path produced it -- and a barrier spanning two
    qubits counting as a two-qubit gate would inflate the objective for
    otherwise identical circuits.
    """
    from qiskit import QuantumCircuit

    from qbalance.metrics.circuit_metrics import extract_circuit_metrics

    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier(0, 1)  # spans two qubits, is not a gate
    circuit.cz(1, 2)
    circuit.delay(16, 0)
    circuit.measure([0, 1, 2], [0, 1, 2])

    assert pipeline._count_two_qubit_ops(circuit) == 2
    assert extract_circuit_metrics(circuit)["two_qubit_ops"] == 2.0
    assert (
        float(pipeline._count_two_qubit_ops(circuit))
        == extract_circuit_metrics(circuit)["two_qubit_ops"]
    )


def test_depth_and_size_count_operations_not_idle_delays():
    """An explicit idle delay is schedule, not a layer of operations."""
    from qiskit import QuantumCircuit

    from qbalance.metrics.circuit_metrics import extract_circuit_metrics
    from qbalance.utils import operation_depth, operation_size

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.delay(160, 0)
    circuit.delay(320, 1)
    circuit.x(0)
    circuit.barrier()
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    # Qiskit counts both delays: h, delay, x, cx, measure on qubit 0.
    assert circuit.depth() == 5 and circuit.size() == 7
    assert operation_depth(circuit) == 4
    assert operation_size(circuit) == 5
    metrics = extract_circuit_metrics(circuit)
    assert (metrics["depth"], metrics["size"]) == (4.0, 5.0)
    # The barrier still synchronizes the wires, as Qiskit's own depth does.
    unsynchronized = circuit.copy_empty_like()
    unsynchronized.h(0)
    unsynchronized.x(0)
    unsynchronized.x(0)
    unsynchronized.barrier()
    unsynchronized.x(1)
    assert operation_depth(unsynchronized) == 4


def test_dynamical_decoupling_does_not_bill_idle_delays_as_depth():
    """Regression: DD's padding delays inflated depth and size.

    Under the default objective each delay layer cost as much as a gate
    layer, so a DD strategy on a Bell circuit scored as twice as deep as the
    same compile without DD, although it runs for exactly as long.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from qbalance.backends import resolve_backend
    from qbalance.utils import operation_depth, operation_size

    backend = resolve_backend("fake:generic:5")
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])

    plain, plain_m = pipeline.compile_ensemble(
        bell, backend, StrategySpec(optimization_level=1)
    )
    (padded,), dd_m = pipeline.compile_ensemble(
        bell,
        backend,
        StrategySpec(optimization_level=1, dynamical_decoupling=True),
    )

    assert dd_m["dd_applied"] is True
    assert padded.count_ops().get("delay", 0) > 0
    assert dd_m["depth"] == operation_depth(padded) < padded.depth()
    assert dd_m["size"] == operation_size(padded) < padded.size()
    # Only the inserted DD pulses are billed, never the delays around them.
    pulses = sum(
        count for name, count in padded.count_ops().items() if name in ("x", "y")
    )
    assert dd_m["size"] == plain_m["size"] + pulses
    assert plain_m["depth"] == plain[0].depth()


def test_measurement_twirl_is_inserted_before_dd_scheduling():
    """Regression: the twirl's X gates were added after DD had scheduled.

    An X inserted before one qubit's measurement after padding pushed that
    measurement past the schedule the padding was built for, leaving an
    idle window no DD sequence covered.  A circuit that is consistently
    scheduled gains no delay when it is scheduled again.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

    from qbalance.backends import resolve_backend

    backend = resolve_backend("fake:generic:5")
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    instances, metrics = pipeline.compile_ensemble(
        qc,
        backend,
        StrategySpec(
            optimization_level=1,
            dynamical_decoupling=True,
            measurement_twirling=True,
            num_twirls=6,
        ),
    )

    assert metrics["dd_applied"] is True
    # Some instance flips only part of the measured qubits.
    assert any(0 < len(flips) < 3 for flips in metrics["measurement_flip_maps"])
    reschedule = PassManager(
        [ALAPScheduleAnalysis(target=backend.target), PadDelay(target=backend.target)]
    )
    for instance in instances:
        again = reschedule.run(instance)
        assert again.count_ops().get("delay", 0) == instance.count_ops().get("delay", 0)


def test_dynamical_decoupling_keeps_the_compiled_layout():
    """Regression: the separate DD pass-manager run dropped the layout.

    Its fresh property set carries no TranspileLayout, so every DD strategy
    returned ``layout=None`` and lost the virtual-to-physical mapping.
    """
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=5, seed=0)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    plain, _ = pipeline.compile_one(qc, backend, StrategySpec(optimization_level=2))
    for spec in (
        StrategySpec(optimization_level=2, dynamical_decoupling=True),
        StrategySpec(
            optimization_level=2,
            dynamical_decoupling=True,
            measurement_twirling=True,
        ),
    ):
        compiled, metrics = pipeline.compile_one(qc, backend, spec)
        assert metrics["dd_applied"] is True
        assert compiled.layout is not None
        assert compiled.layout.initial_index_layout(
            filter_ancillas=True
        ) == plain.layout.initial_index_layout(filter_ancillas=True)


class _CalibratedBackend:
    """Backend stub exposing per-qubit readout error and T1/T2 in seconds."""

    def __init__(self, readout, t1, t2):

        self.num_qubits = len(readout)
        self._readout = readout
        self._t1 = t1
        self._t2 = t2

    def properties(self):

        def item(name, value):
            return types.SimpleNamespace(name=name, value=value)

        qubits = []
        for q in range(self.num_qubits):
            entries = [item("readout_error", self._readout[q])]
            if self._t1[q] is not None:
                entries.append(item("T1", self._t1[q]))
            if self._t2[q] is not None:
                entries.append(item("T2", self._t2[q]))
            qubits.append(entries)
        return types.SimpleNamespace(qubits=qubits)


def _single_qubit_circuit():

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def test_noise_aware_layout_does_not_favour_qubits_missing_coherence_data():
    """Regression: a missing T1/T2 defaulted to 50e3 against values in seconds.

    That default added ~1.0 to the quality score while real coherence times
    (~1e-4 s) added ~1e-9, so an uncalibrated qubit beat every calibrated one
    regardless of readout error.
    """
    backend = _CalibratedBackend(
        readout=[0.01, 0.05],
        t1=[1.2e-4, None],
        t2=[0.9e-4, None],
    )

    layout = nal.noise_aware_initial_layout(backend, _single_qubit_circuit())

    assert list(layout.get_physical_bits()) == [0]


def test_noise_aware_layout_uses_coherence_to_separate_similar_readout():
    """Coherence times must count whatever their units.

    With the old fixed 1e-5 weight, times in seconds contributed ~1e-9, so
    even a 0.05% readout difference outweighed a fourfold coherence gap.
    """
    backend = _CalibratedBackend(
        readout=[0.0200, 0.0205],
        t1=[0.5e-4, 2.0e-4],
        t2=[0.5e-4, 2.0e-4],
    )

    layout = nal.noise_aware_initial_layout(backend, _single_qubit_circuit())

    assert list(layout.get_physical_bits()) == [1]


def _ghz_circuit():

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc


def test_pauli_twirling_is_applied_to_the_compiled_circuit():
    """Regression: twirls were compiled separately and only the best one kept.

    A twirl suppresses noise only averaged over its random instances, and
    twirling *before* transpilation lets optimization re-synthesize the
    twirled blocks.  Every instance must be a target-native twirl of the one
    compiled circuit, implement the same operation, and be returned.
    """
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.quantum_info import Operator

    backend = GenericBackendV2(num_qubits=5, seed=3)
    qc = _ghz_circuit()
    plain, _ = pipeline.compile_ensemble(
        qc, backend, StrategySpec(optimization_level=3, seed_transpiler=4)
    )
    instances, metrics = pipeline.compile_ensemble(
        qc,
        backend,
        StrategySpec(
            optimization_level=3,
            seed_transpiler=4,
            pauli_twirling=True,
            num_twirls=4,
            seed_suppression=9,
        ),
    )

    assert len(plain) == 1
    assert len(instances) == 4
    assert metrics["twirl_instances"] == 4
    supported = set(backend.target.operation_names)
    reference = Operator(plain[0].remove_final_measurements(inplace=False))
    for instance in instances:
        assert set(instance.count_ops()) <= supported
        assert instance.count_ops()["cx"] == plain[0].count_ops()["cx"]
        assert instance.layout is not None
        assert Operator(instance.remove_final_measurements(inplace=False)).equiv(
            reference
        )
    # The instances really are different random twirls.
    assert len({str(instance.data) for instance in instances}) > 1


def test_measurement_twirling_draws_one_flip_pattern_per_instance():
    """Each instance needs its own pattern for the readout error to average."""
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=5, seed=3)
    qc = _ghz_circuit()
    spec = StrategySpec(measurement_twirling=True, num_twirls=6, seed_suppression=5)

    instances, metrics = pipeline.compile_ensemble(qc, backend, spec)

    maps = metrics["measurement_flip_maps"]
    assert len(instances) == len(maps) == 6
    assert metrics["measurement_flip_map"] == maps[0]
    assert len({tuple(sorted(m.items())) for m in maps}) > 1
    # A single-instance strategy with the same seed draws instance 0's pattern.
    _, single = pipeline.compile_one(
        qc, backend, StrategySpec(measurement_twirling=True, seed_suppression=5)
    )
    assert single["measurement_flip_map"] == maps[0]


def test_dynamical_decoupling_reports_the_sequence_actually_applied():
    """A target without ``y`` pads with XX even when XY4 was requested."""
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=5, seed=3)
    assert "y" not in backend.target.operation_names

    _, metrics = pipeline.compile_one(
        _ghz_circuit(),
        backend,
        StrategySpec(optimization_level=1, dynamical_decoupling=True),
    )

    assert metrics["dd_applied"] is True
    assert metrics["dd_sequence_applied"] == "XX"
    assert suppression.dd_sequence_label(suppression._dd_sequence("XY4")) == "XY4"


def test_compile_one_returns_the_first_ensemble_instance(monkeypatch):
    first, second = object(), object()
    monkeypatch.setattr(
        pipeline,
        "compile_ensemble",
        lambda circuit, backend, spec, profile=False: ([first, second], {"d": 1}),
    )

    compiled, metrics = pipeline.compile_one(object(), object(), StrategySpec())

    assert compiled is first
    assert metrics == {"d": 1}


def test_uncut_compiles_report_no_sampling_overhead():
    from qiskit.providers.fake_provider import GenericBackendV2

    _, metrics = pipeline.compile_one(
        _ghz_circuit(), GenericBackendV2(num_qubits=5, seed=3), StrategySpec()
    )

    assert metrics["sampling_overhead"] == 1.0
    assert metrics["twirl_instances"] == 1
    assert metrics["measurement_flip_maps"] == [{}]


def test_zne_compiles_report_the_shots_extrapolation_costs():
    """Regression: a ZNE strategy was billed as if it ran the circuit once.

    ZNE runs every noise factor on the full shot budget and extrapolates, so
    matching an unmitigated run's precision takes ``n * sum(w_i**2)`` times
    the shots.  Reported as 1.0, a ZNE candidate scored exactly like the same
    compile without ZNE and was selected on ties, silently multiplying the
    execution cost.
    """
    from qiskit.providers.fake_provider import GenericBackendV2

    from qbalance.mitigation.zne import zne_sampling_overhead

    # Monte-Carlo-checked values: linear at (1, 3, 5) and (1, 3), quadratic.
    assert zne_sampling_overhead((1.0, 3.0, 5.0)) == pytest.approx(4.375)
    assert zne_sampling_overhead((1.0, 3.0)) == pytest.approx(5.0)
    assert zne_sampling_overhead((1.0, 3.0, 5.0), 2) == pytest.approx(15.65625)
    # Folding realizes (1, 3, 3) here; the repeated point costs shots too.
    assert zne_sampling_overhead((1.0, 2.0, 3.0)) == pytest.approx(7.125)
    assert zne_sampling_overhead((1.0,), 0) == pytest.approx(1.0)
    # Factors realizing 1 share the one unfolded run, so a repeated 1.0 adds
    # neither shots nor information: the estimate is exactly the (1, 3) fit.
    assert zne_sampling_overhead((1.0, 1.0, 3.0)) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="distinct"):
        zne_sampling_overhead((1.0, 2.0, 3.0), 2)

    backend = GenericBackendV2(num_qubits=5, seed=3)
    _, plain = pipeline.compile_one(_ghz_circuit(), backend, StrategySpec())
    _, zne = pipeline.compile_one(_ghz_circuit(), backend, StrategySpec(zne=True))
    assert zne["sampling_overhead"] == pytest.approx(4.375)
    assert {k: zne[k] for k in ("depth", "two_qubit_ops")} == {
        k: plain[k] for k in ("depth", "two_qubit_ops")
    }


def _line_backend_with_broken_coupler():
    """Six-qubit line whose (1, 2) coupler is broken (error 1.0)."""
    from qiskit.circuit.library import CXGate, Measure
    from qiskit.transpiler import InstructionProperties, Target

    target = Target(num_qubits=6)
    cx_props = {}
    for a in range(5):
        error = 1.0 if (a, a + 1) == (1, 2) else 0.01
        cx_props[(a, a + 1)] = InstructionProperties(error=error)
        cx_props[(a + 1, a)] = InstructionProperties(error=error)
    target.add_instruction(CXGate(), cx_props)
    target.add_instruction(
        Measure(), {(q,): InstructionProperties(error=0.01) for q in range(6)}
    )
    return types.SimpleNamespace(
        target=target, num_qubits=6, coupling_map=target.build_coupling_map()
    )


def _chain_circuit(n):

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.measure(range(n), range(n))
    return qc


def test_noise_aware_layout_places_interacting_qubits_on_coupled_pairs():
    """Regression: the layout ignored the coupling map entirely.

    It ranked qubits by calibration alone, so interacting qubits landed far
    apart on sparse devices and routing multiplied the circuit (depth 594 vs
    72 on a heavy-hex snapshot).
    """
    backend = _line_backend_with_broken_coupler()
    qc = _chain_circuit(3)

    layout = nal.noise_aware_initial_layout(backend, qc)

    physical = [layout[qc.qubits[q]] for q in range(3)]
    coupled = {tuple(edge) for edge in backend.coupling_map.get_edges()}
    for a, b in ((0, 1), (1, 2)):
        assert (physical[a], physical[b]) in coupled or (
            physical[b],
            physical[a],
        ) in coupled


def test_noise_aware_layout_avoids_broken_couplers():
    """A coupler with error 1.0 must not carry the circuit's interactions."""
    backend = _line_backend_with_broken_coupler()
    qc = _chain_circuit(3)

    layout = nal.noise_aware_initial_layout(backend, qc)

    physical = {layout[qc.qubits[q]] for q in range(3)}
    # The only chain of three that avoids the broken (1, 2) coupler.
    assert physical == {3, 4, 5} or physical == {2, 3, 4}
    assert not {1, 2} <= physical


def test_noise_aware_compile_routes_no_worse_than_the_chain_needs():
    """On a heavy-hex snapshot a chain needs no routing at all."""
    fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    backend = fake.FakeSherbrooke()
    qc = _chain_circuit(8)

    _, metrics = pipeline.compile_one(
        qc,
        backend,
        StrategySpec(
            optimization_level=2,
            layout_method=pipeline.NOISE_AWARE_LAYOUT,
            routing_method="sabre",
        ),
    )

    assert metrics["two_qubit_ops"] == 7
    assert metrics["estimated_error"] < 0.5


def test_noise_aware_layout_does_not_route_around_a_poor_neighbour():
    """Regression: a qubit with poor readout was skipped at the price of a hop.

    The cost model charged readout where a qubit was placed, but routing
    moves one of an uncoupled pair onto the path between them, so the qubit
    is read out on the skipped qubit anyway -- after three extra gates.  On a
    heavy-hex snapshot an eight-qubit chain came out with 16 two-qubit gates
    instead of 7 and twice the estimated error of Qiskit's own layout.
    """
    from qiskit.circuit.library import CXGate, Measure
    from qiskit.transpiler import InstructionProperties, Target

    # Five-qubit line whose middle qubit reads out poorly: every chain of
    # three adjacent qubits has to include it.
    target = Target(num_qubits=5)
    cx_props = {}
    for a in range(4):
        cx_props[(a, a + 1)] = InstructionProperties(error=0.01)
        cx_props[(a + 1, a)] = InstructionProperties(error=0.01)
    target.add_instruction(CXGate(), cx_props)
    readout = {(q,): InstructionProperties(error=0.01) for q in range(5)}
    readout[(2,)] = InstructionProperties(error=0.3)
    target.add_instruction(Measure(), readout)
    backend = types.SimpleNamespace(
        target=target, num_qubits=5, coupling_map=target.build_coupling_map()
    )
    qc = _chain_circuit(3)

    layout = nal.noise_aware_initial_layout(backend, qc)

    physical = [layout[qc.qubits[q]] for q in range(3)]
    assert all(abs(physical[q] - physical[q + 1]) == 1 for q in range(2))


def test_usable_adjacency_drops_broken_couplers_only():
    adjacency = [{1}, {0, 2}, {1, 3}, {2}]
    errors = {(0, 1): 0.01, (1, 2): 1.0, (2, 3): 0.02}

    assert nal._usable_adjacency(adjacency, errors) == [{1}, {0}, {3}, {2}]
    # The caller's graph is left untouched.
    assert adjacency == [{1}, {0, 2}, {1, 3}, {2}]
    # With every coupler broken there is nothing better to plan with.
    assert nal._usable_adjacency([{1}, {0}], {(0, 1): 1.0}) == [{1}, {0}]


def test_noise_aware_compile_keeps_a_chain_unrouted_on_a_cz_snapshot():
    fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    backend = fake.FakeTorino()
    qc = _chain_circuit(8)

    for level in (1, 2):
        _, metrics = pipeline.compile_one(
            qc,
            backend,
            StrategySpec(
                optimization_level=level,
                layout_method=pipeline.NOISE_AWARE_LAYOUT,
                routing_method="sabre",
            ),
        )
        assert metrics["two_qubit_ops"] == 7
        assert metrics["estimated_error"] < 0.2


def test_noise_aware_layout_sees_the_interactions_of_wider_gates():
    """Regression: gates on three or more qubits were invisible to the layout.

    The layout runs on the circuit as written, where a Toffoli is still one
    three-qubit gate; only literal two-qubit gates were counted, so a
    Toffoli circuit looked interaction-free and was scattered across the
    device (207 two-qubit gates after routing on a heavy-hex snapshot,
    against 66 for SABRE's layout).
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.barrier(0, 3)
    deg, weights = nal._logical_interactions(qc, 4)
    assert weights == {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    assert list(deg) == [2.0, 2.0, 3.0, 1.0]

    # On a line, the Toffoli's three qubits must land on three adjacent
    # qubits, clear of the broken coupler.
    backend = _line_backend_with_broken_coupler()
    toffoli = QuantumCircuit(3)
    toffoli.ccx(0, 1, 2)
    layout = nal.noise_aware_initial_layout(backend, toffoli)
    physical = sorted(layout[toffoli.qubits[q]] for q in range(3))
    assert physical in ([2, 3, 4], [3, 4, 5])


def test_noise_aware_compile_keeps_toffoli_circuits_local():
    fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from qiskit import QuantumCircuit

    backend = fake.FakeSherbrooke()
    qc = QuantumCircuit(6, 6)
    for triple in [(0, 1, 2), (3, 4, 5), (1, 2, 3), (2, 3, 4), (0, 1, 2), (3, 4, 5)]:
        qc.ccx(*triple)
    qc.measure(range(6), range(6))

    def two_qubit_ops(layout_method):
        _, metrics = pipeline.compile_one(
            qc,
            backend,
            StrategySpec(
                optimization_level=1,
                layout_method=layout_method,
                routing_method="sabre",
            ),
        )
        return metrics["two_qubit_ops"]

    # Six Toffolis need 36 CNOTs before routing; SABRE's layout reaches 66.
    assert two_qubit_ops(pipeline.NOISE_AWARE_LAYOUT) <= 1.25 * two_qubit_ops("sabre")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_suppression_strategy_preserves_the_ideal_output(seed):
    """Differential check: compiling, twirling, DD and folding change noise only.

    Each strategy's instances, run noiselessly and untwirled with their own
    flip maps, must reproduce the original circuit's output distribution, and
    so must its ZNE-folded circuits.  Random circuits cover permuted and
    partial measurements, several classical registers, three-qubit gates and
    barriers.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("qiskit_aer")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator

    from qbalance.backends import resolve_backend
    from qbalance.execution.ensemble import instance_flip_maps, merge_counts
    from qbalance.mitigation.zne import fold_global_for_backend

    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 6))
    cut = int(rng.integers(1, n))
    qc = QuantumCircuit(
        QuantumRegister(n), ClassicalRegister(cut, "a"), ClassicalRegister(n, "b")
    )
    for _ in range(10):
        a, b, c = (int(x) for x in rng.choice(n, size=3, replace=False))
        kind = int(rng.integers(0, 5))
        if kind == 0:
            qc.ry(float(rng.uniform(0, 3)), a)
        elif kind == 1:
            qc.cx(a, b)
        elif kind == 2:
            qc.cp(float(rng.uniform(0, 3)), a, b)
        elif kind == 3:
            qc.ccx(a, b, c)
        else:
            qc.barrier(a, b)
    # Permuted qubits into distinct clbits across both registers; the
    # remaining clbits are never written.
    qubits = [int(q) for q in rng.permutation(n)][: n - 1]
    clbits = [int(c) for c in rng.choice(cut + n, size=n - 1, replace=False)]
    for qubit, clbit in zip(qubits, clbits):
        qc.measure(qubit, clbit)

    ideal = AerSimulator()
    backend = resolve_backend("fake:generic:6")
    shots = 8000

    def distribution(counts):
        total = sum(counts.values())
        return {key: value / total for key, value in counts.items()}

    def tvd(p, q):
        return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

    reference = distribution(
        ideal.run(qc, shots=shots, seed_simulator=1).result().get_counts()
    )
    specs = [
        StrategySpec(optimization_level=3, routing_method="sabre"),
        StrategySpec(optimization_level=2, layout_method=pipeline.NOISE_AWARE_LAYOUT),
        StrategySpec(optimization_level=2, pauli_twirling=True, num_twirls=3),
        StrategySpec(
            optimization_level=1,
            dynamical_decoupling=True,
            measurement_twirling=True,
            num_twirls=3,
        ),
    ]
    for spec in specs:
        instances, metrics = pipeline.compile_ensemble(qc, backend, spec)
        flip_maps = instance_flip_maps(metrics, len(instances))
        counts = merge_counts(
            suppression.apply_measurement_untwirl_counts(
                ideal.run(instance, shots=shots // len(instances), seed_simulator=2)
                .result()
                .get_counts(),
                flip_map,
            )
            for instance, flip_map in zip(instances, flip_maps)
        )
        assert tvd(reference, distribution(counts)) < 0.06, spec
    compiled, _ = pipeline.compile_one(qc, backend, StrategySpec())
    folded = fold_global_for_backend(compiled, backend, 3.0)
    counts = ideal.run(folded, shots=shots, seed_simulator=3).result().get_counts()
    assert tvd(reference, distribution(counts)) < 0.06


def test_a_pauli_twirl_that_cannot_be_applied_makes_the_strategy_infeasible(
    monkeypatch,
):
    """Regression: a failed twirl compiled on, untwirled but labeled twirled.

    The candidate then reported ``pauli_twirling=True`` -- and could be
    selected as such -- although no instance was twirled.
    """
    from qiskit.providers.fake_provider import GenericBackendV2

    from qbalance.errors import QBalanceError

    def refuse(*args, **kwargs):
        raise RuntimeError("gate cannot be twirled")

    monkeypatch.setattr(pipeline, "apply_pauli_twirling", refuse)

    with pytest.raises(QBalanceError, match="Pauli twirling failed.*cannot be twirled"):
        pipeline.compile_ensemble(
            _chain_circuit(3),
            GenericBackendV2(num_qubits=3, seed=0),
            StrategySpec(pauli_twirling=True, num_twirls=4),
        )


def test_noise_aware_layout_of_a_circuit_without_qubits_is_empty():
    """Regression: a zero-qubit circuit crashed the placement (IndexError)."""
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    layout = nal.noise_aware_initial_layout(
        GenericBackendV2(num_qubits=5, seed=0), QuantumCircuit(0)
    )

    assert layout is not None
    assert len(layout.get_virtual_bits()) == 0
