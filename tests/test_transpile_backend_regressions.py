from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import CXGate, XGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler import InstructionProperties, Target

from qbalance.backends import aer, fake
from qbalance.errors import QBalanceError
from qbalance.metrics import extract_circuit_metrics
from qbalance.strategies import StrategySpec
from qbalance.transpile import pipeline
from qbalance.transpile.noise_aware_layout import (
    estimate_circuit_error,
    noise_aware_initial_layout,
)
from qbalance.transpile.suppression import apply_measurement_twirling


@pytest.mark.parametrize("target_only", [False, True])
def test_measurement_flip_compiles_in_physical_basis(target_only):
    device = GenericBackendV2(2, basis_gates=["u", "cx"], seed=0)
    backend = SimpleNamespace(target=device.target) if target_only else device
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    compiled, metrics = pipeline.compile_one(
        qc, backend, StrategySpec(measurement_twirling=True, seed_suppression=0)
    )

    assert metrics["measurement_flip_map"] == {0: 1}
    assert compiled.layout is not None
    for entry in compiled.data:
        assert device.target.instruction_supported(
            operation_name=entry.operation.name,
            qargs=tuple(compiled.find_bit(q).index for q in entry.qubits),
        )
    measured = next(e for e in compiled.data if e.operation.name == "measure")
    physical = compiled.find_bit(measured.qubits[0]).index
    state = Statevector.from_instruction(
        compiled.remove_final_measurements(inplace=False)
    )
    assert state.probabilities([physical])[1] == pytest.approx(1.0)


@pytest.mark.parametrize("consumer", ["if", "switch", "store_read", "store_write"])
def test_measurement_flip_respects_implicit_classical_operands(consumer):
    qc = QuantumCircuit(2, 2)
    qc.measure(0, 0)
    first, second = qc.clbits
    branch = QuantumCircuit(1)
    branch.x(0)
    if consumer == "if":
        qc.if_else((qc.cregs[0], 0), branch, None, [1], [])
    elif consumer == "switch":
        qc.switch(first, [(False, branch)], [1], [])
    elif consumer == "store_read":
        qc.store(second, first)
    else:
        qc.store(first, second)

    assert not qc.data[-1].clbits
    twirled, flips = apply_measurement_twirling(qc, seed=0)
    assert twirled == qc
    assert flips == {}


def test_global_target_calibration_prices_every_qubit():
    target = Target(num_qubits=2)
    target.add_instruction(XGate(), {None: InstructionProperties(error=0.3)})
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    assert estimate_circuit_error(SimpleNamespace(target=target), qc) == pytest.approx(
        1 - 0.7**2
    )


def test_compile_time_covers_layout_and_suppression(monkeypatch):
    device = GenericBackendV2(2, seed=0)
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    now = 0.0
    layout = pipeline.noise_aware_initial_layout
    twirl = pipeline.apply_measurement_twirling

    def timed_layout(*args, **kwargs):
        nonlocal now
        now += 2.0
        return layout(*args, **kwargs)

    def timed_twirl(*args, **kwargs):
        nonlocal now
        now += 3.0
        return twirl(*args, **kwargs)

    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=lambda: now))
    monkeypatch.setattr(pipeline, "noise_aware_initial_layout", timed_layout)
    monkeypatch.setattr(pipeline, "apply_measurement_twirling", timed_twirl)
    _, metrics = pipeline.compile_ensemble(
        circuit,
        device,
        StrategySpec(
            measurement_twirling=True,
            num_twirls=2,
            layout_method=pipeline.NOISE_AWARE_LAYOUT,
        ),
    )
    assert metrics["compile_time_s"] == 8.0


@pytest.mark.parametrize(
    "resolver,spec",
    [
        (aer.resolve, "aer:simulator:unexpected"),
        (aer.resolve, "wrong:simulator"),
        (fake.resolve, "wrong:generic:5"),
    ],
)
def test_backend_resolvers_reject_invalid_specs(resolver, spec):
    with pytest.raises(QBalanceError, match="Invalid"):
        resolver(spec)


def test_control_flow_depth_waits_for_ordered_classical_expressions():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure(0, 0)
    qc.store(qc.clbits[1], qc.clbits[0])
    branch = QuantumCircuit(1)
    branch.x(0)
    qc.if_else((qc.clbits[1], 0), branch, None, [1], [])

    assert extract_circuit_metrics(qc)["depth"] == 3.0


def test_layout_for_single_qubit_circuit_can_choose_an_isolated_qubit():
    target = Target(num_qubits=26)
    target.add_instruction(
        CXGate(), {(q, q + 1): InstructionProperties(error=0.001) for q in range(24)}
    )
    target.add_instruction(
        Measure(),
        {(q,): InstructionProperties(error=0.0 if q == 25 else 0.1) for q in range(26)},
    )
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    layout = noise_aware_initial_layout(SimpleNamespace(target=target), circuit)
    assert layout[circuit.qubits[0]] == 25


def test_control_flow_depth_tracks_measurements_through_standalone_variables():
    circuit = QuantumCircuit(2, 1)
    condition = circuit.add_var("condition", False)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.store(condition, circuit.clbits[0])
    with circuit.if_test(condition):
        circuit.x(1)
    assert extract_circuit_metrics(circuit)["depth"] == 3.0
