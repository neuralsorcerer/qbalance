from math import prod

import numpy as np
import pytest

from qbalance.execution.ensemble import instance_flip_maps, run_ensemble, split_shots
from qbalance.mitigation.zne import zne_extrapolate_counts, zne_sampling_overhead


def test_cut_partitions_use_independent_reproducible_samples():
    pytest.importorskip("qiskit_addon_cutting")
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit

    from qbalance.cutting import prepare_cutting_experiment, run_cutting_experiment

    circuit = QuantumCircuit(2, 2)
    circuit.h([0, 1])
    circuit.measure([0, 1], [0, 1])
    experiment = prepare_cutting_experiment(circuit, max_subcircuit_qubits=1)
    simulator = aer.AerSimulator()

    def sample(seed):
        return run_cutting_experiment(
            experiment, experiment.subexperiments, simulator, shots=1, seed=seed
        )

    # Independent |+> qubits have zero parity on average. Restarting the same
    # simulator seed in both partitions produces identical bits for every seed.
    assert {sample(seed) for seed in range(20)} == {-1.0, 1.0}
    assert sample(0) == sample(0)


def test_cutting_backend_without_a_simulator_seed():
    pytest.importorskip("qiskit_addon_cutting")
    from qiskit import QuantumCircuit
    from qiskit.providers.basic_provider import BasicSimulator

    from qbalance.cutting import prepare_cutting_experiment, run_cutting_experiment

    class UnseededBackend(BasicSimulator):
        def run(self, circuits, *, shots, memory):
            return super().run(circuits, shots=shots, memory=memory)

    circuit = QuantumCircuit(2, 2)
    circuit.x([0, 1])
    circuit.measure([0, 1], [0, 1])
    experiment = prepare_cutting_experiment(circuit, max_subcircuit_qubits=1)
    backend = UnseededBackend()
    assert (
        run_cutting_experiment(
            experiment, experiment.subexperiments, backend, shots=4, seed=1
        )
        == 1.0
    )
    assert "run" not in vars(backend)


def test_mthree_preserves_a_classically_stored_bit():
    pytest.importorskip("mthree")
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1, 2)
    circuit.store(circuit.clbits[1], True)
    circuit.x(0)
    circuit.measure(0, 0)
    result = run_ensemble(
        aer.AerSimulator(), [circuit], {}, shots=8, seed=0, mthree=True
    )
    assert result.counts == {"11": 8}
    assert result.mthree_error is None
    assert result.mthree_probs == {"11": 1.0}


def test_mthree_rejects_a_lossy_projection():
    from qbalance.mitigation.mthree import _KeyLayout

    counts = {"00": 1, "11": 1}
    with pytest.raises(ValueError, match="outside the measurement map vary"):
        _KeyLayout(counts, clbits=[0]).project(counts)


@pytest.mark.parametrize("conditional", [False, True])
def test_mthree_rejects_overwritten_readouts_until_a_fresh_measurement(conditional):
    from qiskit import QuantumCircuit

    from qbalance.utils import measured_qubits_by_clbit

    circuit = QuantumCircuit(1, 2)
    circuit.measure(0, 1)
    if conditional:
        with circuit.if_test((circuit.clbits[0], 0)):
            circuit.store(circuit.clbits[1], True)
    else:
        circuit.store(circuit.clbits[1], True)
    with pytest.raises(ValueError, match="overwrite the last measurement"):
        measured_qubits_by_clbit(circuit)
    circuit.measure(0, 1)
    assert measured_qubits_by_clbit(circuit) == [0]


def test_high_degree_zne_overhead_matches_interpolation_weights():
    factors = list(range(1, 24, 2))
    # In a square fit the analytic Lagrange interpolation weights at zero
    # give a reference independent of the least-squares implementation.
    weights = [
        prod(-x / (point - x) for x in factors if x != point) for point in factors
    ]
    expected = len(factors) * sum(w**2 for w in weights)
    assert zne_sampling_overhead(factors, degree=11) == pytest.approx(
        expected, rel=1e-10
    )
    assert zne_extrapolate_counts(factors, [{"00": 10}] * len(factors), degree=11) == {
        "00": 1.0
    }


def test_zne_missing_parity_uses_observed_outcomes_and_preserves_fixed_bits():
    # The right register is unwritten and always zero. The reference contains
    # only even parity, but two odd outcomes were observed at the next factor.
    factors = [5, 1, 3]
    counts = [{"00 0": 10}, {"00 0": 10}, {"10 0": 6, "01 0": 4}]
    result = zne_extrapolate_counts(factors, counts)
    assert result == pytest.approx({"00 0": 2 / 3, "10 0": 0.2, "01 0": 2 / 15})


@pytest.mark.parametrize("count", [20, 30])
def test_zne_rejects_an_unrepresentable_overhead_without_a_warning(count):
    factors = [10**12 + 2 * i + 1 for i in range(count)]
    with pytest.raises(ValueError, match="numerical range"):
        zne_sampling_overhead(factors, degree=count - 1)


@pytest.mark.parametrize("invalid", [True, 1.5])
def test_ensemble_rejects_nonintegral_shot_splits(invalid):
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        split_shots(invalid, 1)
    with pytest.raises(ValueError, match="instances must be a positive integer"):
        split_shots(2, invalid)
    assert split_shots(np.int64(3), np.int64(2)) == [2, 1]


def test_single_instance_refuses_inconsistent_recorded_flip_maps():
    with pytest.raises(ValueError, match="expected 1 measurement flip maps"):
        instance_flip_maps({"measurement_flip_maps": [{}, {0: 1}]}, 1)


def test_run_counts_only_sends_seed_options_the_backend_supports():
    from qiskit import QuantumCircuit
    from qiskit.providers.basic_provider import BasicSimulator

    from qbalance.execution import run_counts

    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    # BasicSimulator.run takes **kwargs but warns on unsupported options.
    assert run_counts(
        BasicSimulator(), circuit, shots=4, seed_simulator=2, seed_transpiler=3
    ) == {"1": 4}
