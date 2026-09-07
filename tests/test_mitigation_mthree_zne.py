# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types

import pytest

from qbalance.mitigation import mthree as mthree_mod
from qbalance.mitigation import zne
from tests.system_stubs import _Circ


def test_mthree_and_zne(monkeypatch):

    mthree = types.ModuleType("mthree")

    class Mit:
        def __init__(self, backend):

            _ = backend

        def cals_from_system(self, measured_qubits, calibration_shots):

            _ = (measured_qubits, calibration_shots)

        def apply_correction(self, raw_counts, measured_qubits):

            _ = measured_qubits
            s = sum(raw_counts.values())
            return types.SimpleNamespace(
                nearest_probability_distribution=lambda: {
                    k: v / s for k, v in raw_counts.items()
                }
            )

    mthree.M3Mitigation = Mit
    monkeypatch.setitem(sys.modules, "mthree", mthree)
    probs = mthree_mod.apply_mthree_mitigation(object(), {"00": 2, "11": 2}, [0, 1], 4)
    assert probs["00"] == 0.5

    c = _Circ()
    f = zne.fold_global(c, 3.2)
    assert getattr(f, "name", "").startswith("circuit_fold") or isinstance(f, _Circ)
    assert zne._counts_to_expval_z({"00": 3, "01": 1}) == pytest.approx(0.5)
    out = zne.zne_extrapolate_counts(
        [1.0, 3.0], [{"00": 2, "11": 2}, {"00": 3, "11": 1}]
    )
    assert pytest.approx(sum(out.values())) == 1.0
    parity_adjusted = zne.zne_extrapolate_counts(
        [1.0, 2.0, 3.0], [{"00": 10}, {"00": 10}, {"01": 10}], degree=2
    )
    assert pytest.approx(sum(parity_adjusted.values())) == 1.0
    assert any(bitstr.count("1") % 2 == 1 for bitstr in parity_adjusted)

    even_adjusted = zne.zne_extrapolate_counts(
        [1.0, 2.0, 3.0], [{"1": 10}, {"1": 10}, {"0": 10}], degree=2
    )
    assert pytest.approx(sum(even_adjusted.values())) == 1.0
    assert any(bitstr.count("1") % 2 == 0 for bitstr in even_adjusted)

    spaced = zne.zne_extrapolate_counts(
        [1.0, 2.0, 3.0], [{"00 0": 10}, {"00 0": 10}, {"00 1": 10}], degree=2
    )
    assert "00 1" in spaced
    assert pytest.approx(sum(spaced.values())) == 1.0

    with pytest.raises(ValueError, match="must have same length"):
        zne.zne_extrapolate_counts([1.0], [{"0": 1}, {"1": 1}])
    with pytest.raises(ValueError, match="factors must be finite"):
        zne.zne_extrapolate_counts([1.0, float("nan")], [{"0": 1}, {"0": 1}])
    with pytest.raises(ValueError, match="distinct values"):
        zne.zne_extrapolate_counts([1.0, 1.0], [{"0": 1}, {"0": 1}], degree=1)
    with pytest.raises(ValueError, match="factors must be >= 1.0"):
        zne.zne_extrapolate_counts([0.5, 1.0], [{"0": 1}, {"0": 1}])
    with pytest.raises(ValueError, match="non-negative integer"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0": 1}, {"0": 1}], degree=-1)
    with pytest.raises(ValueError, match="non-negative integers"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0": -1}, {"0": 1}])
    with pytest.raises(ValueError, match="non-empty mappings"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{}, {"0": 1}])
    with pytest.raises(ValueError, match="non-negative integer"):
        zne.zne_extrapolate_counts([1.0], [{"0": 1}], degree=True)
    with pytest.raises(ValueError, match="at least one shot"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0": 0}, {"0": 1}])
    with pytest.raises(ValueError, match="only binary digits"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0x0": 1}, {"0": 1}])


def test_fold_global_preserves_terminal_measurements():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1, name="measured")
    qc.h(0)
    qc.measure(0, 0)

    folded = zne.fold_global(qc, 3.0)

    assert [inst.operation.name for inst in folded.data] == [
        "h",
        "h",
        "h",
        "measure",
    ]
    assert folded.num_clbits == 1


def test_fold_global_rejects_invalid_scale_and_nonterminal_measurement():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)

    for bad_scale in (0.5, float("nan"), float("inf"), True, "bad"):
        with pytest.raises(ValueError, match="finite real value"):
            zne.fold_global(qc, bad_scale)  # type: ignore[arg-type]

    measured = QuantumCircuit(1, 1)
    measured.h(0)
    measured.measure(0, 0)
    measured.x(0)

    with pytest.raises(ValueError, match="all measurements are terminal"):
        zne.fold_global(measured, 3.0)


def test_fold_global_preserves_terminal_barriers_after_measurements():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.barrier(0)

    folded = zne.fold_global(qc, 3.0)

    assert [inst.operation.name for inst in folded.data] == [
        "h",
        "h",
        "h",
        "measure",
        "barrier",
    ]


def test_zne_extrapolation_restores_the_target_parity_mass():
    """Regression: an empty parity class must share the target mass, not repeat it.

    Every key of a parity class with no sampled mass used to be assigned the
    full target mass, so a reference distribution carrying two zero-count keys
    of that class produced twice the intended mass and an extrapolated
    observable that no longer matched the fit.
    """
    factors = [1.0, 2.0, 3.0]
    counts_per_factor = [
        {"00": 0, "11": 0, "01": 50, "10": 50},
        {"00": 50, "11": 50},
        {"01": 50, "10": 50},
    ]

    probs = zne.zne_extrapolate_counts(factors, counts_per_factor, degree=1)

    assert pytest.approx(sum(probs.values())) == 1.0
    assert all(value >= 0.0 for value in probs.values())
    even_mass = sum(
        value for bitstr, value in probs.items() if zne._parity(bitstr) == 0
    )
    # The linear fit through (-1, +1, -1) extrapolates to -1/3 at zero noise.
    assert even_mass == pytest.approx((1.0 + (-1.0 / 3.0)) / 2.0)
    assert 2.0 * even_mass - 1.0 == pytest.approx(-1.0 / 3.0)


def test_zne_extrapolation_handles_a_missing_parity_class():
    factors = [1.0, 2.0, 3.0]
    counts_per_factor = [
        {"00": 50, "11": 50},
        {"01": 50, "10": 50},
        {"00": 50, "11": 50},
    ]

    probs = zne.zne_extrapolate_counts(factors, counts_per_factor, degree=1)

    assert pytest.approx(sum(probs.values())) == 1.0
    even_mass = sum(
        value for bitstr, value in probs.items() if zne._parity(bitstr) == 0
    )
    assert even_mass == pytest.approx((1.0 + (1.0 / 3.0)) / 2.0)


def test_fold_global_accepts_measurement_twirl_frame_changes():
    """Regression: measurement twirling interleaves X gates with the measurements.

    Rejecting any non-measure instruction after the first measurement made
    ``zne=True`` with ``measurement_twirling=True`` fail for every circuit.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.measure(0, 0)
    qc.x(1)
    qc.measure(1, 1)

    folded = zne.fold_global(qc, 3.0)

    assert [inst.operation.name for inst in folded.data] == [
        "h",
        "cx",
        "x",
        "x",
        "cx",
        "h",
        "h",
        "cx",
        "x",
        "measure",
        "x",
        "measure",
    ]


def test_fold_global_still_rejects_post_measurement_computation():
    from qiskit import QuantumCircuit

    mid_circuit = QuantumCircuit(1, 1)
    mid_circuit.h(0)
    mid_circuit.measure(0, 0)
    mid_circuit.h(0)
    mid_circuit.measure(0, 0)
    with pytest.raises(ValueError, match="all measurements are terminal"):
        zne.fold_global(mid_circuit, 3.0)

    unmeasured_qubit = QuantumCircuit(2, 1)
    unmeasured_qubit.h(0)
    unmeasured_qubit.measure(0, 0)
    unmeasured_qubit.x(1)
    with pytest.raises(ValueError, match="all measurements are terminal"):
        zne.fold_global(unmeasured_qubit, 3.0)


def test_fold_global_preserves_the_circuit_unitary():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    qc.ry(0.7, 0)

    for scale in (1.0, 2.0, 3.0, 5.0):
        folded = zne.fold_global(qc, scale)
        assert Operator(folded).equiv(Operator(qc))


def test_fold_global_for_backend_returns_a_runnable_circuit():
    """Regression: ``U.inverse()`` introduces gates outside the backend basis."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    from qbalance.strategies import StrategySpec
    from qbalance.transpile.pipeline import compile_one

    backend = GenericBackendV2(num_qubits=5, seed=5)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    compiled, _ = compile_one(
        qc,
        backend=backend,
        spec=StrategySpec(optimization_level=2, routing_method="sabre"),
        profile=False,
    )
    supported = set(backend.target.operation_names)
    assert set(compiled.count_ops()) <= supported

    folded = zne.fold_global_for_backend(compiled, backend, 3.0)

    assert set(folded.count_ops()) <= supported
    assert folded.num_qubits == compiled.num_qubits
    assert folded.num_clbits == compiled.num_clbits

    def measurement_map(circuit):
        return {
            circuit.find_bit(inst.clbits[0])
            .index: circuit.find_bit(inst.qubits[0])
            .index
            for inst in circuit.data
            if inst.operation.name == "measure"
        }

    assert measurement_map(folded) == measurement_map(compiled)
    # Folding must still scale the two-qubit gate count.
    assert folded.count_ops()["cx"] > compiled.count_ops()["cx"]
    # Scale 1.0 is a no-op and must not be re-transpiled.
    assert zne.fold_global_for_backend(compiled, backend, 1.0) is compiled


def test_fold_global_for_backend_leaves_unknown_backends_untouched():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    folded = zne.fold_global_for_backend(qc, object(), 3.0)

    assert [inst.operation.name for inst in folded.data] == [
        "h",
        "h",
        "h",
        "measure",
    ]


def test_fake_ibm_backends_resolve_from_lowercase_device_names():
    """Regression: a same-named submodule shadowed the backend class.

    ``qiskit_ibm_runtime.fake_provider`` exposes a submodule per device
    (``manila``) next to the class (``FakeManilaV2``).  The resolver found the
    submodule, tried to call it, and raised instead of trying the next candidate
    spelling -- defeating the lowercase device names the candidate list exists
    to accept.
    """
    pytest.importorskip("qiskit_ibm_runtime")

    from qbalance.backends import resolve_backend

    lowercase = resolve_backend("fake:ibm:manila")
    titlecase = resolve_backend("fake:ibm:Manila")
    explicit = resolve_backend("fake:ibm:FakeManilaV2")

    assert type(lowercase) is type(titlecase) is type(explicit)
    assert lowercase.num_qubits == 5

    from qbalance.errors import QBalanceError

    with pytest.raises(QBalanceError, match="Unknown IBM fake backend"):
        resolve_backend("fake:ibm:definitely_not_a_device")


def test_mthree_mitigation_improves_a_readout_noisy_distribution():
    """The wrapper must feed mthree the qubits its count keys actually use.

    ``_final_measurement_qubits`` orders physical qubits by classical bit, and
    getting that wrong silently degrades the correction rather than failing, so
    check the mitigated distribution really moves toward the ideal one.
    """
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")

    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError

    from qbalance.mitigation.mthree import apply_mthree_mitigation
    from qbalance.strategies import StrategySpec
    from qbalance.transpile.pipeline import compile_one
    from qbalance.workflow.workload import _final_measurement_qubits

    noise = NoiseModel()
    for qubit in range(5):
        wrong_one = 0.02 + 0.04 * qubit
        wrong_zero = 0.01 + 0.02 * qubit
        noise.add_readout_error(
            ReadoutError([[1 - wrong_one, wrong_one], [wrong_zero, 1 - wrong_zero]]),
            [qubit],
        )
    noisy = AerSimulator(noise_model=noise)

    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure(range(3), range(3))
    compiled, _ = compile_one(
        circuit,
        backend=GenericBackendV2(num_qubits=5, seed=1),
        spec=StrategySpec(optimization_level=2, routing_method="sabre"),
        profile=False,
    )

    shots = 8000
    raw = noisy.run(compiled, shots=shots, seed_simulator=5).result().get_counts()
    ideal = (
        AerSimulator()
        .run(compiled, shots=shots, seed_simulator=5)
        .result()
        .get_counts()
    )

    def normalized(counts):
        total = sum(counts.values())
        return {key: value / total for key, value in counts.items()}

    def total_variation(first, second):
        return 0.5 * sum(
            abs(first.get(key, 0.0) - second.get(key, 0.0))
            for key in set(first) | set(second)
        )

    measured = _final_measurement_qubits(compiled)
    assert len(measured) == compiled.num_clbits

    mitigated = apply_mthree_mitigation(
        noisy, raw, measured_qubits=measured, shots=shots
    )
    assert all(isinstance(value, float) for value in mitigated.values())

    ideal_probs = normalized(ideal)
    raw_error = total_variation(ideal_probs, normalized(raw))
    mitigated_error = total_variation(ideal_probs, normalized(mitigated))
    assert mitigated_error < raw_error / 2


def test_zne_extrapolation_rejects_non_string_count_keys():
    """Mutation testing found this validation branch unexercised."""
    with pytest.raises(ValueError, match="counts keys must be non-empty bitstrings"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{0: 5}, {"0": 5}])
    with pytest.raises(ValueError, match="counts keys must be non-empty bitstrings"):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"": 5}, {"0": 5}])
