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
    The frame-change X gates are part of the computation, so they are folded
    with the rest of it (``X X^dagger X`` is still one net X) and every
    measurement stays terminal.
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
        # U
        "h",
        "cx",
        "x",
        "x",
        # U^dagger
        "x",
        "x",
        "cx",
        "h",
        # U
        "h",
        "cx",
        "x",
        "x",
        "measure",
        "measure",
    ]


def test_fold_global_folds_gates_listed_after_another_qubits_measurement():
    """Regression: every gate is folded, whatever its position in ``data``.

    ``circuit.data`` is only a topological order, so a qubit that finishes
    early is measured before gates on other qubits are listed.  Splitting the
    circuit at the first measurement replayed those later gates once, so the
    two-qubit gates below were never folded and ZNE extrapolated a noise curve
    in which most of the circuit's noise was not scaled at all.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.measure(0, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.measure([1, 2], [1, 2])

    folded = zne.fold_global(qc, 3.0)

    assert folded.count_ops()["cx"] == 3
    assert folded.count_ops()["h"] == 6
    names = [inst.operation.name for inst in folded.data]
    assert names[-3:] == ["measure", "measure", "measure"]
    assert "measure" not in names[:-3]
    # Every measurement still reads the same clbit from the same qubit.
    assert [
        (folded.find_bit(i.qubits[0]).index, folded.find_bit(i.clbits[0]).index)
        for i in folded.data
        if i.operation.name == "measure"
    ] == [(0, 0), (1, 1), (2, 2)]
    unitary = qc.remove_final_measurements(inplace=False)
    assert Operator(folded.remove_final_measurements(inplace=False)).equiv(
        Operator(unitary)
    )


def test_fold_global_still_rejects_post_measurement_computation():
    from qiskit import QuantumCircuit

    mid_circuit = QuantumCircuit(1, 1)
    mid_circuit.h(0)
    mid_circuit.measure(0, 0)
    mid_circuit.h(0)
    mid_circuit.measure(0, 0)
    with pytest.raises(ValueError, match="all measurements are terminal"):
        zne.fold_global(mid_circuit, 3.0)

    # A gate listed after a measurement of a *different* qubit is not
    # post-measurement computation: it commutes with that measurement, so it
    # is folded like any other gate.
    unmeasured_qubit = QuantumCircuit(2, 1)
    unmeasured_qubit.h(0)
    unmeasured_qubit.measure(0, 0)
    unmeasured_qubit.x(1)
    folded = zne.fold_global(unmeasured_qubit, 3.0)
    assert [inst.operation.name for inst in folded.data] == [
        "h",
        "x",
        "x",
        "h",
        "h",
        "x",
        "measure",
    ]


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

    ``measured_qubits_by_clbit`` orders physical qubits by classical bit, and
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
    from qbalance.utils import measured_qubits_by_clbit

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

    measured = measured_qubits_by_clbit(compiled)
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


def test_realized_fold_factor_matches_what_fold_global_applies():
    """The extrapolation x-values must be the factors folding really ran at."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)

    for scale in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5):
        k = zne.realized_fold_factor(scale)
        assert k % 2 == 1
        assert k >= scale
        folded = zne.fold_global(qc, scale)
        assert len(folded.data) == k * len(qc.data)

    for bad in (0.5, float("nan"), True, "x"):
        with pytest.raises(ValueError, match="finite real value"):
            zne.realized_fold_factor(bad)  # type: ignore[arg-type]


def test_fold_global_for_backend_keeps_the_compiled_layout():
    """Re-basing must not replace the compile's TranspileLayout.

    The folded circuit runs on the same physical qubits with the same net
    routing permutation, so the original layout still describes it; the
    identity layout the re-basing pass manager stamps on would not.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    from qbalance.strategies import StrategySpec
    from qbalance.transpile.pipeline import compile_one

    backend = GenericBackendV2(num_qubits=5, seed=0)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    compiled, _ = compile_one(qc, backend, StrategySpec(optimization_level=2))
    assert compiled.layout is not None

    folded = zne.fold_global_for_backend(compiled, backend, 3.0)

    assert folded.layout is compiled.layout


def test_mthree_calibrates_generic_backends_on_their_aer_equivalent():
    """Regression: M3 always failed on ``fake:generic`` backends.

    mthree reads ``backend.configuration()``, which ``GenericBackendV2`` lacks.
    Its runs use ``NoiseModel.from_backend`` on Aer, so the Aer equivalent
    carries the same readout channel; any other backend passes through.
    """
    pytest.importorskip("qiskit_aer")
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_aer import AerSimulator

    generic = GenericBackendV2(num_qubits=3, seed=0)
    system = mthree_mod._calibration_system(generic)
    assert isinstance(system, AerSimulator)
    assert callable(system.configuration)

    with_configuration = types.SimpleNamespace(configuration=lambda: None)
    assert mthree_mod._calibration_system(with_configuration) is with_configuration
    other = object()
    assert mthree_mod._calibration_system(other) is other


def test_mthree_mitigation_runs_on_a_generic_backend():
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=3, seed=0)
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    compiled = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
    counts = backend.run(compiled, shots=4000, seed_simulator=1).result().get_counts()

    probs = mthree_mod.apply_mthree_mitigation(
        backend, counts, measured_qubits=[0, 1], calibration_shots=4000
    )

    assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0
    assert probs["11"] >= counts.get("11", 0) / 4000


def _install_fake_mthree(monkeypatch, calls):
    """Install a linear fake M3 whose correction swaps nothing but records calls."""
    mthree = types.ModuleType("mthree")
    classes = types.ModuleType("mthree.classes")

    class Mit:
        def __init__(self, system):

            calls.append(("init", system))

        def cals_from_system(self, qubits, shots):

            calls.append(("cals", list(qubits), shots))

        def apply_correction(self, counts, qubits):

            calls.append(("correct", dict(counts)))
            total = sum(counts.values())
            return {key: value / total for key, value in counts.items()}

    class Quasi(dict):
        def __init__(self, data, shots=None):

            super().__init__(data)
            calls.append(("quasi", dict(data), shots))

        def nearest_probability_distribution(self):

            return dict(self)

    mthree.M3Mitigation = Mit
    classes.QuasiDistribution = Quasi
    mthree.classes = classes
    monkeypatch.setitem(sys.modules, "mthree", mthree)
    monkeypatch.setitem(sys.modules, "mthree.classes", classes)


def test_mitigate_twirled_counts_corrects_each_instance_before_untwirling(
    monkeypatch,
):
    """Each instance is corrected as measured, then untwirled with its own map."""
    calls: list = []
    _install_fake_mthree(monkeypatch, calls)
    backend = types.SimpleNamespace(configuration=lambda: None)

    probs = mthree_mod.mitigate_twirled_counts(
        backend,
        [{"0": 30, "1": 10}, {"0": 5, "1": 15}],
        [{0: 1}, {}],
        measured_qubits=[4],
        calibration_shots=123,
    )

    # Calibrated once, on the backend itself, for the measured qubit.
    assert [c for c in calls if c[0] in ("init", "cals")] == [
        ("init", backend),
        ("cals", [4], 123),
    ]
    # Both instances were corrected on their raw counts.
    assert [c[1] for c in calls if c[0] == "correct"] == [
        {"0": 30, "1": 10},
        {"0": 5, "1": 15},
    ]
    # Instance 0 (flipped) contributes 0.75 to "1"; instance 1 contributes
    # 0.75 to "1" as well; each weighs 1/2 by shots.
    assert probs == pytest.approx({"1": 0.75, "0": 0.25})

    with pytest.raises(ValueError, match="same length"):
        mthree_mod.mitigate_twirled_counts(backend, [{"0": 1}], [], [0])
    with pytest.raises(ValueError, match="at least one shot"):
        mthree_mod.mitigate_twirled_counts(backend, [{"0": 0}], [{}], [0])


def test_mitigate_twirled_counts_recovers_asymmetric_readout():
    """End to end: twirled instances with 20% 1->0 readout error, true 50/50."""
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError

    noise = NoiseModel()
    noise.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.20, 0.80]]))
    sim = AerSimulator(noise_model=noise)

    raw, flips = [], []
    for flip in (1, 0, 1, 0):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        if flip:
            qc.x(0)
        qc.measure(0, 0)
        raw.append(sim.run(qc, shots=50_000, seed_simulator=11).result().get_counts())
        flips.append({0: 1} if flip else {})

    probs = mthree_mod.mitigate_twirled_counts(
        sim, raw, flips, measured_qubits=[0], calibration_shots=100_000
    )

    assert probs["0"] == pytest.approx(0.5, abs=0.02)
    assert probs["1"] == pytest.approx(0.5, abs=0.02)


def test_mthree_handles_counts_from_several_classical_registers():
    """Regression: M3 rejected every circuit with more than one register.

    Qiskit separates registers with spaces in count keys ("10 1"), which
    mthree read as an extra bit: "Bitstring length (4) does not match number
    of qubits (3)".  Separators are stripped for the correction and restored
    on the result, so mitigated keys match the counts they came from.
    """
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError

    noise = NoiseModel()
    noise.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.10, 0.90]]))
    sim = AerSimulator(noise_model=noise)
    qubits = QuantumRegister(3)
    first, second = ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
    qc = QuantumCircuit(qubits, first, second)
    qc.x(0)
    qc.x(2)
    qc.measure(qubits[0], first[0])
    qc.measure(qubits[1], second[0])
    qc.measure(qubits[2], second[1])
    raw = sim.run(qc, shots=20_000, seed_simulator=5).result().get_counts()
    assert all(" " in key for key in raw)

    legacy = mthree_mod.apply_mthree_mitigation(
        sim, raw, [0, 1, 2], calibration_shots=50_000
    )
    twirled = mthree_mod.mitigate_twirled_counts(
        sim, [raw], [{}], measured_qubits=[0, 1, 2], calibration_shots=50_000
    )

    for probs in (legacy, twirled):
        assert all(" " in key for key in probs)
        # Measured x on q0 (register a) and q2 (register b, bit 1).
        assert probs.get("10 1", 0.0) == pytest.approx(1.0, abs=0.02)
        assert raw["10 1"] / 20_000 < 0.85

    assert mthree_mod._with_separators("101", "10 1") == "10 1"
    assert mthree_mod._with_separators("101", None) == "101"
    assert mthree_mod._without_separators({"10 1": 2, "101": 1}) == {"101": 3}


def test_mthree_skips_classical_bits_no_measurement_writes():
    """Regression: M3 failed whenever a classical bit was never measured.

    ``QuantumCircuit(3, 3)`` measuring two qubits yields three-bit count keys
    for two measured qubits, which mthree rejects ("Bitstring length (3) does
    not match number of qubits (2)") -- so every M3 candidate on such a
    circuit failed.  Keys are now projected onto the measured bits and
    expanded back, with the unwritten bits restored as ``0``.
    """
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError

    from qbalance.utils import measured_clbits, measured_qubits_by_clbit

    noise = NoiseModel()
    noise.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.10, 0.90]]))
    sim = AerSimulator(noise_model=noise)
    qc = QuantumCircuit(4, 4)
    qc.x(0)
    qc.x(2)
    qc.measure(0, 1)  # clbits 0 and 2 are never written
    qc.measure(2, 3)
    raw = sim.run(qc, shots=20_000, seed_simulator=5).result().get_counts()

    assert measured_qubits_by_clbit(qc) == [0, 2]
    assert measured_clbits(qc) == [1, 3]
    for probs in (
        mthree_mod.apply_mthree_mitigation(
            sim, raw, [0, 2], calibration_shots=50_000, clbits=[1, 3]
        ),
        mthree_mod.mitigate_twirled_counts(
            sim,
            [raw],
            [{}],
            measured_qubits=[0, 2],
            calibration_shots=50_000,
            clbits=[1, 3],
        ),
    ):
        assert probs.get("1010", 0.0) == pytest.approx(1.0, abs=0.02)
        assert all(key[1] == "0" and key[3] == "0" for key in probs)
    assert raw["1010"] / 20_000 < 0.85

    with pytest.raises(ValueError, match="do not fit"):
        mthree_mod.apply_mthree_mitigation(sim, raw, [0, 2], clbits=[1, 4])


def test_mthree_key_layout_projects_and_expands_measured_bits():
    layout = mthree_mod._KeyLayout({"10 01": 1}, clbits=[0, 3])
    # Key bits, classical bit 0 rightmost: c3=1 c2=0 | c1=0 c0=1.
    assert layout.project({"10 01": 5, "10 00": 2}) == {"11": 5, "10": 2}
    assert layout.expand("11") == "10 01"
    assert layout.expand("01") == "00 01"
    unmapped = mthree_mod._KeyLayout({"101": 1}, clbits=None)
    assert unmapped.project({"101": 3}) == {"101": 3}
    assert unmapped.expand("101") == "101"


def test_parity_expectation_value_matches_the_zne_observable():
    assert zne.parity_expectation_value({"00": 3, "11": 1}) == 1.0
    assert zne.parity_expectation_value({"00": 3, "01": 1}) == pytest.approx(0.5)
    assert zne.parity_expectation_value({"0 1": 0.25, "1 1": 0.75}) == pytest.approx(
        0.5
    )
    with pytest.raises(ValueError, match="positive total weight"):
        zne.parity_expectation_value({})


def test_mthree_on_a_generic_backend_is_reproducible_for_a_seed():
    """The calibration simulator qbalance creates takes the run's seed."""
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(num_qubits=3, seed=0)
    assert mthree_mod._calibration_system(backend, seed=5).options.seed_simulator == 5
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    compiled = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
    counts = backend.run(compiled, shots=2000, seed_simulator=1).result().get_counts()

    def mitigate():

        return mthree_mod.mitigate_twirled_counts(
            backend, [counts], [{}], [0, 1], calibration_shots=2000, seed=5
        )

    assert mitigate() == mitigate()


class _RecordingSimulator:
    """A backend stub with mthree's view of a simulator, recording run options."""

    name = "recording_simulator"
    version = 2

    def __init__(self, simulator=True, name=None):
        self._simulator = simulator
        if name is not None:
            self.name = name
        self.options_seen = []

    def configuration(self):
        return types.SimpleNamespace(simulator=self._simulator)

    def run(self, run_input, **options):
        self.options_seen.append(options)
        return run_input


def test_mthree_seeds_the_calibration_of_every_local_simulator():
    """Regression: only ``fake:generic`` calibrations were seeded.

    mthree runs its calibration circuits through ``backend.run`` without a
    seed, so on an Aer simulator or a ``fake:ibm`` snapshot the mitigated
    result changed from one identical run to the next.  Those are calibrated
    on themselves with every job seeded; the backend object itself is never
    modified, and hardware is passed through untouched.
    """
    simulator = _RecordingSimulator()
    system = mthree_mod._calibration_system(simulator, seed=11)
    assert system is not simulator
    # mthree reads these off the system it is given.
    assert system.name == "recording_simulator"
    assert system.version == 2
    assert system.configuration().simulator is True
    system.run(["circuit"], shots=10)
    assert simulator.options_seen == [{"shots": 10, "seed_simulator": 11}]

    # Device snapshots report the device's configuration but run on Aer.
    snapshot = _RecordingSimulator(simulator=False, name="fake_manila")
    mthree_mod._calibration_system(snapshot, seed=3).run(["circuit"])
    assert snapshot.options_seen == [{"seed_simulator": 3}]

    # Unseeded calls, hardware, and runs without a seed keyword pass through.
    assert mthree_mod._calibration_system(simulator) is simulator
    hardware = _RecordingSimulator(simulator=False, name="ibm_device")
    assert mthree_mod._calibration_system(hardware, seed=3) is hardware
    no_seed = types.SimpleNamespace(
        configuration=lambda: types.SimpleNamespace(simulator=True),
        run=lambda run_input, shots=None: run_input,
    )
    assert mthree_mod._calibration_system(no_seed, seed=3) is no_seed


@pytest.mark.parametrize("spec", ["aer:from_backend:fake:generic:3", "fake:ibm:manila"])
def test_mthree_is_reproducible_for_a_seed_on_any_simulator(spec):
    """The same seed gives the same mitigated distribution on every simulator."""
    pytest.importorskip("mthree")
    pytest.importorskip("qiskit_aer")
    if spec.startswith("fake:ibm"):
        pytest.importorskip("qiskit_ibm_runtime")
    from qiskit import QuantumCircuit, transpile

    from qbalance.backends import resolve_backend

    backend = resolve_backend(spec)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    compiled = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
    counts = backend.run(compiled, shots=2000, seed_simulator=1).result().get_counts()

    def mitigate():

        return mthree_mod.mitigate_twirled_counts(
            backend, [counts], [{}], [0, 1], calibration_shots=2000, seed=5
        )

    assert mitigate() == mitigate()


def test_fold_global_runs_leading_resets_once_ahead_of_the_folded_unitary():
    """Regression: a reset preamble made every circuit using it unfoldable.

    Qiskit keeps a circuit's initial resets at every optimization level, and
    ``reset`` has no inverse, so ZNE failed outright on the common
    ``reset(range(n))`` preamble.  A reset before anything else acts on its
    qubit only re-prepares ``|0>``; it runs once, ahead of the folded unitary,
    just as terminal measurements run once after it.
    """
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.exceptions import CircuitError
    from qiskit.quantum_info import DensityMatrix

    qc = QuantumCircuit(2, 2, global_phase=0.3)
    qc.reset([0, 1])
    qc.barrier()
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.4, 1)
    qc.measure([0, 1], [0, 1])

    folded = zne.fold_global(qc, 3)

    names = [instruction.operation.name for instruction in folded.data]
    assert names[:2] == ["reset", "reset"]
    assert names.count("reset") == 2
    assert names[-2:] == ["measure", "measure"]
    assert folded.count_ops()["cx"] == 3
    assert folded.global_phase == pytest.approx(qc.global_phase)

    def probabilities(circuit):
        unmeasured = circuit.remove_final_measurements(inplace=False)
        return DensityMatrix.from_instruction(unmeasured).probabilities_dict()

    expected, actual = probabilities(qc), probabilities(folded)
    for key in set(expected) | set(actual):
        assert actual.get(key, 0.0) == pytest.approx(expected.get(key, 0.0), abs=1e-9)

    # A reset after another operation on its qubit is part of the computation
    # and still cannot be folded.
    mid = QuantumCircuit(1, 1)
    mid.h(0)
    mid.reset(0)
    mid.x(0)
    mid.measure(0, 0)
    with pytest.raises(CircuitError, match="reset"):
        zne.fold_global(mid, 3)


def test_zne_runs_on_compiled_circuits_with_a_reset_preamble():
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    from qbalance.execution.ensemble import run_ensemble
    from qbalance.strategies import StrategySpec
    from qbalance.transpile.pipeline import compile_ensemble

    backend = GenericBackendV2(num_qubits=3, seed=0)
    qc = QuantumCircuit(2, 2)
    qc.reset([0, 1])
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    spec = StrategySpec(zne=True)
    instances, metrics = compile_ensemble(qc, backend, spec)
    assert instances[0].count_ops()["reset"] == 2

    run = run_ensemble(
        backend, instances, metrics, shots=500, seed=1, zne_factors=spec.zne_factors
    )

    assert run.zne_error is None
    assert run.zne_realized_factors == [1.0, 3.0, 5.0]
    assert run.zne_probs
