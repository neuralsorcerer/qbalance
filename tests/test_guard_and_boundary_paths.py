# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Guard clauses and boundary cases that carry no other coverage.

Most tests here pin a branch that mutation testing showed the suite could
not distinguish from its opposite: validation guards that never saw a
rejected input, fast paths never exercised at their exact boundary, and
defaults nothing depended on.  The rest document a contract that mutation
testing cannot reach at all, noted individually.
"""

from __future__ import annotations

import json
import types

import pytest

from qbalance.dataset import _MAX_ARTIFACT_STEM, _sanitize_artifact_stem
from qbalance.diagnostics.distribution import weighted_cdf
from qbalance.mitigation.zne import _counts_to_expval_z, zne_extrapolate_counts
from qbalance.reports.common import matrix_results
from qbalance.search.pareto import pareto_front
from qbalance.strategies import _strategy_items_from_payload
from qbalance.transpile.pipeline import _backend_basis_gates
from qbalance.transpile.suppression import apply_measurement_untwirl_counts
from qbalance.utils import atomic_write_bytes, dump_json
from qbalance.workflow.workload import _entropy_from_counts


def test_backend_basis_gates_reads_the_target_operation_names():
    """A target's operation names are the basis gates.

    Returning ``None`` here does not fail; it silently drops the basis-gate
    constraint and lets translation pick its own, so the extraction has to be
    pinned directly.
    """
    target = types.SimpleNamespace(operation_names=["cx", "x", "  ", "rz"])

    assert _backend_basis_gates(object(), target) == ["cx", "rz", "x"]


def test_backend_basis_gates_returns_none_for_an_empty_target():
    """An empty name list must read as "no constraint", not an empty basis."""
    target = types.SimpleNamespace(operation_names=[])

    assert _backend_basis_gates(object(), target) is None


def test_counts_to_expval_z_rejects_non_integer_counts():
    """Booleans and floats are not shot counts.

    ``bool`` is a subclass of ``int``, so the explicit bool check is the only
    thing standing between ``{"0": True}`` and a silently wrong expectation
    value.
    """
    with pytest.raises(ValueError, match="non-negative integers"):
        _counts_to_expval_z({"0": True}, validate=True)

    with pytest.raises(ValueError, match="non-negative integers"):
        _counts_to_expval_z({"0": 1.5}, validate=True)

    with pytest.raises(ValueError, match="non-negative integers"):
        _counts_to_expval_z({"0": -1}, validate=True)


def test_zne_extrapolate_accepts_a_zero_degree_fit():
    """degree=0 is a constant fit, not an invalid degree.

    The distinct-values requirement is explicitly skipped for degree 0, so the
    degree guard must reject only negative degrees.
    """
    out = zne_extrapolate_counts(
        [1.0, 3.0], [{"0": 8, "1": 2}, {"0": 6, "1": 4}], degree=0
    )

    # A constant fit extrapolates to the mean expectation value (0.6, 0.2) ->
    # 0.4, which puts 0.7 of the mass on the even-parity key.
    assert out == {"0": pytest.approx(0.7), "1": pytest.approx(0.3)}


def test_zne_extrapolate_rejects_a_negative_degree():
    with pytest.raises(ValueError, match="non-negative integer"):
        zne_extrapolate_counts([1.0], [{"0": 1}], degree=-1)


def test_pareto_front_keeps_candidates_that_tie_on_every_key():
    """Equal points do not dominate each other.

    Domination needs "no worse on all keys AND strictly better on one".  If a
    tie is ever read as strict improvement, whichever candidate is compared
    second silently evicts the first.

    ``pareto_front`` deduplicates metric vectors before the dominance loop, so
    no mutation of that loop can change this result -- the guarantee rests on
    the deduplication, and this test is what pins it.
    """
    items = [
        ("a", {"depth": 1.0, "err": 2.0}),
        ("b", {"depth": 1.0, "err": 2.0}),
    ]

    assert pareto_front(items, ["depth", "err"]) == [0, 1]


def test_pareto_front_still_drops_a_strictly_dominated_candidate():
    """The tie rule must not cost real domination."""
    items = [
        ("a", {"depth": 1.0, "err": 2.0}),
        ("b", {"depth": 2.0, "err": 3.0}),
    ]

    assert pareto_front(items, ["depth", "err"]) == [0]


def test_entropy_ignores_outcomes_that_were_never_observed():
    """A zero count contributes no entropy.

    ``0 * log2(0)`` is ``nan``, so a single unobserved key would poison the
    metric for the whole distribution.
    """
    assert _entropy_from_counts({"0": 10, "1": 0}) == 0.0
    assert _entropy_from_counts({"0": 5, "1": 5}) == pytest.approx(1.0)


def test_sanitize_artifact_stem_keeps_a_stem_of_exactly_the_maximum_length():
    """The length cap is inclusive; shortening at the boundary is wrong."""
    stem = "a" * _MAX_ARTIFACT_STEM

    assert _sanitize_artifact_stem(stem, fallback="fallback") == stem


def test_sanitize_artifact_stem_shortens_one_character_past_the_maximum():
    stem = "a" * (_MAX_ARTIFACT_STEM + 1)

    shortened = _sanitize_artifact_stem(stem, fallback="fallback")

    assert shortened != stem
    assert len(shortened) <= _MAX_ARTIFACT_STEM


def test_atomic_write_creates_missing_parent_directories(tmp_path):
    """Callers pass nested cache paths that do not exist yet."""
    target = tmp_path / "deep" / "nested" / "payload.bin"

    atomic_write_bytes(target, b"data")

    assert target.read_bytes() == b"data"


def test_dump_json_sorts_keys(tmp_path):
    """Stable key order keeps rewritten artifacts diffable and reproducible."""
    path = tmp_path / "out.json"

    dump_json(path, {"zeta": 1, "alpha": 2, "mu": 3})

    text = path.read_text(encoding="utf-8")
    assert text.index('"alpha"') < text.index('"mu"') < text.index('"zeta"')
    assert json.loads(text) == {"alpha": 2, "mu": 3, "zeta": 1}


def test_matrix_results_rejects_an_empty_backend_name():
    """An empty backend name is as unusable as a missing one."""
    payload = {
        "results": [
            {"backend": "", "strategy": {"optimization_level": 1}, "metrics": {}}
        ]
    }

    with pytest.raises(ValueError, match="invalid 'backend'"):
        matrix_results(payload)


def test_strategy_items_falls_through_to_a_bare_spec_payload():
    """A payload that is itself a spec has neither 'spec' nor 'strategy'."""
    payload = {"optimization_level": 2, "layout_method": "sabre"}

    assert _strategy_items_from_payload(payload) == [payload]


def test_weighted_cdf_aggregates_a_two_point_support_of_equal_values():
    """Two identical samples are one support point.

    The "fewer than two points" fast path skips duplicate aggregation, so it
    must not swallow the smallest input that actually has a duplicate.
    """
    xs, cdf = weighted_cdf([1.0, 1.0])

    assert xs.tolist() == [1.0]
    assert cdf.tolist() == [pytest.approx(1.0)]


def test_untwirl_flips_the_highest_classical_bit():
    """Clbit indices are little-endian, so the last one is the leftmost char.

    That maps to offset zero in the display string -- the exact edge of the
    in-range check, and the one position an off-by-one guard would skip.
    """
    counts = {"000": 7}

    assert apply_measurement_untwirl_counts(counts, {2: 1}) == {"100": 7}
    assert apply_measurement_untwirl_counts(counts, {0: 1}) == {"001": 7}


def test_untwirl_ignores_a_classical_bit_beyond_the_key_width():
    """An out-of-range clbit must be skipped, not wrapped around."""
    assert apply_measurement_untwirl_counts({"000": 7}, {9: 1}) == {"000": 7}


def test_untwirl_preserves_register_separators():
    """Count keys with multiple registers carry spaces that are not bits."""
    assert apply_measurement_untwirl_counts({"00 0": 4}, {0: 1}) == {"00 1": 4}


def test_stage_pass_manager_adds_layout_passes_only_when_given_a_layout():
    """The fallback pipeline installs a layout only when it has both parts.

    ``SetLayout``/``ApplyLayout`` need an initial layout *and* a target to
    allocate ancillas against.  Appending them without a layout would apply a
    null layout; skipping them when one was supplied silently discards the
    noise-aware placement that was just computed.
    """
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import Layout

    from qbalance.strategies import StrategySpec
    from qbalance.transpile.pipeline import _generate_stage_pm

    backend = GenericBackendV2(num_qubits=3, seed=1)
    circuit = QuantumCircuit(2)
    layout = Layout.from_intlist([0, 1], *circuit.qregs)

    def pass_names(pm):

        # Qiskit 2.x exposes the schedule through the flow controller, whose
        # tasks nest; leaves are the individual passes.
        found = []

        def walk(node):

            tasks = getattr(node, "tasks", None)
            if tasks is None:
                found.append(type(node).__name__)
                return
            for task in tasks:
                walk(task)

        walk(pm.to_flow_controller())
        return found

    with_layout = pass_names(_generate_stage_pm(backend, StrategySpec(), layout))
    without_layout = pass_names(_generate_stage_pm(backend, StrategySpec()))

    assert "SetLayout" in with_layout
    assert "ApplyLayout" in with_layout
    assert "SetLayout" not in without_layout
    assert "ApplyLayout" not in without_layout


def test_zne_extrapolate_rejects_a_non_integer_degree():
    """Degree is a polynomial order, so bools and floats are not degrees.

    ``bool`` is a subclass of ``int``, so the explicit bool check is the only
    thing that keeps ``degree=True`` from being read as a linear fit.
    """
    for bad in (True, 1.5, "1"):
        with pytest.raises(ValueError, match="non-negative integer"):
            zne_extrapolate_counts([1.0, 2.0], [{"0": 1}, {"0": 1}], degree=bad)


def test_fold_global_rounds_the_scale_up_to_an_odd_factor():
    """Global folding builds U (U-dagger U)^r, so the factor must be odd.

    ``reps`` is ``(k - 1) // 2``.  An even ``k`` therefore folds one
    repetition short -- k=2 collapses to no folding at all -- which scales the
    noise by the wrong amount and quietly biases the extrapolation instead of
    failing.
    """
    from qiskit import QuantumCircuit

    from qbalance.mitigation.zne import fold_global

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    base_ops = len(circuit.data)

    for scale, factor in ((2.0, 3), (3.0, 3), (4.0, 5)):
        folded = fold_global(circuit, scale)
        assert folded.name.endswith(f"_fold{factor}")
        assert len(folded.data) == factor * base_ops

    # scale 1.0 is a no-op and returns the circuit itself.
    assert fold_global(circuit, 1.0) is circuit


def test_rebase_skips_a_circuit_whose_width_differs_from_the_backend():
    """Re-basing needs an identity layout, which needs matching widths.

    ``initial_layout=range(circuit.num_qubits)`` only describes the backend
    when the two agree, so a mismatch has to be left alone rather than
    silently re-laid-out onto different physical qubits.
    """
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2

    from qbalance.mitigation.zne import _rebase_to_backend

    backend = GenericBackendV2(num_qubits=5, seed=1)
    narrow = QuantumCircuit(2)
    narrow.h(0)

    assert _rebase_to_backend(narrow, backend) is narrow


def test_fold_global_rejects_a_measurement_that_is_not_terminal():
    """Folding replays the circuit, so a re-used measurement bit is not safe.

    A measurement is terminal only if nothing afterwards touches either the
    qubit it measured or the clbit it wrote.  Requiring both to be re-used
    before objecting would let ``U (U-dagger U)^r`` replay a mid-circuit
    measurement and quietly produce a different computation.
    """
    from qiskit import QuantumCircuit

    from qbalance.mitigation.zne import fold_global

    # The clbit is written twice, from different qubits.
    shared_clbit = QuantumCircuit(2, 1)
    shared_clbit.h(0)
    shared_clbit.measure(0, 0)
    shared_clbit.measure(1, 0)

    # The qubit is measured twice, into different clbits.
    shared_qubit = QuantumCircuit(1, 2)
    shared_qubit.h(0)
    shared_qubit.measure(0, 0)
    shared_qubit.measure(0, 1)

    for circuit in (shared_clbit, shared_qubit):
        with pytest.raises(ValueError, match="all measurements are terminal"):
            fold_global(circuit, 3.0)
