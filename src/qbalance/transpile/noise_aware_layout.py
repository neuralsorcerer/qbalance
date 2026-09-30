# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from qbalance.logging import get_logger
from qbalance.utils import SCHEDULING_DIRECTIVES, bit_index, instruction_parts

log = get_logger(__name__)


_DIRECTIVE_NAMES = SCHEDULING_DIRECTIVES


def _backend_target(backend: Any) -> Any:
    """Return the BackendV2 transpiler target when one is exposed."""
    return getattr(backend, "target", None)


def _target_instruction_error(
    target: Any, name: str, qubits: tuple[int, ...]
) -> Optional[float]:
    """Read an instruction error rate from a BackendV2 target, best effort."""
    if target is None:
        return None
    try:
        properties_map = target[name]
        props = properties_map.get(qubits)
        if props is None:
            return None
        return _coerce_error_rate(getattr(props, "error", None))
    except Exception:
        return None


def _target_qubit_property(target: Any, q: int, attr: str) -> Optional[float]:
    """Read a per-qubit property (e.g. t1/t2) from a BackendV2 target."""
    if target is None:
        return None
    try:
        qubit_properties = getattr(target, "qubit_properties", None)
        if qubit_properties is None:
            return None
        return _coerce_finite_float(getattr(qubit_properties[q], attr, None))
    except Exception:
        return None


def _safe_get_qubit_readout_error(backend: Any, q: int) -> Optional[float]:
    """Return qubit ``q``'s calibrated readout error, clipped to [0, 1], or ``None``.

    The target's ``measure`` error is preferred, then a legacy
    ``readout_error`` property.  ``None`` means no usable value, never an
    exception.
    """
    # BackendV2: readout error lives on the target's measure instruction.
    error = _target_instruction_error(_backend_target(backend), "measure", (q,))
    if error is not None:
        return error

    try:
        props = backend.properties()
        if props is None:
            return None
        # Qiskit properties: readout_error in qubit properties
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "readout_error":
                return _coerce_error_rate(item.value)
    except Exception:
        return None
    return None


def _safe_get_t1(backend: Any, q: int) -> Optional[float]:
    """Return qubit ``q``'s calibrated T1, or ``None`` when unavailable.

    Targets report seconds and legacy properties usually microseconds, so
    compare values only within one backend.
    """
    value = _target_qubit_property(_backend_target(backend), q, "t1")
    if value is not None:
        return value

    try:
        props = backend.properties()
        if props is None:
            return None
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "T1":
                return _coerce_finite_float(item.value)
    except Exception:
        return None
    return None


def _safe_get_t2(backend: Any, q: int) -> Optional[float]:
    """Return qubit ``q``'s calibrated T2, or ``None`` when unavailable.

    Targets report seconds and legacy properties usually microseconds, so
    compare values only within one backend.
    """
    value = _target_qubit_property(_backend_target(backend), q, "t2")
    if value is not None:
        return value

    try:
        props = backend.properties()
        if props is None:
            return None
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "T2":
                return _coerce_finite_float(item.value)
    except Exception:
        return None
    return None


def _safe_get_2q_error(backend: Any, gate: str, q0: int, q1: int) -> Optional[float]:
    """Return the calibrated error of ``gate`` on ``(q0, q1)``, or ``None``.

    The target is read in both qubit orders, since some list only one
    direction, before a legacy ``gate_error`` property.  The error is
    clipped to [0, 1]; ``None`` means no usable value, never an exception.
    """
    # BackendV2: gate errors live on the target, keyed by qubit tuple.  Try
    # the reversed direction too because some targets only list one direction.
    target = _backend_target(backend)
    error = _target_instruction_error(target, gate, (q0, q1))
    if error is None:
        error = _target_instruction_error(target, gate, (q1, q0))
    if error is not None:
        return error

    try:
        props = backend.properties()
        if props is None:
            return None
        g = props.gate_error(gate, [q0, q1])
        if g is None:
            return None
        return _coerce_error_rate(g)
    except Exception:
        return None


def _coerce_finite_float(value: Any) -> Optional[float]:
    """Return a finite float, or None when the value is not usable."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _coerce_error_rate(value: Any) -> Optional[float]:
    """Return a finite probability-like error rate clipped to [0, 1]."""
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(rate):
        return None
    return float(np.clip(rate, 0.0, 1.0))


def estimate_circuit_error(backend: Any, circuit: Any) -> float:
    """Return the probability that at least one instruction of ``circuit`` fails.

    That is ``1 - prod(1 - e_i)`` over the instructions, with ``e_i`` the
    backend's calibrated error for the instruction on its qubits.  An
    instruction without calibration data counts as 1e-3 (one qubit), 1e-2
    (two qubits) or 2e-2 (a measurement), and one on three or more qubits
    always as 1e-3.  Barriers and delays are not billed, and idle
    decoherence is not modelled.

    Args:
        backend: Backend whose calibration data prices the instructions.
        circuit: Circuit compiled for ``backend``.

    Returns:
        The estimate, in ``[0, 1]``; 1.0 when the circuit cannot be read.
    """
    # 1 - Π(1-e_i) approximation
    total_survival = 1.0
    try:
        target = _backend_target(backend)
        for entry in circuit.data:
            inst, qargs, _ = instruction_parts(entry)
            name = getattr(inst, "name", "").lower()
            if name in _DIRECTIVE_NAMES:
                # Barriers/delays are scheduling directives, not error channels;
                # a two-qubit barrier must not be billed as a two-qubit gate.
                continue
            if len(qargs) == 2:
                q0 = bit_index(circuit, qargs[0])
                q1 = bit_index(circuit, qargs[1])
                e = _safe_get_2q_error(backend, name, q0, q1)
                if e is None:
                    e = 0.01
                total_survival *= 1.0 - e
            elif name == "measure" and len(qargs) == 1:
                q0 = bit_index(circuit, qargs[0])
                e = _safe_get_qubit_readout_error(backend, q0)
                if e is None:
                    e = 0.02
                total_survival *= 1.0 - e
            elif len(qargs) == 1:
                # 1q gate errors: prefer target calibration, else 0.001
                q0 = bit_index(circuit, qargs[0])
                e = _target_instruction_error(target, name, (q0,))
                if e is None:
                    e = 0.001
                total_survival *= 1.0 - e
            elif len(qargs) > 0:
                # multi-qubit (>2) operations: conservative default
                total_survival *= 1.0 - 0.001
    except Exception:
        return 1.0
    if not np.isfinite(total_survival):
        return 1.0
    return float(np.clip(1.0 - total_survival, 0.0, 1.0))


# Weight of each median-normalized coherence time in the physical-qubit score.
# A qubit at twice the median T1 gains 0.001, the order of a 0.1% readout
# error difference, so coherence separates qubits of similar readout quality
# without overriding a real readout advantage.
_COHERENCE_WEIGHT = 1e-3


def _fill_missing_with_median(
    values: list[Optional[float]], *, fallback: float
) -> np.ndarray:
    """Replace missing entries with the median of the reported ones."""
    reported = [v for v in values if v is not None]
    neutral = float(np.median(reported)) if reported else fallback
    return np.asarray([neutral if v is None else v for v in values], dtype=float)


def _relative_to_median(values: list[Optional[float]]) -> np.ndarray:
    """Return values divided by their median; missing or unusable ones map to 1."""
    reported = [v for v in values if v is not None and v > 0.0]
    if not reported:
        return np.ones(len(values), dtype=float)
    median = float(np.median(reported))
    return np.asarray(
        [v / median if v is not None and v > 0.0 else 1.0 for v in values],
        dtype=float,
    )


# A qubit pair that is not coupled is bridged by routing: one SWAP -- three
# two-qubit gates -- per extra hop, per interaction.
_SWAP_GATE_COUNT = 3.0
# Placement is grown from this many of the most promising start qubits.
_MAX_LAYOUT_STARTS = 24
_DEFAULT_EDGE_ERROR = 0.01
# IBM calibration snapshots report an out-of-service coupler with error 1.0.
_BROKEN_EDGE_ERROR = 1.0
# Cost of an interaction between qubits no usable coupler path connects.
_UNREACHABLE_COST = 1e6


def _coupling_adjacency(backend: Any, num_physical: int) -> Optional[List[Set[int]]]:
    """Undirected adjacency of the backend's coupling map, or None if unconstrained."""
    cmap = getattr(backend, "coupling_map", None)
    if cmap is None:
        build = getattr(_backend_target(backend), "build_coupling_map", None)
        if callable(build):
            try:
                cmap = build()
            except Exception:
                cmap = None
    if cmap is None:
        return None
    get_edges = getattr(cmap, "get_edges", None)
    try:
        edges = list(get_edges() if callable(get_edges) else cmap)
    except Exception:
        return None
    adjacency: List[Set[int]] = [set() for _ in range(num_physical)]
    for edge in edges:
        a, b = (int(q) for q in edge)
        if a != b and 0 <= a < num_physical and 0 <= b < num_physical:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency if any(adjacency) else None


# Hop distances depend on the coupling graph alone, and every compile for one
# backend -- each candidate, each cut subexperiment -- shares it.
_HOP_DISTANCE_CACHE: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], np.ndarray] = {}
_HOP_DISTANCE_CACHE_SIZE = 16


def _hop_distances(adjacency: List[Set[int]]) -> np.ndarray:
    """All-pairs hop distance on the coupling graph (inf when disconnected).

    The result is cached per graph and returned read-only.
    """
    key = (
        len(adjacency),
        tuple(
            sorted(
                (a, b)
                for a, neighbors in enumerate(adjacency)
                for b in neighbors
                if a < b
            )
        ),
    )
    cached = _HOP_DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    dist = _compute_hop_distances(adjacency)
    dist.setflags(write=False)
    if len(_HOP_DISTANCE_CACHE) >= _HOP_DISTANCE_CACHE_SIZE:
        _HOP_DISTANCE_CACHE.pop(next(iter(_HOP_DISTANCE_CACHE)))
    _HOP_DISTANCE_CACHE[key] = dist
    return dist


def _compute_hop_distances(adjacency: List[Set[int]]) -> np.ndarray:
    """Breadth-first all-pairs hop distance (inf when disconnected)."""
    size = len(adjacency)
    dist = np.full((size, size), np.inf)
    for source in range(size):
        dist[source, source] = 0.0
        frontier = [source]
        hops = 0.0
        while frontier:
            hops += 1.0
            following = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if dist[source, neighbor] == np.inf:
                        dist[source, neighbor] = hops
                        following.append(neighbor)
            frontier = following
    return dist


def _edge_errors(
    backend: Any, adjacency: List[Set[int]]
) -> Dict[Tuple[int, int], float]:
    """Lowest calibrated two-qubit gate error per coupled pair (either direction)."""
    target = _backend_target(backend)
    names: List[str] = []
    if target is not None:
        for name in getattr(target, "operation_names", ()):
            try:
                operation = target.operation_from_name(name)
            except Exception:
                continue
            if getattr(operation, "num_qubits", None) == 2:
                names.append(str(name))
    errors: Dict[Tuple[int, int], float] = {}
    for a, neighbors in enumerate(adjacency):
        for b in neighbors:
            if b < a:
                continue
            found = [
                error
                for name in names
                for pair in ((a, b), (b, a))
                for error in [_target_instruction_error(target, name, pair)]
                if error is not None
            ]
            if found:
                errors[(a, b)] = min(found)
    return errors


def _usable_adjacency(
    adjacency: List[Set[int]], edge_errors: Dict[Tuple[int, int], float]
) -> List[Set[int]]:
    """Return ``adjacency`` without the couplers calibrated as broken.

    A broken coupler cannot carry an interaction, so it must not count as a
    connection either: otherwise two qubits across it look adjacent, and a
    pair two hops apart through it looks one SWAP away.  When every coupler is
    broken there is nothing better to plan with, and ``adjacency`` is kept.
    """
    usable = [set(neighbors) for neighbors in adjacency]
    for (a, b), error in edge_errors.items():
        if error >= _BROKEN_EDGE_ERROR:
            usable[a].discard(b)
            usable[b].discard(a)
    return usable if any(usable) else adjacency


def _logical_interactions(
    circuit: Any, n: int
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """Interaction degree per logical qubit and weight per interacting pair.

    The layout is chosen for the circuit as written, before transpilation
    decomposes its gates, so a gate on three or more qubits (a Toffoli, say)
    is still whole here.  It decomposes into two-qubit gates among its qubits,
    so every pair of its qubits interacts; counting only literal two-qubit
    gates made such circuits look interaction-free and scattered them.
    """
    deg = np.zeros(n, dtype=float)
    weights: Dict[Tuple[int, int], float] = {}
    for entry in circuit.data:
        inst, qargs, _ = instruction_parts(entry)
        if getattr(inst, "name", "").lower() in _DIRECTIVE_NAMES:
            continue
        if len(qargs) < 2:
            continue
        qubits = sorted({bit_index(circuit, qubit) for qubit in qargs})
        for position, a in enumerate(qubits):
            deg[a] += len(qubits) - 1
            for b in qubits[position + 1 :]:
                weights[(a, b)] = weights.get((a, b), 0.0) + 1.0
    return deg, weights


def _placement_order(
    n: int, deg: np.ndarray, weights: Dict[Tuple[int, int], float]
) -> List[int]:
    """Busiest logical qubit first, then always the one most tied to those placed."""
    partners: List[Dict[int, float]] = [dict() for _ in range(n)]
    for (a, b), weight in weights.items():
        partners[a][b] = weight
        partners[b][a] = weight
    order: List[int] = []
    remaining = set(range(n))
    while remaining:
        placed = set(order)
        chosen = max(
            remaining,
            key=lambda q: (
                sum(w for k, w in partners[q].items() if k in placed),
                deg[q],
                -q,
            ),
        )
        order.append(chosen)
        remaining.discard(chosen)
    return order


def _connected_placement(
    n: int,
    deg: np.ndarray,
    weights: Dict[Tuple[int, int], float],
    node_error: np.ndarray,
    adjacency: List[Set[int]],
    edge_errors: Dict[Tuple[int, int], float],
) -> Optional[List[int]]:
    """Greedy placement that grows along the coupling graph.

    Logical qubits are placed in :func:`_placement_order`.  Each goes to the
    free physical qubit that needs the fewest routing hops to the partners
    already placed -- adjacent to them whenever a free neighbour exists --
    and, among those, the one with the lowest cost in error units: its
    readout error plus, per interaction, the pair's calibrated gate error, or
    for an uncoupled pair ``h`` hops apart, one typical gate error, three
    (a SWAP) per extra hop, and the readout error routing adds by moving one
    of the pair onto a qubit of the path.  Routing relocates a qubit, so
    skipping a poor neighbour to reach a better qubit beyond it saves
    nothing: the qubit is read out wherever the SWAPs leave it.  The
    placement is grown from several start qubits and the one with the lowest
    total cost wins.

    Args:
        n: Number of logical qubits.
        deg: Interaction degree per logical qubit.
        weights: Interaction count per logical qubit pair.
        node_error: Per-physical-qubit error score (readout, coherence-adjusted).
        adjacency: Usable coupling graph (see :func:`_usable_adjacency`).
        edge_errors: Lowest calibrated two-qubit gate error per coupled pair.

    Returns:
        The physical qubit for every logical qubit, or ``None`` when no start
        yields a complete placement.
    """
    if n == 0:
        return []
    size = len(adjacency)
    dist = _hop_distances(adjacency)
    usable_errors = [
        error for (a, b), error in edge_errors.items() if b in adjacency[a]
    ]
    typical = float(np.median(usable_errors)) if usable_errors else _DEFAULT_EDGE_ERROR

    def relocation(p: int, q: int, hops: float) -> float:
        # Routing brings one of the pair next to the other along a shortest
        # path, and that qubit is then read out where it ends up.
        best: Optional[float] = None
        for mover, fixed in ((p, q), (q, p)):
            for m in adjacency[fixed]:
                if dist[mover, m] == hops - 1.0:
                    delta = float(node_error[m]) - float(node_error[mover])
                    if best is None or delta < best:
                        best = delta
        return max(0.0, best) if best is not None else 0.0

    def pair_cost(p: int, q: int) -> float:
        hops = float(dist[p, q])
        if hops == 1.0:
            return edge_errors.get((min(p, q), max(p, q)), typical)
        if not np.isfinite(hops):
            return _UNREACHABLE_COST
        return (
            typical + (hops - 1.0) * _SWAP_GATE_COUNT * typical + relocation(p, q, hops)
        )

    # Routing hops beyond the first, per physical pair.  Every entry is an
    # integer (or the unreachable cost), so the weighted sums below are exact
    # whatever order they are summed in, and ties compare reliably.
    extra_hops = np.where(np.isfinite(dist), dist - 1.0, _UNREACHABLE_COST)

    partners: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for (a, b), weight in weights.items():
        partners[a].append((b, weight))
        partners[b].append((a, weight))
    order = _placement_order(n, deg, weights)

    def local_score(p: int) -> float:
        incident = sorted(
            edge_errors.get((min(p, q), max(p, q)), typical) for q in adjacency[p]
        )
        return float(node_error[p]) + (incident[0] if incident else _UNREACHABLE_COST)

    starts = sorted(range(size), key=lambda p: (local_score(p), p))[:_MAX_LAYOUT_STARTS]
    best: Optional[Tuple[float, List[int]]] = None
    for start in starts:
        placement = {order[0]: start}
        is_free = np.ones(size, dtype=bool)
        is_free[start] = False
        total = float(node_error[start])
        for logical in order[1:]:
            placed_partners = [
                (placement[k], w) for k, w in partners[logical] if k in placement
            ]
            free = np.flatnonzero(is_free)
            if free.size == 0:
                break
            # Routing hops decide first, so the error cost -- the expensive
            # part -- is only needed for the qubits tied at the fewest hops.
            if placed_partners:
                hops = extra_hops[
                    np.ix_(free, [other for other, _ in placed_partners])
                ] @ (np.asarray([w for _, w in placed_partners], dtype=float))
            else:
                hops = np.zeros(free.size)
            candidate = -1
            candidate_cost: Optional[float] = None
            for physical in free[hops == hops.min()]:
                physical = int(physical)
                cost = float(node_error[physical]) + sum(
                    w * pair_cost(physical, other) for other, w in placed_partners
                )
                if candidate_cost is None or cost < candidate_cost:
                    candidate_cost, candidate = cost, physical
            assert candidate_cost is not None  # some free qubit has the fewest hops
            placement[logical] = candidate
            is_free[candidate] = False
            total += candidate_cost
        if len(placement) != n:
            continue
        if best is None or total < best[0]:
            best = (total, [placement[q] for q in range(n)])
    return None if best is None else best[1]


def noise_aware_initial_layout(backend: Any, circuit: Any) -> Optional[Any]:
    """Choose an initial layout from calibration data and the coupling map.

    Physical qubits are scored by readout error, with T1/T2 (relative to the
    backend median) as a tie-breaker.  When the backend has a coupling map,
    logical qubits are placed greedily along it -- busiest first, each next to
    the partners it interacts with whenever a free neighbour exists -- over
    the couplers that are not calibrated as broken, choosing among equally
    connected qubits by readout error, calibrated two-qubit gate error on the
    coupled pairs used, and the routing an uncoupled pair would need (see
    :func:`_connected_placement`).  Ignoring connectivity scattered
    interacting qubits across sparse devices and let routing multiply the
    circuit (depth 594 vs 72 on a heavy-hex snapshot), and ignoring gate
    errors let it land on broken couplers.  Without a coupling map the
    best-scoring qubits are simply assigned to the busiest logical qubits.

    Args:
        backend: Backend (or backend-like handle) providing calibration data.
        circuit: QuantumCircuit to lay out.

    Returns:
        A :class:`~qiskit.transpiler.Layout`, or ``None`` when one cannot be
        built (unknown sizes, or a circuit wider than the backend).

    Raises:
        None.
    """
    try:
        from qiskit.transpiler import Layout
    except Exception:  # pragma: no cover
        return None

    n = getattr(circuit, "num_qubits", None)
    if n is None:
        return None

    # logical activity: interaction graph degree (directives are not gates)
    deg, weights = _logical_interactions(circuit, n)

    phys_n = getattr(backend, "num_qubits", None)
    if phys_n is None:
        try:
            phys_n = backend.configuration().num_qubits
        except Exception:
            return None
    if phys_n < n:
        return None

    readout = [_safe_get_qubit_readout_error(backend, q) for q in range(phys_n)]
    t1s = [_safe_get_t1(backend, q) for q in range(phys_n)]
    t2s = [_safe_get_t2(backend, q) for q in range(phys_n)]

    # Coherence times are compared relative to the backend's own median, which
    # makes the term independent of units: Targets report seconds (~1e-4)
    # while legacy properties report microseconds.  A qubit missing a value is
    # treated as median (neutral) rather than handed an absolute default that
    # could dwarf -- or be dwarfed by -- the calibrated qubits' values.
    ro = _fill_missing_with_median(readout, fallback=0.02)
    t1 = _relative_to_median(t1s)
    t2 = _relative_to_median(t2s)
    # quality higher is better: readout fidelity first, coherence a tie-breaker
    qualities = (1.0 - ro) + _COHERENCE_WEIGHT * (t1 + t2)

    physical_for_logical: Optional[List[int]] = None
    adjacency = _coupling_adjacency(backend, phys_n)
    if adjacency is not None:
        edge_errors = _edge_errors(backend, adjacency)
        physical_for_logical = _connected_placement(
            n,
            deg,
            weights,
            node_error=1.0 - qualities,
            adjacency=_usable_adjacency(adjacency, edge_errors),
            edge_errors=edge_errors,
        )
    if physical_for_logical is None:
        # Stable sorts break ties by index, so equally ranked qubits map in a
        # platform-independent order.
        logical_order = list(np.argsort(-deg, kind="stable"))  # most active first
        physical_order = list(np.argsort(-qualities, kind="stable"))  # best first
        physical_for_logical = [0] * n
        for lq, pq in zip(logical_order, physical_order[:n]):
            physical_for_logical[int(lq)] = int(pq)

    layout = Layout()
    for lq, pq in enumerate(physical_for_logical):
        layout[circuit.qubits[lq]] = (
            backend.qubits[pq] if hasattr(backend, "qubits") else int(pq)
        )
    return layout
