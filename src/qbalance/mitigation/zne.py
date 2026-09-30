# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from qbalance.logging import get_logger
from qbalance.utils import (
    SCHEDULING_DIRECTIVES,
    instruction_parts,
    is_terminal_measurement,
    shares_bit,
)

log = get_logger(__name__)

# Scheduling directives: they neither change the state nor read a classical bit,
# so they never disqualify a trailing measurement block.
_SUFFIX_TRANSPARENT_OPS = SCHEDULING_DIRECTIVES


_NON_TERMINAL_MEASUREMENT_ERROR = (
    "Global folding supports circuits with measurements only when all "
    "measurements are terminal."
)


def _split_terminal_suffix(
    circuit: Any,
) -> tuple[Any, list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]]]:
    """Return the invertible part of ``circuit`` plus its terminal measurements.

    Every measurement must be terminal (see
    :func:`~qbalance.utils.is_terminal_measurement`).
    The split is decided per qubit rather than at the first measurement in
    ``circuit.data``: instruction order is only a topological order, so a qubit
    that finishes early is routinely measured before gates on other qubits are
    listed.  Cutting at the first measurement would replay those later gates
    once instead of folding them, silently leaving part of the circuit's noise
    unscaled.  A terminal measurement commutes with everything listed after it
    (nothing later shares its qubit or clbit), so moving all of them behind the
    folded unitary preserves the computation.  Directives listed after a
    qubit's terminal measurement stay behind it as well.
    """
    copy_empty_like = getattr(circuit, "copy_empty_like", None)
    if not callable(copy_empty_like):
        return circuit, []

    data = [instruction_parts(entry) for entry in list(getattr(circuit, "data", []))]
    unitary = copy_empty_like()
    terminal: list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]] = []
    measured_qubits: list[Any] = []
    for index, (inst, qargs, cargs) in enumerate(data):
        name = getattr(inst, "name", "")
        if name == "measure":
            if not is_terminal_measurement(data, index):
                raise ValueError(_NON_TERMINAL_MEASUREMENT_ERROR)
            terminal.append((inst, qargs, cargs))
            measured_qubits.extend(qargs)
            continue
        if (
            name in _SUFFIX_TRANSPARENT_OPS
            and qargs
            and all(shares_bit((qubit,), measured_qubits) for qubit in qargs)
        ):
            terminal.append((inst, qargs, cargs))
            continue
        unitary.append(inst, qargs, cargs)

    return unitary, terminal


def _validated_scale(scale: Any) -> float:
    """Return ``scale`` as a float after checking it is a finite value >= 1."""
    if isinstance(scale, (bool, np.bool_)):
        raise ValueError("scale must be a finite real value >= 1.0")
    try:
        scale_f = float(scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale must be a finite real value >= 1.0") from exc
    if not np.isfinite(scale_f) or scale_f < 1.0:
        raise ValueError("scale must be a finite real value >= 1.0")
    return scale_f


def realized_fold_factor(scale: float) -> int:
    """Return the noise-scale factor :func:`fold_global` actually applies.

    Global folding builds ``U (U^dagger U)^r`` and therefore only realizes odd
    integer factors: a requested ``scale`` is rounded up to the next odd
    integer (``2.0 -> 3``, ``3.0 -> 3``, ``4.0 -> 5``) and ``1.0`` stays ``1``.
    Extrapolation must use these realized factors -- fitting against the
    requested values would place, for example, the ``2.0`` and ``3.0`` samples
    at different noise levels even though both ran at ``3``.

    Args:
        scale: Requested noise scale factor, ``>= 1.0``.

    Returns:
        The odd integer folding factor.

    Raises:
        ValueError: If ``scale`` is not a finite real value ``>= 1.0``.
    """
    scale_f = _validated_scale(scale)
    if scale_f == 1.0:
        return 1
    k = int(np.ceil(scale_f))
    if k % 2 == 0:
        k += 1
    return k


def zne_sampling_overhead(factors: Sequence[float], degree: int = 1) -> float:
    """Return the factor by which ZNE multiplies the shots a precision needs.

    Every factor runs on the full shot budget, and the extrapolated value is a
    fixed linear combination ``sum(w_i * y_i)`` of the measured points (the
    intercept of the least-squares polynomial fit), so its variance is
    ``sum(w_i**2)`` times that of one point.  Matching the precision of a
    single unmitigated run therefore takes ``n * sum(w_i**2)`` times the shots
    for ``n`` factors: 4.375 for the default linear fit at factors
    ``(1, 3, 5)``, and about 15.7 for a quadratic one.  The fit uses the
    factors folding actually realizes (see :func:`realized_fold_factor`).

    Args:
        factors: Requested noise scale factors, each ``>= 1.0``.
        degree (default: 1): Degree of the extrapolation polynomial.

    Returns:
        The shot multiplier, ``>= 1.0``.

    Raises:
        ValueError: If a factor is invalid or the realized factors are too few
            for ``degree``.
    """
    realized = np.asarray([realized_fold_factor(f) for f in factors], dtype=float)
    if isinstance(degree, bool) or int(degree) != degree or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    if np.unique(realized).size < int(degree) + 1:
        raise ValueError("not enough distinct realized factors for the degree")
    design = np.vander(realized, int(degree) + 1, increasing=True)
    intercept_weights = np.linalg.pinv(design)[0]
    return float(realized.size * np.sum(intercept_weights**2))


def fold_global(circuit: Any, scale: float) -> Any:
    """Globally fold ``circuit`` to scale its noise by an odd integer factor.

    Args:
        circuit: QuantumCircuit to fold.  Every measurement must be terminal.
        scale: Requested noise scale factor, ``>= 1.0``; it is rounded up to the
            odd factor reported by :func:`realized_fold_factor`.

    Returns:
        ``circuit`` itself when the realized factor is 1, otherwise the folded
        circuit ``U (U^dagger U)^r`` followed by the terminal measurements.

    Raises:
        ValueError: If ``scale`` is invalid or a measurement is not terminal.
    """
    k = realized_fold_factor(scale)
    if k == 1:
        return circuit

    base, terminal_suffix = _split_terminal_suffix(circuit)
    qc = base.copy()
    inv = base.inverse()
    # construct: U (U^dag U)^{(k-1)/2}
    out = base.copy()
    reps = (k - 1) // 2
    for _ in range(reps):
        out = out.compose(inv).compose(qc)

    for inst, qargs, cargs in terminal_suffix:
        out.append(inst, qargs, cargs)

    out.name = f"{getattr(circuit,'name','circuit')}_fold{k}"
    return out


def _rebase_to_backend(circuit: Any, backend: Any) -> Any:
    """Re-express an already-compiled circuit in the backend's native basis.

    Global folding appends ``U.inverse()``, which introduces adjoint gates that
    are not part of the backend basis (an ``sx`` basis gains ``sxdg``), so a
    folded circuit is rejected at execution even though the circuit it folded
    was fully compiled.  Re-running the preset pass manager at optimization
    level 0 with the identity layout restores a runnable basis while leaving the
    folding, the qubit layout, and the measurement clbit mapping intact -- all
    of which the ZNE extrapolation depends on to compare counts across factors.

    Returns the circuit unchanged when the backend cannot be described to the
    preset pass manager, or when the circuit is not sized for this backend
    (re-transpiling would then relayout it and shift the count-key bit order).
    """
    try:
        from qiskit.providers import BackendV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except Exception:  # pragma: no cover - qiskit always provides these
        return circuit

    if not isinstance(backend, BackendV2):
        return circuit

    num_qubits = getattr(circuit, "num_qubits", None)
    if not isinstance(num_qubits, int) or num_qubits != getattr(
        backend, "num_qubits", None
    ):
        return circuit

    try:
        pass_manager = generate_preset_pass_manager(
            optimization_level=0,
            backend=backend,
            initial_layout=list(range(num_qubits)),
        )
        return pass_manager.run(circuit)
    except Exception as e:
        log.warning("Could not rebase folded circuit to the backend basis: %s", e)
        return circuit


def fold_global_for_backend(circuit: Any, backend: Any, scale: float) -> Any:
    """Fold a compiled circuit and return it runnable on ``backend``.

    Args:
        circuit: Compiled QuantumCircuit to fold.
        backend: Backend the folded circuit will be executed on.
        scale: Noise scale factor, ``>= 1.0``.

    Returns:
        The folded circuit, expressed in the backend's native basis.

    Raises:
        ValueError: If ``scale`` is not a finite real value ``>= 1.0``.
    """
    folded = fold_global(circuit, scale)
    if folded is circuit:
        return circuit
    rebased = _rebase_to_backend(folded, backend)
    # Re-basing runs a fresh preset pass manager, which stamps the identity
    # layout it was given onto the result.  The folded circuit still acts on
    # the same physical qubits with the same net routing permutation --
    # (U^dagger U)^r is the identity, permutation included -- so the compiled
    # circuit's own TranspileLayout remains the accurate description.
    if hasattr(circuit, "_layout") and rebased is not circuit:
        rebased._layout = circuit._layout
    return rebased


def _bit_positions(bitstr: str) -> list[int]:
    """Return positions of binary digits in a Qiskit count key."""
    return [idx for idx, char in enumerate(bitstr) if char in {"0", "1"}]


def _parity(bitstr: str) -> int:
    """Return even/odd parity for binary digits in a Qiskit count key."""
    return sum(1 for char in bitstr if char == "1") % 2


def _synthetic_parity_key(template: str | None, *, odd: bool) -> str:
    """Create an all-zero/easy-odd key preserving count-key spacing when possible."""
    if not template:
        return "1" if odd else "0"

    chars = ["0" if char in {"0", "1"} else char for char in template]
    positions = _bit_positions(template)
    if not positions:
        return "1" if odd else "0"
    if odd:
        chars[positions[-1]] = "1"
    return "".join(chars)


def _counts_to_expval_z(counts: Mapping[str, int], *, validate: bool = False) -> float:
    """Return the Z-parity expectation of ``counts``.

    Args:
        counts: Counts keyed by Qiskit bitstrings.
        validate (default: False): Check that ``counts`` is non-empty, its
            values are non-negative integers with a positive total, and its
            keys hold binary digits (and register-separating spaces) only.
            Unvalidated, an empty mapping gives 0.

    Raises:
        ValueError: If ``validate`` is set and a check fails.
    """
    if validate and not counts:
        raise ValueError("counts entries must be non-empty mappings")

    shots = 0
    s = 0.0
    for bitstr, c in counts.items():
        if validate:
            if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
                raise ValueError("counts values must be non-negative integers")
            if c < 0:
                raise ValueError("counts values must be non-negative integers")
        if validate:
            if not isinstance(bitstr, str) or not bitstr:
                raise ValueError("counts keys must be non-empty bitstrings")
            if not _bit_positions(bitstr):
                raise ValueError("counts keys must contain at least one binary digit")
            if any(char not in {"0", "1", " "} for char in bitstr):
                raise ValueError(
                    "counts keys must contain only binary digits and spaces"
                )

        parity = _parity(bitstr)
        shots += int(c)
        s += (1.0 if parity == 0 else -1.0) * c

    if validate:
        if shots <= 0:
            raise ValueError("each counts entry must contain at least one shot")
    else:
        shots = shots or 1
    return s / shots


def parity_expectation_value(distribution: Mapping[str, float]) -> float:
    """Return the Z-parity expectation of a counts or probability mapping.

    The value is ``<Z x ... x Z>`` over every measured bit: the weight of
    even-parity bitstrings minus that of odd-parity ones, normalized by the
    total weight.  It is the observable ZNE extrapolates, so it is also what
    uncut, mitigated and cut executions report for comparison.

    Args:
        distribution: Counts or (quasi-)probabilities keyed by Qiskit bitstrings.

    Returns:
        The parity expectation in ``[-1, 1]`` for a probability distribution.

    Raises:
        ValueError: If the mapping carries no positive total weight.
    """
    total = 0.0
    signed = 0.0
    for bitstr, weight in distribution.items():
        value = float(weight)
        total += value
        signed += value if _parity(bitstr) == 0 else -value
    if not total > 0.0:
        raise ValueError("distribution must carry a positive total weight")
    return signed / total


def zne_extrapolate_counts(
    factors: Sequence[float],
    counts_per_factor: Sequence[Dict[str, int]],
    degree: int = 1,
) -> Dict[str, float]:
    """Extrapolate counts measured at several noise scales to zero noise.

    The Z-parity expectation of each counts mapping is fitted with a
    least-squares polynomial of ``degree`` in the noise factor and
    evaluated at 0.  The returned distribution is the one measured at the
    factor closest to 1 with its even- and odd-parity mass rescaled (each
    class keeping its shape) so its parity equals that value, clipped to
    ``[-1, 1]``.  A parity class the reference counts never sampled gets
    one synthetic key, so any target is reachable.

    Args:
        factors: Noise scale factors, each ``>= 1``; pass the realized ones
            (see :func:`realized_fold_factor`).
        counts_per_factor: Counts measured at each factor.
        degree (default: 1): Degree of the fitted polynomial.

    Returns:
        Probabilities summing to 1, keyed by bitstring.

    Raises:
        ValueError: If ``degree`` is not a non-negative integer, the lengths
            differ, there are fewer than ``degree + 1`` (distinct) factors,
            a factor is not finite or is below 1, or a counts mapping fails
            the checks of :func:`_counts_to_expval_z`.
    """
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise ValueError("degree must be a non-negative integer")
    if degree < 0:
        raise ValueError("degree must be a non-negative integer")

    if len(factors) != len(counts_per_factor):
        raise ValueError("factors and counts_per_factor must have same length")
    if len(factors) < degree + 1:
        raise ValueError("not enough points for requested polynomial degree")

    xs = np.asarray(factors, dtype=float)
    if not np.all(np.isfinite(xs)):
        raise ValueError("factors must be finite real numbers")
    if np.any(xs < 1.0):
        raise ValueError("factors must be >= 1.0")
    if degree > 0 and np.unique(xs).size < degree + 1:
        raise ValueError("factors must contain at least degree + 1 distinct values")

    ys = np.asarray(
        [_counts_to_expval_z(c, validate=True) for c in counts_per_factor], dtype=float
    )

    coeffs = np.polyfit(xs, ys, deg=degree)
    # value at x=0 is constant term (last)
    y0 = float(np.polyval(coeffs, 0.0))

    # Build a pseudo distribution from the least-noisy (factor closest to 1) counts,
    # then gently adjust parity mass to match y0.
    idx0 = int(np.argmin(np.abs(xs - 1.0)))
    base = counts_per_factor[idx0]
    shots = sum(base.values()) or 1
    probs = {k: v / shots for k, v in base.items()}

    # Adjust parity mass.  The extrapolated observable constrains the total
    # even/odd parity probability, so keep the shape within each existing parity
    # class when possible and create the missing complementary class only when
    # the reference counts never sampled it.
    even_keys = [b for b in probs if _parity(b) == 0]
    odd_keys = [b for b in probs if _parity(b) != 0]
    even_mass = sum(probs[b] for b in even_keys)
    odd_mass = sum(probs[b] for b in odd_keys)
    # expval = even - odd => target even = (1+exp)/2
    target_even = max(0.0, min(1.0, (1.0 + y0) / 2.0))
    target_odd = 1.0 - target_even

    template = next(iter(probs), None)
    if target_even > 0.0 and not even_keys:
        even_keys = [_synthetic_parity_key(template, odd=False)]
        probs[even_keys[0]] = 0.0
    if target_odd > 0.0 and not odd_keys:
        odd_keys = [_synthetic_parity_key(template, odd=True)]
        probs[odd_keys[0]] = 0.0

    # Rescale each parity class to its target mass.  When a class carries no
    # sampled mass its shape is unknown, so the target is spread uniformly over
    # that class -- assigning the full target to every key would multiply the
    # class mass by the number of keys in it.
    for keys, mass, target in (
        (even_keys, even_mass, target_even),
        (odd_keys, odd_mass, target_odd),
    ):
        if not keys:
            continue
        if mass > 0.0:
            for b in keys:
                probs[b] = probs[b] * target / mass
        else:
            share = target / len(keys)
            for b in keys:
                probs[b] = share

    # renormalize against roundoff and degenerate polynomial outputs.
    s = sum(probs.values()) or 1.0
    probs = {k: float(v / s) for k, v in probs.items()}

    return probs
