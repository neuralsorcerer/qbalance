# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from qbalance.logging import get_logger
from qbalance.utils import instruction_parts, shares_bit

log = get_logger(__name__)

# Scheduling directives: they neither change the state nor read a classical bit,
# so they never disqualify a trailing measurement block.
_SUFFIX_TRANSPARENT_OPS = {"barrier", "delay"}


_NON_TERMINAL_MEASUREMENT_ERROR = (
    "Global folding supports circuits with measurements only when all "
    "measurements are terminal."
)


def _validate_terminal_suffix(
    suffix: list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]],
) -> None:
    """Reject a trailing block that is not a pure terminal measurement block.

    ``fold_global`` folds only the prefix before the first measurement and
    replays the suffix once, so the suffix must not contain computation whose
    noise should have been scaled:

    * every measurement must be terminal -- no later instruction may act on the
      measured qubit or overwrite its classical bit; and
    * every other operation must act only on qubits the suffix goes on to
      measure.  That admits the single-qubit frame changes measurement twirling
      inserts before each measurement, while still rejecting mid-circuit
      measurement feeding further computation.
    """
    for index, (inst, qargs, cargs) in enumerate(suffix):
        name = getattr(inst, "name", "")
        if name in _SUFFIX_TRANSPARENT_OPS:
            continue

        later = suffix[index + 1 :]
        if name == "measure":
            for later_inst, later_qargs, later_cargs in later:
                if getattr(later_inst, "name", "") in _SUFFIX_TRANSPARENT_OPS:
                    continue
                if shares_bit(qargs, later_qargs) or shares_bit(cargs, later_cargs):
                    raise ValueError(_NON_TERMINAL_MEASUREMENT_ERROR)
            continue

        measured_later = [
            later_qargs
            for later_inst, later_qargs, _ in later
            if getattr(later_inst, "name", "") == "measure"
        ]
        for qubit in qargs:
            if not any(shares_bit((qubit,), targets) for targets in measured_later):
                raise ValueError(_NON_TERMINAL_MEASUREMENT_ERROR)


def _split_terminal_suffix(
    circuit: Any,
) -> tuple[Any, list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]]]:
    """Return an invertible prefix plus the terminal measurement suffix."""
    copy_empty_like = getattr(circuit, "copy_empty_like", None)
    if not callable(copy_empty_like):
        return circuit, []

    data = [instruction_parts(entry) for entry in list(getattr(circuit, "data", []))]
    first_measure = next(
        (
            index
            for index, (inst, _, _) in enumerate(data)
            if getattr(inst, "name", "") == "measure"
        ),
        None,
    )

    prefix = data if first_measure is None else data[:first_measure]
    suffix = [] if first_measure is None else data[first_measure:]
    _validate_terminal_suffix(suffix)

    unitary = copy_empty_like()
    for inst, qargs, cargs in prefix:
        unitary.append(inst, qargs, cargs)

    return unitary, suffix


def fold_global(circuit: Any, scale: float) -> Any:
    """Fold global used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        scale: Scale value consumed by this routine.

    Returns:
        Any with the computed result.

    Raises:
        None.
    """
    if isinstance(scale, (bool, np.bool_)):
        raise ValueError("scale must be a finite real value >= 1.0")
    try:
        scale_f = float(scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale must be a finite real value >= 1.0") from exc
    if not np.isfinite(scale_f) or scale_f < 1.0:
        raise ValueError("scale must be a finite real value >= 1.0")
    if scale_f == 1.0:
        return circuit

    # odd integer close to scale
    k = int(np.ceil(scale_f))
    if k % 2 == 0:
        k += 1

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
    return _rebase_to_backend(folded, backend)


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
    """Internal helper that counts to expval z.

    Args:
        counts: Counts value consumed by this routine.
        validate (default: False): Validate value consumed by this routine.

    Returns:
        float with the computed result.

    Raises:
        ValueError: Raised when input validation fails or a dependent operation cannot be completed.
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


def zne_extrapolate_counts(
    factors: Sequence[float],
    counts_per_factor: Sequence[Dict[str, int]],
    degree: int = 1,
) -> Dict[str, float]:
    """Zne extrapolate counts used by the qbalance workflow.

    Args:
        factors: Factors value consumed by this routine.
        counts_per_factor: Counts per factor value consumed by this routine.
        degree (default: 1): Degree value consumed by this routine.

    Returns:
        Dict[str, float] with the computed result.

    Raises:
        ValueError: Raised when input validation fails or a dependent operation cannot be completed.
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

    # Build a pseudo distribution from the noisiest (factor closest to 1) counts,
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
