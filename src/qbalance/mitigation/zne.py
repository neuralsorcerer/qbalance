# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from qbalance.logging import get_logger
from qbalance.utils import instruction_parts

log = get_logger(__name__)


def _split_terminal_suffix(
    circuit: Any,
) -> tuple[Any, list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]]]:
    """Return an invertible prefix plus a terminal measurement/barrier suffix."""
    copy_empty_like = getattr(circuit, "copy_empty_like", None)
    if not callable(copy_empty_like):
        return circuit, []

    suffix_seen = False
    terminal_suffix_ops = {"barrier", "delay", "measure"}
    unitary = copy_empty_like()
    suffix: list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]] = []
    for entry in list(getattr(circuit, "data", [])):
        inst, qargs, cargs = instruction_parts(entry)
        inst_name = getattr(inst, "name", "")
        if inst_name == "measure":
            suffix_seen = True

        if suffix_seen:
            if inst_name not in terminal_suffix_ops:
                raise ValueError(
                    "Global folding supports circuits with measurements only when all measurements are terminal."
                )
            suffix.append((inst, qargs, cargs))
            continue

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
    even_mass = sum(p for b, p in probs.items() if _parity(b) == 0)
    odd_mass = 1.0 - even_mass
    # expval = even - odd => target even = (1+exp)/2
    target_even = max(0.0, min(1.0, (1.0 + y0) / 2.0))

    template = next(iter(probs), None)
    if target_even > 0.0 and even_mass == 0.0:
        probs[_synthetic_parity_key(template, odd=False)] = 0.0
    if target_even < 1.0 and odd_mass == 0.0:
        probs[_synthetic_parity_key(template, odd=True)] = 0.0

    for b in list(probs.keys()):
        if _parity(b) == 0:
            probs[b] = (
                probs[b] * target_even / even_mass if even_mass > 0.0 else target_even
            )
        else:
            probs[b] = (
                probs[b] * (1.0 - target_even) / odd_mass
                if odd_mass > 0.0
                else 1.0 - target_even
            )

    # renormalize against roundoff and degenerate polynomial outputs.
    s = sum(probs.values()) or 1.0
    probs = {k: float(v / s) for k, v in probs.items()}

    return probs
