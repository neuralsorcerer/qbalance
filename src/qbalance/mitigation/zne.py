# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from qbalance.logging import get_logger

log = get_logger(__name__)


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
    if scale <= 1.0:
        return circuit

    # odd integer close to scale
    k = int(np.ceil(scale))
    if k % 2 == 0:
        k += 1

    qc = circuit.copy()
    inv = circuit.inverse()
    # construct: U (U^dag U)^{(k-1)/2}
    out = circuit.copy()
    reps = (k - 1) // 2
    for _ in range(reps):
        out = out.compose(inv).compose(qc)
    out.name = f"{getattr(circuit,'name','circuit')}_fold{k}"
    return out


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
        parity = bitstr.count("1") % 2
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

    # Adjust parity mass
    even_mass = sum(p for b, p in probs.items() if b.count("1") % 2 == 0)
    odd_mass = 1.0 - even_mass
    # expval = even - odd => target even = (1+exp)/2
    target_even = max(0.0, min(1.0, (1.0 + y0) / 2.0))
    if even_mass > 0 and odd_mass > 0:
        scale_even = target_even / even_mass
        scale_odd = (1.0 - target_even) / odd_mass
        for b in list(probs.keys()):
            if b.count("1") % 2 == 0:
                probs[b] *= scale_even
            else:
                probs[b] *= scale_odd
        # renormalize
        s = sum(probs.values()) or 1.0
        probs = {k: float(v / s) for k, v in probs.items()}

    return probs
