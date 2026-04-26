# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from numbers import Integral
from typing import Any, Dict, Optional

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger

log = get_logger(__name__)


def _ensure_backend(backend: Any) -> Any:
    # If backend can't run, try to wrap with AerSimulator.from_backend

    """Internal helper that ensure backend.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.

    Returns:
        Any with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if hasattr(backend, "run"):
        return backend
    try:
        from qiskit_aer import AerSimulator
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "Backend has no .run(); install qbalance[aer] to execute with AerSimulator"
        ) from e
    return AerSimulator.from_backend(backend)


def _prepare_run_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Internal helper that prepare run kwargs.

    Args:
        func: Func value consumed by this routine.
        kwargs: Kwargs value consumed by this routine.

    Returns:
        Dict[str, Any] with the computed result.

    Raises:
        None.
    """
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params):
        return dict(kwargs)

    allowed = {
        param.name
        for param in params
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def run_counts(
    backend: Any,
    circuit: Any,
    shots: int = 1024,
    seed_simulator: Optional[int] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, int]:
    """Execute counts and return the collected results.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        shots (default: 1024): Number of shots used when executing circuits on a backend.
        seed_simulator (default: None): Seed forwarded to simulator backends when supported.
        seed_transpiler (default: None): Seed forwarded to transpiler-aware backend run methods when supported.

    Returns:
        Dict[str, int] with the computed result.

    Raises:
        ValueError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if not isinstance(shots, Integral) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if seed_simulator is not None and (
        not isinstance(seed_simulator, Integral) or isinstance(seed_simulator, bool)
    ):
        raise ValueError("seed_simulator must be an integer or None.")
    if seed_transpiler is not None and (
        not isinstance(seed_transpiler, Integral) or isinstance(seed_transpiler, bool)
    ):
        raise ValueError("seed_transpiler must be an integer or None.")

    b = _ensure_backend(backend)
    run_callable = b.run

    run_kwargs: Dict[str, Any] = {"shots": shots}
    if seed_simulator is not None:
        run_kwargs["seed_simulator"] = seed_simulator
    if seed_transpiler is not None:
        run_kwargs["seed_transpiler"] = seed_transpiler

    call_kwargs = _prepare_run_kwargs(run_callable, run_kwargs)

    job = run_callable(circuit, **call_kwargs)
    result = job.result()

    try:
        return dict(result.get_counts(circuit))
    except TypeError:
        return dict(result.get_counts())
