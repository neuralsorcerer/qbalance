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
from qbalance.utils import instruction_parts

log = get_logger(__name__)


def _ensure_backend(backend: Any) -> Any:
    """Return ``backend`` if it can run circuits, else an Aer simulator of it.

    A backend without ``run`` is wrapped in ``AerSimulator.from_backend``.

    Raises:
        OptionalDependencyError: If wrapping is needed and qiskit-aer is not
            installed.
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
    """Return the subset of ``kwargs`` that ``func`` accepts.

    Every keyword is kept when ``func`` takes ``**kwargs`` or its signature
    cannot be inspected.
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


def _measures_nothing(circuit: Any) -> bool:
    """Whether ``circuit`` is a real circuit without a single measurement."""
    data = getattr(circuit, "data", None)
    if data is None or not hasattr(circuit, "num_clbits"):
        return False  # not a QuantumCircuit; let the backend decide
    return not _contains_measurement(data)


def _contains_measurement(data: Any) -> bool:
    """Whether ``data`` holds a measurement, inside control-flow blocks too.

    A dynamic circuit can measure only within an ``if_else`` or ``for_loop``
    body; those measurements still produce counts.
    """
    for entry in data:
        operation = instruction_parts(entry)[0]
        if getattr(operation, "name", "") == "measure":
            return True
        for block in getattr(operation, "blocks", None) or ():
            if _contains_measurement(getattr(block, "data", None) or ()):
                return True
    return False


def run_counts(
    backend: Any,
    circuit: Any,
    shots: int = 1024,
    seed_simulator: Optional[int] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, int]:
    """Run ``circuit`` on ``backend`` and return its counts.

    A backend without ``run`` is executed through
    ``AerSimulator.from_backend``.  The seeds are passed only to ``run``
    methods that accept them.

    Args:
        backend: Backend to run on.
        circuit: Compiled circuit with at least one measurement
            (measurements inside control-flow blocks count).
        shots (default: 1024): Number of shots, a positive integer.
        seed_simulator (default: None): Simulator seed.
        seed_transpiler (default: None): Transpiler seed, for backends
            whose ``run`` transpiles.

    Returns:
        Counts keyed by Qiskit bitstrings.

    Raises:
        ValueError: If ``shots`` is not a positive integer, a seed is not an
            integer or ``None`` (booleans are rejected), or the circuit
            measures nothing.
        OptionalDependencyError: If the backend has no ``run`` and
            qiskit-aer is not installed.
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

    if _measures_nothing(circuit):
        # Backends answer with an opaque "No counts for experiment <object at
        # 0x...>" long after the job was submitted.
        raise ValueError(
            f"Circuit {getattr(circuit, 'name', '')!r} has no measurements, "
            "so executing it yields no counts."
        )

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
