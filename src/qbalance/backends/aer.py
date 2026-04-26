# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from qbalance.backends.resolver import resolve_backend
from qbalance.errors import OptionalDependencyError, QBalanceError
from qbalance.logging import get_logger

log = get_logger(__name__)


def resolve(spec: str) -> Any:
    """Resolve a backend/plugin implementation from user-provided configuration.

    Args:
        spec: Strategy/backend specification controlling compilation behavior.

    Returns:
        Any with the computed result.

    Raises:
        QBalanceError: Raised when input validation fails or a dependent operation cannot be completed.
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise QBalanceError(f"Invalid aer backend spec: {spec!r}")

    try:
        from qiskit_aer import AerSimulator
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit-aer is required for aer backends (install qbalance[aer])"
        ) from e

    mode = parts[1]
    if mode == "simulator":
        return AerSimulator()

    if mode == "from_backend":
        if len(parts) < 3:
            raise QBalanceError("aer:from_backend requires a nested backend spec")
        nested = ":".join(parts[2:])
        backend = resolve_backend(nested)
        return AerSimulator.from_backend(backend)

    raise QBalanceError(f"Unknown aer backend mode: {mode!r}")
