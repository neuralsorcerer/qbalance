# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

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
    if len(parts) < 3:
        raise QBalanceError(f"Invalid fake backend spec: {spec!r}")

    mode = parts[1]
    if mode == "generic":
        try:
            n = int(parts[2])
        except ValueError as e:
            raise QBalanceError(f"Invalid qubit count in {spec!r}") from e

        try:
            from qiskit.providers.fake_provider import GenericBackendV2
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError(
                "qiskit is required for fake backends (GenericBackendV2)"
            ) from e

        return GenericBackendV2(num_qubits=n)

    if mode == "ibm":
        name = parts[2]
        try:
            from qiskit.providers.fake_provider import fake_backend
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError("qiskit is required for fake backends") from e
        try:
            return fake_backend(name)
        except Exception as e:
            raise QBalanceError(f"Could not resolve fake backend {name!r}: {e}") from e

    raise QBalanceError(f"Unknown fake backend mode: {mode!r}")
