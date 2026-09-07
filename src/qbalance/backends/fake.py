# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from qbalance.errors import OptionalDependencyError, QBalanceError
from qbalance.logging import get_logger

log = get_logger(__name__)

# Fixed default calibration seed for fake:generic backends.  Without an
# explicit seed GenericBackendV2 draws random calibration data on every
# instantiation, which breaks run-to-run reproducibility and invalidates
# compile-cache entries keyed by the (stable) backend name.
_DEFAULT_GENERIC_SEED = 0


def _resolve_generic(spec: str, parts: list[str]) -> Any:
    """Build a deterministic GenericBackendV2 from ``fake:generic:N[:SEED]``."""
    if len(parts) not in (3, 4):
        raise QBalanceError(f"Invalid generic fake backend spec: {spec!r}")
    try:
        n = int(parts[2])
    except ValueError as e:
        raise QBalanceError(f"Invalid qubit count in {spec!r}") from e
    if n < 2:
        raise QBalanceError("fake:generic requires at least 2 qubits")

    seed = _DEFAULT_GENERIC_SEED
    if len(parts) == 4:
        try:
            seed = int(parts[3])
        except ValueError as e:
            raise QBalanceError(f"Invalid calibration seed in {spec!r}") from e

    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit is required for fake backends (GenericBackendV2)"
        ) from e

    try:
        try:
            return GenericBackendV2(num_qubits=n, seed=seed)
        except TypeError:
            # Older qiskit releases (and lightweight test stubs) do not accept
            # a calibration seed; fall back to the legacy constructor.
            return GenericBackendV2(num_qubits=n)
    except Exception as e:
        raise QBalanceError(f"Could not create generic fake backend: {e}") from e


def _resolve_ibm(spec: str, parts: list[str]) -> Any:
    """Resolve ``fake:ibm:NAME`` to an IBM fake-device snapshot backend."""
    if len(parts) != 3:
        raise QBalanceError(f"Invalid IBM fake backend spec: {spec!r}")
    name = parts[2]

    # Legacy path: some installations expose a callable factory on
    # qiskit.providers.fake_provider.  Use it when present.
    try:
        fake_provider_module = import_module("qiskit.providers.fake_provider")
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required for fake backends") from e
    factory = getattr(fake_provider_module, "fake_backend", None)
    # Guard against the same shadowing: qiskit.providers.fake_provider.fake_backend
    # is a submodule, not the legacy factory function.
    if callable(factory) and not isinstance(factory, ModuleType):
        try:
            return factory(name)
        except Exception as e:
            raise QBalanceError(f"Could not resolve fake backend {name!r}: {e}") from e

    # Modern path: IBM device snapshots live in qiskit-ibm-runtime.
    try:
        ibm_fake_provider = import_module("qiskit_ibm_runtime.fake_provider")
    except Exception as e:
        raise OptionalDependencyError(
            "qiskit-ibm-runtime is required for fake:ibm backends "
            "(pip install qiskit-ibm-runtime)"
        ) from e

    candidates = [name]
    cleaned = name.strip()
    if cleaned and not cleaned.lower().startswith("fake"):
        title = cleaned[0].upper() + cleaned[1:]
        candidates.extend([f"Fake{title}", f"Fake{title}V2"])
    for candidate in candidates:
        backend_cls = getattr(ibm_fake_provider, candidate, None)
        if not isinstance(backend_cls, type):
            # qiskit_ibm_runtime.fake_provider exposes a submodule per device
            # ("manila") alongside the backend class ("FakeManilaV2").  Skipping
            # non-classes keeps that submodule from shadowing the class the next
            # candidate spelling resolves to, which is the whole point of trying
            # several spellings.
            continue
        try:
            return backend_cls()
        except Exception as e:
            raise QBalanceError(f"Could not resolve fake backend {name!r}: {e}") from e
    raise QBalanceError(
        f"Unknown IBM fake backend {name!r} (no matching class in "
        "qiskit_ibm_runtime.fake_provider)"
    )


def resolve(spec: str) -> Any:
    """Resolve a backend/plugin implementation from user-provided configuration.

    Supported forms are ``fake:generic:N`` and ``fake:generic:N:SEED`` for
    deterministic :class:`GenericBackendV2` instances, and ``fake:ibm:NAME``
    for IBM fake-device snapshots.

    Args:
        spec: Strategy/backend specification controlling compilation behavior.

    Returns:
        Any with the computed result.

    Raises:
        QBalanceError: Raised when input validation fails or a dependent operation cannot be completed.
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) < 3 or any(part == "" for part in parts[:3]):
        raise QBalanceError(f"Invalid fake backend spec: {spec!r}")

    mode = parts[1]
    if mode == "generic":
        return _resolve_generic(spec, parts)

    if mode == "ibm":
        return _resolve_ibm(spec, parts)

    raise QBalanceError(f"Unknown fake backend mode: {mode!r}")
