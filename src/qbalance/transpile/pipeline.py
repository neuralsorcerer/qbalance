# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
from importlib import import_module
from typing import Any, Dict, Tuple

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger
from qbalance.strategies import StrategySpec
from qbalance.transpile.noise_aware_layout import (
    estimate_circuit_error,
    noise_aware_initial_layout,
)
from qbalance.transpile.profiling import ProfileReport, make_callback
from qbalance.transpile.suppression import (
    apply_measurement_twirling,
    apply_pauli_twirling,
    build_dd_pass_manager,
)

log = get_logger(__name__)


def _generate_pm(backend: Any, spec: StrategySpec):
    """Internal helper that generate pm.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        spec: Strategy/backend specification controlling compilation behavior.

    Returns:
        Computed value produced by this routine.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit preset pass managers are required (qiskit>=1.0)"
        ) from e

    init_layout = None
    if spec.layout_method == "qbalance_noise_aware":
        init_layout = noise_aware_initial_layout(
            backend, None
        )  # will be overridden later per-circuit

    pm = generate_preset_pass_manager(
        optimization_level=spec.optimization_level,
        backend=backend,
        layout_method=(
            None
            if spec.layout_method in (None, "qbalance_noise_aware")
            else spec.layout_method
        ),
        routing_method=spec.routing_method,
        translation_method=spec.translation_method,
        seed_transpiler=spec.seed_transpiler,
        initial_layout=init_layout,
    )
    return pm


def compile_one(
    circuit: Any,
    backend: Any,
    spec: StrategySpec,
    profile: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Compile one used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        spec: Strategy/backend specification controlling compilation behavior.
        profile (default: False): Whether pass-level transpiler profiling is enabled.

    Returns:
        Tuple[Any, Dict[str, Any]] with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        import_module("qiskit.converters")
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit required") from e

    # Suppression before compile (twirling can be done pre or post; we keep pre-compile)
    twirled_ensemble = [circuit]
    if spec.pauli_twirling:
        twirled_ensemble = apply_pauli_twirling(
            circuit,
            num_twirls=max(1, spec.num_twirls),
            seed=spec.seed_suppression,
            target=getattr(backend, "target", None),
        )

    profile_report = ProfileReport()

    # choose best in ensemble by objective proxy (depth + estimated error)
    best_score = float("inf")
    best = None
    best_metrics = None

    for tw in twirled_ensemble:
        pm = _generate_pm(backend, spec)
        # noise-aware init layout per circuit
        if spec.layout_method == "qbalance_noise_aware":
            try:
                il = noise_aware_initial_layout(backend, tw)
                if il is not None:
                    # regenerate pass manager with initial_layout
                    from qiskit.transpiler.preset_passmanagers import (
                        generate_preset_pass_manager,
                    )

                    pm = generate_preset_pass_manager(
                        optimization_level=spec.optimization_level,
                        backend=backend,
                        layout_method=None,
                        routing_method=spec.routing_method,
                        translation_method=spec.translation_method,
                        seed_transpiler=spec.seed_transpiler,
                        initial_layout=il,
                    )
            except Exception:
                pass

        cb = make_callback(profile_report) if profile else None
        t0 = time.time()
        out = pm.run(tw, callback=cb) if cb is not None else pm.run(tw)
        t1 = time.time()

        dd_applied = False
        if spec.dynamical_decoupling:
            try:
                dd_pm = build_dd_pass_manager(backend, spec.dd_sequence)
                out = dd_pm.run(out)
                dd_applied = True
            except Exception as e:
                log.warning("DD insertion failed (continuing without DD): %s", e)

        flip_map: Dict[int, int] = {}
        if spec.measurement_twirling:
            try:
                out, flip_map = apply_measurement_twirling(
                    out, seed=spec.seed_suppression
                )
            except Exception as e:
                log.warning("Measurement twirling failed (continuing): %s", e)

        m = {
            "compile_time_s": float(t1 - t0),
            "depth": int(out.depth()),
            "size": int(out.size()),
            "width": int(out.num_qubits),
            "two_qubit_ops": int(
                sum(1 for inst, qargs, _ in out.data if len(qargs) == 2)
            ),
            "dd_applied": bool(dd_applied),
            "measurement_flip_map": flip_map,
        }
        try:
            m["estimated_error"] = float(estimate_circuit_error(backend, out))
        except Exception:
            m["estimated_error"] = None

        # score for selection within twirling ensemble
        estimated_error = m.get("estimated_error")
        err_value = (
            float(estimated_error) if isinstance(estimated_error, (int, float)) else 0.0
        )
        depth_value = m.get("depth", 0)
        depth_score = (
            float(depth_value) if isinstance(depth_value, (int, float)) else 0.0
        )
        score = depth_score + 10.0 * err_value
        if score < best_score:
            best_score = score
            best = out
            best_metrics = m

    assert best is not None and best_metrics is not None
    if profile:
        best_metrics["pass_profile"] = profile_report.to_json()

    return best, best_metrics
