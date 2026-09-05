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
from qbalance.utils import backend_display_name, instruction_parts

log = get_logger(__name__)

_DIRECTIVE_NAMES = {"barrier", "delay"}

# qbalance-specific layout name.  Qiskit does not know this method; it is
# realized by handing the computed layout to the preset pass manager as an
# ``initial_layout`` instead.
NOISE_AWARE_LAYOUT = "qbalance_noise_aware"


def _count_two_qubit_ops(circuit: Any) -> int:
    """Count two-qubit gate operations, excluding scheduling directives.

    Barriers can span exactly two qubits; counting them as two-qubit gates
    would inflate the objective for otherwise identical circuits.
    """
    count = 0
    for entry in circuit.data:
        inst, qargs, _ = instruction_parts(entry)
        if len(qargs) == 2 and getattr(inst, "name", "") not in _DIRECTIVE_NAMES:
            count += 1
    return count


def _backend_basis_gates(backend: Any, target: Any) -> list[str] | None:
    """Return backend basis gates for Qiskit stage generators when available."""
    if target is not None:
        raw_names = getattr(target, "operation_names", None)
        if callable(raw_names):
            raw_names = raw_names()
        names = sorted(str(name) for name in (raw_names or ()) if str(name).strip())
        return names or None

    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        try:
            raw_basis = getattr(configuration(), "basis_gates", None)
        except Exception:
            raw_basis = None
        if raw_basis:
            names = sorted(str(name) for name in raw_basis if str(name).strip())
            return names or None
    return None


def _append_stage(pass_manager: Any, stage: Any) -> Any:
    """Append a generated Qiskit stage pass manager with stub compatibility."""
    try:
        pass_manager += stage
        return pass_manager
    except TypeError:
        append = getattr(pass_manager, "append", None)
        if callable(append):
            append(stage)
            return pass_manager
        raise


def _preset_layout_method(spec: StrategySpec) -> str | None:
    """Return the Qiskit preset layout method requested by a strategy.

    ``qbalance_noise_aware`` is not a Qiskit layout plugin, so it maps to "no
    preset layout method"; the noise-aware layout reaches the preset pass
    manager through ``initial_layout``.
    """
    if spec.layout_method in (None, NOISE_AWARE_LAYOUT):
        return None
    return spec.layout_method


def _supports_preset_pass_manager(backend: Any) -> bool:
    """Return True when Qiskit's preset pass manager can target ``backend``.

    Lightweight stubs and pre-BackendV2 objects do not carry the transpiler
    ``Target`` the preset generator needs; those fall back to the
    translation-only stage pipeline below.
    """
    try:
        from qiskit.providers import BackendV2
        from qiskit.transpiler import Target
    except Exception:  # pragma: no cover - qiskit always provides these
        return False

    if isinstance(backend, BackendV2):
        return True
    return isinstance(getattr(backend, "target", None), Target)


def _generate_stage_pm(backend: Any, spec: StrategySpec, initial_layout: Any = None):
    """Build a translation-only pass manager for backends without a Target.

    This fallback cannot honor ``optimization_level`` or perform routing, so it
    is used only when :func:`_supports_preset_pass_manager` rejects the backend.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        spec: Strategy/backend specification controlling compilation behavior.
        initial_layout (default: None): Layout applied before translation, when available.

    Returns:
        Computed value produced by this routine.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import (
            ApplyLayout,
            EnlargeWithAncilla,
            FullAncillaAllocation,
            SetLayout,
        )
        from qiskit.transpiler.preset_passmanagers import (
            generate_translation_passmanager,
            generate_unroll_3q,
        )
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit preset pass-manager stage generators are required (qiskit>=1.0)"
        ) from e

    target = getattr(backend, "target", None)
    basis_gates = _backend_basis_gates(backend, target)
    translation_method = spec.translation_method or "translator"

    pm = PassManager()
    if initial_layout is not None and target is not None:
        pm.append(SetLayout(initial_layout))
        pm.append(FullAncillaAllocation(target))
        pm.append(EnlargeWithAncilla())
        pm.append(ApplyLayout())

    pm = _append_stage(pm, generate_unroll_3q(target=target, basis_gates=basis_gates))
    pm = _append_stage(
        pm,
        generate_translation_passmanager(
            target=target,
            basis_gates=basis_gates,
            method=translation_method,
        ),
    )
    return pm


def _generate_pm(backend: Any, spec: StrategySpec, initial_layout: Any = None):
    """Build the compilation pass manager for one strategy.

    Qiskit's preset pass manager is what actually honors ``optimization_level``,
    ``layout_method``, ``routing_method``, ``translation_method`` and
    ``seed_transpiler``, and what maps the circuit onto the backend coupling
    map.  Backends that cannot be described to it (BackendV1-style objects and
    test stubs) fall back to the translation-only stage pipeline.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        spec: Strategy/backend specification controlling compilation behavior.
        initial_layout (default: None): Explicit initial layout, used for the
            ``qbalance_noise_aware`` layout method.

    Returns:
        Pass manager that compiles a circuit for ``backend`` under ``spec``.

    Raises:
        OptionalDependencyError: Raised when qiskit pass-manager builders are unavailable.
        TranspilerError: Raised when the strategy names an unknown layout,
            routing, or translation method.
    """
    if _supports_preset_pass_manager(backend):
        try:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError(
                "qiskit preset pass managers are required (qiskit>=1.0)"
            ) from e

        return generate_preset_pass_manager(
            optimization_level=spec.optimization_level,
            backend=backend,
            layout_method=_preset_layout_method(spec),
            routing_method=spec.routing_method,
            translation_method=spec.translation_method,
            seed_transpiler=spec.seed_transpiler,
            initial_layout=initial_layout,
        )

    log.warning(
        "Backend %s exposes no transpiler Target; falling back to translation-only "
        "compilation, which ignores optimization_level, layout, and routing.",
        backend_display_name(backend),
    )
    return _generate_stage_pm(backend, spec, initial_layout=initial_layout)


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

    # Only the noise-aware layout varies per twirled circuit; every other
    # strategy yields the same pass manager, and building one for a large
    # backend is not free, so build it once for the whole ensemble.
    uses_noise_aware_layout = spec.layout_method == NOISE_AWARE_LAYOUT
    shared_pm = None if uses_noise_aware_layout else _generate_pm(backend, spec)

    for tw in twirled_ensemble:
        if uses_noise_aware_layout:
            initial_layout = None
            try:
                initial_layout = noise_aware_initial_layout(backend, tw)
            except Exception as e:
                log.warning(
                    "Noise-aware layout failed (continuing with the default layout): %s",
                    e,
                )
            pm = _generate_pm(backend, spec, initial_layout=initial_layout)
        else:
            pm = shared_pm

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
            "two_qubit_ops": int(_count_two_qubit_ops(out)),
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
