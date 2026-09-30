# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
from importlib import import_module
from typing import Any, Dict, List, Tuple

from qbalance.errors import OptionalDependencyError, QBalanceError
from qbalance.logging import get_logger
from qbalance.mitigation.zne import zne_sampling_overhead
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
    dd_sequence_label,
    resolve_dd_sequence,
)
from qbalance.utils import (
    SCHEDULING_DIRECTIVES,
    backend_display_name,
    instruction_parts,
    operation_depth,
    operation_size,
)

log = get_logger(__name__)

_DIRECTIVE_NAMES = SCHEDULING_DIRECTIVES

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
        backend: Backend to translate for; its basis gates are the
            operations of its ``target``, or else the ``basis_gates`` of
            its ``configuration()``.
        spec: Strategy; only ``translation_method`` applies (default
            ``"translator"``).
        initial_layout (default: None): Layout applied before translation,
            when the backend has a ``target``.

    Returns:
        A pass manager that applies the layout (when given), unrolls gates on
        three or more qubits, and translates to the backend's basis.

    Raises:
        OptionalDependencyError: If qiskit's stage generators are unavailable.
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
        backend: Backend to compile for.
        spec: Strategy/backend specification controlling compilation behavior.
        initial_layout (default: None): Explicit initial layout, used for the
            ``qbalance_noise_aware`` layout method.

    Returns:
        Pass manager that compiles a circuit for ``backend`` under ``spec``.

    Raises:
        OptionalDependencyError: If qiskit's pass-manager builders are unavailable.
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


def _restore_layout(result: Any, source: Any) -> Any:
    """Carry ``source``'s TranspileLayout over to ``result`` when it was dropped.

    A separate ``PassManager.run`` (dynamical decoupling, re-translation) starts
    from an empty property set, so its output carries no TranspileLayout even
    though it keeps the qubits the compile placed.  Without this every such
    strategy loses the virtual-to-physical mapping callers need for observables.
    """
    if getattr(result, "_layout", None) is None:
        layout = getattr(source, "_layout", None)
        if layout is not None:
            result._layout = layout
    return result


def _instance_seed(seed: Any, index: int) -> Any:
    """Return the suppression seed for ensemble instance ``index``.

    Instance 0 keeps the strategy's own seed; later instances get distinct,
    reproducible seeds so each draws its own random flip pattern.
    """
    return None if seed is None else int(seed) + index


def _circuit_metrics(backend: Any, circuit: Any) -> Dict[str, Any]:
    """Structural and estimated-error metrics of one compiled circuit."""
    metrics: Dict[str, Any] = {
        # Idle delays (dynamical decoupling pads every idle window with them)
        # are schedule, not operations: counting them would make a scheduled
        # circuit look deeper than the identical unscheduled one.
        "depth": operation_depth(circuit),
        "size": operation_size(circuit),
        "width": int(circuit.num_qubits),
        "two_qubit_ops": int(_count_two_qubit_ops(circuit)),
    }
    try:
        metrics["estimated_error"] = float(estimate_circuit_error(backend, circuit))
    except Exception:
        metrics["estimated_error"] = None
    return metrics


def _twirl_instance_count(spec: StrategySpec) -> int:
    """Number of randomized instances a strategy compiles into."""
    if spec.pauli_twirling or spec.measurement_twirling:
        return max(1, int(spec.num_twirls))
    return 1


def compile_ensemble(
    circuit: Any,
    backend: Any,
    spec: StrategySpec,
    profile: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Compile ``circuit`` once and expand it into its twirled instances.

    The circuit is transpiled a single time.  Pauli twirling is then applied
    to the compiled, target-native circuit -- as Qiskit recommends for
    :func:`~qiskit.circuit.pauli_twirl_2q_gates`, whose ``target`` keeps the
    inserted Paulis native.  Twirling *before* transpilation, as earlier
    releases did, lets optimization re-synthesize the twirled two-qubit blocks
    and erase the twirl.  Measurement twirling and then dynamical decoupling
    are applied to every instance, each instance drawing its own flip pattern;
    DD comes last because it schedules the circuit it pads.

    A twirl only suppresses noise on average over its random instances, so all
    of them are returned; executing a single one gains nothing.  Without
    twirling the ensemble has one instance.

    Args:
        circuit: QuantumCircuit to compile.
        backend: Backend (or backend-like handle) to compile for.
        spec: Strategy controlling compilation and suppression.
        profile (default: False): Record pass-level transpiler profiling.

    Returns:
        ``(instances, metrics)``.  ``metrics`` describes the whole ensemble:
        structural metrics and ``estimated_error`` are the maxima over the
        instances, ``measurement_flip_maps`` holds one flip map per instance
        (``measurement_flip_map`` repeats the first), and ``twirl_instances``
        is the ensemble size.

    Raises:
        OptionalDependencyError: If qiskit is unavailable.
    """
    try:
        import_module("qiskit.converters")
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit required") from e

    profile_report = ProfileReport()
    initial_layout = None
    if spec.layout_method == NOISE_AWARE_LAYOUT:
        try:
            initial_layout = noise_aware_initial_layout(backend, circuit)
        except Exception as e:
            log.warning(
                "Noise-aware layout failed (continuing with the default layout): %s",
                e,
            )
    pm = _generate_pm(backend, spec, initial_layout=initial_layout)

    cb = make_callback(profile_report) if profile else None
    t0 = time.time()
    compiled = pm.run(circuit, callback=cb) if cb is not None else pm.run(circuit)
    t1 = time.time()

    target = getattr(backend, "target", None)
    count = _twirl_instance_count(spec)
    instances: List[Any] = [compiled]
    if spec.pauli_twirling:
        try:
            twirled = apply_pauli_twirling(
                compiled,
                num_twirls=count,
                seed=spec.seed_suppression,
                target=target,
            )
            if target is None:
                # Without a Target the inserted Paulis are not synthesized to
                # the backend basis; re-translate so every instance still runs.
                retranslate = _generate_stage_pm(backend, spec)
                twirled = [retranslate.run(instance) for instance in twirled]
            instances = [_restore_layout(instance, compiled) for instance in twirled]
        except Exception as e:
            # Continuing untwirled would report -- and let selection pick -- a
            # pauli_twirling strategy that twirled nothing.  Like any other
            # strategy the backend cannot compile, it is infeasible instead.
            raise QBalanceError(f"Pauli twirling failed for this backend: {e}") from e
    if spec.measurement_twirling and len(instances) < count:
        # Measurement twirling alone still needs one instance per flip pattern.
        instances = instances + [compiled] * (count - len(instances))

    dd_pm = None
    dd_label = None
    if spec.dynamical_decoupling:
        try:
            dd_pm = build_dd_pass_manager(backend, spec.dd_sequence)
        except Exception as e:
            log.warning("DD insertion failed (continuing without DD): %s", e)
        else:
            try:
                dd_label = dd_sequence_label(
                    resolve_dd_sequence(backend, spec.dd_sequence)
                )
            except Exception:
                dd_label = None

    finished: List[Any] = []
    flip_maps: List[Dict[int, int]] = []
    dd_flags: List[bool] = []
    for index, instance in enumerate(instances):
        out = instance
        flip_map: Dict[int, int] = {}
        if spec.measurement_twirling:
            try:
                out, flip_map = apply_measurement_twirling(
                    out, seed=_instance_seed(spec.seed_suppression, index)
                )
            except Exception as e:
                log.warning("Measurement twirling failed (continuing): %s", e)

        # DD schedules the circuit and pads its idle windows, so it runs last:
        # a gate inserted afterwards (a measurement-twirl X) would shift the
        # measurement it precedes past the schedule the padding was built for.
        dd_applied = False
        if dd_pm is not None:
            try:
                out = _restore_layout(dd_pm.run(out), instance)
                dd_applied = True
            except Exception as e:
                log.warning("DD insertion failed (continuing without DD): %s", e)

        finished.append(out)
        flip_maps.append(flip_map)
        dd_flags.append(dd_applied)

    per_instance = [_circuit_metrics(backend, out) for out in finished]
    errors = [
        m["estimated_error"] for m in per_instance if m["estimated_error"] is not None
    ]
    metrics: Dict[str, Any] = {
        "compile_time_s": float(t1 - t0),
        "depth": max(m["depth"] for m in per_instance),
        "size": max(m["size"] for m in per_instance),
        "width": max(m["width"] for m in per_instance),
        "two_qubit_ops": max(m["two_qubit_ops"] for m in per_instance),
        "estimated_error": max(errors) if errors else None,
        "dd_applied": bool(dd_flags) and all(dd_flags),
        "measurement_flip_map": flip_maps[0],
        "measurement_flip_maps": flip_maps,
        "twirl_instances": len(finished),
        # The shots a given precision costs relative to running the circuit
        # once: 1 unless ZNE must also run it at every noise factor and
        # extrapolate (cutting overrides this with its quasi-probability cost).
        "sampling_overhead": (
            zne_sampling_overhead(spec.zne_factors, spec.zne_degree)
            if spec.zne
            else 1.0
        ),
    }
    if metrics["dd_applied"] and dd_label is not None:
        metrics["dd_sequence_applied"] = dd_label
    if profile:
        metrics["pass_profile"] = profile_report.to_json()
    return finished, metrics


def compile_one(
    circuit: Any,
    backend: Any,
    spec: StrategySpec,
    profile: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Compile ``circuit`` and return its first instance with ensemble metrics.

    Equivalent to :func:`compile_ensemble` for strategies without twirling.
    With twirling the metrics describe the whole ensemble; run every instance
    from :func:`compile_ensemble` to obtain the twirled result.

    Args:
        circuit: QuantumCircuit instance to compile.
        backend: Backend object (or backend-like handle) to compile for.
        spec: Strategy specification controlling compilation behavior.
        profile (default: False): Whether pass-level transpiler profiling is enabled.

    Returns:
        ``(compiled_circuit, metrics)``.

    Raises:
        OptionalDependencyError: If qiskit is unavailable.
    """
    instances, metrics = compile_ensemble(circuit, backend, spec, profile=profile)
    return instances[0], metrics
