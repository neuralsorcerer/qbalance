# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from qbalance.errors import OptionalDependencyError, QBalanceError
from qbalance.logging import get_logger
from qbalance.utils import (
    SCHEDULING_DIRECTIVES,
    bit_index,
    instruction_parts,
    is_terminal_measurement,
)

log = get_logger(__name__)


def find_cuts_best_effort(
    circuit: Any,
    max_subcircuit_qubits: int,
    max_backjumps: int = 10_000,
    max_gamma: float = 1e6,
) -> Tuple[Any, Dict[str, Any]]:
    """Find gate and wire cuts that fit ``circuit`` into narrower subcircuits.

    This is ``qiskit_addon_cutting.find_cuts`` with the given optimization
    parameters and subcircuit width.

    Args:
        circuit: Circuit to cut; it must carry no classical bits.
        max_subcircuit_qubits: Widest subcircuit allowed.
        max_backjumps (default: 10000): Search budget of the cut finder.
        max_gamma (default: 1000000.0): Largest sampling overhead the cut
            finder may accept.

    Returns:
        ``(cut_circuit, metadata)``: the circuit with its cut instructions,
        and the finder's metadata (``cuts``, ``sampling_overhead``, ...).

    Raises:
        OptionalDependencyError: If qiskit-addon-cutting is not installed.
        QBalanceError: If the installed qiskit-addon-cutting does not
            expose the automated cut finder.
    """
    try:
        import qiskit_addon_cutting
    except Exception as e:
        raise OptionalDependencyError(
            "qiskit-addon-cutting is required (install qbalance[cutting])"
        ) from e

    # The automated cut finder lives at the package root.  Keep the legacy
    # submodule path as a fallback, but never report a missing symbol as a
    # missing package: that message sent users to reinstall a dependency they
    # already had.
    try:
        DeviceConstraints = qiskit_addon_cutting.DeviceConstraints
        OptimizationParameters = qiskit_addon_cutting.OptimizationParameters
        find_cuts = qiskit_addon_cutting.find_cuts
    except AttributeError:
        try:
            from qiskit_addon_cutting.cutting import (  # type: ignore[import-not-found,no-redef]
                DeviceConstraints,
                OptimizationParameters,
                find_cuts,
            )
        except Exception as e:
            raise QBalanceError(
                "Installed qiskit-addon-cutting does not expose the automated cut "
                f"finder (find_cuts/OptimizationParameters/DeviceConstraints): {e}"
            ) from e

    optimization = OptimizationParameters(
        max_backjumps=max_backjumps, max_gamma=max_gamma
    )
    subcircuit_qubits = int(max_subcircuit_qubits)
    try:
        constraints = DeviceConstraints(qubits_per_subcircuit=subcircuit_qubits)
    except TypeError:
        # Older releases spelled the same constraint differently.
        constraints = DeviceConstraints(max_subcircuit_width=subcircuit_qubits)

    cut_circuit, meta = find_cuts(
        circuit, optimization=optimization, constraints=constraints
    )
    return cut_circuit, dict(meta)


# Exact quasi-probability expansion enumerates every term of every cut; past
# this many subexperiments compiling and running them is impractical, and the
# sampling overhead is prohibitive anyway (4 CNOT cuts already cost 9**4).
MAX_SUBEXPERIMENTS = 1024


@dataclass
class CuttingExperiment:
    """Subexperiments that together reconstruct a cut circuit's Z-parity.

    Args:
        subexperiments: Subexperiment circuits per partition label.
        coefficients: Quasi-probability coefficients, one per sample.
        subobservables: Observable per partition label.
        measured_qubits: Qubits the original circuit measured; the parity
            observable is ``Z`` on each of them.  A measured qubit no gate acts
            on stays in ``|0>`` and contributes exactly ``+1``, so it is left
            out of the cut problem.
        metadata: JSON-safe description of the cut (``cut_count``,
            ``cut_types``, ``sampling_overhead``, ``num_subcircuits``,
            ``subcircuit_widths``, ``num_subexperiments``).
    """

    subexperiments: Dict[Hashable, List[Any]]
    coefficients: List[Tuple[float, Any]]
    subobservables: Dict[Hashable, Any]
    measured_qubits: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _unitary_part(circuit: Any) -> Tuple[Any, List[int]]:
    """Return ``circuit`` without its terminal measurements, plus the measured qubits.

    qiskit-addon-cutting rejects circuits with classical bits and gates on
    more than two qubits, so measurements are split off (they define the
    observable instead), scheduling directives dropped, and wider gates
    unrolled.
    """
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroll3qOrMore

    data = [instruction_parts(entry) for entry in circuit.data]
    unitary = QuantumCircuit(
        circuit.num_qubits,
        name=getattr(circuit, "name", None),
        global_phase=getattr(circuit, "global_phase", 0.0),
    )
    measured: List[int] = []
    for index, (inst, qargs, cargs) in enumerate(data):
        name = getattr(inst, "name", "")
        if name == "measure":
            if not is_terminal_measurement(data, index):
                raise ValueError("Circuit cutting supports only terminal measurements.")
            measured.append(bit_index(circuit, qargs[0]))
            continue
        if name in SCHEDULING_DIRECTIVES:
            continue
        if cargs:
            raise ValueError(
                f"Circuit cutting does not support {name!r}, which uses classical bits."
            )
        unitary.append(inst, [bit_index(circuit, qubit) for qubit in qargs])
    return PassManager([Unroll3qOrMore()]).run(unitary), measured


def _restrict_to_active_qubits(unitary: Any) -> Tuple[Any, List[int]]:
    """Return ``unitary`` over only the qubits some gate acts on, and those qubits.

    qiskit-addon-cutting labels a qubit no gate touches ``None`` when it
    partitions a circuit and then fails looking that partition up, so idle
    qubits have to be removed before the cut problem is built.  The compact
    circuit's qubit ``i`` is the ``i``-th entry of the returned list.
    """
    from qiskit import QuantumCircuit

    data = [instruction_parts(entry) for entry in unitary.data]
    active = sorted(
        {bit_index(unitary, qubit) for _, qargs, _ in data for qubit in qargs}
    )
    position = {qubit: index for index, qubit in enumerate(active)}
    compact = QuantumCircuit(
        len(active),
        name=getattr(unitary, "name", None),
        global_phase=getattr(unitary, "global_phase", 0.0),
    )
    for inst, qargs, _ in data:
        compact.append(inst, [position[bit_index(unitary, qubit)] for qubit in qargs])
    return compact, active


def _canonical_subexperiment(subexperiment: Any, name: str) -> Any:
    """Return ``subexperiment`` under fixed circuit and quantum-register names.

    qiskit-addon-cutting names the circuits and quantum registers it builds
    from Qiskit's global counters (``circuit-48`` with register ``q2`` in one
    run, ``circuit-68`` with ``q0`` in the next).  QPY records both names, so
    an identical subexperiment never matched its compile-cache entry: every
    rerun recompiled all of them and, with the compile time in the objective,
    re-ranked cut candidates on fresh timings.  Renaming makes identical
    subexperiments identical circuits.  The classical registers, which the
    reconstruction reads results by, are kept as they are.
    """
    from qiskit import QuantumCircuit, QuantumRegister

    cregs = list(subexperiment.cregs)
    if sum(register.size for register in cregs) != subexperiment.num_clbits:
        # Classical bits outside any register: keep the circuit as built.
        return subexperiment
    canonical = QuantumCircuit(
        QuantumRegister(subexperiment.num_qubits, "q"), *cregs, name=name
    )
    canonical.compose(
        subexperiment,
        qubits=list(range(subexperiment.num_qubits)),
        clbits=list(range(subexperiment.num_clbits)),
        inplace=True,
    )
    canonical.metadata = dict(subexperiment.metadata or {})
    return canonical


def prepare_cutting_experiment(
    circuit: Any,
    max_subcircuit_qubits: int,
    max_backjumps: int = 10_000,
    max_gamma: float = 1e6,
    max_subexperiments: int = MAX_SUBEXPERIMENTS,
) -> Optional[CuttingExperiment]:
    """Cut ``circuit`` into subcircuits of at most ``max_subcircuit_qubits`` qubits.

    The observable is the Z-parity of the measured qubits -- the quantity ZNE
    extrapolates -- and every quasi-probability term is enumerated exactly
    (``num_samples=inf``), which keeps the experiment deterministic.

    Args:
        circuit: Circuit to cut; every measurement must be terminal.
        max_subcircuit_qubits: Width limit for each subcircuit.
        max_backjumps (default: 10000): Cut-finder search budget.
        max_gamma (default: 1e6): Largest sampling overhead the cut finder accepts.
        max_subexperiments (default: MAX_SUBEXPERIMENTS): Refuse cuts that
            would need more subexperiments than this.

    Returns:
        The cutting experiment, or ``None`` when the circuit is no wider than
        ``max_subcircuit_qubits`` (or no gate acts on any qubit) and so runs
        uncut.  A wider circuit whose connected parts each fit needs no cut
        but is still split into those parts, one subcircuit each.

    Raises:
        OptionalDependencyError: If qiskit-addon-cutting is not installed.
        ValueError: If the circuit has non-terminal measurements or classically
            controlled operations, or the cut needs too many subexperiments.
    """
    try:
        import qiskit_addon_cutting as addon
        from qiskit.quantum_info import PauliList
    except Exception as e:
        raise OptionalDependencyError(
            "qiskit-addon-cutting is required (install qbalance[cutting])"
        ) from e

    unitary, measured = _unitary_part(circuit)
    if unitary.num_qubits <= int(max_subcircuit_qubits):
        return None
    compact, active = _restrict_to_active_qubits(unitary)
    if not active:
        # Every qubit stays in |0>: there is nothing to cut, and the parity of
        # the measured qubits is exactly +1 whichever way it runs.
        return None
    cut_circuit, meta = find_cuts_best_effort(
        compact, max_subcircuit_qubits, max_backjumps=max_backjumps, max_gamma=max_gamma
    )
    cuts = [tuple(cut) for cut in meta.get("cuts", [])]

    width = compact.num_qubits
    measured_set = set(measured)
    # Idle measured qubits read |0>, i.e. Z = +1, so the observable covers the
    # active measured qubits only.  Pauli labels are little-endian: the
    # rightmost character is qubit 0.
    observed = {index for index, qubit in enumerate(active) if qubit in measured_set}
    label = "".join(
        "Z" if width - 1 - position in observed else "I" for position in range(width)
    )
    wired = addon.cut_wires(cut_circuit)
    observables = addon.expand_observables(PauliList([label]), compact, wired)
    problem = addon.partition_problem(circuit=wired, observables=observables)

    terms = 1
    for basis in problem.bases:
        terms *= len(basis.maps)
    expected = terms * len(problem.subcircuits)
    if expected > max_subexperiments:
        raise ValueError(
            f"Cutting into {max_subcircuit_qubits}-qubit subcircuits needs "
            f"{expected} subexperiments, more than the limit of {max_subexperiments}."
        )

    subexperiments, coefficients = addon.generate_cutting_experiments(
        circuits=problem.subcircuits,
        observables=problem.subobservables,
        num_samples=np.inf,
    )
    labels = sorted(problem.subcircuits, key=str)
    metadata: Dict[str, Any] = {
        "cut_count": len(cuts),
        "cut_types": [str(cut[0]) for cut in cuts],
        "sampling_overhead": float(meta.get("sampling_overhead", 1.0)),
        "num_subcircuits": len(labels),
        "subcircuit_widths": [
            int(problem.subcircuits[lab].num_qubits) for lab in labels
        ],
        "num_subexperiments": int(sum(len(v) for v in subexperiments.values())),
    }
    base_name = str(getattr(circuit, "name", None) or "circuit")
    return CuttingExperiment(
        subexperiments={
            lab: [
                _canonical_subexperiment(sub, f"{base_name}_cut_{lab}")
                for sub in subexperiments[lab]
            ]
            for lab in labels
        },
        coefficients=list(coefficients),
        subobservables=dict(problem.subobservables),
        measured_qubits=sorted(measured_set),
        metadata=metadata,
    )


def combine_subexperiment_metrics(
    metrics: Sequence[Mapping[str, Any]], experiment: CuttingExperiment
) -> Dict[str, Any]:
    """Describe a cut candidate by its compiled subexperiments.

    Every subexperiment is an independent circuit on the device, so per-shot
    quality is that of the worst one: structural metrics and
    ``estimated_error`` are the maxima.  ``compile_time_s`` is the total, and
    the cut's own description -- notably ``sampling_overhead``, the factor by
    which shots must grow for the same precision -- is merged in.

    Args:
        metrics: Compile metrics of every subexperiment.
        experiment: The experiment the subexperiments belong to.

    Returns:
        Candidate metrics for the cut strategy.

    Raises:
        ValueError: If ``metrics`` is empty.
    """
    if not metrics:
        raise ValueError("a cutting experiment needs at least one subexperiment")
    errors = [
        float(m["estimated_error"])
        for m in metrics
        if isinstance(m.get("estimated_error"), (int, float))
    ]
    combined: Dict[str, Any] = {
        "compile_time_s": float(
            sum(float(m.get("compile_time_s", 0.0)) for m in metrics)
        ),
        "depth": max(int(m["depth"]) for m in metrics),
        "size": max(int(m["size"]) for m in metrics),
        "width": max(int(m["width"]) for m in metrics),
        "two_qubit_ops": max(int(m["two_qubit_ops"]) for m in metrics),
        "estimated_error": max(errors) if errors else None,
        "dd_applied": all(bool(m.get("dd_applied")) for m in metrics),
        "measurement_flip_map": {},
        "measurement_flip_maps": [{}],
        "twirl_instances": 1,
    }
    labels = {m.get("dd_sequence_applied") for m in metrics}
    if combined["dd_applied"] and len(labels) == 1 and None not in labels:
        combined["dd_sequence_applied"] = labels.pop()
    profiles = [m.get("pass_profile") for m in metrics]
    mappings = [p for p in profiles if isinstance(p, Mapping)]
    if len(mappings) == len(profiles):
        combined["pass_profile"] = {
            "total_time_s": float(
                sum(float(p.get("total_time_s", 0.0)) for p in mappings)
            ),
            "passes": [entry for p in mappings for entry in p.get("passes", [])],
        }
    combined.update(experiment.metadata)
    return combined


def run_cutting_experiment(
    experiment: CuttingExperiment,
    compiled: Mapping[Hashable, Sequence[Any]],
    backend: Any,
    shots: int,
    seed: Optional[int] = None,
) -> float:
    """Run compiled subexperiments and reconstruct the circuit's Z-parity.

    Args:
        experiment: The cutting experiment.
        compiled: Compiled subexperiments per partition label, in the order of
            ``experiment.subexperiments``.
        backend: Backend to run on.
        shots: Shots per subexperiment.
        seed (default: None): Simulator seed, forwarded only to backends whose
            ``run`` accepts ``seed_simulator``.

    Returns:
        The reconstructed ``<Z x ... x Z>`` over the measured qubits.

    Raises:
        ValueError: If the original circuit measured nothing.
    """
    if not experiment.measured_qubits:
        raise ValueError(
            "Circuit has no measurements, so a cut execution has nothing to "
            "reconstruct."
        )
    from qiskit.primitives import BackendSamplerV2
    from qiskit_addon_cutting import reconstruct_expectation_values

    from qbalance.execution.runner import _ensure_backend, _prepare_run_kwargs

    runnable = _ensure_backend(backend)
    options: Dict[str, Any] = {"default_shots": int(shots)}
    if seed is not None and _prepare_run_kwargs(runnable.run, {"seed_simulator": seed}):
        options["seed_simulator"] = int(seed)
    sampler = BackendSamplerV2(backend=runnable, options=options)
    results = {
        lab: sampler.run(list(compiled[lab])).result()
        for lab in experiment.subexperiments
    }
    values = reconstruct_expectation_values(
        results, experiment.coefficients, experiment.subobservables
    )
    return float(values[0])


def evaluate_cutting_experiment(
    experiment: CuttingExperiment,
    backend: Any,
    compile_subexperiment: Callable[[Any], Tuple[Sequence[Any], Mapping[str, Any]]],
    *,
    execute: bool,
    shots: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compile every subexperiment of a cut and optionally run the cut.

    Args:
        experiment: The cutting experiment.
        backend: Backend to run on.
        compile_subexperiment: Compiles one subexperiment circuit into
            ``(instances, metrics)``.  A cut strategy cannot twirl, so its
            first instance is the one that runs.
        execute: Whether to run the subexperiments and reconstruct.
        shots: Shots per subexperiment.
        seed (default: None): Simulator seed, as for ``run_cutting_experiment``.

    Returns:
        The metrics of ``combine_subexperiment_metrics``, plus
        ``cut_parity_expval`` and ``cut_shots_total`` after a run or
        ``exec_error`` when the run failed.  When a subexperiment cannot be
        compiled, ``compile_error`` together with the cut's description.
    """
    compiled: Dict[Hashable, List[Any]] = {}
    sub_metrics: List[Mapping[str, Any]] = []
    try:
        for label, circuits in experiment.subexperiments.items():
            compiled[label] = []
            for sub in circuits:
                sub_instances, sub_m = compile_subexperiment(sub)
                compiled[label].append(sub_instances[0])
                sub_metrics.append(sub_m)
    except Exception as e:
        return {"compile_error": str(e), **experiment.metadata}

    m = combine_subexperiment_metrics(sub_metrics, experiment)
    if execute:
        try:
            m["cut_parity_expval"] = run_cutting_experiment(
                experiment, compiled, backend, shots=shots, seed=seed
            )
            m["cut_shots_total"] = int(shots) * int(m["num_subexperiments"])
        except Exception as e:
            m["exec_error"] = str(e)
    return m
