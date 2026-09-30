# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Executing a twirled ensemble of compiled instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from qbalance.execution.runner import run_counts
from qbalance.mitigation.mthree import mitigate_twirled_counts
from qbalance.mitigation.zne import (
    fold_global_for_backend,
    realized_fold_factor,
    zne_extrapolate_counts,
)
from qbalance.transpile.suppression import (
    apply_measurement_untwirl_counts,
    normalize_measurement_flip_map,
)
from qbalance.utils import measured_clbits, measured_qubits_by_clbit


def split_shots(shots: int, instances: int) -> List[int]:
    """Split ``shots`` across ``instances`` as evenly as possible.

    The first ``shots % instances`` instances receive one extra shot, so the
    shares always sum to ``shots``.  When there are more instances than shots
    the surplus instances get zero shots and are simply not run.

    Args:
        shots: Total shot budget, ``>= 1``.
        instances: Number of ensemble instances, ``>= 1``.

    Returns:
        One shot count per instance.

    Raises:
        ValueError: If either argument is not positive.
    """
    if shots < 1 or instances < 1:
        raise ValueError("shots and instances must both be positive")
    base, extra = divmod(int(shots), int(instances))
    return [base + (1 if index < extra else 0) for index in range(instances)]


def instance_flip_maps(
    metrics: Mapping[str, Any], instances: int
) -> List[Dict[int, int]]:
    """Return the measurement-twirl flip map of every ensemble instance.

    Args:
        metrics: Compile metrics carrying ``measurement_flip_maps`` (one map per
            instance) or, for single-instance results, ``measurement_flip_map``.
        instances: Number of compiled instances.

    Returns:
        One normalized flip map per instance.

    Raises:
        ValueError: If the recorded flip maps do not match the instance count.
    """
    maps = metrics.get("measurement_flip_maps")
    if isinstance(maps, list) and len(maps) == instances:
        return [normalize_measurement_flip_map(flip_map) for flip_map in maps]
    if instances == 1:
        return [normalize_measurement_flip_map(metrics.get("measurement_flip_map"))]
    raise ValueError(
        f"expected {instances} measurement flip maps, found "
        f"{len(maps) if isinstance(maps, list) else 'none'}"
    )


def merge_counts(distributions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Sum counts (or weights) key by key across instances."""
    merged: Dict[str, Any] = {}
    for distribution in distributions:
        for key, value in distribution.items():
            merged[key] = merged.get(key, 0) + value
    return merged


@dataclass
class EnsembleRun:
    """Outcome of running one compiled ensemble, with optional mitigation.

    Attributes:
        counts: Untwirled counts of every instance, merged.
        mthree_probs: M3-mitigated quasi-probabilities, when M3 was requested
            and succeeded.
        mthree_error: Why M3 failed, when it was requested and did.
        zne_realized_factors: Noise scale factors the folded circuits actually
            ran at, recorded once every folded run succeeded.
        zne_probs: Zero-noise extrapolated probabilities, when ZNE was
            requested and succeeded.
        zne_error: Why ZNE failed, when it was requested and did.
    """

    counts: Dict[str, Any]
    mthree_probs: Optional[Dict[str, float]] = None
    mthree_error: Optional[str] = None
    zne_realized_factors: Optional[List[float]] = None
    zne_probs: Optional[Dict[str, float]] = None
    zne_error: Optional[str] = None


def run_ensemble(
    backend: Any,
    instances: Sequence[Any],
    metrics: Mapping[str, Any],
    *,
    shots: int,
    seed: int,
    mthree: bool = False,
    zne_factors: Sequence[float] = (),
    zne_degree: int = 1,
) -> EnsembleRun:
    """Run a compiled ensemble and apply the requested mitigation.

    A twirl only suppresses noise averaged over its instances, so the shot
    budget is split across all of them (see ``split_shots``), instance ``i``
    runs with ``seed_simulator=seed + i``, and each instance's counts are
    untwirled with its own flip map before they are merged.

    M3 corrects each instance's counts as measured and untwirls afterwards,
    because its calibration describes the physical readout channel.  ZNE folds
    every instance at each factor and reruns it with the same share and seed;
    a factor that realizes 1 reuses the unfolded runs.  The fit is against the
    realized factors, not the requested ones they were rounded from.

    Args:
        backend: Backend to run on.
        instances: Compiled ensemble instances.
        metrics: Their compile metrics, carrying the measurement flip maps.
        shots: Total shot budget, split across the instances.
        seed: Base simulator seed.
        mthree (default: False): Whether to apply M3 readout mitigation.
        zne_factors (default: ()): ZNE noise scale factors; empty skips ZNE.
        zne_degree (default: 1): Polynomial degree of the ZNE fit.

    Returns:
        The merged counts plus each requested mitigation's estimate or error.

    Raises:
        Exception: Whatever the unmitigated runs raise.  A mitigation failure
            is recorded in the result instead, so it keeps the counts.
    """
    flip_maps = instance_flip_maps(metrics, len(instances))
    runs = [
        (index, instance, flip_map, share)
        for index, (instance, flip_map, share) in enumerate(
            zip(instances, flip_maps, split_shots(shots, len(instances)))
        )
        if share > 0
    ]
    raw_counts = [
        run_counts(backend, instance, shots=share, seed_simulator=seed + index)
        for index, instance, _, share in runs
    ]
    counts = merge_counts(
        apply_measurement_untwirl_counts(raw, flip_map)
        for raw, (_, _, flip_map, _) in zip(raw_counts, runs)
    )
    result = EnsembleRun(counts=counts)
    if mthree:
        try:
            # Untwirling first would hand M3 bits whose 0->1 and 1->0 error
            # rates are swapped wherever a twirl flip was applied.
            result.mthree_probs = mitigate_twirled_counts(
                backend,
                raw_counts,
                [flip_map for _, _, flip_map, _ in runs],
                measured_qubits=measured_qubits_by_clbit(instances[0]),
                seed=seed,
                clbits=measured_clbits(instances[0]),
            )
        except Exception as e:
            result.mthree_error = str(e)
    if zne_factors:
        try:
            counts_per_factor = []
            for factor in zne_factors:
                if realized_fold_factor(factor) == 1:
                    # Unfolded: the very runs (circuits, shares and seeds)
                    # already made above.
                    counts_per_factor.append(counts)
                    continue
                folded_counts = []
                for index, instance, flip_map, share in runs:
                    folded = fold_global_for_backend(instance, backend, factor)
                    folded_raw = run_counts(
                        backend, folded, shots=share, seed_simulator=seed + index
                    )
                    folded_counts.append(
                        apply_measurement_untwirl_counts(folded_raw, flip_map)
                    )
                counts_per_factor.append(merge_counts(folded_counts))
            realized = [float(realized_fold_factor(f)) for f in zne_factors]
            result.zne_realized_factors = realized
            result.zne_probs = zne_extrapolate_counts(
                realized, counts_per_factor, degree=zne_degree
            )
        except Exception as e:
            result.zne_error = str(e)
    return result
