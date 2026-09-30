# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import List

from qbalance.strategies import StrategySpec

# Instances per measurement-twirled candidate, as for the Pauli-twirled ones.
_MEASUREMENT_TWIRLS = 8


def default_candidate_strategies(
    max_candidates: int = 24, seed: int = 0
) -> List[StrategySpec]:
    """Return the built-in candidate strategies ``adjust`` searches by default.

    The pool holds 23 distinct strategies: optimization levels 0 to 3 with
    default routing, SABRE routing, SABRE layout and routing, and the
    noise-aware layout with SABRE routing; at level 2 with SABRE routing,
    Pauli twirling, dynamical decoupling, measurement twirling, all three
    combined, M3 with measurement twirling, and ZNE with measurement
    twirling (8 twirl instances each); and a cutting candidate with 4-qubit
    subcircuits.  ``StrategySpec(optimization_level=0)`` stays first and
    the rest are shuffled with ``random.Random(seed)``.

    Args:
        max_candidates (default: 24): How many strategies to return, at most.
        seed (default: 0): Seed of the shuffle.

    Returns:
        The first ``max_candidates`` strategies of the shuffled pool; ``[]``
        when ``max_candidates <= 0``.

    Raises:
        ValueError: If max_candidates is not an integer, or if seed is not an integer.
    """
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
        raise ValueError("max_candidates must be an integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    if max_candidates <= 0:
        return []

    uniq: List[StrategySpec] = []
    seen = set()

    def _add(spec: StrategySpec) -> bool:
        """Record ``spec`` unless seen; return whether the pool is now full."""
        if spec in seen:
            return False
        seen.add(spec)
        uniq.append(spec)
        return len(uniq) >= max_candidates

    # Build full candidate pool first; later shuffle deterministically by seed.
    pool: List[StrategySpec] = []

    # Basic compilation sweep
    for opt in (0, 1, 2, 3):
        pool.append(StrategySpec(optimization_level=opt))
        pool.append(StrategySpec(optimization_level=opt, routing_method="sabre"))
        pool.append(
            StrategySpec(
                optimization_level=opt, layout_method="sabre", routing_method="sabre"
            )
        )
        pool.append(
            StrategySpec(
                optimization_level=opt,
                layout_method="qbalance_noise_aware",
                routing_method="sabre",
            )
        )

    # Suppression variants
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            pauli_twirling=True,
            num_twirls=8,
        )
    )
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            dynamical_decoupling=True,
            dd_sequence="XY4",
        )
    )
    # A measurement twirl symmetrizes readout error only averaged over its
    # flip patterns; a single instance is one fixed pattern, which merely
    # relabels which outcomes suffer the larger error.
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            measurement_twirling=True,
            num_twirls=_MEASUREMENT_TWIRLS,
        )
    )

    # Combine: twirling + DD
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            pauli_twirling=True,
            num_twirls=8,
            dynamical_decoupling=True,
            dd_sequence="XY4",
            measurement_twirling=True,
        )
    )

    # Mitigation toggles (execution stage required)
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            mthree=True,
            measurement_twirling=True,
            num_twirls=_MEASUREMENT_TWIRLS,
        )
    )
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            zne=True,
            measurement_twirling=True,
            num_twirls=_MEASUREMENT_TWIRLS,
        )
    )

    # Cutting (optional).  Subexperiments are small, and at optimization
    # level 1 Qiskit keeps a small circuit on qubits 0.. whenever that trivial
    # layout fits the coupling map, whatever their error rates; the noise-aware
    # layout places every subexperiment on good qubits instead.
    pool.append(
        StrategySpec(
            optimization_level=1,
            layout_method="qbalance_noise_aware",
            routing_method="sabre",
            cutting=True,
            max_subcircuit_qubits=4,
        )
    )

    if len(pool) > 1:
        head, tail = pool[0], pool[1:]
        random.Random(seed).shuffle(tail)
        pool = [head, *tail]

    for spec in pool:
        if _add(spec):
            return uniq

    return uniq
