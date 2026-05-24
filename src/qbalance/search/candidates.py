# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import List

from qbalance.strategies import StrategySpec


def default_candidate_strategies(
    max_candidates: int = 24, seed: int = 0
) -> List[StrategySpec]:
    """Return the default candidate strategies configuration used by qbalance.

    Args:
        max_candidates (default: 24): Max candidates value consumed by this routine.
        seed (default: 0): Seed used for deterministic randomization.

    Returns:
        List[StrategySpec] with the computed result.

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
        """Internal helper that add.

        Args:
            spec: Strategy/backend specification controlling compilation behavior.

        Returns:
            bool with the computed result.

        Raises:
            None.
        """
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
    pool.append(
        StrategySpec(
            optimization_level=2, routing_method="sabre", measurement_twirling=True
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
        )
    )
    pool.append(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            zne=True,
            measurement_twirling=True,
        )
    )

    # Cutting (optional)
    pool.append(
        StrategySpec(optimization_level=1, cutting=True, max_subcircuit_qubits=4)
    )

    if len(pool) > 1:
        head, tail = pool[0], pool[1:]
        random.Random(seed).shuffle(tail)
        pool = [head, *tail]

    for spec in pool:
        if _add(spec):
            return uniq

    return uniq
