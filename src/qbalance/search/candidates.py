# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
        None.
    """
    if max_candidates <= 0:
        return []

    _ = seed  # reserved for future deterministic candidate randomization

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

    # Basic compilation sweep
    for opt in (0, 1, 2, 3):
        if _add(StrategySpec(optimization_level=opt)):
            return uniq
        if _add(StrategySpec(optimization_level=opt, routing_method="sabre")):
            return uniq
        if _add(
            StrategySpec(
                optimization_level=opt, layout_method="sabre", routing_method="sabre"
            )
        ):
            return uniq
        if _add(
            StrategySpec(
                optimization_level=opt,
                layout_method="qbalance_noise_aware",
                routing_method="sabre",
            )
        ):
            return uniq

    # Suppression variants
    if _add(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            pauli_twirling=True,
            num_twirls=8,
        )
    ):
        return uniq
    if _add(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            dynamical_decoupling=True,
            dd_sequence="XY4",
        )
    ):
        return uniq
    if _add(
        StrategySpec(
            optimization_level=2, routing_method="sabre", measurement_twirling=True
        )
    ):
        return uniq

    # Combine: twirling + DD
    if _add(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            pauli_twirling=True,
            num_twirls=8,
            dynamical_decoupling=True,
            dd_sequence="XY4",
            measurement_twirling=True,
        )
    ):
        return uniq

    # Mitigation toggles (execution stage required)
    if _add(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            mthree=True,
            measurement_twirling=True,
        )
    ):
        return uniq
    if _add(
        StrategySpec(
            optimization_level=2,
            routing_method="sabre",
            zne=True,
            measurement_twirling=True,
        )
    ):
        return uniq

    # Cutting (optional)
    _add(StrategySpec(optimization_level=1, cutting=True, max_subcircuit_qubits=4))

    return uniq
