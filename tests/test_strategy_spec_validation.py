# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from qbalance.strategies import StrategySpec


@pytest.mark.parametrize("bad", [0, -1, False])
def test_strategy_spec_rejects_invalid_num_twirls(bad):
    with pytest.raises(ValueError, match="num_twirls"):
        StrategySpec(num_twirls=bad)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"zne": True, "zne_factors": ()}, "non-empty"),
        ({"zne": True, "zne_factors": (1.0, 0.5, 2.0)}, ">= 1.0"),
        ({"zne": True, "zne_factors": (1.0, 3.0, 2.0)}, "sorted"),
        ({"zne": True, "zne_factors": (2.0, 3.0, 4.0)}, "include 1.0"),
        ({"zne": True, "zne_degree": 0}, "zne_degree must be >= 1"),
        (
            {"zne": True, "zne_factors": (1.0, 2.0), "zne_degree": 2},
            r"less than len\(zne_factors\)",
        ),
    ],
)
def test_strategy_spec_zne_validations_when_enabled(kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        StrategySpec(**kwargs)


@pytest.mark.parametrize("bad", [-1])
def test_strategy_spec_rejects_invalid_zne_degree_globally(bad):
    with pytest.raises(ValueError, match="zne_degree"):
        StrategySpec(zne_degree=bad)


def test_strategy_spec_requires_cutting_qubit_budget_when_enabled():
    with pytest.raises(ValueError, match="max_subcircuit_qubits"):
        StrategySpec(cutting=True)


@pytest.mark.parametrize("bad", [0, -1, False])
def test_strategy_spec_rejects_invalid_cutting_qubit_budget(bad):
    with pytest.raises(ValueError, match="max_subcircuit_qubits"):
        StrategySpec(cutting=True, max_subcircuit_qubits=bad)


@pytest.mark.parametrize("bad", [-1, 3, 5])
def test_strategy_spec_resilience_level_bounds(bad):
    with pytest.raises(ValueError, match="resilience_level"):
        StrategySpec(resilience_level=bad)


def test_strategy_spec_accepts_valid_combination():
    spec = StrategySpec(
        zne=True,
        zne_factors=(1.0, 2.0, 3.0),
        zne_degree=1,
        cutting=True,
        max_subcircuit_qubits=5,
        resilience_level=2,
    )
    assert spec.zne is True
    assert spec.max_subcircuit_qubits == 5


def test_strategy_spec_allows_default_zne_fields_when_disabled():
    spec = StrategySpec(zne=False, zne_degree=0)
    assert spec.zne is False
    assert spec.zne_degree == 0
