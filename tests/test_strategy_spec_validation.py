# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pydantic
import pytest

from qbalance.strategies import StrategySpec, coerce_strategy_specs


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
        max_subcircuit_qubits=5,
        resilience_level=2,
    )
    assert spec.zne is True
    assert spec.max_subcircuit_qubits == 5

    cut = StrategySpec(
        cutting=True,
        max_subcircuit_qubits=5,
        dynamical_decoupling=True,
        optimization_level=3,
    )
    assert cut.cutting is True


def test_strategy_spec_allows_default_zne_fields_when_disabled():
    spec = StrategySpec(zne=False, zne_degree=0)
    assert spec.zne is False
    assert spec.zne_degree == 0


def test_strategy_spec_rejects_unknown_fields():
    """Regression: a misspelled key silently produced a default strategy.

    qbalance exists to compare configurations, so dropping an unknown key would
    make several visibly different entries collapse into one identical spec and
    a report claiming they performed the same.
    """
    for kwargs in (
        {"optimisation_level": 0},
        {"routing": "sabre"},
        {"dd": True},
        {"bogus": 1},
    ):
        with pytest.raises(pydantic.ValidationError):
            StrategySpec(**kwargs)

    specs = [
        {"optimisation_level": 0},
        {"optimization_level": 1},
    ]
    with pytest.raises(ValueError):
        coerce_strategy_specs(specs)


def test_strategy_spec_accepts_and_round_trips_every_field():
    full = {
        "optimization_level": 2,
        "layout_method": "sabre",
        "routing_method": "sabre",
        "translation_method": "translator",
        "seed_transpiler": 1,
        "pauli_twirling": True,
        "num_twirls": 4,
        "dynamical_decoupling": True,
        "dd_sequence": "XY4",
        "measurement_twirling": True,
        "seed_suppression": 2,
        "mthree": True,
        "zne": True,
        "zne_factors": (1.0, 3.0, 5.0),
        "zne_degree": 2,
        # cutting cannot combine with twirling or mitigation; the cutting
        # combinations are covered by test_cutting_rejects_unsupported_knobs.
        "cutting": False,
        "max_subcircuit_qubits": 3,
        "resilience_level": 1,
    }
    spec = StrategySpec(**full)

    assert StrategySpec(**spec.model_dump()) == spec
    assert StrategySpec(**json.loads(spec.model_dump_json())) == spec


def test_dd_sequence_is_validated_and_normalized():
    """An unknown sequence used to compile as XY4 while reported as itself."""
    assert StrategySpec(dd_sequence=" xx ").dd_sequence == "XX"
    assert StrategySpec(dd_sequence="yy") == StrategySpec(dd_sequence="YY")
    assert StrategySpec().dd_sequence == "XY4"

    for bad in ("XY8", "", None, 4):
        with pytest.raises(ValueError, match="dd_sequence must be"):
            StrategySpec(dd_sequence=bad)


@pytest.mark.parametrize(
    "knob", ["pauli_twirling", "measurement_twirling", "mthree", "zne"]
)
def test_cutting_rejects_unsupported_knobs(knob):
    """A cut circuit yields an expectation value, not one count distribution.

    Twirl flips, M3 and ZNE folding all act on a single count distribution,
    so combining them with cutting used to be accepted and then silently do
    nothing.
    """
    with pytest.raises(
        ValueError, match=f"cutting=True cannot be combined with {knob}"
    ):
        StrategySpec(cutting=True, max_subcircuit_qubits=4, **{knob: True})

    assert StrategySpec(
        cutting=True, max_subcircuit_qubits=4, dynamical_decoupling=True
    )


@pytest.mark.parametrize("field", ["seed_transpiler", "seed_suppression"])
def test_seeds_must_be_non_negative(field):
    """Negative seeds reach numpy/Qiskit RNGs that reject them mid-compile."""
    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        StrategySpec(**{field: -1})

    assert getattr(StrategySpec(**{field: 0}), field) == 0
    assert getattr(StrategySpec(**{field: None}), field) is None


def test_zne_degree_must_fit_the_realized_fold_factors():
    """(1, 2, 3) folds at (1, 3, 3): two noise levels cannot fit a quadratic."""
    with pytest.raises(ValueError, match="distinct fold factors"):
        StrategySpec(zne=True, zne_factors=(1.0, 2.0, 3.0), zne_degree=2)

    assert StrategySpec(zne=True, zne_factors=(1.0, 3.0, 5.0), zne_degree=2)
    # The default realizes three distinct levels.
    assert StrategySpec().zne_factors == (1.0, 3.0, 5.0)
    assert StrategySpec(zne=True, zne_degree=2).zne_degree == 2
