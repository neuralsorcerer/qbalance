# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math

import pytest

from qbalance.strategies import StrategySpec, coerce_strategy_specs, load_strategy_specs


def test_load_strategy_specs_accepts_wrapped_list_validates_and_dedupes(tmp_path):
    path = tmp_path / "strategies.json"
    path.write_text(
        json.dumps(
            {
                "strategies": [
                    {"optimization_level": 0},
                    {"optimization_level": 0},
                    {"optimization_level": 2, "zne": True, "zne_factors": [1.0, 2.0]},
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_strategy_specs(path)

    assert [s.optimization_level for s in specs] == [0, 2]
    assert specs[1].zne_factors == (1.0, 2.0)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"strategies": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_strategy_specs(bad)


def test_load_strategy_specs_accepts_saved_results_shapes(tmp_path):
    selections = tmp_path / "results.json"
    selections.write_text(
        json.dumps(
            {
                "selections": {
                    "bell": {"spec": {"optimization_level": 3}},
                    "ghz": {"spec": {"optimization_level": 1}},
                }
            }
        ),
        encoding="utf-8",
    )
    assert [s.optimization_level for s in load_strategy_specs(selections)] == [3, 1]

    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "results": [
                    {"strategy": {"optimization_level": 2}},
                    {"strategy": {"optimization_level": 2}},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [s.optimization_level for s in load_strategy_specs(matrix)] == [2]


def test_load_strategy_specs_rejects_bad_json_shapes(tmp_path):
    not_json = tmp_path / "not_json.json"
    not_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid strategy JSON"):
        load_strategy_specs(not_json)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Could not read"):
        load_strategy_specs(missing)

    bad_container = tmp_path / "bad_container.json"
    bad_container.write_text(
        json.dumps({"strategies": {"optimization_level": 1}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="strategies.*list"):
        load_strategy_specs(bad_container)

    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(ValueError, match="entry 0"):
        load_strategy_specs(bad_entry)


def test_coerce_strategy_specs_accepts_specs_dicts_and_generators():
    specs = coerce_strategy_specs(
        iter([StrategySpec(optimization_level=1), {"optimization_level": 3}])
    )

    assert [s.optimization_level for s in specs] == [1, 3]

    with pytest.raises(ValueError, match="at least one"):
        coerce_strategy_specs([])
    with pytest.raises(ValueError, match="iterable"):
        coerce_strategy_specs("not-a-strategy")  # type: ignore[arg-type]


def test_strategy_spec_rejects_bool_and_non_finite_edge_cases():
    with pytest.raises(ValueError, match="optimization_level"):
        StrategySpec(optimization_level=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resilience_level"):
        StrategySpec(resilience_level=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        StrategySpec(zne=True, zne_factors=(1.0, math.nan, 2.0))
    with pytest.raises(ValueError, match="max_subcircuit_qubits"):
        StrategySpec(max_subcircuit_qubits=False)  # type: ignore[arg-type]
