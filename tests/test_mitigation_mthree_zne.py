# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types

import pytest

from qbalance.mitigation import mthree as mthree_mod
from qbalance.mitigation import zne
from tests.system_stubs import _Circ


def test_mthree_and_zne(monkeypatch):

    mthree = types.ModuleType("mthree")

    class Mit:
        def __init__(self, backend):

            _ = backend

        def cals_from_system(self, measured_qubits, calibration_shots):

            _ = (measured_qubits, calibration_shots)

        def apply_correction(self, raw_counts, measured_qubits):

            _ = measured_qubits
            s = sum(raw_counts.values())
            return types.SimpleNamespace(
                nearest_probability_distribution=lambda: {
                    k: v / s for k, v in raw_counts.items()
                }
            )

    mthree.M3Mitigation = Mit
    monkeypatch.setitem(sys.modules, "mthree", mthree)
    probs = mthree_mod.apply_mthree_mitigation(object(), {"00": 2, "11": 2}, [0, 1], 4)
    assert probs["00"] == 0.5

    c = _Circ()
    f = zne.fold_global(c, 3.2)
    assert getattr(f, "name", "").startswith("circuit_fold") or isinstance(f, _Circ)
    assert zne._counts_to_expval_z({"00": 3, "01": 1}) == pytest.approx(0.5)
    out = zne.zne_extrapolate_counts(
        [1.0, 3.0], [{"00": 2, "11": 2}, {"00": 3, "11": 1}]
    )
    assert pytest.approx(sum(out.values())) == 1.0
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0], [{"0": 1}, {"1": 1}])
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0, float("nan")], [{"0": 1}, {"0": 1}])
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0, 1.0], [{"0": 1}, {"0": 1}], degree=1)
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([0.5, 1.0], [{"0": 1}, {"0": 1}])
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0": 1}, {"0": 1}], degree=-1)
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0": -1}, {"0": 1}])
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0, 2.0], [{}, {"0": 1}])
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0], [{"0": 1}], degree=True)
    with pytest.raises(ValueError):
        zne.zne_extrapolate_counts([1.0, 2.0], [{"0": 0}, {"0": 1}])
