# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from qbalance.execution import runner
from tests.system_stubs import _Circ, _Job


def test_execution_runner_paths(monkeypatch):

    b = types.SimpleNamespace(run=lambda c, shots, seed_simulator=None: _Job({"0": 2}))
    assert runner.run_counts(b, _Circ(), shots=2) == {"0": 2}

    class NoSeedBackend:
        def run(self, circuit, shots):

            _ = (circuit, shots)
            return _Job({"1": 3})

    assert runner.run_counts(NoSeedBackend(), _Circ(), shots=3) == {"1": 3}

    aer = types.ModuleType("qiskit_aer")
    aer.AerSimulator = type("A", (), {"from_backend": staticmethod(lambda b: b)})
    monkeypatch.setitem(sys.modules, "qiskit_aer", aer)
    assert runner._ensure_backend(object()) is not None


def test_run_counts_validates_inputs():

    b = type(
        "Backend", (), {"run": staticmethod(lambda c, shots, **k: _Job({"0": shots}))}
    )

    with pytest.raises(ValueError, match="positive integer"):
        runner.run_counts(b, _Circ(), shots=0)

    with pytest.raises(ValueError, match="positive integer"):
        runner.run_counts(b, _Circ(), shots=True)

    with pytest.raises(ValueError, match="seed_simulator"):
        runner.run_counts(b, _Circ(), shots=1, seed_simulator=1.5)

    with pytest.raises(ValueError, match="seed_simulator"):
        runner.run_counts(b, _Circ(), shots=1, seed_simulator=False)

    with pytest.raises(ValueError, match="seed_transpiler"):
        runner.run_counts(b, _Circ(), shots=1, seed_transpiler=object())

    with pytest.raises(ValueError, match="seed_transpiler"):
        runner.run_counts(b, _Circ(), shots=1, seed_transpiler=True)


def test_run_counts_passes_seed_transpiler_and_filters_unsupported_seed_kwargs():

    seen = {}

    class BackendWithSeedTranspiler:
        def run(self, circuit, shots, seed_transpiler=None, seed_simulator=None):

            _ = (circuit, shots, seed_simulator)
            seen["seed_transpiler"] = seed_transpiler
            return _Job({"10": 1})

    out = runner.run_counts(
        BackendWithSeedTranspiler(), _Circ(), shots=1, seed_transpiler=7
    )
    assert out == {"10": 1}
    assert seen["seed_transpiler"] == 7

    class BackendNoSeeds:
        def run(self, circuit, shots):

            _ = (circuit, shots)
            return _Job({"11": 2})

    out2 = runner.run_counts(
        BackendNoSeeds(), _Circ(), shots=2, seed_transpiler=9, seed_simulator=3
    )
    assert out2 == {"11": 2}


def test_run_counts_does_not_swallow_backend_type_errors():

    class BackendRaisesTypeError:
        def run(self, circuit, shots):

            _ = (circuit, shots)
            raise TypeError("backend exploded")

    with pytest.raises(TypeError, match="backend exploded"):
        runner.run_counts(BackendRaisesTypeError(), _Circ(), shots=1)


def test_run_counts_accepts_numpy_integral_inputs():

    b = type(
        "Backend",
        (),
        {"run": staticmethod(lambda c, shots, **k: _Job({"0": int(shots)}))},
    )

    out = runner.run_counts(
        b,
        _Circ(),
        shots=np.int64(2),
        seed_simulator=np.int64(3),
        seed_transpiler=np.int64(5),
    )
    assert out == {"0": 2}


def test_prepare_run_kwargs_keeps_kwargs_for_opaque_callables(monkeypatch):

    original_signature = runner.inspect.signature

    def fake_signature(func):

        if func is opaque_run:
            raise ValueError("opaque callable")
        return original_signature(func)

    seen = {}

    def opaque_run(circuit, **kwargs):

        _ = circuit
        seen.update(kwargs)
        return _Job({"0": kwargs["shots"]})

    backend = type("Backend", (), {"run": staticmethod(opaque_run)})
    monkeypatch.setattr(runner.inspect, "signature", fake_signature)

    out = runner.run_counts(backend, _Circ(), shots=4, seed_simulator=3)
    assert out == {"0": 4}
    assert seen["shots"] == 4
    assert seen["seed_simulator"] == 3
