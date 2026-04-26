# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types

from qbalance.strategies import StrategySpec
from qbalance.transpile import noise_aware_layout as nal
from qbalance.transpile import pipeline, suppression
from qbalance.workflow import workload as wl
from tests.system_stubs import _I, _PM, _Q, _Circ


def test_finalize_full_coverage_paths(monkeypatch, tmp_path):
    # noise-aware helpers with None properties and missing gate errors

    backend_none_props = types.SimpleNamespace(properties=lambda: None)
    assert nal._safe_get_qubit_readout_error(backend_none_props, 0) is None
    assert nal._safe_get_t1(backend_none_props, 0) is None
    assert nal._safe_get_t2(backend_none_props, 0) is None
    assert nal._safe_get_2q_error(backend_none_props, "cx", 0, 1) is None

    class PropsNoneGate:
        qubits = [[types.SimpleNamespace(name="other", value=1.0)]]

        @staticmethod
        def gate_error(name, pair):

            _ = (name, pair)
            return None

    be = types.SimpleNamespace(properties=lambda: PropsNoneGate())
    c = _Circ()
    c.data = [
        (_I("x"), [_Q(0)], []),
        (_I("cx"), [_Q(0), _Q(1)], []),
        (_I("measure"), [_Q(0)], []),
    ]
    assert nal.estimate_circuit_error(be, c) > 0

    class NoPhys:
        @staticmethod
        def properties():

            return types.SimpleNamespace(qubits=[[]])

        @staticmethod
        def configuration():

            raise RuntimeError("no config")

    assert nal.noise_aware_initial_layout(NoPhys(), _Circ()) is None

    # suppression non-list twirl return and XX sequence
    circuit_mod = types.ModuleType("qiskit.circuit")
    circuit_mod.pauli_twirl_2q_gates = lambda circuit, seed, num_twirls, target: circuit
    monkeypatch.setitem(sys.modules, "qiskit.circuit", circuit_mod)
    out = suppression.apply_pauli_twirling(_Circ(), num_twirls=1)
    assert isinstance(out, list) and len(out) == 1

    lib = types.ModuleType("qiskit.circuit.library")
    lib.XGate = type("XGate", (), {})
    lib.YGate = type("YGate", (), {})
    monkeypatch.setitem(sys.modules, "qiskit.circuit.library", lib)
    assert len(suppression._dd_sequence("XX")) == 2

    # pipeline exception-swallow branch around noise-aware override
    ppm = types.ModuleType("qiskit.transpiler.preset_passmanagers")
    ppm.generate_preset_pass_manager = lambda **kwargs: _PM(_Circ())
    monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", ppm)
    conv = types.ModuleType("qiskit.converters")
    conv.circuit_to_dag = lambda c: c
    monkeypatch.setitem(sys.modules, "qiskit.converters", conv)
    monkeypatch.setattr(pipeline, "_generate_pm", lambda backend, spec: _PM(_Circ()))
    monkeypatch.setattr(
        pipeline,
        "noise_aware_initial_layout",
        lambda backend, tw: (_ for _ in ()).throw(RuntimeError("layout")),
    )
    _, m = pipeline.compile_one(
        _Circ(),
        types.SimpleNamespace(
            target=types.SimpleNamespace(operation_names=["x", "cx"])
        ),
        StrategySpec(layout_method="qbalance_noise_aware"),
        profile=False,
    )
    assert m["depth"] >= 0

    # workload bandit while-loop and successful mitigation/zne top-prob assignments
    rec = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    dsroot = tmp_path / "ds3"
    dsroot.mkdir()
    (dsroot / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dsroot / "c0.qpy").write_bytes(b"x")
    ds = wl.CircuitDataset(dsroot, [rec])
    monkeypatch.setattr(ds, "load_circuits", lambda: [_Circ()])
    monkeypatch.setattr(
        wl,
        "resolve_backend",
        lambda b: types.SimpleNamespace(name=lambda: "bk", num_qubits=2),
    )
    monkeypatch.setattr(
        wl,
        "default_candidate_strategies",
        lambda max_candidates, seed: [
            StrategySpec(mthree=True, zne=True, seed_transpiler=i) for i in range(8)
        ],
    )

    class BanditFull:
        def observe(self, *a, **k):

            return None

        def propose(self, candidates, rng):

            _ = rng
            return candidates[0]

    monkeypatch.setattr(wl, "BanditSearcher", BanditFull)
    monkeypatch.setattr(
        wl, "compile_one", lambda *a, **k: (_Circ(), {"measurement_flip_map": {}})
    )
    monkeypatch.setattr(wl, "load_compiled", lambda entry: None)
    monkeypatch.setattr(wl, "save_compiled", lambda entry, compiled, m: None)
    monkeypatch.setattr(wl, "run_counts", lambda *a, **k: {"00": 9, "11": 1})
    monkeypatch.setattr(
        wl, "apply_measurement_untwirl_counts", lambda counts, flip_map: counts
    )
    monkeypatch.setattr(
        wl, "apply_mthree_mitigation", lambda *a, **k: {"00": 0.8, "11": 0.2}
    )
    monkeypatch.setattr(wl, "fold_global", lambda compiled, f: compiled)
    monkeypatch.setattr(
        wl, "zne_extrapolate_counts", lambda *a, **k: {"00": 0.7, "11": 0.3}
    )

    bw = (
        wl.Workload.from_dataset(ds)
        .set_target("fake:generic:2")
        .adjust(search="bandit", execute=True, max_candidates=8)
    )
    metrics = bw.selections["c0"].metrics
    assert metrics["mitigated_top_prob"] == 0.8
    assert metrics["zne_top_prob"] == 0.7
