# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sys
import types

import pytest

import qbalance._version as version
from qbalance import builtin_data, cache, plugins
from qbalance.backends import aer, fake
from qbalance.backends import resolver as backend_resolver
from qbalance.errors import OptionalDependencyError, QBalanceError
from qbalance.utils import (
    default_cache_dir,
    dump_json,
    load_json,
    stable_hash_bytes,
    stable_hash_str,
)


class _EP:
    def __init__(self, name, loader):

        self.name = name
        self._loader = loader

    def load(self):

        return self._loader()


class _EPS:
    def __init__(self, mapping):

        self._mapping = mapping

    def select(self, group):

        return self._mapping.get(group, [])


def test_version_module_exposes_expected_version():

    assert version.__version__ == "0.1.0"


def test_utils_hash_and_json_helpers(tmp_path):

    digest = stable_hash_bytes(b"abc")
    assert digest == stable_hash_str("abc")

    path = tmp_path / "x" / "d.json"
    dump_json(path, {"b": 2, "a": 1})
    assert load_json(path) == {"a": 1, "b": 2}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(bad)


def test_default_cache_dir_uses_app_name():

    assert default_cache_dir("qbalance").name == "qbalance"


def _install_fake_qiskit_for_cache(monkeypatch):

    qiskit = types.ModuleType("qiskit")
    qpy = types.SimpleNamespace()

    def dump(circuit, fh):

        fh.write(f"c:{circuit}".encode("utf-8"))

    def load(fh):

        return [fh.read().decode("utf-8")]

    qpy.dump = dump
    qpy.load = load
    qiskit.qpy = qpy
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)


def test_cache_roundtrip_and_helpers(tmp_path, monkeypatch):

    _install_fake_qiskit_for_cache(monkeypatch)

    cdir = cache.cache_dir(tmp_path)
    assert cdir == tmp_path / "cache"
    entry = cache.get_entry("abcdef", tmp_path)
    assert entry.dir == tmp_path / "cache" / "ab" / "abcdef"

    assert cache.load_compiled(entry) is None
    assert cache.fingerprint_circuit("circ") == stable_hash_bytes(b"c:circ")

    cache.save_compiled(entry, "circ", {"m": 1})
    compiled, meta = cache.load_compiled(entry)
    assert compiled == "c:circ"
    assert meta == {"m": 1}


def test_cache_optional_dependency_errors(tmp_path, monkeypatch):

    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    entry = cache.get_entry("ab", tmp_path)
    with pytest.raises(OptionalDependencyError):
        cache.fingerprint_circuit("x")
    with pytest.raises(OptionalDependencyError):
        cache.save_compiled(entry, "x", {})

    entry.dir.mkdir(parents=True, exist_ok=True)
    (entry.dir / "meta.json").write_text("{}", encoding="utf-8")
    (entry.dir / "compiled.qpy").write_bytes(b"x")
    with pytest.raises(OptionalDependencyError):
        cache.load_compiled(entry)


def test_list_plugins_sorts_names(monkeypatch):

    eps = _EPS(
        {
            "qbalance.backends": [_EP("z", lambda: None), _EP("a", lambda: None)],
            "qbalance.objectives": [_EP("o", lambda: None)],
            "qbalance.reports": [],
        }
    )
    monkeypatch.setattr(plugins, "entry_points", lambda: eps)
    out = plugins.list_plugins()
    assert out["qbalance.backends"] == ["a", "z"]
    assert out["qbalance.objectives"] == ["o"]


def test_backend_resolver_load_and_resolve(monkeypatch):

    def good_loader():

        return lambda spec: {"spec": spec}

    def bad_loader():

        raise RuntimeError("boom")

    eps = _EPS({"qbalance.backends": [_EP("ok", good_loader), _EP("bad", bad_loader)]})
    monkeypatch.setattr(backend_resolver, "entry_points", lambda: eps)
    backend_resolver._PLUGINS = None

    plugins_map = backend_resolver._load_backend_plugins()
    assert "ok" in plugins_map

    backend_resolver._PLUGINS = {"ok": lambda spec: f"resolved:{spec}"}
    assert backend_resolver.resolve_backend("ok:foo") == "resolved:ok:foo"
    obj = object()
    assert backend_resolver.resolve_backend(obj) is obj
    with pytest.raises(QBalanceError):
        backend_resolver.resolve_backend("missing:foo")


def _install_fake_provider(monkeypatch):

    pkg = types.ModuleType("qiskit.providers.fake_provider")

    class GenericBackendV2:
        def __init__(self, num_qubits):

            self.num_qubits = num_qubits

    def fake_backend(name):

        if name == "bad":
            raise RuntimeError("not found")
        return {"name": name}

    pkg.GenericBackendV2 = GenericBackendV2
    pkg.fake_backend = fake_backend
    monkeypatch.setitem(sys.modules, "qiskit.providers.fake_provider", pkg)


def test_fake_backend_resolve_paths(monkeypatch):

    _install_fake_provider(monkeypatch)
    assert fake.resolve("fake:generic:7").num_qubits == 7
    assert fake.resolve("fake:ibm:FakeManilaV2") == {"name": "FakeManilaV2"}

    with pytest.raises(QBalanceError):
        fake.resolve("fake:generic:not-int")
    with pytest.raises(QBalanceError):
        fake.resolve("fake")
    with pytest.raises(QBalanceError):
        fake.resolve("fake:unknown:x")
    with pytest.raises(QBalanceError):
        fake.resolve("fake:ibm:bad")


def test_fake_backend_optional_dependency(monkeypatch):

    monkeypatch.setitem(
        sys.modules,
        "qiskit.providers.fake_provider",
        types.ModuleType("qiskit.providers.fake_provider"),
    )
    with pytest.raises(OptionalDependencyError):
        fake.resolve("fake:generic:2")


def _install_qiskit_aer(monkeypatch):

    m = types.ModuleType("qiskit_aer")

    class AerSimulator:
        def __init__(self):

            self.kind = "sim"

        @classmethod
        def from_backend(cls, backend):

            return {"backend": backend}

    m.AerSimulator = AerSimulator
    monkeypatch.setitem(sys.modules, "qiskit_aer", m)


def test_aer_backend_resolve_paths(monkeypatch):

    _install_qiskit_aer(monkeypatch)
    monkeypatch.setattr(aer, "resolve_backend", lambda spec: f"nested:{spec}")
    assert aer.resolve("aer:simulator").kind == "sim"
    assert aer.resolve("aer:from_backend:fake:generic:5") == {
        "backend": "nested:fake:generic:5"
    }
    with pytest.raises(QBalanceError):
        aer.resolve("aer")
    with pytest.raises(QBalanceError):
        aer.resolve("aer:from_backend")
    with pytest.raises(QBalanceError):
        aer.resolve("aer:unknown")


def test_aer_backend_optional_dependency(monkeypatch):

    monkeypatch.setitem(sys.modules, "qiskit_aer", types.ModuleType("qiskit_aer"))
    with pytest.raises(OptionalDependencyError):
        aer.resolve("aer:simulator")


def test_builtin_make_tiny_and_get_dataset_dir(tmp_path, monkeypatch):

    qiskit = types.ModuleType("qiskit")

    class QuantumCircuit:
        def __init__(self, n, c, name):

            self.n = n
            self.c = c
            self.name = name
            self.ops = []

        def h(self, i):

            self.ops.append(("h", i))

        def cx(self, a, b):

            self.ops.append(("cx", a, b))

        def cp(self, angle, a, b):

            self.ops.append(("cp", float(angle), a, b))

        def measure(self, q, c):

            self.ops.append(("measure", tuple(q), tuple(c)))

    qiskit.QuantumCircuit = QuantumCircuit
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)

    circuits = builtin_data._make_tiny()
    assert [c.name for c in circuits] == ["bell", "ghz3", "qft4"]

    monkeypatch.setattr(builtin_data, "user_data_dir", lambda app: str(tmp_path))
    saved = {}

    def fake_save_dataset(root, circuits, overwrite):

        saved["root"] = root
        saved["count"] = len(circuits)
        (root / "qbalance_dataset.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "qbalance_dataset.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(builtin_data, "save_dataset", fake_save_dataset)
    root = builtin_data.get_builtin_dataset_dir("tiny")
    assert root == tmp_path / "datasets" / "tiny"
    assert saved["count"] == 3

    assert builtin_data.get_builtin_dataset_dir("tiny") == root
    with pytest.raises(KeyError):
        builtin_data.get_builtin_dataset_dir("unknown")


def test_main_entrypoint_runs_cli(monkeypatch):

    cli_mod = types.ModuleType("qbalance.cli")
    called = {"v": 0}

    def app():

        called["v"] += 1

    cli_mod.app = app
    monkeypatch.setitem(sys.modules, "qbalance.cli", cli_mod)
    monkeypatch.delitem(sys.modules, "qbalance.__main__", raising=False)
    __import__("qbalance.__main__")
    assert called["v"] == 1
