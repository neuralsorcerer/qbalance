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

from qbalance import builtin_data, cache, plugins
from qbalance import utils as utils_module
from qbalance.backends import aer, fake
from qbalance.backends import resolver as backend_resolver
from qbalance.errors import OptionalDependencyError, QBalanceError
from qbalance.utils import (
    atomic_write_bytes,
    bit_index,
    default_cache_dir,
    dump_json,
    load_json,
    shares_bit,
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


def test_bit_index_prefers_circuit_find_bit_and_validates_fallbacks():

    bit = types.SimpleNamespace(index=5, _index=6)
    circuit = types.SimpleNamespace(
        find_bit=lambda b: types.SimpleNamespace(index=2) if b is bit else None
    )
    assert bit_index(circuit, bit) == 2

    fallback_bit = types.SimpleNamespace(_index=3)
    assert bit_index(types.SimpleNamespace(), fallback_bit) == 3

    with pytest.raises(AttributeError):
        bit_index(types.SimpleNamespace(), object())
    with pytest.raises(ValueError):
        bit_index(types.SimpleNamespace(), types.SimpleNamespace(index=-1))


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

    cache.save_compiled(entry, "circ", {"m": 1, "measurement_flip_map": {0: 1}})
    compiled, meta = cache.load_compiled(entry)
    assert compiled == "c:circ"
    assert meta == {"m": 1, "measurement_flip_map": {0: 1}}


def test_cache_optional_dependency_errors(tmp_path, monkeypatch):

    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    monkeypatch.delitem(sys.modules, "qiskit.qpy", raising=False)
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
    # monkeypatch restores the module-level plugin cache after the test; a bare
    # assignment would leak the stub table into every later test.
    monkeypatch.setattr(backend_resolver, "_PLUGINS", None)

    plugins_map = backend_resolver._load_backend_plugins()
    assert "ok" in plugins_map

    monkeypatch.setattr(
        backend_resolver, "_PLUGINS", {"ok": lambda spec: f"resolved:{spec}"}
    )
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
    with pytest.raises(QBalanceError, match="at least 2"):
        fake.resolve("fake:generic:1")
    with pytest.raises(QBalanceError):
        fake.resolve("fake:generic:2:extra")
    with pytest.raises(QBalanceError):
        fake.resolve("fake::2")
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


def test_bit_index_reports_a_negative_resolved_index():
    """Regression: the negative-index guard was swallowed by its own except."""

    class NegativeFinder:
        @staticmethod
        def find_bit(bit):
            _ = bit
            return types.SimpleNamespace(index=-3)

    with pytest.raises(ValueError):
        bit_index(NegativeFinder(), object())

    class BrokenFinder:
        @staticmethod
        def find_bit(bit):
            raise RuntimeError("no lookup")

    # An unusable find_bit still falls through to the attribute probes.
    assert bit_index(BrokenFinder(), types.SimpleNamespace(_index=4)) == 4


def test_shares_bit_matches_unhashable_bits():
    left = types.SimpleNamespace(name="q0")
    right = types.SimpleNamespace(name="q1")

    assert shares_bit((left,), (right, left))
    assert shares_bit((left,), (types.SimpleNamespace(name="q0"),))
    assert not shares_bit((left,), (right,))
    assert not shares_bit((), (left,))


def test_atomic_write_bytes_replaces_without_leaving_partial_files(tmp_path):
    target = tmp_path / "sub" / "payload.bin"

    atomic_write_bytes(target, b"first")
    assert target.read_bytes() == b"first"

    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
    assert [p.name for p in target.parent.iterdir()] == ["payload.bin"]


def test_atomic_write_bytes_keeps_the_previous_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "payload.bin"
    atomic_write_bytes(target, b"original")

    def boom(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(utils_module.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"replacement")

    assert target.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["payload.bin"]


def test_dump_json_is_atomic(tmp_path, monkeypatch):
    """Regression: a half-written meta.json used to abort every later run."""
    target = tmp_path / "meta.json"
    dump_json(target, {"a": 1})
    assert load_json(target) == {"a": 1}

    def boom(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(utils_module.os, "replace", boom)
    with pytest.raises(OSError):
        dump_json(target, {"a": 2})

    assert load_json(target) == {"a": 1}


def test_get_logger_does_not_double_emit_when_the_host_configures_logging():
    """Regression: a handler per module plus propagation printed twice."""
    import logging

    from qbalance.logging import LOGGER_NAME, get_logger

    package_logger = logging.getLogger(LOGGER_NAME)
    module_logger = get_logger("qbalance.for_test")

    # Only the package logger carries qbalance's handler, if any.
    assert module_logger.handlers == []
    assert len(package_logger.handlers) <= 1
    # Records still reach the package logger's level configuration.
    assert module_logger.getEffectiveLevel() <= logging.WARNING


def test_resolve_backend_accepts_the_same_spacing_its_plugins_do():
    """Regression: the resolver rejected "fake : generic : 5" as kind 'fake '.

    The fake plugin strips each spec part deliberately, so dispatch has to strip
    the kind too; otherwise the error claims an available kind is unknown.
    """
    pytest.importorskip("qiskit")

    from qbalance.backends import resolve_backend

    spaced = resolve_backend(" fake : generic : 5 ")
    plain = resolve_backend("fake:generic:5")
    assert spaced.num_qubits == plain.num_qubits == 5

    for spec in ("", "   "):
        with pytest.raises(QBalanceError, match="non-empty"):
            resolve_backend(spec)

    with pytest.raises(QBalanceError, match="Unknown backend kind"):
        resolve_backend("definitely_not_a_kind:1")
