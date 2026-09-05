# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import pathlib
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
    backend_display_name,
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


def test_atomic_write_bytes_retries_transient_replace_permission_errors(
    tmp_path, monkeypatch
):
    """Windows may briefly lock a destination while another reader closes it."""
    target = tmp_path / "payload.bin"
    real_replace = utils_module.os.replace
    attempts = 0

    def intermittently_locked(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("destination is temporarily locked")
        real_replace(src, dst)

    monkeypatch.setattr(utils_module.os, "replace", intermittently_locked)
    monkeypatch.setattr(utils_module.time, "sleep", lambda _delay: None)

    atomic_write_bytes(target, b"complete payload")

    assert attempts == 3
    assert target.read_bytes() == b"complete payload"
    assert [p.name for p in tmp_path.iterdir()] == ["payload.bin"]


def test_atomic_write_bytes_serializes_replacements_of_the_same_path(
    tmp_path, monkeypatch
):
    """Windows replacements must not race one another before retrying readers."""
    import threading

    target = tmp_path / "payload.bin"
    real_replace = utils_module.os.replace
    barrier = threading.Barrier(4)
    state_lock = threading.Lock()
    active_replacements = 0
    most_active_replacements = 0

    def observed_replace(src, dst):
        nonlocal active_replacements, most_active_replacements
        with state_lock:
            active_replacements += 1
            most_active_replacements = max(
                most_active_replacements, active_replacements
            )
        try:
            # Give another writer a chance to overlap if path locking regresses.
            threading.Event().wait(0.005)
            real_replace(src, dst)
        finally:
            with state_lock:
                active_replacements -= 1

    monkeypatch.setattr(utils_module.os, "replace", observed_replace)

    def writer(payload):
        barrier.wait()
        atomic_write_bytes(target, payload)

    threads = [threading.Thread(target=writer, args=(bytes([i]),)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert most_active_replacements == 1
    assert target.read_bytes() in {bytes([i]) for i in range(4)}


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


def test_atomic_write_bytes_is_safe_under_concurrent_writers(tmp_path):
    """Concurrent runs share one compile cache, so a torn file must be impossible.

    Several writers race on the same destination; every observed value must be
    one writer's complete payload, never a mixture, and no temporary file may
    survive.
    """
    import threading

    target = tmp_path / "entry.bin"
    payloads = [bytes([index]) * 4096 for index in range(1, 9)]
    observed: list[bytes] = []
    barrier = threading.Barrier(len(payloads))
    errors: list[BaseException] = []

    def writer(payload):
        try:
            barrier.wait()
            for _ in range(20):
                atomic_write_bytes(target, payload)
                if target.exists():
                    observed.append(target.read_bytes())
        except BaseException as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(p,)) for p in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert observed
    complete = set(payloads)
    assert all(value in complete for value in observed)
    assert [p.name for p in tmp_path.iterdir()] == ["entry.bin"]


def test_only_the_backend_entry_point_group_is_resolved(monkeypatch):
    """Pin which plugin groups actually extend behavior.

    ``plugins list`` reports three entry-point groups, but only
    ``qbalance.backends`` is loaded: a registration there becomes a usable
    backend spec, while the objectives and reports groups are inventory that
    nothing resolves by name. The docs say so, so a test should notice if that
    ever changes in either direction.
    """
    from qbalance.backends import resolver as backend_resolver

    listed = plugins.list_plugins()
    assert set(listed) == {
        "qbalance.backends",
        "qbalance.objectives",
        "qbalance.reports",
    }

    # A backend registration is loaded and drives spec resolution.
    sentinel = object()
    monkeypatch.setattr(backend_resolver, "_PLUGINS", None)
    monkeypatch.setattr(
        backend_resolver,
        "_load_backend_plugins",
        lambda: {"mock": lambda spec: (sentinel, spec)},
    )
    assert backend_resolver.resolve_backend("mock:5") == (sentinel, "mock:5")

    # Nothing in the package loads the other two groups.
    source_root = pathlib.Path(backend_resolver.__file__).resolve().parents[1]
    loaders = [
        path
        for path in source_root.rglob("*.py")
        if ".load()" in path.read_text(encoding="utf-8")
    ]
    assert [p.name for p in loaders] == ["resolver.py"]


def test_package_exposes_a_version_consistent_with_its_metadata():
    """Regression: qbalance.__version__ did not exist.

    ``_version.py`` was written to hold it but nothing imported the module, so
    the near-universal ``package.__version__`` convention was unavailable and
    the file was dead. The assertions also catch the version in
    ``_version.py`` drifting from the one ``pyproject.toml`` ships, which
    nothing else enforces.
    """
    import qbalance
    from qbalance._version import __version__ as in_tree_version

    assert hasattr(qbalance, "__version__")
    assert "__version__" in qbalance.__all__
    assert qbalance.__version__ == in_tree_version

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as installed_version

    try:
        distribution_version = installed_version("qbalance")
    except PackageNotFoundError:  # pragma: no cover - source checkout only
        pytest.skip("qbalance is not installed in this environment")
    assert qbalance.__version__ == distribution_version
    assert in_tree_version == distribution_version


def test_load_compiled_treats_a_half_written_entry_as_a_miss(tmp_path):
    """A cache entry is only usable when both of its files are present.

    An interrupted save leaves one of ``meta.json`` / ``compiled.qpy`` behind.
    Recompiling is always correct; reading the surviving half and then failing
    on the missing one would turn a stale cache into a hard error.
    """
    entry = cache.get_entry("abcdef0123", root=tmp_path)
    entry.dir.mkdir(parents=True, exist_ok=True)

    (entry.dir / "meta.json").write_text("{}", encoding="utf-8")
    assert cache.load_compiled(entry) is None

    (entry.dir / "meta.json").unlink()
    (entry.dir / "compiled.qpy").write_bytes(b"")
    assert cache.load_compiled(entry) is None


def test_atomic_write_cleanup_does_not_mask_the_original_error(tmp_path, monkeypatch):
    """Removing the temp file must never replace the failure being reported.

    The cleanup runs while an exception is already propagating.  If the
    temporary file is gone by then -- a concurrent sweep of stale ``.tmp``
    files, say -- an unguarded unlink raises FileNotFoundError and the real
    cause is lost.
    """

    def _vanish_then_fail(src, dst):

        pathlib.Path(src).unlink()
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _vanish_then_fail)

    with pytest.raises(OSError, match="replace failed"):
        utils_module.atomic_write_bytes(tmp_path / "out.bin", b"data")


def test_backend_display_name_handles_both_backend_conventions():
    """BackendV2 exposes name as a string; BackendV1-style as a method.

    A bare getattr returns the bound method for the second kind, so anything
    naming a backend for a human would print a repr with a memory address --
    and those are exactly the backends most likely to need naming.
    """

    class V2Style:
        name = "fake_kyoto"

    class V1Style:
        def name(self):

            return "ibmq_fake_device"

    class Anonymous:
        pass

    class BlankName:
        name = ""

    assert backend_display_name(V2Style()) == "fake_kyoto"
    assert backend_display_name(V1Style()) == "ibmq_fake_device"
    assert backend_display_name(Anonymous()) == "Anonymous"
    # An empty name is no name; fall back rather than print nothing.
    assert backend_display_name(BlankName()) == "BlankName"


def test_default_cache_dir_does_not_duplicate_the_app_name(monkeypatch):
    """platformdirs may or may not already end in the app name.

    Appending unconditionally yields ``.../qbalance/qbalance``, whose ``name``
    is still "qbalance" -- so checking only the last component cannot tell the
    two apart.  Pin the whole path for both shapes.
    """
    monkeypatch.setattr(
        utils_module, "user_cache_dir", lambda app: f"/base/cache/{app}"
    )
    assert utils_module.default_cache_dir("qbalance") == pathlib.Path(
        "/base/cache/qbalance"
    )

    # A platform whose cache dir does not already carry the app name.
    monkeypatch.setattr(utils_module, "user_cache_dir", lambda app: "/base/Caches")
    assert utils_module.default_cache_dir("qbalance") == pathlib.Path(
        "/base/Caches/qbalance"
    )


@pytest.mark.parametrize(
    "name",
    ["../escaped", "..", "/absolute/path", "nested/name", "", "tiny/../other"],
)
def test_builtin_dataset_name_is_validated_before_touching_the_filesystem(
    tmp_path, monkeypatch, name
):
    """An unknown name must not create anything on the way to being rejected.

    The name is joined straight into a path, so "/abs" discards the data
    directory entirely and ".." climbs out of it.  The rejection used to come
    *after* mkdir(parents=True), so those directories were created first and
    only then did the KeyError arrive.
    """
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(builtin_data, "user_data_dir", lambda app: str(data_root))

    def _must_not_save(*args, **kwargs):

        raise AssertionError("save_dataset must not run for an unknown dataset")

    monkeypatch.setattr(builtin_data, "save_dataset", _must_not_save)

    with pytest.raises(KeyError, match="Unknown built-in dataset"):
        builtin_data.get_builtin_dataset_dir(name)

    # Nothing anywhere under tmp_path gained a directory.
    assert [p for p in tmp_path.rglob("*") if p.is_dir()] == [data_root]


def test_package_logger_defers_to_a_host_that_configured_logging():
    """qbalance installs a handler only when the application installed none.

    Both halves matter.  Installing one on top of the host's duplicates every
    record; installing none when the host has no logging leaves the package
    silent.  The existing regression test cannot pin either, because under
    pytest the root logger's state is whatever the run happens to leave.
    """
    import logging

    from qbalance.logging import LOGGER_NAME, get_logger

    package_logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(package_logger.handlers)
    saved_propagate = package_logger.propagate
    saved_level = package_logger.level
    saved_root = list(logging.root.handlers)
    try:
        # Host owns logging: qbalance must stay out of the way and propagate.
        package_logger.handlers.clear()
        package_logger.propagate = True
        logging.root.handlers[:] = [logging.NullHandler()]
        get_logger("qbalance.probe_host")
        assert package_logger.handlers == []
        assert package_logger.propagate is True

        # No host logging: qbalance installs exactly one handler and stops
        # propagating, so a later basicConfig cannot make records double.
        package_logger.handlers.clear()
        package_logger.propagate = True
        logging.root.handlers[:] = []
        get_logger("qbalance.probe_bare")
        assert len(package_logger.handlers) == 1
        assert package_logger.propagate is False

        # Idempotent: a second call must not stack another handler.
        get_logger("qbalance.probe_again")
        assert len(package_logger.handlers) == 1
    finally:
        package_logger.handlers[:] = saved_handlers
        package_logger.propagate = saved_propagate
        package_logger.setLevel(saved_level)
        logging.root.handlers[:] = saved_root
