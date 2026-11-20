from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from qbalance.cache import get_entry, load_compiled, save_compiled
from qbalance.dataset import CircuitDataset, load_dataset, save_dataset
from qbalance.reports.common import strategy_key


def test_failed_cache_update_keeps_previous_circuit_and_metadata(tmp_path):
    entry = get_entry("a" * 64, tmp_path)
    save_compiled(entry, QuantumCircuit(1, name="old"), {"source": "old"})

    with pytest.raises(TypeError):
        save_compiled(entry, QuantumCircuit(2, name="new"), {"source": object()})

    circuit, meta = load_compiled(entry)
    assert circuit.name == meta["source"] == "old"


def test_first_dataset_writer_wins_without_overwrite(tmp_path, monkeypatch):
    destination = tmp_path / "dataset"
    dumped = Barrier(2, timeout=15)
    original_dump = qpy.dump

    def simultaneous_dump(circuit, stream):
        original_dump(circuit, stream)
        dumped.wait()

    monkeypatch.setattr(qpy, "dump", simultaneous_dump)

    def save(name):
        try:
            save_dataset(destination, [QuantumCircuit(1, name=name)])
        except FileExistsError:
            return None
        return name

    with ThreadPoolExecutor(max_workers=2) as pool:
        saved = list(pool.map(save, ["first", "second"]))
    (winner,) = [name for name in saved if name is not None]
    assert load_dataset(destination).names() == [winner]


def test_dataset_rejects_complex_fraction_and_cyclic_metadata(tmp_path):
    with pytest.raises(ValueError, match="frac_train"):
        CircuitDataset(tmp_path, []).split(frac_train=np.complex128(0.5 + 1j))

    metadata = {}
    metadata["self"] = metadata
    with pytest.raises(
        ValueError, match="metadata entry at index 0.*JSON-serializable"
    ):
        save_dataset(tmp_path / "dataset", [QuantumCircuit(1)], [metadata])
    assert not (tmp_path / "dataset").exists()


def test_report_key_preserves_zne_fold_thresholds_and_field_boundaries():
    def zne_label(middle_factor):
        return strategy_key(
            {
                "optimization_level": 1,
                "zne": True,
                "zne_factors": [1.0, middle_factor, 7.0],
            }
        )

    assert zne_label(2.9999999) != zne_label(3.0000001)
    assert strategy_key({"layout_method": "a,route=b"}) != strategy_key(
        {"layout_method": "a", "routing_method": "b"}
    )


@pytest.mark.parametrize(
    "key", ["", ".", "..suffix", "nested/path", r"nested\path", "C:entry"]
)
def test_cache_keys_stay_under_the_cache_root(tmp_path, key):
    with pytest.raises(ValueError, match="cache key"):
        get_entry(key, tmp_path)


def test_cached_files_can_share_a_directory_and_detect_mismatched_versions(tmp_path):
    from qbalance.cache import CacheEntry

    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    entry = CacheEntry("keyA", tmp_path)
    save_compiled(entry, QuantumCircuit(1, name="old"), {"name": "old"})
    old_bytes = (tmp_path / "compiled.qpy").read_bytes()
    save_compiled(entry, QuantumCircuit(2, name="new"), {"name": "new"})
    assert unrelated.read_text(encoding="utf-8") == "keep"

    (tmp_path / "compiled.qpy").write_bytes(old_bytes)
    assert load_compiled(entry) is None

    # Legacy on-disk entries stored a metrics object directly in meta.json.
    (tmp_path / "meta.json").write_text('{"name": "old"}', encoding="utf-8")
    circuit, metrics = load_compiled(entry)
    assert circuit.name == metrics["name"] == "old"


def test_cache_read_during_first_update_of_legacy_metadata(tmp_path, monkeypatch):
    from pathlib import Path

    from qbalance import cache

    entry = get_entry("keyA", tmp_path)
    entry.dir.mkdir(parents=True)
    metadata = entry.dir / "meta.json"
    metadata.write_text('{"source": "legacy"}', encoding="utf-8")
    with (entry.dir / "compiled.qpy").open("wb") as stream:
        qpy.dump(QuantumCircuit(1, name="legacy"), stream)

    original_read = Path.read_text
    original_write = cache.atomic_write_bytes
    between_writes = []

    def observe_publication(path, payload):
        original_write(path, payload)
        if not between_writes:
            between_writes.append(load_compiled(entry))

    def finish_update_after_reading_metadata(path, *args, **kwargs):
        data = original_read(path, *args, **kwargs)
        if path == metadata:
            monkeypatch.setattr(Path, "read_text", original_read)
            save_compiled(entry, QuantumCircuit(2, name="new"), {"source": "new"})
        return data

    monkeypatch.setattr(cache, "atomic_write_bytes", observe_publication)
    monkeypatch.setattr(Path, "read_text", finish_update_after_reading_metadata)
    circuit, metrics = load_compiled(entry)
    assert circuit.name == metrics["source"] == "legacy"
    assert between_writes == [None]
    circuit, metrics = load_compiled(entry)
    assert circuit.name == metrics["source"] == "new"


@pytest.mark.parametrize("name", ["report.md", "report.html"])
def test_report_preserves_its_input_matrix(tmp_path, name):
    from qbalance.reports import render_html, render_markdown

    original = '{"results": []}'
    path = tmp_path / name
    path.write_text(original, encoding="utf-8")
    if name.endswith(".html"):
        pytest.importorskip("jinja2")
    render = render_html if name.endswith(".html") else render_markdown
    with pytest.raises(ValueError, match="overwrite the input matrix"):
        render(path, tmp_path)
    assert path.read_text(encoding="utf-8") == original


def test_markdown_renders_multiline_labels_as_literal_text(tmp_path):
    import json

    from qbalance.reports import render_markdown

    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "backend": "line one\n# two",
                        "strategy": {"layout_method": "tick`\ncell|end"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    lines = render_markdown(path, tmp_path).read_text(encoding="utf-8").splitlines()
    assert "## Backend: `line one # two`" in lines
    (row,) = [line for line in lines if line.startswith("| `")]
    assert "`` optNone,layout=tick` cell\\|end ``" in row


def test_aggregate_preserves_finite_mean_when_the_sum_overflows():
    from qbalance.reports.common import aggregate

    values = [{"metrics": {"depth": 1e308}} for _ in range(2)]
    assert aggregate(values)["depth"] == 1e308


def test_split_archive_cannot_overwrite_a_circuit_in_the_other_split(tmp_path):
    from qbalance.workflow.workload import BalancedWorkload

    dataset = save_dataset(
        tmp_path / "dataset", [QuantumCircuit(1, name=name) for name in ("a", "b")]
    )
    train, test = dataset.split(frac_train=0.5)
    destination = test.root / test.records[0].artifact
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="source dataset"):
        BalancedWorkload(train, "fake:generic:2", {}).to_download(
            destination, overwrite=True
        )
    assert destination.read_bytes() == before


def test_save_uses_filesystem_parent_after_directory_symlink(tmp_path):
    from qbalance.utils import atomic_write_bytes

    nested = tmp_path / "nested"
    real = nested / "target"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks unavailable")

    parent = link / ".."
    # Use the actual filesystem behavior, since Windows may normalize ".."
    # before following a directory symlink.
    (parent / "probe").touch()
    expected_parent = nested if (nested / "probe").exists() else tmp_path
    saved = save_dataset(parent / "dataset", [QuantumCircuit(1, name="sample")])
    assert load_dataset(saved.root).names() == ["sample"]
    assert (expected_parent / "dataset").is_dir()

    atomic_write_bytes(parent / "data" / "value", b"saved")
    assert (expected_parent / "data" / "value").read_bytes() == b"saved"
