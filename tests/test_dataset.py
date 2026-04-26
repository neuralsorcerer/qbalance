# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from qbalance.dataset import (
    DATASET_INDEX,
    CircuitDataset,
    CircuitRecord,
    _build_unique_artifact,
    load_data,
    load_dataset,
    save_dataset,
)
from qbalance.errors import OptionalDependencyError


class _DummyCircuit:
    def __init__(self, name: str) -> None:

        self.name = name


class _DummyQuantumCircuit:
    @staticmethod
    def from_qasm_file(path: str) -> str:

        return f"loaded:{Path(path).name}"


def _install_fake_qiskit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    qpy_load_result: list[object] | None = None,
) -> None:

    qiskit = types.ModuleType("qiskit")
    qpy = types.SimpleNamespace()

    def dump(circuit: _DummyCircuit, handle: io.BufferedWriter) -> None:

        payload = json.dumps({"name": circuit.name}).encode("utf-8")
        handle.write(payload)

    def load(handle: io.BufferedReader) -> list[object]:

        _ = handle.read()
        return ["loaded_qpy"] if qpy_load_result is None else qpy_load_result

    qpy.dump = dump
    qpy.load = load
    qiskit.qpy = qpy
    qiskit.QuantumCircuit = _DummyQuantumCircuit
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)


def test_build_unique_artifact_skips_existing_suffixes():

    used = {"dup.qpy", "dup_1.qpy"}
    assert _build_unique_artifact("dup", used) == "dup_2.qpy"


def test_save_dataset_rejects_misaligned_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)

    with pytest.raises(ValueError, match="same length"):
        save_dataset(tmp_path / "dataset", [_DummyCircuit("a")], metadata=[])


def test_save_dataset_rejects_existing_directory_without_overwrite(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_dataset(dataset_dir, [_DummyCircuit("a")])


def test_save_dataset_disambiguates_colliding_names(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)

    dataset = save_dataset(
        tmp_path / "dataset",
        [_DummyCircuit("dup"), _DummyCircuit("dup_1"), _DummyCircuit("dup")],
        metadata=[{"i": 1}, {"i": 2}, {"i": 3}],
    )

    assert [record.artifact for record in dataset.records] == [
        "dup.qpy",
        "dup_1.qpy",
        "dup_2.qpy",
    ]


def test_save_dataset_uses_fallback_name_and_empty_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)

    unnamed = types.SimpleNamespace(name="")
    dataset = save_dataset(tmp_path / "dataset", [unnamed], metadata=[None])

    assert dataset.records[0].name == "circuit_0"
    assert dataset.records[0].metadata == {}


def test_save_dataset_sanitizes_artifact_name_components(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)

    dataset = save_dataset(
        tmp_path / "dataset",
        [
            _DummyCircuit("../unsafe/name"),
            _DummyCircuit(".."),
            _DummyCircuit("name with spaces"),
        ],
    )

    assert [record.artifact for record in dataset.records] == [
        "unsafe_name.qpy",
        "circuit_1.qpy",
        "name_with_spaces.qpy",
    ]


def test_save_dataset_coerces_non_string_circuit_name(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)

    circuit = types.SimpleNamespace(name=123)
    dataset = save_dataset(tmp_path / "dataset", [circuit])

    assert dataset.records[0].name == "123"
    assert dataset.records[0].artifact == "123.qpy"


def test_save_dataset_preserves_zero_like_names(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)

    circuit = types.SimpleNamespace(name=0)
    dataset = save_dataset(tmp_path / "dataset", [circuit])

    assert dataset.records[0].name == "0"
    assert dataset.records[0].artifact == "0.qpy"


def test_save_dataset_raises_optional_dependency_error_when_qiskit_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))

    with pytest.raises(OptionalDependencyError, match="qiskit"):
        save_dataset(tmp_path / "dataset", [_DummyCircuit("a")], overwrite=True)


def test_load_dataset_round_trip_index(tmp_path):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {
                        "name": "c0",
                        "artifact": "c0.qpy",
                        "format": "qpy",
                        "metadata": {"split": "train"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_dir)

    assert dataset.root == dataset_dir
    assert dataset.names() == ["c0"]
    assert dataset.records[0].metadata == {"split": "train"}


def test_load_data_delegates_to_builtin_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):

    monkeypatch.setattr(
        "qbalance.builtin_data.get_builtin_dataset_dir", lambda name: tmp_path
    )
    monkeypatch.setattr(
        "qbalance.dataset.load_dataset", lambda path: CircuitDataset(path, [])
    )

    out = load_data("demo")

    assert isinstance(out, CircuitDataset)
    assert out.root == tmp_path


def test_load_dataset_rejects_missing_records_list(tmp_path):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(
        json.dumps({"version": 1}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="records"):
        load_dataset(dataset_dir)


def test_load_dataset_rejects_non_object_index(tmp_path):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(json.dumps(["invalid"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_dataset(dataset_dir)


def test_load_dataset_rejects_non_mapping_record_entries(tmp_path):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(
        json.dumps({"version": 1, "records": ["invalid"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index 0"):
        load_dataset(dataset_dir)


def test_load_dataset_rejects_record_with_missing_required_fields(tmp_path):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(
        json.dumps({"version": 1, "records": [{"name": "c0", "format": "qpy"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_dataset(dataset_dir)


def test_load_dataset_rejects_non_object_metadata(tmp_path):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {
                        "name": "c0",
                        "artifact": "c0.qpy",
                        "format": "qpy",
                        "metadata": ["not", "a", "dict"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata"):
        load_dataset(dataset_dir)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"name": "", "artifact": "c0.qpy", "format": "qpy"}, "invalid name"),
        (
            {"name": "c0", "artifact": "", "format": "qpy"},
            "invalid artifact",
        ),
        (
            {"name": "c0", "artifact": "../c0.qpy", "format": "qpy"},
            "unsafe artifact",
        ),
        (
            {"name": "c0", "artifact": "/tmp/c0.qpy", "format": "qpy"},
            "unsafe artifact",
        ),
        (
            {"name": "c0", "artifact": "..\\c0.qpy", "format": "qpy"},
            "unsafe artifact",
        ),
        (
            {"name": "c0", "artifact": "nested/c0.qpy", "format": "qpy"},
            "unsafe artifact",
        ),
        (
            {"name": "c0", "artifact": "./c0.qpy", "format": "qpy"},
            "unsafe artifact",
        ),
        (
            {"name": "c0", "artifact": "c0.qpy\u0000", "format": "qpy"},
            "unsafe artifact",
        ),
        (
            {"name": "c0", "artifact": "c0.qpy", "format": "bin"},
            "unsupported format",
        ),
    ],
)
def test_load_dataset_rejects_invalid_required_field_values(tmp_path, record, message):

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / DATASET_INDEX).write_text(
        json.dumps({"version": 1, "records": [record]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_dataset(dataset_dir)


def test_dataset_iter_len_names_and_split():

    records = [
        CircuitRecord(name="a", artifact="a.qpy", format="qpy"),
        CircuitRecord(name="b", artifact="b.qpy", format="qpy"),
    ]
    dataset = CircuitDataset(Path("."), records)

    assert len(dataset) == 2
    assert dataset.names() == ["a", "b"]
    assert list(dataset.iter_records()) == records

    train, test = dataset.split(seed=3, frac_train=0.5)
    assert len(train) == 1
    assert len(test) == 1


@pytest.mark.parametrize("frac_train", [-0.1, 1.1, float("inf"), float("nan")])
def test_dataset_split_rejects_out_of_range_or_nonfinite_fraction(frac_train: float):

    dataset = CircuitDataset(Path("."), [CircuitRecord("a", "a.qpy", "qpy")])

    with pytest.raises(ValueError, match="frac_train"):
        dataset.split(frac_train=frac_train)


def test_dataset_split_handles_empty_dataset_without_rng_work():

    dataset = CircuitDataset(Path("."), [])

    train, test = dataset.split(seed=7, frac_train=0.5)

    assert train.records == []
    assert test.records == []


def test_dataset_split_supports_fraction_boundaries():

    records = [CircuitRecord(str(i), f"{i}.qpy", "qpy") for i in range(3)]
    dataset = CircuitDataset(Path("."), records)

    train, test = dataset.split(seed=11, frac_train=0.0)
    assert train.records == []
    assert test.records == records

    train, test = dataset.split(seed=11, frac_train=1.0)
    assert train.records == records
    assert test.records == []


@pytest.mark.parametrize("frac_train", [[0.5], True, False, "0.5", b"0.5"])
def test_dataset_split_rejects_nonscalar_or_boolean_fraction(frac_train):

    dataset = CircuitDataset(Path("."), [CircuitRecord("a", "a.qpy", "qpy")])

    with pytest.raises(ValueError, match="frac_train"):
        dataset.split(frac_train=frac_train)


def test_dataset_split_accepts_numpy_float_scalar():

    records = [CircuitRecord("a", "a.qpy", "qpy"), CircuitRecord("b", "b.qpy", "qpy")]
    dataset = CircuitDataset(Path("."), records)

    train, test = dataset.split(seed=0, frac_train=np.float64(0.5))

    assert len(train) == 1
    assert len(test) == 1


def test_dataset_split_uses_rounding_for_small_datasets():

    one = CircuitDataset(Path("."), [CircuitRecord("a", "a.qpy", "qpy")])
    train, test = one.split(seed=0, frac_train=0.8)
    assert len(train) == 1
    assert len(test) == 0

    two = CircuitDataset(
        Path("."),
        [
            CircuitRecord("a", "a.qpy", "qpy"),
            CircuitRecord("b", "b.qpy", "qpy"),
        ],
    )
    train, test = two.split(seed=0, frac_train=0.8)
    assert len(train) == 2
    assert len(test) == 0


def test_dataset_split_rejects_overflowing_fraction_value():

    dataset = CircuitDataset(Path("."), [CircuitRecord("a", "a.qpy", "qpy")])

    with pytest.raises(ValueError, match="frac_train"):
        dataset.split(frac_train=10**400)


def test_load_circuits_loads_qpy_and_qasm(tmp_path, monkeypatch: pytest.MonkeyPatch):

    _install_fake_qiskit(monkeypatch)
    qpy_path = tmp_path / "a.qpy"
    qasm_path = tmp_path / "b.qasm"
    qpy_path.write_bytes(b"payload")
    qasm_path.write_text("OPENQASM 2.0;", encoding="utf-8")

    dataset = CircuitDataset(
        tmp_path,
        [
            CircuitRecord(name="a", artifact="a.qpy", format="qpy"),
            CircuitRecord(name="b", artifact="b.qasm", format="qasm"),
        ],
    )

    loaded = dataset.load_circuits()

    assert loaded == ["loaded_qpy", "loaded:b.qasm"]


def test_load_circuits_rejects_unknown_format(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch)
    dataset = CircuitDataset(
        tmp_path,
        [CircuitRecord(name="a", artifact="a.bin", format="bin")],
    )

    with pytest.raises(ValueError, match="Unknown circuit format"):
        dataset.load_circuits()


def test_load_circuits_rejects_empty_qpy_payload(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    _install_fake_qiskit(monkeypatch, qpy_load_result=[])
    (tmp_path / "a.qpy").write_bytes(b"payload")

    dataset = CircuitDataset(
        tmp_path,
        [CircuitRecord(name="a", artifact="a.qpy", format="qpy")],
    )

    with pytest.raises(ValueError, match="Empty QPY file"):
        dataset.load_circuits()


def test_load_circuits_raises_optional_dependency_error_when_qiskit_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):

    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    dataset = CircuitDataset(tmp_path, [])

    with pytest.raises(OptionalDependencyError, match="qiskit"):
        dataset.load_circuits()
