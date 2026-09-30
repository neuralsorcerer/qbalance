# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger
from qbalance.utils import (
    dump_json,
    load_json,
    replacing_directory,
    stable_hash_str,
)

log = get_logger(__name__)


@dataclass
class CircuitRecord:
    """A single circuit artifact and metadata entry.

    Args:
        name: Logical circuit name.
        artifact: Relative artifact filename under the dataset root.
        format: Serialization format, currently ``"qpy"`` or ``"qasm"``.
        metadata: Free-form JSON-serializable metadata for this record.

    Returns:
        A populated dataclass instance.

    Raises:
        TypeError: Raised by dataclass construction when required fields are missing.

    Examples:
        >>> record = CircuitRecord(name="bell", artifact="bell.qpy", format="qpy")
        >>> record.artifact
        'bell.qpy'
    """

    name: str
    artifact: str
    format: str  # "qpy" | "qasm"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitDataset:
    """Dataset of serialized circuits with metadata.

    Args:
        root: Root directory containing circuit artifacts and dataset index.
        records: Ordered record entries describing each artifact.

    Returns:
        A dataset object that can be iterated, split, and deserialized.

    Raises:
        TypeError: Raised by dataclass construction when required fields are missing.

    Examples:
        >>> ds = CircuitDataset(Path("."), [CircuitRecord("c0", "c0.qpy", "qpy")])
        >>> len(ds)
        1
    """

    root: Path
    records: List[CircuitRecord]

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def names(self) -> List[str]:
        """Return the record names, in dataset order."""
        return [r.name for r in self.records]

    def iter_records(self) -> Iterable[CircuitRecord]:
        """Yield the records in dataset order."""
        yield from self.records

    def load_circuits(self) -> List[Any]:
        """Deserialize every record's circuit, in dataset order.

        A QPY artifact holding several circuits contributes its first one.

        Returns:
            One circuit per record.

        Raises:
            OptionalDependencyError: If qiskit is not installed.
            ValueError: If a record has an unknown format, or its artifact is
                missing, unreadable, or empty.  The message names the record.
        """
        try:
            # ``from`` imports the submodule: ``qiskit.qpy`` is only an
            # attribute once something has imported it, which qiskit 2.0
            # itself does not do.
            from qiskit import QuantumCircuit, qpy
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError("qiskit is required to load circuits") from e

        circuits: List[Any] = []
        for index, rec in enumerate(self.records):
            path = self.root / rec.artifact
            if rec.format not in ("qpy", "qasm"):
                raise ValueError(f"Unknown circuit format: {rec.format}")

            # A truncated or corrupt artifact (an interrupted copy, a partially
            # extracted bundle) otherwise surfaces as a bare struct error with
            # no clue which file is at fault, while every other failure this
            # loader can hit names the offending record.
            try:
                if rec.format == "qpy":
                    with path.open("rb") as f:
                        loaded = qpy.load(f)
                else:
                    loaded = [QuantumCircuit.from_qasm_file(str(path))]
            except Exception as e:
                raise ValueError(
                    f"Record at index {index} ({rec.name!r}) has an unreadable "
                    f"{rec.format} artifact {rec.artifact!r}: {e}"
                ) from e
            if not loaded:
                raise ValueError(f"Empty QPY file: {path}")
            circuits.append(loaded[0])
        return circuits

    def split(
        self, seed: int = 0, frac_train: float = 0.8
    ) -> Tuple["CircuitDataset", "CircuitDataset"]:
        """Split the records at random into a train and a test dataset.

        The records are shuffled with ``numpy.random.default_rng(seed)`` and the
        first ``round(frac_train * len(self))`` of them (Python's ``round``,
        halves to even) form the train half.  ``frac_train`` of 0 or 1 puts
        every record, unshuffled, in one half.  Both halves share this
        dataset's root.

        Args:
            seed (default: 0): Seed of the shuffle.
            frac_train (default: 0.8): Fraction of records for the train half.

        Returns:
            ``(train, test)``.

        Raises:
            ValueError: If ``frac_train`` is not a finite, non-boolean number
                in ``[0, 1]``.
        """
        fraction_error = "frac_train must be a finite numeric non-boolean scalar in the inclusive range [0, 1]."
        if isinstance(
            frac_train, (bool, np.bool_, str, bytes, bytearray)
        ) or not np.isscalar(frac_train):
            raise ValueError(fraction_error)

        try:
            frac = float(cast(float, frac_train))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(fraction_error) from exc
        if not np.isfinite(frac):
            raise ValueError(fraction_error)
        if frac < 0.0 or frac > 1.0:
            raise ValueError("frac_train must be in the inclusive range [0, 1].")

        n_records = len(self.records)
        # Fast paths avoid allocation and RNG work for deterministic outputs.
        if n_records == 0:
            return CircuitDataset(self.root, []), CircuitDataset(self.root, [])
        if frac == 0.0:
            return CircuitDataset(self.root, []), CircuitDataset(
                self.root, list(self.records)
            )
        if frac == 1.0:
            return CircuitDataset(self.root, list(self.records)), CircuitDataset(
                self.root, []
            )

        rng = np.random.default_rng(seed)
        idx = np.arange(n_records)
        rng.shuffle(idx)
        cut = int(round(frac * n_records))
        cut = min(max(cut, 0), n_records)
        train = [self.records[i] for i in idx[:cut]]
        test = [self.records[i] for i in idx[cut:]]
        return CircuitDataset(self.root, train), CircuitDataset(self.root, test)


DATASET_INDEX = "qbalance_dataset.json"
_SAFE_ARTIFACT_STEM = re.compile(r"[^A-Za-z0-9._-]+")

# Circuit names are arbitrary user data, but artifact stems become real
# filenames.  Common filesystems cap a single path component at 255 bytes, and
# the stem still has to fit a ``.qpy`` extension plus a de-duplicating ``_N``
# suffix, so keep it comfortably short on every platform.
_MAX_ARTIFACT_STEM = 120
_STEM_DIGEST_LENGTH = 8

# Device names Windows reserves in every directory, with or without an
# extension: writing "con.qpy" there opens the console instead of a file.
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _sanitize_artifact_stem(value: str, *, fallback: str) -> str:
    """Return a filesystem-safe artifact stem derived from ``value``.

    Args:
        value: Source value used to derive a stem.
        fallback: Fallback stem when the source is empty after sanitization.

    Returns:
        Sanitized stem safe for use as a single filename component.  The stem
        is ASCII-only and length-bounded; an over-long name is truncated and
        given a short digest of the original so distinct names sharing a prefix
        stay distinguishable.  A stem Windows reserves as a device name
        (``CON``, ``COM1``, ...) is prefixed with ``_``.
    """
    # Normalize separators and common traversal patterns before replacing unsupported chars.
    stem = value.replace("/", "_").replace("\\", "_")
    stem = _SAFE_ARTIFACT_STEM.sub("_", stem).strip("._-")
    stem = stem or fallback

    if len(stem) > _MAX_ARTIFACT_STEM:
        digest = stable_hash_str(value)[:_STEM_DIGEST_LENGTH]
        keep = _MAX_ARTIFACT_STEM - _STEM_DIGEST_LENGTH - 1
        stem = f"{stem[:keep].rstrip('._-')}-{digest}"
    # Windows matches reserved names on the part before the first dot, so
    # prefix rather than suffix to move the whole stem off the device name.
    if stem.split(".", 1)[0].upper() in _WINDOWS_RESERVED_STEMS:
        stem = f"_{stem}"
    return stem


def _is_safe_artifact_path(artifact: str) -> bool:
    """Validate that an artifact path is a safe single relative filename."""
    if "/" in artifact or "\\" in artifact or "\x00" in artifact:
        return False

    artifact_path = Path(artifact)
    if artifact_path.is_absolute():
        return False

    parts = artifact_path.parts
    if len(parts) != 1:
        return False

    component = parts[0]
    return component not in {"", ".", ".."}


def _build_unique_artifact(base_name: str, used_artifacts: set[str]) -> str:
    """Return an artifact filename not yet taken, ignoring case.

    macOS and Windows filesystems are case-insensitive by default, so
    ``Bell.qpy`` and ``bell.qpy`` are one file there: the second circuit would
    overwrite the first while both records still pointed at it.  Uniqueness is
    therefore decided on the case-folded name.

    Args:
        base_name: Sanitized artifact stem.
        used_artifacts: Case-folded artifact names already assigned.

    Returns:
        ``base_name.qpy``, or ``base_name_N.qpy`` for the smallest free
        ``N >= 1`` (original case preserved).
    """
    suffix = 0
    while True:
        artifact = f"{base_name}.qpy" if suffix == 0 else f"{base_name}_{suffix}.qpy"
        if artifact.casefold() not in used_artifacts:
            return artifact
        suffix += 1


def _build_unique_name(base_name: str, used_names: set[str]) -> str:
    """Return a unique dataset record name while preserving the first name."""
    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        if name not in used_names:
            return name
        suffix += 1


def _validate_metadata_json_keys(value: Any, *, index: int, path: str) -> None:
    """Reject metadata mappings that JSON would silently rewrite."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"metadata entry at index {index} has a non-string key at {path}."
                )
            _validate_metadata_json_keys(nested, index=index, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for nested_index, nested in enumerate(value):
            _validate_metadata_json_keys(
                nested, index=index, path=f"{path}[{nested_index}]"
            )


def _normalize_metadata_entry(index: int, metadata: Any) -> Dict[str, Any]:
    """Validate and deep-copy a save_dataset metadata entry."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata entry at index {index} must be a dict or None.")

    _validate_metadata_json_keys(metadata, index=index, path="metadata")
    try:
        normalized = json.loads(json.dumps(metadata, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata entry at index {index} must be JSON-serializable."
        ) from exc

    if not isinstance(normalized, dict):  # Defensive: root was checked above.
        raise ValueError(f"metadata entry at index {index} must be a dict or None.")
    return cast(Dict[str, Any], normalized)


def save_dataset(
    dataset_dir: Path,
    circuits: Sequence[Any],
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    overwrite: bool = False,
) -> CircuitDataset:
    """Save circuits as a dataset directory and return the dataset.

    Each circuit is written to its own QPY file, and the index
    ``qbalance_dataset.json`` records its name, file and metadata.  A
    circuit's name becomes its record name (``circuit_<i>`` when it has
    none), with ``_1``, ``_2``, ... appended to repeats; file names are
    sanitized, length-bounded and unique ignoring case.  The dataset is
    assembled in a temporary directory and moved into place at the end, so
    a failure leaves no partial dataset and, with ``overwrite=True``, keeps
    the existing one.

    Args:
        dataset_dir: Directory to create.
        circuits: The circuits to save.
        metadata (default: None): One JSON-serializable dict (or ``None``)
            per circuit, stored with its record.
        overwrite (default: False): Whether to replace an existing
            ``dataset_dir``.

    Returns:
        The saved dataset.

    Raises:
        FileExistsError: If ``dataset_dir`` exists and ``overwrite`` is
            false.
        ValueError: If ``metadata`` does not match ``circuits`` in length,
            or an entry is neither a dict nor ``None``, has a non-string
            key, or is not JSON-serializable.
        OptionalDependencyError: If qiskit is not installed.
    """
    dataset_dir = Path(dataset_dir)
    # is_symlink() catches a dangling link, which exists() reports as absent.
    if (dataset_dir.exists() or dataset_dir.is_symlink()) and not overwrite:
        raise FileExistsError(f"{dataset_dir} exists (use overwrite=True)")

    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required to save circuits") from e

    md: List[Any] = list(metadata) if metadata is not None else [None] * len(circuits)
    if len(md) != len(circuits):
        raise ValueError("metadata must have the same length as circuits.")

    # Assembled beside dataset_dir and swapped in at the end, so a failure
    # leaves no partial dataset and keeps an existing one intact.
    with replacing_directory(dataset_dir) as tmp_dir:
        records: List[CircuitRecord] = []
        used_names: set[str] = set()
        used_artifacts: set[str] = set()

        for i, (qc, m) in enumerate(zip(circuits, md)):
            raw_name = getattr(qc, "name", None)
            base_name = f"circuit_{i}" if raw_name is None else str(raw_name)
            if not base_name:
                base_name = f"circuit_{i}"
            name = _build_unique_name(base_name, used_names)
            used_names.add(name)

            record_metadata = _normalize_metadata_entry(i, m)

            safe_stem = _sanitize_artifact_stem(name, fallback=f"circuit_{i}")
            artifact = _build_unique_artifact(safe_stem, used_artifacts)
            used_artifacts.add(artifact.casefold())
            out = tmp_dir / artifact
            with out.open("wb") as f:
                qpy.dump(qc, f)
            records.append(
                CircuitRecord(
                    name=name,
                    artifact=artifact,
                    format="qpy",
                    metadata=record_metadata,
                )
            )

        dump_json(
            tmp_dir / DATASET_INDEX,
            {"version": 1, "records": [r.__dict__ for r in records]},
        )

    return CircuitDataset(dataset_dir, records)


def load_dataset(dataset_dir: Path) -> CircuitDataset:
    """Load the dataset saved in ``dataset_dir``.

    The index is validated in full before anything is deserialized: every
    record needs a unique non-empty ``name``, a unique ``artifact`` that is
    a plain file name present in the directory, a ``format`` of ``"qpy"``
    or ``"qasm"``, and ``metadata`` that is an object or ``null``.

    Args:
        dataset_dir: Directory holding ``qbalance_dataset.json``.

    Returns:
        The dataset; circuits load on demand with
        :meth:`CircuitDataset.load_circuits`.

    Raises:
        OSError: If the index cannot be read (``FileNotFoundError`` when
            ``dataset_dir`` holds none).
        ValueError: If the index is not valid JSON or violates the rules
            above.
    """
    dataset_dir = Path(dataset_dir)
    idx = load_json(dataset_dir / DATASET_INDEX)
    if not isinstance(idx, dict):
        raise ValueError("Dataset index must be a JSON object.")

    records_data = idx.get("records")
    if not isinstance(records_data, list):
        raise ValueError("Dataset index must contain a 'records' list.")

    records: List[CircuitRecord] = []
    required_fields = {"name", "artifact", "format"}
    allowed_formats = {"qpy", "qasm"}
    seen_names: set[str] = set()
    seen_artifacts: set[str] = set()
    for i, raw_record in enumerate(records_data):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Record at index {i} must be a JSON object.")

        missing = required_fields - raw_record.keys()
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(
                f"Record at index {i} is missing required fields: {missing_fields}."
            )

        name = raw_record["name"]
        artifact = raw_record["artifact"]
        fmt = raw_record["format"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"Record at index {i} has invalid name.")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError(f"Record at index {i} has invalid artifact path.")
        if not _is_safe_artifact_path(artifact):
            raise ValueError(f"Record at index {i} has unsafe artifact path.")
        if not isinstance(fmt, str) or fmt not in allowed_formats:
            raise ValueError(f"Record at index {i} has unsupported format: {fmt!r}.")

        if name in seen_names:
            raise ValueError(
                f"Record at index {i} has duplicate circuit name: {name!r}."
            )
        seen_names.add(name)

        if artifact in seen_artifacts:
            raise ValueError(
                f"Record at index {i} has duplicate artifact path: {artifact!r}."
            )
        seen_artifacts.add(artifact)

        metadata = raw_record.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Record at index {i} has non-object metadata.")

        artifact_path = dataset_dir / artifact
        if not artifact_path.exists():
            raise ValueError(
                f"Record at index {i} points to missing artifact: {artifact!r}."
            )
        if not artifact_path.is_file():
            raise ValueError(
                f"Record at index {i} points to non-file artifact: {artifact!r}."
            )

        records.append(
            CircuitRecord(
                name=name,
                artifact=artifact,
                format=fmt,
                metadata=metadata,
            )
        )

    return CircuitDataset(dataset_dir, records)


def load_data(name: str) -> CircuitDataset:
    """Load a built-in dataset, materializing it on first use.

    Args:
        name: Built-in dataset name; ``"tiny"`` is the only one.

    Returns:
        The dataset.

    Raises:
        KeyError: If ``name`` is not a built-in dataset.
    """
    from qbalance.builtin_data import get_builtin_dataset_dir

    return load_dataset(get_builtin_dataset_dir(name))
