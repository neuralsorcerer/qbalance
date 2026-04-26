# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger
from qbalance.utils import dump_json, load_json

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
        """Return the number of records contained in the dataset.

        Args:
            None.

        Returns:
            int with the computed result.

        Raises:
            None.
        """
        return len(self.records)

    def names(self) -> List[str]:
        """Names used by the qbalance workflow.

        Args:
            None.

        Returns:
            List[str] with the computed result.

        Raises:
            None.
        """
        return [r.name for r in self.records]

    def iter_records(self) -> Iterable[CircuitRecord]:
        """Iter records used by the qbalance workflow.

        Args:
            None.

        Returns:
            Iterable[CircuitRecord] with the computed result.

        Raises:
            None.
        """
        yield from self.records

    def load_circuits(self) -> List[Any]:
        """Load circuits from serialized data or persisted storage.

        Args:
            None.

        Returns:
            List[Any] with the computed result.

        Raises:
            OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        try:
            from qiskit import QuantumCircuit, qpy
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError("qiskit is required to load circuits") from e

        circuits: List[Any] = []
        for rec in self.records:
            path = self.root / rec.artifact
            if rec.format == "qpy":
                with path.open("rb") as f:
                    loaded = qpy.load(f)
                if not loaded:
                    raise ValueError(f"Empty QPY file: {path}")
                circuits.append(loaded[0])
            elif rec.format == "qasm":
                circuits.append(QuantumCircuit.from_qasm_file(str(path)))
            else:
                raise ValueError(f"Unknown circuit format: {rec.format}")
        return circuits

    def split(
        self, seed: int = 0, frac_train: float = 0.8
    ) -> Tuple["CircuitDataset", "CircuitDataset"]:
        """Split used by the qbalance workflow.

        Args:
            seed (default: 0): Seed used for deterministic randomization.
            frac_train (default: 0.8): Frac train value consumed by this routine.

        Returns:
            Tuple['CircuitDataset', 'CircuitDataset'] with the computed result.

        Raises:
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
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


def _sanitize_artifact_stem(value: str, *, fallback: str) -> str:
    """Return a filesystem-safe artifact stem.

    Args:
        value: Source value used to derive a stem.
        fallback: Fallback stem when the source is empty after sanitization.

    Returns:
        Sanitized stem safe for use as a single filename component.

    Raises:
        None.
    """
    # Normalize separators and common traversal patterns before replacing unsupported chars.
    stem = value.replace("/", "_").replace("\\", "_")
    stem = _SAFE_ARTIFACT_STEM.sub("_", stem).strip("._-")
    return stem or fallback


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
    """Internal helper that build unique artifact.

    Args:
        base_name: Base name value consumed by this routine.
        used_artifacts: Used artifacts value consumed by this routine.

    Returns:
        str with the computed result.

    Raises:
        None.
    """
    suffix = 0
    while True:
        artifact = f"{base_name}.qpy" if suffix == 0 else f"{base_name}_{suffix}.qpy"
        if artifact not in used_artifacts:
            return artifact
        suffix += 1


def save_dataset(
    dataset_dir: Path,
    circuits: Sequence[Any],
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    overwrite: bool = False,
) -> CircuitDataset:
    """Persist dataset and return a reference to the saved artifact.

    Args:
        dataset_dir: Directory containing the dataset index and circuit artifacts.
        circuits: Iterable of QuantumCircuit objects to serialize or process.
        metadata (default: None): Metadata value consumed by this routine.
        overwrite (default: False): Whether existing files/directories may be replaced.

    Returns:
        CircuitDataset with the computed result.

    Raises:
        FileExistsError: Raised when input validation fails or a dependent operation cannot be completed.
        ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"{dataset_dir} exists (use overwrite=True)")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required to save circuits") from e

    md: List[Optional[Dict[str, Any]]] = (
        list(metadata) if metadata is not None else [None] * len(circuits)
    )
    if len(md) != len(circuits):
        raise ValueError("metadata must have the same length as circuits.")

    records: List[CircuitRecord] = []
    used_artifacts: set[str] = set()

    for i, (qc, m) in enumerate(zip(circuits, md)):
        raw_name = getattr(qc, "name", None)
        name = f"circuit_{i}" if raw_name is None else str(raw_name)
        if not name:
            name = f"circuit_{i}"
        safe_stem = _sanitize_artifact_stem(name, fallback=f"circuit_{i}")
        artifact = _build_unique_artifact(safe_stem, used_artifacts)
        used_artifacts.add(artifact)
        out = dataset_dir / artifact
        with out.open("wb") as f:
            qpy.dump(qc, f)
        records.append(
            CircuitRecord(name=name, artifact=artifact, format="qpy", metadata=m or {})
        )

    dump_json(
        dataset_dir / DATASET_INDEX,
        {"version": 1, "records": [r.__dict__ for r in records]},
    )
    return CircuitDataset(dataset_dir, records)


def load_dataset(dataset_dir: Path) -> CircuitDataset:
    """Load dataset from serialized data or persisted storage.

    Args:
        dataset_dir: Directory containing the dataset index and circuit artifacts.

    Returns:
        CircuitDataset with the computed result.

    Raises:
        None.
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

        metadata = raw_record.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Record at index {i} has non-object metadata.")

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
    """Load data from serialized data or persisted storage.

    Args:
        name: Name/identifier for a circuit, dataset, or lookup record.

    Returns:
        CircuitDataset with the computed result.

    Raises:
        None.
    """
    from qbalance.builtin_data import get_builtin_dataset_dir

    return load_dataset(get_builtin_dataset_dir(name))
