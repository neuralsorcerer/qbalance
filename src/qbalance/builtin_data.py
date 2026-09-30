# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from platformdirs import user_data_dir

from qbalance.dataset import save_dataset
from qbalance.logging import get_logger

log = get_logger(__name__)

# Names accepted by get_builtin_dataset_dir; each becomes a path component.
_BUILTIN_DATASETS = frozenset({"tiny"})


def _make_tiny() -> List[Any]:
    """Return the circuits of the ``tiny`` dataset: ``bell``, ``ghz3`` and ``qft4``."""
    import numpy as np
    from qiskit import QuantumCircuit

    circuits: List[QuantumCircuit] = []

    qc1 = QuantumCircuit(2, 2, name="bell")
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure([0, 1], [0, 1])
    circuits.append(qc1)

    qc2 = QuantumCircuit(3, 3, name="ghz3")
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.cx(1, 2)
    qc2.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc2)

    qc3 = QuantumCircuit(4, 4, name="qft4")
    for i in range(4):
        qc3.h(i)
        for j in range(i + 1, 4):
            qc3.cp(np.pi / (2 ** (j - i)), j, i)
    qc3.measure(range(4), range(4))
    circuits.append(qc3)

    return circuits


def get_builtin_dataset_dir(name: str) -> Path:
    """Return the directory of a built-in dataset, writing it on first use.

    The dataset lives under the platform's user data directory
    (``platformdirs.user_data_dir("qbalance")/datasets/NAME``) and is
    reused once its index exists.

    Args:
        name: Built-in dataset name; ``"tiny"`` is the only one.

    Returns:
        The dataset directory.

    Raises:
        KeyError: If ``name`` is not a built-in dataset.
    """
    # Validate before the name reaches the filesystem.  It is joined straight
    # into a path, so "/abs" would discard the data directory entirely and
    # "../.." would climb out of it -- and mkdir(parents=True) ran before the
    # name was ever checked, creating those directories on the way to raising.
    if name not in _BUILTIN_DATASETS:
        raise KeyError(f"Unknown built-in dataset: {name}")

    root = Path(user_data_dir("qbalance")) / "datasets" / name
    if (root / "qbalance_dataset.json").exists():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)

    save_dataset(root, _make_tiny(), overwrite=True)
    return root
