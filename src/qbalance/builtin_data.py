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


def _make_tiny() -> List[Any]:
    """Internal helper that make tiny.

    Args:
        None.

    Returns:
        List[Any] with the computed result.

    Raises:
        None.
    """
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
    """Return builtin dataset dir for the provided inputs.

    Args:
        name: Name/identifier for a circuit, dataset, or lookup record.

    Returns:
        Path with the computed result.

    Raises:
        KeyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    root = Path(user_data_dir("qbalance")) / "datasets" / name
    if (root / "qbalance_dataset.json").exists():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)

    if name == "tiny":
        circuits = _make_tiny()
    else:
        raise KeyError(f"Unknown built-in dataset: {name}")

    save_dataset(root, circuits, overwrite=True)
    return root
