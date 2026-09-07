# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from qbalance._version import __version__ as _fallback_version
from qbalance.dataset import CircuitDataset, load_data, load_dataset, save_dataset
from qbalance.objectives import Objective, default_objective, load_objective
from qbalance.strategies import (
    Strategy,
    StrategySpec,
    coerce_strategy_specs,
    load_strategy_specs,
)
from qbalance.workflow.workload import (
    BalancedWorkload,
    Workload,
    load_balanced_workload,
)

try:
    # Prefer the installed distribution metadata, which is authoritative for
    # the environment actually in use; _version.py is the in-tree fallback for
    # running straight from a source checkout.
    __version__ = _installed_version("qbalance")
except _PackageNotFoundError:  # pragma: no cover - only without an install
    __version__ = _fallback_version

__all__ = [
    "__version__",
    "CircuitDataset",
    "load_dataset",
    "save_dataset",
    "load_data",
    "Objective",
    "default_objective",
    "load_objective",
    "Workload",
    "BalancedWorkload",
    "load_balanced_workload",
    "Strategy",
    "StrategySpec",
    "coerce_strategy_specs",
    "load_strategy_specs",
]
