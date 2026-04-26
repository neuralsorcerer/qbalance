# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Advanced qbalance API example.

This script demonstrates:
- choosing an explicit objective,
- running a grid search with Pareto filtering,
- inspecting covariate distance diagnostics.
"""

from __future__ import annotations

import logging

from qbalance import Workload, load_data
from qbalance.objectives import Objective


def main() -> None:
    logging.disable(logging.ERROR)
    ds = load_data("tiny")
    wl = Workload.from_dataset(ds).set_target("fake:generic:5")
    objective = Objective(
        weights={"depth": 0.4, "two_qubit_ops": 0.4, "compile_time_s": 0.2}
    )
    balanced = wl.adjust(
        objective=objective,
        search="grid",
        pareto=True,
        max_candidates=20,
        execute=False,
        profile=False,
        seed=7,
    )
    print(balanced.summary())
    print("covars:", balanced.covars())


if __name__ == "__main__":
    main()
