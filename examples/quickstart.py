# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from qbalance import Workload, load_data


def main() -> None:
    logging.disable(logging.ERROR)
    ds = load_data("tiny")
    wl = Workload.from_dataset(ds).set_target("fake:generic:5")
    bw = wl.adjust(
        search="bandit", pareto=True, max_candidates=24, execute=False, profile=False
    )
    print(bw.summary())


if __name__ == "__main__":
    main()
