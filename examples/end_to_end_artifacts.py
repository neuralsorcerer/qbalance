# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end artifact generation example."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from qbalance import Workload, load_data


def main() -> None:
    logging.disable(logging.ERROR)
    with TemporaryDirectory() as td:
        out = Path(td)
        ds = load_data("tiny")
        wl = Workload.from_dataset(ds).set_target("fake:generic:5")
        balanced = wl.adjust(
            search="bandit", pareto=True, max_candidates=12, execute=False
        )

        run_dir = out / "balanced"
        zip_path = out / "balanced.zip"
        balanced.save(run_dir, overwrite=True)
        balanced.to_download(zip_path, overwrite=True)

        print("saved_dir:", run_dir)
        print("zip_path:", zip_path)
        print("summary_head:", balanced.summary().splitlines()[0])


if __name__ == "__main__":
    main()
