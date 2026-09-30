# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PassProfile:
    """One pass of a profiled compile: its class name, run time and index."""

    name: str
    time_s: float
    index: int


@dataclass
class ProfileReport:
    """The passes of a profiled compile, in the order they ran."""

    passes: List[PassProfile] = field(default_factory=list)

    def total_time_s(self) -> float:
        """Return the summed run time of the recorded passes, in seconds."""
        return sum(p.time_s for p in self.passes)

    def to_json(self) -> Dict[str, Any]:
        """Return the report as JSON-ready data.

        ``{"total_time_s": t, "passes": [{"name": n, "time_s": t, "index": i}]}``
        """
        return {
            "total_time_s": self.total_time_s(),
            "passes": [
                {"name": p.name, "time_s": p.time_s, "index": p.index}
                for p in self.passes
            ],
        }


def make_callback(report: ProfileReport):
    """Return a pass-manager callback that records every pass into ``report``.

    Pass it as ``callback=`` to ``PassManager.run`` (or ``transpile``);
    each call appends the pass's class name, run time and running index.
    """

    def _cb(**kwargs):
        """Append the pass Qiskit reports to ``report``."""
        pass_ = kwargs.get("pass_")
        t = float(kwargs.get("time", 0.0))
        idx = int(kwargs.get("count", -1))
        name = (
            getattr(pass_, "__class__", type(pass_)).__name__
            if pass_ is not None
            else "UnknownPass"
        )
        report.passes.append(PassProfile(name=name, time_s=t, index=idx))

    return _cb
