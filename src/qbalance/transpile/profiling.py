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
    name: str
    time_s: float
    index: int


@dataclass
class ProfileReport:
    passes: List[PassProfile] = field(default_factory=list)

    def total_time_s(self) -> float:
        """Total time s used by the qbalance workflow.

        Args:
            None.

        Returns:
            float with the computed result.

        Raises:
            None.
        """
        return sum(p.time_s for p in self.passes)

    def to_json(self) -> Dict[str, Any]:
        """To json used by the qbalance workflow.

        Args:
            None.

        Returns:
            Dict[str, Any] with the computed result.

        Raises:
            None.
        """
        return {
            "total_time_s": self.total_time_s(),
            "passes": [
                {"name": p.name, "time_s": p.time_s, "index": p.index}
                for p in self.passes
            ],
        }


def make_callback(report: ProfileReport):
    """Make callback used by the qbalance workflow.

    Args:
        report: Report value consumed by this routine.

    Returns:
        Computed value produced by this routine.

    Raises:
        None.
    """

    def _cb(**kwargs):
        """Internal helper that cb.

        Args:
            **kwargs: Kwargs value consumed by this routine.

        Returns:
            Computed value produced by this routine.

        Raises:
            None.
        """
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
