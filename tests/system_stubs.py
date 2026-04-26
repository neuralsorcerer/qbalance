# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import types


class _Q:
    def __init__(self, idx: int):

        self.index = idx


class _I:
    def __init__(self, name: str):

        self.name = name


class _Circ:
    def __init__(self):

        self.num_qubits = 2
        self.qubits = [_Q(0), _Q(1)]
        self.data = [(_I("cx"), [_Q(0), _Q(1)], []), (_I("measure"), [_Q(0)], [])]

    def depth(self):

        return 3

    def size(self):

        return 4

    def copy(self):

        c = _Circ()
        c.data = list(self.data)
        return c

    def inverse(self):

        return self

    def compose(self, other):

        _ = other
        return self


class _PM:
    def __init__(self, out):

        self.out = out

    def run(self, circ, callback=None):

        if callback is not None:
            callback(pass_=object(), time=0.1, count=1)
        _ = circ
        return self.out


class _Job:
    def __init__(self, counts):

        self._counts = counts

    def result(self):

        return types.SimpleNamespace(get_counts=lambda: self._counts)
