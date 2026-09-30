# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger

log = get_logger(__name__)


def _calibration_system(backend: Any, seed: Optional[int] = None) -> Any:
    """Return the backend M3 should run its calibration circuits on.

    mthree reads ``backend.configuration()``, which Qiskit's
    ``GenericBackendV2`` (every ``fake:generic`` spec) does not provide, so M3
    failed on those backends every time.  ``GenericBackendV2`` executes on an
    Aer simulator carrying ``NoiseModel.from_backend(self)``, which is exactly
    what ``AerSimulator.from_backend`` builds, so calibrating on the latter
    measures the same readout channel.  Every other backend is returned
    unchanged: a hardware backend's calibration must come from the device.
    The simulator created here is private, so it can take ``seed`` without
    side effects, which makes the calibration -- and the mitigated result --
    reproducible.
    """
    if callable(getattr(backend, "configuration", None)):
        return backend
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception:  # pragma: no cover - qiskit always provides it
        return backend
    if not isinstance(backend, GenericBackendV2):
        return backend
    try:
        from qiskit_aer import AerSimulator
    except Exception:
        # Without Aer, GenericBackendV2 runs noiselessly and there is no
        # readout channel to calibrate; let mthree report the problem.
        return backend
    simulator = AerSimulator.from_backend(backend)
    if seed is not None:
        simulator.set_options(seed_simulator=int(seed))
    return simulator


def _register_separators(counts: Mapping[str, Any]) -> Optional[str]:
    """Return a count key to copy register separators from, if keys have any.

    Qiskit separates classical registers with spaces in count keys ("10 1"),
    but mthree requires one character per measured qubit.  Every key of one
    circuit's counts shares the same register layout, so any key serves as
    the template.
    """
    for key in counts:
        return key if " " in key else None
    return None


def _without_separators(counts: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``counts`` keyed by bitstrings with register separators removed."""
    merged: Dict[str, Any] = {}
    for key, value in counts.items():
        bits = key.replace(" ", "")
        merged[bits] = merged.get(bits, 0) + value
    return merged


def _with_separators(bits: str, template: Optional[str]) -> str:
    """Re-insert ``template``'s register separators into a bare bitstring."""
    if template is None or len(bits) != len(template.replace(" ", "")):
        return bits
    digits = iter(bits)
    return "".join(char if char == " " else next(digits) for char in template)


class _KeyLayout:
    """Where the bits M3 corrects sit inside Qiskit count keys.

    A count key has one character per classical bit, classical bit 0
    rightmost, with a space between registers.  M3 needs exactly one character
    per measured qubit, so keys are projected onto the measured bits for the
    correction and expanded back afterwards: separators return, and a
    classical bit no measurement writes -- always ``0`` -- is restored as ``0``.
    """

    def __init__(self, counts: Mapping[str, Any], clbits: Optional[Sequence[int]]):
        template = next(iter(counts), "")
        self.template = _register_separators(counts)
        self.width = len(template.replace(" ", ""))
        self.clbits = None if clbits is None else [int(c) for c in clbits]
        if self.clbits is not None and any(
            c < 0 or c >= self.width for c in self.clbits
        ):
            raise ValueError(
                f"clbits {self.clbits} do not fit count keys of {self.width} bits"
            )

    def project(self, counts: Mapping[str, Any]) -> Dict[str, Any]:
        """Return ``counts`` keyed by the measured bits only, in M3's order."""
        if self.clbits is None:
            return _without_separators(counts)
        projected: Dict[str, Any] = {}
        for key, value in counts.items():
            bits = key.replace(" ", "")
            reduced = "".join(bits[len(bits) - 1 - c] for c in reversed(self.clbits))
            projected[reduced] = projected.get(reduced, 0) + value
        return projected

    def expand(self, reduced: str) -> str:
        """Return the full count key for a key over the measured bits."""
        if self.clbits is None:
            return _with_separators(reduced, self.template)
        chars = ["0"] * self.width
        for index, clbit in enumerate(self.clbits):
            chars[self.width - 1 - clbit] = reduced[len(reduced) - 1 - index]
        return _with_separators("".join(chars), self.template)


def apply_mthree_mitigation(
    backend: Any,
    raw_counts: Dict[str, int],
    measured_qubits: List[int],
    shots: Optional[int] = None,
    calibration_shots: int = 10_000,
    clbits: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Apply M3 readout mitigation to one counts mapping.

    The readout of ``measured_qubits`` is calibrated on the backend (for
    ``GenericBackendV2``, which mthree cannot calibrate, on
    ``AerSimulator.from_backend(backend)``), the counts are corrected, and
    the result is projected onto the nearest probability distribution.
    Pass counts exactly as measured, before any untwirling.

    Args:
        backend: Backend the counts were measured on.
        raw_counts: Counts as returned by the backend.
        measured_qubits: Physical qubit feeding each classical bit, ordered by
            classical bit index.  Getting this order wrong degrades the
            correction silently rather than failing.
        shots (default: None): Accepted for call-site symmetry and recorded by
            callers; mthree derives the shot count from ``raw_counts`` and takes
            its calibration budget from ``calibration_shots``, so this value is
            not used here.
        calibration_shots (default: 10000): Shot budget for mthree's calibration.
        clbits (default: None): Classical bit each entry of ``measured_qubits``
            writes (see :func:`qbalance.utils.measured_clbits`).  Needed when
            the counts carry classical bits no measurement writes; ``None``
            means every bit of the keys is measured, in order.

    Returns:
        Mitigated probabilities keyed like ``raw_counts``: register
        separators are kept, and a classical bit no measurement writes stays
        ``0``.

    Raises:
        OptionalDependencyError: If mthree is not installed.
        ValueError: If ``clbits`` names a bit outside the count keys.
        Exception: Whatever mthree raises when it cannot calibrate the
            backend (``GenericBackendV2`` without qiskit-aer, for one).
    """
    try:
        import mthree
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "mthree is required (install qbalance[mitigation])"
        ) from e

    mit = mthree.M3Mitigation(_calibration_system(backend))
    mit.cals_from_system(measured_qubits, calibration_shots)
    layout = _KeyLayout(raw_counts, clbits)
    quasi = mit.apply_correction(layout.project(raw_counts), measured_qubits)
    probs = quasi.nearest_probability_distribution()  # best-effort true probs
    # Convert to python dict, keyed like the counts that came in.
    return {layout.expand(k): float(v) for k, v in probs.items()}


def mitigate_twirled_counts(
    backend: Any,
    raw_counts: Sequence[Dict[str, int]],
    flip_maps: Sequence[Dict[int, int]],
    measured_qubits: List[int],
    calibration_shots: int = 10_000,
    seed: Optional[int] = None,
    clbits: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """M3-correct a measurement-twirled ensemble and combine the instances.

    Each instance carries its own twirl flip pattern, so the instances cannot
    be pooled before mitigation: M3's calibration describes the physical
    readout channel, which only matches the counts as measured.  Every
    instance is therefore corrected on its raw counts, untwirled with its own
    flip map, and weighted by its shot share; the combined quasi-distribution
    is projected onto the nearest probability distribution once, at the end.

    Args:
        backend: Backend the instances ran on.
        raw_counts: Raw counts per instance, as returned by the backend.
        flip_maps: Measurement-twirl flip map per instance.
        measured_qubits: Physical qubit feeding each classical bit, ordered by
            classical bit index (shared by all instances).
        calibration_shots (default: 10000): Shot budget for M3's calibration.
        seed (default: None): Seed for the calibration simulator qbalance
            creates for ``GenericBackendV2``; other backends calibrate as they
            run.
        clbits (default: None): Classical bit each entry of ``measured_qubits``
            writes; needed when the counts carry classical bits no measurement
            writes.  ``None`` means every bit of the keys is measured, in order.

    Returns:
        Mitigated, untwirled probabilities keyed by bitstring.

    Raises:
        OptionalDependencyError: If mthree is not installed.
        ValueError: If ``raw_counts`` and ``flip_maps`` differ in length or hold
            no shots.
    """
    try:
        import mthree
        from mthree.classes import QuasiDistribution
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "mthree is required (install qbalance[mitigation])"
        ) from e
    from qbalance.transpile.suppression import apply_measurement_untwirl_counts

    if len(raw_counts) != len(flip_maps):
        raise ValueError("raw_counts and flip_maps must have the same length")
    total = sum(sum(counts.values()) for counts in raw_counts)
    if total <= 0:
        raise ValueError("raw_counts must contain at least one shot")

    mit = mthree.M3Mitigation(_calibration_system(backend, seed))
    mit.cals_from_system(measured_qubits, calibration_shots)
    combined: Dict[str, float] = {}
    for counts, flip_map in zip(raw_counts, flip_maps):
        instance_shots = sum(counts.values())
        if instance_shots <= 0:
            continue
        layout = _KeyLayout(counts, clbits)
        quasi = mit.apply_correction(layout.project(counts), measured_qubits)
        untwirled = apply_measurement_untwirl_counts(
            {layout.expand(key): float(value) for key, value in quasi.items()},
            flip_map,
        )
        weight = instance_shots / total
        for key, value in untwirled.items():
            combined[key] = combined.get(key, 0.0) + weight * value
    probs = QuasiDistribution(combined, shots=total).nearest_probability_distribution()
    return {key: float(value) for key, value in probs.items()}
