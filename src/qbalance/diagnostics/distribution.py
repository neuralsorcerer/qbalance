# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Tuple

import numpy as np


def _normalized_cumsum(weights: np.ndarray) -> np.ndarray:
    """Return the cumulative sum of normalized weights, ending at exactly 1.

    Normalize by the accumulated total so trailing zero-weight points keep
    the same probability even when cumulative addition rounds above or below 1.
    """
    cdf = np.cumsum(weights, dtype=float)
    if cdf.size:
        cdf /= cdf[-1]
    return cdf


def _as_1d_float_array(x: Iterable[float], *, name: str) -> np.ndarray:
    """Return ``x`` as a one-dimensional float array.

    Args:
        x: Numeric samples: an array-like or any iterable, generators included.
        name: What ``x`` is, for the error messages.

    Returns:
        The values as a 1-D ``float`` array.  Finiteness is not checked.

    Raises:
        ValueError: If a value is not a real number or ``x`` is not 1-D.
    """
    try:
        raw = np.asarray(x)
        if raw.ndim == 0 and raw.dtype == object and isinstance(x, Iterable):
            raw = np.asarray(list(x))
        if np.iscomplexobj(raw) or (
            raw.dtype == object and any(np.iscomplexobj(v) for v in raw.flat)
        ):
            raise ValueError("complex values are not real")
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only finite real numbers.") from exc

    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional iterable of real numbers.")
    return values


def _to_np(
    x: Iterable[float], w: Iterable[float] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return validated samples and their weights normalized to sum to 1.

    Negative weights count as 0.  Without weights, or when none is
    positive, every sample weighs the same.

    Args:
        x: Samples.
        w (default: None): One weight per sample.

    Returns:
        ``(values, weights)`` as float arrays of equal length.

    Raises:
        ValueError: If ``x`` is empty, not 1-D, or not finite, or ``w`` is
            not finite or differs from ``x`` in length.
    """
    values = _as_1d_float_array(x, name="Input samples")
    if values.size == 0:
        raise ValueError("Input samples must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Input samples must be finite real numbers.")

    if w is None:
        weights = np.full(values.shape, 1.0 / values.size, dtype=float)
        return values, weights

    weights = _as_1d_float_array(w, name="Weights")
    if weights.shape != values.shape:
        raise ValueError("Weights must have the same length as samples.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite real numbers.")

    # Clamp negative entries into a new array: ``np.asarray`` returns a float
    # input unchanged, so clamping in place would rewrite the caller's weights.
    weights = np.maximum(weights, 0.0)
    max_weight = float(np.max(weights))
    if max_weight <= 0.0:
        weights = np.full(values.shape, 1.0 / values.size, dtype=float)
        return values, weights

    # Scale by the maximum first so summation cannot overflow for very large
    # finite weights and tiny values are preserved proportionally.
    scaled = weights / max_weight
    total_scaled = float(np.sum(scaled, dtype=float))
    if total_scaled <= 0.0:
        weights = np.full(values.shape, 1.0 / values.size, dtype=float)
    else:
        weights = scaled / total_scaled
    return values, weights


def weighted_cdf(
    x: Iterable[float], w: Iterable[float] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the weighted empirical CDF of ``x``.

    Negative weights count as 0; without weights, or when none is positive,
    every sample weighs the same.

    Args:
        x: Samples; non-empty, one-dimensional and finite.
        w (default: None): One finite weight per sample.

    Returns:
        ``(support, cdf)``: the sorted distinct sample values, and the CDF at
        each of them (the last entry is exactly 1).

    Raises:
        ValueError: If the samples or weights are invalid.
    """
    xs, weights = _weighted_support(x, w)
    return xs, _normalized_cumsum(weights)


def _weighted_support(
    x: Iterable[float], w: Iterable[float] | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ordered distinct sample values and the mass at each one."""
    values, weights = _to_np(x, w)
    order = np.argsort(values)
    xs_sorted = values[order]
    ws_sorted = weights[order]

    # Fast path: if support has fewer than two points there can be no
    # duplicates to aggregate.
    if xs_sorted.size < 2:
        return xs_sorted, ws_sorted

    # Locate the start index of every run of identical sorted support values.
    # Reusing these boundaries avoids an extra uniqueness pass over data that is
    # already sorted and aggregates duplicate points in O(n).
    run_start = np.empty(xs_sorted.size, dtype=bool)
    run_start[0] = True
    run_start[1:] = xs_sorted[1:] != xs_sorted[:-1]
    first_idx = np.flatnonzero(run_start)
    if first_idx.size == xs_sorted.size:
        return xs_sorted, ws_sorted

    # Aggregate repeated support points so downstream CDF evaluation does not
    # perform redundant binary-search work.
    unique_xs = xs_sorted[first_idx]
    summed_weights = np.add.reduceat(ws_sorted, first_idx)
    return unique_xs, summed_weights


def _cdf_on_grid(grid: np.ndarray, xs: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Evaluate the step CDF ``(xs, cdf)`` at every point of ``grid``.

    The CDF is right-continuous and 0 before its first support point.
    """
    if grid is xs:
        return cdf
    idx = np.searchsorted(xs, grid, side="right") - 1
    out = np.zeros(grid.size, dtype=float)
    valid = idx >= 0
    out[valid] = cdf[idx[valid]]
    return out


def _aligned_cdfs(
    x1: Iterable[float],
    x2: Iterable[float],
    w1: Iterable[float] | None = None,
    w2: Iterable[float] | None = None,
    *,
    center_upper_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two weighted empirical CDFs evaluated on a common grid.

    The grid is the union of both supports, on which both CDFs are exact
    step functions. For distances, ``center_upper_tail`` subtracts one from
    both CDFs wherever both exceed one half. Evaluating those entries as
    negative survival probabilities preserves their difference even when
    both ordinary CDFs round to one.

    Raises:
        ValueError: If either sample or its weights are invalid.
    """
    xs1, mass1 = _weighted_support(x1, w1)
    xs2, mass2 = _weighted_support(x2, w2)
    # Fast path for equal supports avoids unnecessary concatenation,
    # de-duplication, and repeated binary searches.
    if xs1.shape == xs2.shape and np.array_equal(xs1, xs2):
        grid = xs1
        xs2 = xs1
    # If supports do not overlap, concatenate directly; if they touch at one
    # endpoint, skip one duplicate boundary value.
    elif xs1[-1] <= xs2[0]:
        grid = np.concatenate((xs1, xs2[1:] if xs1[-1] == xs2[0] else xs2))
    elif xs2[-1] <= xs1[0]:
        grid = np.concatenate((xs2, xs1[1:] if xs2[-1] == xs1[0] else xs1))
    else:
        # ``np.union1d`` uses optimized NumPy routines and remains faster than
        # Python-level merge loops for realistic support sizes.
        grid = np.union1d(xs1, xs2)
    cdf1 = _cdf_on_grid(grid, xs1, _normalized_cumsum(mass1))
    cdf2 = _cdf_on_grid(grid, xs2, _normalized_cumsum(mass2))
    if center_upper_tail:
        upper = (cdf1 > 0.5) & (cdf2 > 0.5)
        upper_grid = grid[upper]
        for cdf, xs, mass in ((cdf1, xs1, mass1), (cdf2, xs2, mass2)):
            # P(X > x_i) excludes the mass at x_i. The upper grid starts
            # within both supports, so the usual right-continuous lookup fits.
            survival = np.append(_normalized_cumsum(mass[::-1])[-2::-1], 0.0)
            cdf[upper] = -_cdf_on_grid(upper_grid, xs, survival)
    return grid, cdf1, cdf2


def _integer_scale(values: np.ndarray) -> tuple[list[int], int]:
    """Express binary floats as integers with a common power-of-two denominator."""
    ratios = [float(value).as_integer_ratio() for value in values]
    scale = max(denominator for _, denominator in ratios)
    return [numerator * (scale // divisor) for numerator, divisor in ratios], scale


def _exact_distance(
    grid: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    w1: np.ndarray | None,
    w2: np.ndarray | None,
    metric: str,
) -> float:
    """Compute ill-conditioned CDF differences before rounding their probabilities."""

    def masses(x: np.ndarray, w: np.ndarray | None) -> dict[float, int]:
        if w is None or np.max(w) <= 0.0:
            weights = [1] * len(x)
        else:
            weights, _ = _integer_scale(np.maximum(w, 0.0))
        result: dict[float, int] = {}
        for point, weight in zip(x, weights):
            result[point] = result.get(point, 0) + weight
        return result

    a, b = masses(x1, w1), masses(x2, w2)
    total_a, total_b = sum(a.values()), sum(b.values())
    denominator = total_a * total_b
    coords, coord_scale = _integer_scale(grid)
    cumulative = 0
    result = 0
    for i, point in enumerate(grid[:-1]):
        cumulative += a.get(point, 0) * total_b - b.get(point, 0) * total_a
        if metric == "ks":
            result = max(result, abs(cumulative))
        else:
            height = cumulative**2 if metric == "cvm" else abs(cumulative)
            result += (coords[i + 1] - coords[i]) * height

    if metric != "ks":
        denominator = coord_scale * (denominator**2 if metric == "cvm" else denominator)
    try:
        return result / denominator
    except OverflowError as exc:
        raise ValueError(
            "Integrated area is not finite; check value/grid scales."
        ) from exc


def _distance(
    x1: Iterable[float],
    x2: Iterable[float],
    w1: Iterable[float] | None,
    w2: Iterable[float] | None,
    metric: str,
) -> float:
    # Keep the original weights for cases where the two CDFs nearly cancel,
    # including generator inputs that cannot be iterated a second time.
    a = _as_1d_float_array(x1, name="Input samples")
    b = _as_1d_float_array(x2, name="Input samples")
    wa = None if w1 is None else _as_1d_float_array(w1, name="Weights")
    wb = None if w2 is None else _as_1d_float_array(w2, name="Weights")
    grid, c1, c2 = _aligned_cdfs(a, b, wa, wb, center_upper_tail=True)
    delta = np.abs(c1 - c2)

    def measure(heights: np.ndarray) -> float:
        if metric == "ks":
            return float(np.max(heights))
        return _integrate_piecewise_constant(heights, grid, squared=metric == "cvm")

    if wa is None and wb is None:
        return measure(delta)
    # Extremely disparate weights can underflow during normalization before
    # any subtraction is attempted.
    if any(
        float(np.min(w[w > 0])) / float(np.max(w)) < np.finfo(float).tiny
        for w in (wa, wb)
        if w is not None and np.any(w > 0)
    ):
        return _exact_distance(grid, a, b, wa, wb, metric)

    result = measure(delta)
    # An interval whose difference is smaller than sqrt(eps) times the
    # operands may have lost half its significant digits. Compare the
    # magnitude of these uncertain intervals with the whole statistic, so an
    # isolated crossing in two otherwise different CDFs stays vectorized.
    bound = np.maximum(np.abs(c1), np.abs(c2)) * np.sqrt(np.finfo(float).eps)
    uncertain = (bound > 0.0) & (delta <= bound)
    if np.any(uncertain) and measure(np.where(uncertain, bound, 0.0)) >= result:
        return _exact_distance(grid, a, b, wa, wb, metric)
    return result


def ks_1d(
    x1: Iterable[float],
    x2: Iterable[float],
    w1: Iterable[float] | None = None,
    w2: Iterable[float] | None = None,
) -> float:
    """Return the Kolmogorov-Smirnov statistic between two weighted samples.

    That is ``sup_t |F1(t) - F2(t)|`` for the weighted empirical CDFs of
    :func:`weighted_cdf`.

    Args:
        x1: First sample.
        x2: Second sample.
        w1 (default: None): Weights of ``x1``.
        w2 (default: None): Weights of ``x2``.

    Returns:
        The statistic, in ``[0, 1]``.

    Raises:
        ValueError: If a sample or its weights are invalid (see
            :func:`weighted_cdf`).
    """
    return _distance(x1, x2, w1, w2, "ks")


def _integrate_piecewise_constant(
    values: np.ndarray, grid: np.ndarray, *, squared: bool = False
) -> float:
    """Integrate the step heights (or their squares) over consecutive grid intervals.

    Args:
        values: Function value on each grid interval (the last is unused).
        grid: Non-decreasing interval boundaries, the same length as
            ``values``.
        squared: Integrate the squared heights for the Cramér-von Mises distance.

    Returns:
        The integral; 0 for fewer than two grid points.

    Raises:
        ValueError: If the shapes differ, an entry is not finite, the grid
            decreases, or the integral is not finite.
    """
    if values.shape != grid.shape:
        raise ValueError("values and grid must have the same shape.")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(grid))):
        raise ValueError("values and grid must contain only finite real numbers.")
    if grid.size < 2:
        return 0.0

    if np.any(grid[1:] < grid[:-1]):
        raise ValueError("grid must be monotonically non-decreasing.")

    heights = values[:-1]
    # Multiply by the interval width before squaring the CDF difference:
    # a tiny probability squared alone can underflow on a very wide interval.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        terms = np.diff(grid) * heights
        if squared:
            terms *= heights
        area = float(np.sum(terms, dtype=float))
    if np.isfinite(area):
        return area

    # Opposite-sign finite endpoints can have an unrepresentable difference,
    # even when multiplying by a small height gives a finite area. On this
    # exceptional path accumulate the binary floats exactly before rounding.
    total = Fraction()
    for lower, upper, height in zip(grid[:-1], grid[1:], heights):
        width = Fraction(float(upper)) - Fraction(float(lower))
        value = Fraction(float(height))
        total += width * (value * value if squared else value)
    try:
        return float(total)
    except OverflowError as exc:
        raise ValueError(
            "Integrated area is not finite; check value/grid scales."
        ) from exc


def cvm_1d(
    x1: Iterable[float],
    x2: Iterable[float],
    w1: Iterable[float] | None = None,
    w2: Iterable[float] | None = None,
) -> float:
    """Return the Cramér-von Mises distance between two weighted samples.

    That is ``integral (F1(t) - F2(t))**2 dt`` for the weighted empirical
    CDFs of :func:`weighted_cdf`, integrated over ``t``, so it carries the
    units of the samples.

    Args:
        x1: First sample.
        x2: Second sample.
        w1 (default: None): Weights of ``x1``.
        w2 (default: None): Weights of ``x2``.

    Returns:
        The distance, ``>= 0``.

    Raises:
        ValueError: If a sample or its weights are invalid, or the integral
            is not finite.
    """
    return _distance(x1, x2, w1, w2, "cvm")


def emd_1d(
    x1: Iterable[float],
    x2: Iterable[float],
    w1: Iterable[float] | None = None,
    w2: Iterable[float] | None = None,
) -> float:
    """Return the earth mover's (Wasserstein-1) distance between two samples.

    That is ``integral |F1(t) - F2(t)| dt`` for the weighted empirical CDFs
    of :func:`weighted_cdf`.

    Args:
        x1: First sample.
        x2: Second sample.
        w1 (default: None): Weights of ``x1``.
        w2 (default: None): Weights of ``x2``.

    Returns:
        The distance, ``>= 0``, in the units of the samples.

    Raises:
        ValueError: If a sample or its weights are invalid, or the integral
            is not finite.
    """
    return _distance(x1, x2, w1, w2, "emd")
