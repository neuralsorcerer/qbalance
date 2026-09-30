# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _normalized_cumsum(weights: np.ndarray) -> np.ndarray:
    """Return the cumulative sum of normalized weights, ending at exactly 1.

    Pinning the last entry absorbs the round-off of summing the weights.
    """
    cdf = np.cumsum(weights, dtype=float)
    if cdf.size:
        cdf[-1] = 1.0
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
        values = np.asarray(x, dtype=float)
    except TypeError:
        # Some iterables (for example generators) are not directly array-like.
        # ``np.fromiter`` avoids creating an intermediate list for these cases.
        try:
            values = np.fromiter(x, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite real numbers.") from exc
    except ValueError as exc:
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
    values, weights = _to_np(x, w)
    order = np.argsort(values)
    xs_sorted = values[order]
    ws_sorted = weights[order]

    # Fast path: if support has fewer than two points there can be no
    # duplicates to aggregate.
    if xs_sorted.size < 2:
        return xs_sorted, _normalized_cumsum(ws_sorted)

    # Locate the start index of every run of identical sorted support values.
    # Reusing these boundaries avoids an extra uniqueness pass over data that is
    # already sorted and aggregates duplicate points in O(n).
    run_start = np.empty(xs_sorted.size, dtype=bool)
    run_start[0] = True
    run_start[1:] = xs_sorted[1:] != xs_sorted[:-1]
    first_idx = np.flatnonzero(run_start)
    if first_idx.size == xs_sorted.size:
        return xs_sorted, _normalized_cumsum(ws_sorted)

    # Aggregate repeated support points so downstream CDF evaluation does not
    # perform redundant binary-search work.
    unique_xs = xs_sorted[first_idx]
    summed_weights = np.add.reduceat(ws_sorted, first_idx)
    cdf = _normalized_cumsum(summed_weights)
    return unique_xs, cdf


def _cdf_on_grid(grid: np.ndarray, xs: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Evaluate the step CDF ``(xs, cdf)`` at every point of ``grid``.

    The CDF is right-continuous and 0 before its first support point.
    """
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two weighted empirical CDFs evaluated on a common grid.

    The grid is the union of both supports, on which both CDFs are exact
    step functions.

    Raises:
        ValueError: If either sample or its weights are invalid.
    """
    xs1, c1 = weighted_cdf(x1, w1)
    xs2, c2 = weighted_cdf(x2, w2)
    # Fast path for equal supports avoids unnecessary concatenation,
    # de-duplication, and repeated binary searches.
    if xs1.shape == xs2.shape and np.array_equal(xs1, xs2):
        return xs1, c1, c2

    # If supports do not overlap, concatenate directly; if they touch at one
    # endpoint, skip one duplicate boundary value.
    if xs1[-1] <= xs2[0]:
        grid = np.concatenate((xs1, xs2[1:] if xs1[-1] == xs2[0] else xs2))
    elif xs2[-1] <= xs1[0]:
        grid = np.concatenate((xs2, xs1[1:] if xs2[-1] == xs1[0] else xs1))
    else:
        # ``np.union1d`` uses optimized NumPy routines and remains faster than
        # Python-level merge loops for realistic support sizes.
        grid = np.union1d(xs1, xs2)
    cdf1 = _cdf_on_grid(grid, xs1, c1)
    cdf2 = _cdf_on_grid(grid, xs2, c2)
    return grid, cdf1, cdf2


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
    _, cdf1, cdf2 = _aligned_cdfs(x1, x2, w1, w2)
    return float(np.max(np.abs(cdf1 - cdf2)))


def _integrate_piecewise_constant(values: np.ndarray, grid: np.ndarray) -> float:
    """Integrate the step function taking ``values[i]`` on ``[grid[i], grid[i+1])``.

    Both factors are scaled by their maxima before multiplying, so large
    finite values and widths cannot overflow on the way to a finite result.

    Args:
        values: Function value on each grid interval (the last is unused).
        grid: Non-decreasing interval boundaries, the same length as
            ``values``.

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

    widths = np.diff(grid)
    if np.any(widths < 0.0):
        raise ValueError("grid must be monotonically non-decreasing.")

    left = values[:-1]
    max_left = float(np.max(np.abs(left)))
    max_width = float(np.max(np.abs(widths)))
    if max_left == 0.0 or max_width == 0.0:
        return 0.0

    # Normalize both factors to avoid transient overflow in the element-wise
    # products, then recover scale using an exponent-safe representation.
    scaled_left = left / max_left
    scaled_widths = widths / max_width

    # Use wider accumulation when the platform provides it; otherwise fall back
    # to float accumulation without changing semantics.
    sum_dtype = (
        np.longdouble if np.finfo(np.longdouble).max > np.finfo(float).max else float
    )
    scaled_sum = float(np.sum(scaled_left * scaled_widths, dtype=sum_dtype))
    if not np.isfinite(scaled_sum):
        raise ValueError("Integrated area is not finite; check value/grid scales.")

    left_mantissa, left_exp = np.frexp(max_left)
    width_mantissa, width_exp = np.frexp(max_width)
    scale_mantissa = left_mantissa * width_mantissa
    scale_exp = left_exp + width_exp

    with np.errstate(over="ignore", invalid="ignore"):
        area = float(np.ldexp(scale_mantissa * scaled_sum, scale_exp))
    if not np.isfinite(area):
        raise ValueError("Integrated area is not finite; check value/grid scales.")
    return area


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
    grid, cdf1, cdf2 = _aligned_cdfs(x1, x2, w1, w2)
    return _integrate_piecewise_constant((cdf1 - cdf2) ** 2, grid)


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
    grid, cdf1, cdf2 = _aligned_cdfs(x1, x2, w1, w2)
    return _integrate_piecewise_constant(np.abs(cdf1 - cdf2), grid)
