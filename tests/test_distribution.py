# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from qbalance.diagnostics import distribution
from qbalance.diagnostics.distribution import cvm_1d, emd_1d, ks_1d


def test_metrics_zero_for_identical():

    x = [0, 1, 2, 3, 4]
    assert emd_1d(x, x) == 0.0
    assert ks_1d(x, x) == 0.0
    assert cvm_1d(x, x) == 0.0


def test_ks_bounds():

    x1 = [0, 0, 0, 0]
    x2 = [1, 1, 1, 1]
    k = ks_1d(x1, x2)
    assert 0.0 <= k <= 1.0


def test_emd_positive():

    x1 = [0, 0, 0, 0]
    x2 = [2, 2, 2, 2]
    assert emd_1d(x1, x2) > 0


def test_weighted_metrics_expected_values():

    x1 = [0.0, 1.0]
    x2 = [1.0, 2.0]
    w1 = [0.75, 0.25]
    w2 = [0.25, 0.75]

    assert ks_1d(x1, x2, w1, w2) == pytest.approx(0.75)
    assert emd_1d(x1, x2, w1, w2) == pytest.approx(1.5)
    assert cvm_1d(x1, x2, w1, w2) == pytest.approx(1.125)


def test_invalid_inputs_raise_value_error():

    with pytest.raises(ValueError, match="non-empty"):
        ks_1d([], [1])

    with pytest.raises(ValueError, match="same length"):
        emd_1d([0, 1], [0, 1], w1=[1])

    with pytest.raises(ValueError, match="finite"):
        cvm_1d([0.0, float("nan")], [1.0, 2.0])

    with pytest.raises(ValueError, match="finite"):
        emd_1d([0.0, 1.0], [1.0, 2.0], w1=[0.5, float("inf")])


def test_non_1d_inputs_raise_value_error():

    with pytest.raises(ValueError, match="one-dimensional"):
        ks_1d([[0, 1], [2, 3]], [0, 1, 2, 3])

    with pytest.raises(ValueError, match="one-dimensional"):
        emd_1d([0, 1], [0, 1], w1=[[0.5, 0.5]])


def test_generator_input_is_supported_via_materialization():

    x = (float(i) for i in [0, 2, 4])
    values = distribution._as_1d_float_array(x, name="Input samples")
    np.testing.assert_allclose(values, np.array([0.0, 2.0, 4.0]))


def test_weights_fallback_to_uniform_when_clipped_total_is_zero():

    values, weights = distribution._to_np([1.0, 3.0], w=[-2.0, 0.0])
    np.testing.assert_allclose(values, np.array([1.0, 3.0]))
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


def test_weight_normalization_avoids_overflow_and_preserves_ratios():

    _, weights = distribution._to_np([0.0, 1.0], w=[1e308, 5e307])
    np.testing.assert_allclose(weights, np.array([2.0 / 3.0, 1.0 / 3.0]))


def test_weighted_cdf_with_huge_weights_remains_well_formed():

    xs, cdf = distribution.weighted_cdf([0.0, 1.0], [1e308, 5e307])
    np.testing.assert_allclose(xs, np.array([0.0, 1.0]))
    np.testing.assert_allclose(cdf, np.array([2.0 / 3.0, 1.0]))


def test_weighted_cdf_sorts_and_accumulates_weights():

    xs, cdf = distribution.weighted_cdf([3.0, 1.0, 2.0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(xs, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(cdf, np.array([0.3, 0.8, 1.0]))


def test_weighted_cdf_collapses_duplicate_support_points():

    xs, cdf = distribution.weighted_cdf([1.0, 1.0, 2.0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(xs, np.array([1.0, 2.0]))
    np.testing.assert_allclose(cdf, np.array([0.5, 1.0]))


def test_to_np_mixed_sign_weights_clip_and_renormalize_positive_mass():

    values, weights = distribution._to_np([0.0, 1.0, 2.0], w=[-3.0, 2.0, 6.0])
    np.testing.assert_allclose(values, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(weights, np.array([0.0, 0.25, 0.75]))


def test_weighted_cdf_all_duplicate_support_collapses_to_single_point():

    xs, cdf = distribution.weighted_cdf([5.0, 5.0, 5.0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(xs, np.array([5.0]))
    np.testing.assert_allclose(cdf, np.array([1.0]))


def test_cdf_on_grid_zero_before_support():

    grid = np.array([-1.0, 1.5, 3.0])
    xs = np.array([1.0, 2.0])
    cdf = np.array([0.4, 1.0])
    out = distribution._cdf_on_grid(grid, xs, cdf)
    np.testing.assert_allclose(out, np.array([0.0, 0.4, 1.0]))


def test_single_point_grid_integration_paths_return_zero():

    x = [1.0]
    assert emd_1d(x, x) == 0.0
    assert cvm_1d(x, x) == 0.0
    assert (
        distribution._integrate_piecewise_constant(np.array([1.0]), np.array([0.0]))
        == 0.0
    )


def test_integrate_piecewise_constant_rejects_invalid_inputs():

    with pytest.raises(ValueError, match="same shape"):
        distribution._integrate_piecewise_constant(
            np.array([1.0, 2.0]), np.array([0.0])
        )

    with pytest.raises(ValueError, match="monotonically"):
        distribution._integrate_piecewise_constant(
            np.array([1.0, 2.0]), np.array([1.0, 0.0])
        )


def test_integrate_piecewise_constant_singleton_grid_rejects_non_finite_values():

    with pytest.raises(ValueError, match="finite real numbers"):
        distribution._integrate_piecewise_constant(
            np.array([float("nan")]), np.array([float("inf")])
        )


def test_integrate_piecewise_constant_rejects_non_finite_inputs():

    with pytest.raises(ValueError, match="finite real numbers"):
        distribution._integrate_piecewise_constant(
            np.array([1.0, float("nan")]), np.array([0.0, 1.0])
        )

    with pytest.raises(ValueError, match="finite real numbers"):
        distribution._integrate_piecewise_constant(
            np.array([1.0, 2.0]), np.array([0.0, float("inf")])
        )


def test_integrate_piecewise_constant_rejects_non_finite_integral_result():

    with pytest.raises(ValueError, match="not finite"):
        distribution._integrate_piecewise_constant(
            np.array([1e308, 1e308]), np.array([0.0, 2.0])
        )


def test_integrate_piecewise_constant_handles_large_cancelling_terms_without_warning():

    area = distribution._integrate_piecewise_constant(
        np.array([1e308, -1e308, 0.0]), np.array([0.0, 2.0, 4.0])
    )
    assert area == 0.0


def test_aligned_cdfs_returns_expected_shapes_and_values():

    grid, cdf1, cdf2 = distribution._aligned_cdfs([0.0, 1.0], [1.0, 2.0])
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cdf1, np.array([0.5, 1.0, 1.0]))
    np.testing.assert_allclose(cdf2, np.array([0.0, 0.5, 1.0]))


def test_aligned_cdfs_with_matching_support_values():

    grid, cdf1, cdf2 = distribution._aligned_cdfs(
        [0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [0.2, 0.3, 0.5], [0.5, 0.2, 0.3]
    )
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cdf1, np.array([0.2, 0.5, 1.0]))
    np.testing.assert_allclose(cdf2, np.array([0.2, 0.5, 1.0]))


def test_aligned_cdfs_with_strictly_separated_supports_in_order():

    grid, cdf1, cdf2 = distribution._aligned_cdfs([0.0, 1.0], [2.0, 3.0])
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(cdf1, np.array([0.5, 1.0, 1.0, 1.0]))
    np.testing.assert_allclose(cdf2, np.array([0.0, 0.0, 0.5, 1.0]))


def test_aligned_cdfs_with_strictly_separated_supports_reverse_order():

    grid, cdf1, cdf2 = distribution._aligned_cdfs([3.0, 4.0], [0.0, 1.0])
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 3.0, 4.0]))
    np.testing.assert_allclose(cdf1, np.array([0.0, 0.0, 0.5, 1.0]))
    np.testing.assert_allclose(cdf2, np.array([0.5, 1.0, 1.0, 1.0]))


def test_aligned_cdfs_with_touching_support_boundary():

    grid, cdf1, cdf2 = distribution._aligned_cdfs([0.0, 1.0], [1.0, 2.0])
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cdf1, np.array([0.5, 1.0, 1.0]))
    np.testing.assert_allclose(cdf2, np.array([0.0, 0.5, 1.0]))


def test_aligned_cdfs_with_touching_support_boundary_reverse_order():

    grid, cdf1, cdf2 = distribution._aligned_cdfs([1.0, 2.0], [0.0, 1.0])
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cdf1, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(cdf2, np.array([0.5, 1.0, 1.0]))


def test_aligned_cdfs_grid_matches_union_for_random_inputs():

    rng = np.random.default_rng(1234)
    for _ in range(200):
        x1 = rng.integers(-5, 6, size=20).astype(float)
        x2 = rng.integers(-5, 6, size=20).astype(float)
        grid, _, _ = distribution._aligned_cdfs(x1, x2)

        xs1, _ = distribution.weighted_cdf(x1)
        xs2, _ = distribution.weighted_cdf(x2)
        np.testing.assert_array_equal(grid, np.union1d(xs1, xs2))


def test_weighted_cdf_ends_exactly_at_one_under_roundoff():

    xs, cdf = distribution.weighted_cdf([0.0, 1.0, 2.0], [0.1, 0.2, 0.7])
    np.testing.assert_allclose(xs, np.array([0.0, 1.0, 2.0]))
    assert cdf[-1] == 1.0


def test_aligned_cdfs_end_exactly_at_one():

    grid, cdf1, cdf2 = distribution._aligned_cdfs(
        [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]
    )
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0]))
    assert cdf1[-1] == 1.0
    assert cdf2[-1] == 1.0


def test_as_1d_float_array_rejects_non_numeric_content():

    with pytest.raises(ValueError, match="finite real numbers"):
        distribution._as_1d_float_array(["invalid"], name="Input samples")


def test_as_1d_float_array_rejects_non_numeric_generator_content():

    with pytest.raises(ValueError, match="finite real numbers"):
        distribution._as_1d_float_array(
            (value for value in ["invalid"]), name="Input samples"
        )
